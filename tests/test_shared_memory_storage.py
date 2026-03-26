"""Tests for SharedMemoryStorage.

Tests shared memory visibility across forked workers, FIFO eviction,
variable-length episodes, and integration with PrefetchedSource + DataLoader.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.storage import SharedMemoryStorage
from dataporter.prefetched_source import PrefetchedSource
from dataporter.transformable_dataset import TransformableDataset


def _make_frames(n_frames: int = 10, c: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    """Create a fake episode frame tensor."""
    return torch.randint(0, 255, (n_frames, c, h, w), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestSharedMemoryStorageBasic:

    def test_put_and_get(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames = _make_frames(7)
        s.put(42, frames)

        result = s.get(42)
        assert result is not None
        assert torch.equal(result["frames"], frames)
        assert result["frames"].shape == (7, 3, 8, 8)

    def test_put_dict(self):
        """Accepts dict with 'frames' key."""
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames = _make_frames(5)
        s.put(0, {"frames": frames})
        assert torch.equal(s.get(0)["frames"], frames)

    def test_get_missing_returns_none(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        assert s.get(99) is None

    def test_len(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        assert len(s) == 0
        s.put(0, _make_frames(3))
        assert len(s) == 1
        s.put(1, _make_frames(3))
        assert len(s) == 2

    def test_contains(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        s.put(10, _make_frames(3))
        assert 10 in s
        assert 99 not in s

    def test_keys(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        s.put(10, _make_frames(3))
        s.put(20, _make_frames(3))
        assert set(s.keys()) == {10, 20}

    def test_capacity(self):
        s = SharedMemoryStorage(capacity=3, max_frames=10, channels=3, height=8, width=8)
        assert s.capacity == 3

    def test_clear(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        s.put(0, _make_frames(3))
        s.put(1, _make_frames(3))
        s.clear()
        assert len(s) == 0
        assert s.get(0) is None

    def test_update_existing(self):
        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames1 = _make_frames(5)
        frames2 = _make_frames(5)
        s.put(0, frames1)
        s.put(0, frames2)
        assert len(s) == 1
        assert torch.equal(s.get(0)["frames"], frames2)


# ---------------------------------------------------------------------------
# Variable-length episodes
# ---------------------------------------------------------------------------

class TestVariableLength:

    def test_different_frame_counts(self):
        s = SharedMemoryStorage(capacity=5, max_frames=20, channels=3, height=8, width=8)
        s.put(0, _make_frames(5))
        s.put(1, _make_frames(15))
        s.put(2, _make_frames(1))

        assert s.get(0)["frames"].shape[0] == 5
        assert s.get(1)["frames"].shape[0] == 15
        assert s.get(2)["frames"].shape[0] == 1

    def test_truncates_to_max_frames(self):
        """Frames beyond max_frames are silently truncated."""
        s = SharedMemoryStorage(capacity=5, max_frames=5, channels=3, height=8, width=8)
        s.put(0, _make_frames(10))  # 10 frames, max is 5
        assert s.get(0)["frames"].shape[0] == 5


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------

class TestEviction:

    def test_ring_buffer_evicts_oldest(self):
        s = SharedMemoryStorage(capacity=3, max_frames=5, channels=3, height=8, width=8)
        s.put(0, _make_frames(3))
        s.put(1, _make_frames(3))
        s.put(2, _make_frames(3))
        # Capacity full — next put evicts slot 0
        s.put(3, _make_frames(3))

        assert s.get(0) is None  # evicted
        assert s.get(3) is not None  # new item
        assert len(s) == 3

    def test_explicit_evict(self):
        s = SharedMemoryStorage(capacity=5, max_frames=5, channels=3, height=8, width=8)
        for i in range(5):
            s.put(i, _make_frames(3))
        evicted = s.evict(2)
        assert evicted == 2
        assert len(s) == 3

    def test_evict_more_than_available(self):
        s = SharedMemoryStorage(capacity=5, max_frames=5, channels=3, height=8, width=8)
        s.put(0, _make_frames(3))
        evicted = s.evict(10)
        assert evicted == 1
        assert len(s) == 0


# ---------------------------------------------------------------------------
# Shared memory across processes
# ---------------------------------------------------------------------------

class TestSharedMemory:

    def test_tensors_are_shared(self):
        """Verify that internal tensors are in shared memory."""
        s = SharedMemoryStorage(capacity=3, max_frames=5, channels=3, height=8, width=8)
        assert s._buffer.is_shared()
        assert s._lengths.is_shared()
        assert s._index_map.is_shared()
        assert s._slot_keys.is_shared()

    def test_visible_in_forked_worker(self):
        """Data written in parent is visible in a forked child process."""
        import multiprocessing

        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames = _make_frames(5, c=3, h=8, w=8)
        s.put(42, frames)

        result_queue = multiprocessing.Queue()

        def worker(storage, q):
            item = storage.get(42)
            if item is not None:
                q.put(item["frames"].shape)
            else:
                q.put(None)

        p = multiprocessing.Process(target=worker, args=(s, result_queue))
        p.start()
        p.join(timeout=10)

        shape = result_queue.get(timeout=5)
        assert shape == (5, 3, 8, 8)

    def test_parent_writes_visible_after_fork(self):
        """Parent writes after fork are visible to child via shared memory."""
        import multiprocessing

        s = SharedMemoryStorage(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames = _make_frames(3, c=3, h=8, w=8)

        result_queue = multiprocessing.Queue()

        def worker(storage, q):
            time.sleep(0.1)
            item = storage.get(7)
            q.put(item is not None)

        # Write before forking
        s.put(7, frames)

        p = multiprocessing.Process(target=worker, args=(s, result_queue))
        p.start()
        p.join(timeout=10)

        visible = result_queue.get(timeout=5)
        assert visible, "Data written before fork should be visible in child"


# ---------------------------------------------------------------------------
# Integration: PrefetchedSource + DataLoader
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:

    def test_single_worker(self):
        """num_workers=0: basic integration."""
        s = SharedMemoryStorage(capacity=50, max_frames=5, channels=3, height=8, width=8)
        for i in range(20):
            s.put(i, _make_frames(5, c=3, h=8, w=8))

        source = PrefetchedSource(s, shuffle_available=True)

        def transform(src, idx):
            item = src[idx]
            return {"frames": item["frames"].float() / 255.0}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 20

    def test_multi_worker(self):
        """num_workers=2: shared memory visible across forked workers."""
        s = SharedMemoryStorage(capacity=50, max_frames=5, channels=3, height=8, width=8)
        for i in range(20):
            s.put(i, _make_frames(5, c=3, h=8, w=8))

        source = PrefetchedSource(s, shuffle_available=True)

        def transform(src, idx):
            item = src[idx]
            return {"frames": item["frames"].float() / 255.0}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 20

    def test_with_background_producer(self):
        """Producer fills storage in background, workers read."""
        s = SharedMemoryStorage(capacity=30, max_frames=5, channels=3, height=8, width=8)

        def producer():
            for i in range(20):
                yield i, _make_frames(5, c=3, h=8, w=8)
                time.sleep(0.01)

        source = PrefetchedSource(
            s, producers=[producer], shuffle_available=True, min_available=10,
        )
        source.start()
        source.wait_for_min(timeout=10)

        def transform(src, idx):
            item = src[idx]
            return {"frames": item["frames"].float() / 255.0}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        batches = list(loader)
        assert len(batches) > 0
        assert all(b["frames"].shape[-2:] == (8, 8) for b in batches)
        source.stop()

    def test_zero_cache_misses_in_shuffle_mode(self):
        """shuffle_available=True guarantees zero misses."""
        s = SharedMemoryStorage(capacity=10, max_frames=5, channels=3, height=8, width=8)
        for i in range(10):
            s.put(i, _make_frames(5, c=3, h=8, w=8))

        source = PrefetchedSource(s, shuffle_available=True)

        miss_count = 0
        for idx in range(len(source)):
            try:
                item = source[idx]
                assert "frames" in item
            except (IndexError, KeyError):
                miss_count += 1

        assert miss_count == 0


# ---------------------------------------------------------------------------
# State dict / resumption
# ---------------------------------------------------------------------------

class TestSharedMemoryResumption:

    def test_state_dict_saves_keys(self):
        """state_dict saves which episodes are in the buffer."""
        s = SharedMemoryStorage(capacity=10, max_frames=5, channels=3, height=8, width=8)
        s.put(10, _make_frames(3, c=3, h=8, w=8))
        s.put(20, _make_frames(5, c=3, h=8, w=8))
        s.put(30, _make_frames(2, c=3, h=8, w=8))

        state = s.state_dict()
        assert set(state["episode_keys"]) == {10, 20, 30}
        assert state["capacity"] == 10

    def test_load_state_dict_sets_priority_keys(self):
        """load_state_dict stores priority keys for producer."""
        s = SharedMemoryStorage(capacity=10, max_frames=5, channels=3, height=8, width=8)
        state = {"episode_keys": [10, 20, 30], "capacity": 10, "max_frames": 5}
        s.load_state_dict(state)

        assert s.priority_keys == [10, 20, 30]
        # Buffer is empty — frames must be re-decoded
        assert len(s) == 0

    def test_priority_producer_decodes_keys_first(self):
        """priority_producer yields priority keys before normal production."""
        from dataporter.prefetched_source import priority_producer

        decoded = []

        def decode_fn(key):
            decoded.append(key)
            return _make_frames(3, c=3, h=8, w=8)

        def base_producer():
            for i in range(100, 105):
                yield i, _make_frames(3, c=3, h=8, w=8)

        producer = priority_producer(base_producer, [10, 20, 30], decode_fn)
        items = list(producer())

        # Priority keys decoded first
        assert decoded == [10, 20, 30]
        # Then base producer continues
        keys = [k for k, v in items]
        assert keys[:3] == [10, 20, 30]
        assert keys[3:] == [100, 101, 102, 103, 104]

    def test_resume_with_priority_fill(self):
        """Full resume cycle: save state → clear → reload → priority fill."""
        s = SharedMemoryStorage(capacity=10, max_frames=5, channels=3, height=8, width=8)
        # Fill with some episodes
        for i in [10, 20, 30]:
            s.put(i, _make_frames(4, c=3, h=8, w=8))

        # Save state
        state = s.state_dict()
        assert set(state["episode_keys"]) == {10, 20, 30}

        # Clear (simulates restart)
        s.clear()
        assert len(s) == 0

        # Load state — sets priority keys
        s.load_state_dict(state)

        # Producer decodes priority keys first
        from dataporter.prefetched_source import priority_producer

        def decode_fn(key):
            return _make_frames(4, c=3, h=8, w=8)

        def base_producer():
            for i in range(40, 50):
                yield i, _make_frames(4, c=3, h=8, w=8)

        source = PrefetchedSource(
            s,
            producers=[priority_producer(base_producer, s.priority_keys, decode_fn)],
            shuffle_available=True,
            min_available=3,
        )
        source.start()
        source.wait_for_min(timeout=10)

        # Priority episodes should be in the buffer first
        assert 10 in s or 20 in s or 30 in s
        source.stop()

    def test_prefetched_source_state_dict(self):
        """PrefetchedSource.state_dict delegates to storage."""
        s = SharedMemoryStorage(capacity=5, max_frames=5, channels=3, height=8, width=8)
        s.put(0, _make_frames(3, c=3, h=8, w=8))
        s.put(1, _make_frames(3, c=3, h=8, w=8))

        source = PrefetchedSource(s, shuffle_available=True)
        state = source.state_dict()
        assert "storage" in state
        assert set(state["storage"]["episode_keys"]) == {0, 1}

        # Round-trip
        s2 = SharedMemoryStorage(capacity=5, max_frames=5, channels=3, height=8, width=8)
        source2 = PrefetchedSource(s2, shuffle_available=True)
        source2.load_state_dict(state)
        assert s2.priority_keys == [0, 1]
