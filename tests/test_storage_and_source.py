"""Tests for Storage backends and PrefetchedSource.

Tests both ShardStorage (disk) and MemoryStorage (in-memory), plus
PrefetchedSource with background producers and fallback.
"""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.storage import MemoryStorage, ShardStorage
from dataporter.prefetched_source import PrefetchedSource
from dataporter.transformable_dataset import TransformableDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text_shard(path: Path, texts: list[str]):
    schema = pa.schema([("text", pa.string())])
    table = pa.table({"text": texts}, schema=schema)
    pq.write_table(table, str(path), compression="zstd")


def _write_shards(tmp_path, n=3, docs_per_shard=20):
    for i in range(n):
        _write_text_shard(
            tmp_path / f"shard_{i:06d}.parquet",
            [f"shard{i}_doc{j}" for j in range(docs_per_shard)],
        )


# ---------------------------------------------------------------------------
# ShardStorage
# ---------------------------------------------------------------------------

class TestShardStorage:

    def test_read(self, tmp_path):
        _write_shards(tmp_path, n=2, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert len(s) == 20
        item = s.get(0)
        assert item is not None
        assert "shard0" in item["text"]

    def test_grows_with_new_shards(self, tmp_path):
        _write_shards(tmp_path, n=2, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert len(s) == 20

        _write_text_shard(tmp_path / "shard_000002.parquet", ["new"] * 10)
        time.sleep(0.02)
        assert len(s) == 30

    def test_deferred_eviction(self, tmp_path):
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        s = ShardStorage(tmp_path, refresh_interval=0.01)
        assert s.shard_count == 3

        s.schedule_eviction(tmp_path / "shard_000000.parquet")
        assert (tmp_path / "shard_000000.parquet").exists()  # not yet

        time.sleep(0.02)
        _ = len(s)  # triggers refresh
        assert not (tmp_path / "shard_000000.parquet").exists()
        assert s.shard_count == 2

    def test_auto_eviction(self, tmp_path):
        _write_shards(tmp_path, n=10, docs_per_shard=5)
        # Limit to ~half the total size
        total = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
        half_gb = (total * 0.55) / 1_073_741_824  # 55% of total = ~5.5 shards
        s = ShardStorage(tmp_path, refresh_interval=0.01, max_cache_gb=half_gb)
        assert s.shard_count <= 6
        assert s.shard_count >= 4

    def test_eviction_keeps_up_with_fast_writer(self, tmp_path):
        """Shards written between refreshes don't accumulate unbounded.

        Regression test: previously, shards accumulated to 73GB because
        eviction only ran during refresh() (30s interval). The prefetcher
        wrote 10-20 shards in that window, and over 12h they piled up.

        This test simulates the real scenario: shards are written
        continuously while get() is called regularly. At no point should
        the on-disk count exceed max_shards by more than a small margin.
        """
        _write_shards(tmp_path, n=5, docs_per_shard=5)
        # Size-based limit: allow ~5 shards worth (use 5.5x shard size)
        one_shard = list(tmp_path.glob("*.parquet"))[0].stat().st_size
        limit_gb = (one_shard * 5.5) / 1_073_741_824
        s = ShardStorage(tmp_path, refresh_interval=60, max_cache_gb=limit_gb)
        assert s.shard_count == 5

        max_seen = 5
        # Simulate 10 rounds of: write 3 shards, sleep to exceed 0.5s
        # throttle, then call get() to trigger eviction
        for batch in range(10):
            for j in range(3):
                idx = 5 + batch * 3 + j
                _write_text_shard(
                    tmp_path / f"shard_{idx:06d}.parquet",
                    [f"doc_{k}" for k in range(5)],
                )
            time.sleep(0.6)  # exceed 0.5s eviction throttle
            s.get(0)
            on_disk = len(list(tmp_path.glob("*.parquet")))
            max_seen = max(max_seen, on_disk)

        # Final count should be near max_shards
        final_count = len(list(tmp_path.glob("*.parquet")))
        assert final_count <= 8, (
            f"Final: {final_count} shards on disk (expected <= 8 with max_shards=5)"
        )
        # Peak: max_shards + up to 3 shards written between eviction checks
        assert max_seen <= 10, (
            f"Peak: {max_seen} shards on disk (expected <= 10). "
            f"Eviction isn't keeping up with writer."
        )

    def test_max_cache_gb_evicts_by_size(self, tmp_path):
        """max_cache_gb evicts oldest shards when total size exceeds limit."""
        # Write 10 shards, each ~1KB
        for i in range(10):
            _write_text_shard(
                tmp_path / f"shard_{i:06d}.parquet",
                [f"doc_{j}" for j in range(100)],
            )

        total_bytes = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
        half_gb = total_bytes / 2 / 1_073_741_824  # half the total in GB

        s = ShardStorage(tmp_path, refresh_interval=0.01, max_cache_gb=half_gb)
        # Initial refresh should evict to under the size limit
        remaining_bytes = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
        assert remaining_bytes <= total_bytes / 2 + 10000  # small tolerance
        assert s.shard_count < 10

    def test_wraps_index(self, tmp_path):
        _write_shards(tmp_path, n=1, docs_per_shard=5)
        s = ShardStorage(tmp_path)
        assert s.get(0) == s.get(5)  # wraps

    def test_contains(self, tmp_path):
        _write_shards(tmp_path, n=1, docs_per_shard=5)
        s = ShardStorage(tmp_path)
        assert 0 in s
        assert 4 in s
        assert 5 not in s

    def test_empty_dir(self, tmp_path):
        s = ShardStorage(tmp_path)
        assert len(s) == 0
        assert s.get(0) is None


# ---------------------------------------------------------------------------
# MemoryStorage
# ---------------------------------------------------------------------------

class TestMemoryStorage:

    def test_put_and_get(self):
        s = MemoryStorage(capacity=5)
        s.put(0, "hello")
        s.put(1, "world")
        assert s.get(0) == "hello"
        assert s.get(1) == "world"
        assert len(s) == 2

    def test_lru_eviction(self):
        s = MemoryStorage(capacity=3)
        s.put(0, "a")
        s.put(1, "b")
        s.put(2, "c")
        s.put(3, "d")  # evicts 0 (oldest)
        assert s.get(0) is None
        assert s.get(1) == "b"
        assert len(s) == 3

    def test_access_refreshes_lru(self):
        s = MemoryStorage(capacity=3)
        s.put(0, "a")
        s.put(1, "b")
        s.put(2, "c")
        _ = s.get(0)  # refresh 0 → now 1 is oldest
        s.put(3, "d")  # evicts 1 (oldest)
        assert s.get(0) == "a"  # still here
        assert s.get(1) is None  # evicted

    def test_contains(self):
        s = MemoryStorage()
        s.put(42, "data")
        assert 42 in s
        assert 99 not in s

    def test_unlimited_capacity(self):
        s = MemoryStorage(capacity=None)
        for i in range(1000):
            s.put(i, f"item_{i}")
        assert len(s) == 1000

    def test_explicit_evict(self):
        s = MemoryStorage()
        for i in range(5):
            s.put(i, f"item_{i}")
        evicted = s.evict(2)
        assert evicted == 2
        assert len(s) == 3

    def test_keys(self):
        s = MemoryStorage()
        s.put(10, "a")
        s.put(20, "b")
        assert set(s.keys()) == {10, 20}

    def test_clear(self):
        s = MemoryStorage()
        s.put(0, "a")
        s.clear()
        assert len(s) == 0

    def test_put_updates_existing(self):
        s = MemoryStorage()
        s.put(0, "old")
        s.put(0, "new")
        assert s.get(0) == "new"
        assert len(s) == 1


# ---------------------------------------------------------------------------
# PrefetchedSource — no producers (read-only)
# ---------------------------------------------------------------------------

class TestPrefetchedSourceReadOnly:

    def test_shard_storage(self, tmp_path):
        _write_shards(tmp_path, n=2, docs_per_shard=10)
        storage = ShardStorage(tmp_path)
        source = PrefetchedSource(storage)
        assert len(source) == 20
        assert "shard0" in source[0]["text"]

    def test_memory_storage(self):
        storage = MemoryStorage()
        storage.put(0, {"text": "hello"})
        storage.put(1, {"text": "world"})
        source = PrefetchedSource(storage)
        assert len(source) == 2
        assert source[0] == {"text": "hello"}

    def test_miss_raises_without_fallback(self):
        storage = MemoryStorage()
        source = PrefetchedSource(storage)
        with pytest.raises(IndexError):
            source[0]

    def test_fallback_on_miss(self):
        storage = MemoryStorage()
        source = PrefetchedSource(
            storage,
            fallback=lambda idx: {"text": f"fallback_{idx}"},
        )
        item = source[42]
        assert item == {"text": "fallback_42"}
        # Fallback results are NOT written back to storage (avoids race
        # conditions when multiple DataLoader workers share the storage)
        assert storage.get(42) is None


# ---------------------------------------------------------------------------
# PrefetchedSource — with background producers
# ---------------------------------------------------------------------------

class TestPrefetchedSourceWithProducers:

    def test_pre_filled_read(self):
        """Pre-filled MemoryStorage + PrefetchedSource (no producers)."""
        storage = MemoryStorage(capacity=100)
        for i in range(50):
            storage.put(i, {"value": i})

        source = PrefetchedSource(storage)
        assert len(source) == 50
        assert source[0] == {"value": 0}

    def test_multiple_sources_pre_filled(self):
        """Pre-filled from multiple sources."""
        storage = MemoryStorage(capacity=200)
        for i in range(50):
            storage.put(i, {"src": "a", "idx": i})
        for i in range(50, 100):
            storage.put(i, {"src": "b", "idx": i})

        source = PrefetchedSource(storage)
        assert source[0] == {"src": "a", "idx": 0}
        assert source[50] == {"src": "b", "idx": 50}

    def test_capacity_eviction(self):
        from dataporter.storage import SharedMemoryStorage
        storage = SharedMemoryStorage(
            capacity=10, max_frames=3, channels=3, height=8, width=8, max_keys=100,
        )

        def producer():
            import torch
            for i in range(50):
                yield i, torch.randint(0, 255, (3, 3, 8, 8), dtype=torch.uint8)
                time.sleep(0.01)

        source = PrefetchedSource(storage, producers=[producer])
        source.start()
        time.sleep(0.5)
        source.stop()

        assert len(storage) <= 10

    def test_stop_terminates_producers(self):
        from dataporter.storage import SharedMemoryStorage
        storage = SharedMemoryStorage(
            capacity=100, max_frames=3, channels=3, height=8, width=8, max_keys=200,
        )

        def slow_producer():
            import torch
            for i in range(10000):
                yield i % 100, torch.randint(0, 255, (3, 3, 8, 8), dtype=torch.uint8)
                time.sleep(0.1)

        source = PrefetchedSource(storage, producers=[slow_producer])
        source.start()
        time.sleep(0.3)
        source.stop()
        assert len(storage) < 100


# ---------------------------------------------------------------------------
# Integration: PrefetchedSource + TransformableDataset
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_text_pipeline(self, tmp_path):
        """ShardStorage → PrefetchedSource → TransformableDataset → DataLoader."""
        _write_shards(tmp_path, n=3, docs_per_shard=50)
        storage = ShardStorage(tmp_path)
        source = PrefetchedSource(storage)

        def transform(src, idx):
            item = src[idx]
            tokens = list(item["text"].encode("utf-8"))[:32]
            # Pad to fixed length for batching
            tokens = tokens + [0] * (32 - len(tokens))
            ids = torch.tensor(tokens, dtype=torch.long)
            return {"input_ids": ids}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

        total = sum(b["input_ids"].shape[0] for b in loader)
        assert total == 150

    def test_memory_pipeline(self):
        """MemoryStorage → PrefetchedSource → TransformableDataset → DataLoader."""
        storage = MemoryStorage()
        for i in range(100):
            storage.put(i, {"frames": torch.randn(3, 8, 8)})

        source = PrefetchedSource(storage)

        def transform(src, idx):
            return {"frames": src[idx]["frames"]}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 100

    def test_with_pre_filled_and_dataloader(self):
        """Pre-filled MemoryStorage + DataLoader."""
        storage = MemoryStorage(capacity=200)
        for i in range(100):
            storage.put(i, {"value": float(i)})

        source = PrefetchedSource(storage)

        def transform(src, idx):
            item = src[idx]
            return {"value": torch.tensor([item["value"]])}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

        batches = list(loader)
        assert len(batches) == 10


# ---------------------------------------------------------------------------
# Shuffle-from-available mode
# ---------------------------------------------------------------------------

class TestShuffleFromAvailable:

    def test_len_reflects_available_items(self):
        """__len__ returns count of items in storage, not total dataset."""
        storage = MemoryStorage(capacity=100)
        for i in range(10):
            storage.put(i * 100, {"value": i})  # sparse keys

        source = PrefetchedSource(storage, shuffle_available=True)
        assert len(source) == 10

    def test_zero_cache_misses(self):
        """Every __getitem__ call hits — no misses possible."""
        storage = MemoryStorage(capacity=50)
        for i in range(50):
            storage.put(i, {"frame": torch.randn(3, 8, 8)})

        source = PrefetchedSource(storage, shuffle_available=True)
        # Access every index in shuffled order
        for idx in range(len(source)):
            item = source[idx]
            assert "frame" in item  # always hits

    def test_maps_sequential_idx_to_storage_keys(self):
        """DataLoader idx 0,1,2 maps to actual storage keys."""
        storage = MemoryStorage()
        storage.put(100, {"ep": 100})
        storage.put(200, {"ep": 200})
        storage.put(300, {"ep": 300})

        source = PrefetchedSource(storage, shuffle_available=True)
        assert len(source) == 3
        # Each access returns a valid item from the storage
        for i in range(3):
            item = source[i]
            assert item["ep"] in {100, 200, 300}

    def test_grows_as_producer_fills(self):
        """__len__ increases as background producer adds items."""
        from dataporter.storage import SharedMemoryStorage
        storage = SharedMemoryStorage(
            capacity=30, max_frames=3, channels=3, height=8, width=8, max_keys=100,
        )

        def producer():
            import torch
            for i in range(20):
                yield i, torch.randint(0, 255, (3, 3, 8, 8), dtype=torch.uint8)
                time.sleep(0.02)

        source = PrefetchedSource(
            storage, producers=[producer], shuffle_available=True,
            min_available=5, keys_refresh_interval=0.01,
        )
        source.start()
        source.wait_for_min(timeout=10)

        assert len(source) >= 5
        time.sleep(0.5)
        assert len(source) >= 15
        source.stop()

    def test_wraps_index_modulo(self):
        """Indices beyond len() wrap around (DataLoader may overshoot)."""
        storage = MemoryStorage()
        storage.put(0, {"v": "a"})
        storage.put(1, {"v": "b"})

        source = PrefetchedSource(storage, shuffle_available=True)
        # With 2 items, idx=2 wraps to same position as idx=0
        item_0 = source[0]
        item_2 = source[2]
        # Both should be valid items (not errors)
        assert "v" in item_0
        assert "v" in item_2

    def test_survives_eviction_during_read(self):
        """If a key is evicted between __len__ and __getitem__, retry works."""
        storage = MemoryStorage(capacity=5)
        for i in range(5):
            storage.put(i, {"value": i})

        source = PrefetchedSource(storage, shuffle_available=True, keys_refresh_interval=0.01)
        assert len(source) == 5

        # Evict one item externally
        storage.evict(1)
        time.sleep(0.02)  # let interval-based refresh pick up the change

        # Should still work — retry refreshes key list
        items = [source[i] for i in range(len(source))]
        assert len(items) == 4

    def test_with_dataloader(self):
        """Full integration: MemoryStorage + shuffle + DataLoader."""
        storage = MemoryStorage(capacity=100)
        for i in range(50):
            storage.put(i, {"frames": torch.randn(3, 8, 8)})

        source = PrefetchedSource(storage, shuffle_available=True)

        def transform(src, idx):
            return {"frames": src[idx]["frames"]}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 50

    def test_empty_buffer_raises(self):
        """Accessing empty buffer raises IndexError."""
        storage = MemoryStorage()
        source = PrefetchedSource(storage, shuffle_available=True)
        with pytest.raises(IndexError, match="No data available"):
            source[0]

    def test_wait_for_min(self):
        """wait_for_min blocks until enough items are loaded."""
        from dataporter.storage import SharedMemoryStorage
        storage = SharedMemoryStorage(
            capacity=30, max_frames=3, channels=3, height=8, width=8, max_keys=100,
        )

        def slow_producer():
            import torch
            for i in range(20):
                yield i, torch.randint(0, 255, (3, 3, 8, 8), dtype=torch.uint8)
                time.sleep(0.05)

        source = PrefetchedSource(
            storage, producers=[slow_producer],
            shuffle_available=True, min_available=10,
            keys_refresh_interval=0.01,
        )
        source.start()
        source.wait_for_min(timeout=10)
        assert len(source) >= 10
        source.stop()


# ---------------------------------------------------------------------------
# Coverage: all items eventually seen
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_all_items_seen_over_time(self):
        """With a cycling producer, all items eventually enter the buffer."""
        from dataporter.storage import SharedMemoryStorage
        n_items = 50
        buffer_size = 15

        storage = SharedMemoryStorage(
            capacity=buffer_size, max_frames=3, channels=3, height=8, width=8,
            max_keys=n_items * 2,
        )
        seen_keys = set()

        def cycling_producer():
            import random, torch
            rng = random.Random(42)
            while True:
                order = list(range(n_items))
                rng.shuffle(order)
                for key in order:
                    yield key, torch.randint(0, 255, (3, 3, 8, 8), dtype=torch.uint8)
                    time.sleep(0.001)

        source = PrefetchedSource(
            storage, producers=[cycling_producer],
            shuffle_available=True, min_available=buffer_size,
            keys_refresh_interval=0.01,
        )
        source.start()
        source.wait_for_min(timeout=10)

        for _ in range(200):
            keys = storage.keys()
            seen_keys.update(keys)
            time.sleep(0.01)

        source.stop()

        coverage = len(seen_keys) / n_items
        assert coverage >= 0.5, (
            f"Only {coverage:.0%} coverage — expected >= 50%. "
            f"Seen {len(seen_keys)}/{n_items}"
        )


# ---------------------------------------------------------------------------
# Process-mode PrefetchedSource
# ---------------------------------------------------------------------------

class TestProcessMode:

    def test_process_producer_fills_shared_storage(self):
        """use_process=True: producer runs in forked child, fills SharedMemoryStorage."""
        from dataporter.storage import SharedMemoryStorage

        storage = SharedMemoryStorage(
            capacity=20, max_frames=5, channels=3, height=8, width=8, max_keys=100,
        )

        def producer():
            import torch
            for i in range(10):
                frames = torch.randint(0, 255, (5, 3, 8, 8), dtype=torch.uint8)
                yield i, frames
                time.sleep(0.01)

        source = PrefetchedSource(
            storage, producers=[producer], shuffle_available=True,
            min_available=5, keys_refresh_interval=0.01,
        )
        source.start()
        source.wait_for_min(timeout=10)

        assert len(storage) >= 5
        item = source[0]
        assert "frames" in item
        source.stop()

    def test_process_mode_stop_terminates(self):
        """Process-mode producers terminate cleanly on stop()."""
        from dataporter.storage import SharedMemoryStorage

        storage = SharedMemoryStorage(
            capacity=100, max_frames=5, channels=3, height=8, width=8, max_keys=200,
        )

        def slow_producer():
            import torch
            for i in range(10000):
                frames = torch.randint(0, 255, (5, 3, 8, 8), dtype=torch.uint8)
                yield i % 100, frames
                time.sleep(0.1)

        source = PrefetchedSource(
            storage, producers=[slow_producer],
        )
        source.start()
        time.sleep(0.3)
        source.stop()  # should terminate within timeout
        assert not any(w.is_alive() for w in source._workers)

    def test_default_process_mode_with_shared_storage(self):
        """Producers always run as processes (no thread mode)."""
        from dataporter.storage import SharedMemoryStorage

        storage = SharedMemoryStorage(
            capacity=20, max_frames=5, channels=3, height=8, width=8, max_keys=100,
        )

        def producer():
            import torch
            for i in range(10):
                frames = torch.randint(0, 255, (5, 3, 8, 8), dtype=torch.uint8)
                yield i, frames
                time.sleep(0.01)

        source = PrefetchedSource(
            storage, producers=[producer], shuffle_available=True,
            min_available=5, keys_refresh_interval=0.01,
        )
        source.start()
        source.wait_for_min(timeout=10)

        assert len(storage) >= 5
        source.stop()

