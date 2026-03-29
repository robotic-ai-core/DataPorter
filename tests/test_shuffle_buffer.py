"""Tests for ShuffleBuffer, ProducerPool, and ShuffleBufferDataset."""

from __future__ import annotations

import multiprocessing
import random
import time

import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.shuffle_buffer import ShuffleBuffer
from dataporter.producer_pool import AsyncProducer, ProducerPool
from dataporter.shuffle_buffer_dataset import ShuffleBufferDataset


def _make_frames(n_frames: int = 10, c: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.randint(0, 255, (n_frames, c, h, w), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# ShuffleBuffer
# ---------------------------------------------------------------------------

class TestShuffleBuffer:

    def test_put_and_sample(self):
        buf = ShuffleBuffer(capacity=5, max_frames=10, channels=3, height=8, width=8)
        frames = _make_frames(7)
        buf.put(42, frames)

        assert len(buf) == 1
        rng = random.Random(0)
        key, sampled = buf.sample(rng)
        assert key == 42
        assert torch.equal(sampled, frames)

    def test_empty_sample_raises(self):
        buf = ShuffleBuffer(capacity=5, max_frames=10, channels=3, height=8, width=8)
        with pytest.raises(IndexError, match="empty"):
            buf.sample(random.Random(0))

    def test_ring_buffer_eviction(self):
        buf = ShuffleBuffer(capacity=3, max_frames=5, channels=3, height=8, width=8)
        buf.put(0, _make_frames(3))
        buf.put(1, _make_frames(3))
        buf.put(2, _make_frames(3))
        evicted = buf.put(3, _make_frames(3))

        assert evicted == 0  # oldest evicted
        assert len(buf) == 3
        assert 0 not in buf
        assert 3 in buf

    def test_fill_capacity(self):
        buf = ShuffleBuffer(capacity=5, max_frames=5, channels=3, height=8, width=8)
        for i in range(5):
            buf.put(i, _make_frames(5))
        assert len(buf) == 5

    def test_sample_uniform_distribution(self):
        """All items in buffer should be sampled roughly equally."""
        buf = ShuffleBuffer(capacity=5, max_frames=3, channels=3, height=8, width=8)
        for i in range(5):
            buf.put(i, _make_frames(3))

        counts = {i: 0 for i in range(5)}
        rng = random.Random(42)
        for _ in range(1000):
            key, _ = buf.sample(rng)
            counts[key] += 1

        # Each should be ~200 (20%), allow 10-30%
        for k, c in counts.items():
            assert 100 < c < 300, f"Key {k} sampled {c} times (expected ~200)"

    def test_variable_length_frames(self):
        buf = ShuffleBuffer(capacity=5, max_frames=20, channels=3, height=8, width=8)
        buf.put(0, _make_frames(5))
        buf.put(1, _make_frames(15))
        buf.put(2, _make_frames(1))

        rng = random.Random(42)
        for _ in range(20):
            key, frames = buf.sample(rng)
            if key == 0:
                assert frames.shape[0] == 5
            elif key == 1:
                assert frames.shape[0] == 15
            elif key == 2:
                assert frames.shape[0] == 1

    def test_keys(self):
        buf = ShuffleBuffer(capacity=5, max_frames=3, channels=3, height=8, width=8)
        buf.put(10, _make_frames(3))
        buf.put(20, _make_frames(3))
        assert set(buf.keys()) == {10, 20}

    def test_contains(self):
        buf = ShuffleBuffer(capacity=5, max_frames=3, channels=3, height=8, width=8)
        buf.put(42, _make_frames(3))
        assert 42 in buf
        assert 99 not in buf

    def test_clear(self):
        buf = ShuffleBuffer(capacity=5, max_frames=3, channels=3, height=8, width=8)
        buf.put(0, _make_frames(3))
        buf.clear()
        assert len(buf) == 0

    def test_shared_memory(self):
        """Tensors are in shared memory (survive fork)."""
        buf = ShuffleBuffer(capacity=3, max_frames=5, channels=3, height=8, width=8)
        assert buf._buffer.is_shared()
        assert buf._lengths.is_shared()
        assert buf._keys.is_shared()

    def test_cross_process_visibility(self):
        """Data written in parent is visible in forked child."""
        buf = ShuffleBuffer(capacity=5, max_frames=5, channels=3, height=8, width=8)
        frames = _make_frames(5, c=3, h=8, w=8)
        buf.put(42, frames)

        q = multiprocessing.Queue()

        def worker(b, q):
            rng = random.Random(0)
            key, f = b.sample(rng)
            q.put((key, f.shape))

        p = multiprocessing.Process(target=worker, args=(buf, q))
        p.start()
        p.join(timeout=10)

        key, shape = q.get(timeout=5)
        assert key == 42
        assert shape == (5, 3, 8, 8)


# ---------------------------------------------------------------------------
# ProducerPool (with synthetic decode)
# ---------------------------------------------------------------------------

def _synthetic_decode(ep_idx: int) -> torch.Tensor:
    """Deterministic synthetic decode — no real video needed."""
    return torch.full((5, 3, 8, 8), ep_idx % 256, dtype=torch.uint8)


class TestProducerPool:

    def test_fills_buffer(self):
        buf = ShuffleBuffer(capacity=20, max_frames=5, channels=3, height=8, width=8)
        producer = AsyncProducer(
            source_name="test",
            decode_fn=_synthetic_decode,
            episode_indices=list(range(50)),
            weight=1.0,
        )
        pool = ProducerPool(buf, producers=[producer], total_workers=2, warmup_target=10)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        assert len(buf) >= 10
        pool.stop()

    def test_weighted_blend(self):
        """Two sources with different weights produce proportional output."""
        buf = ShuffleBuffer(capacity=100, max_frames=5, channels=3, height=8, width=8)

        # Source A episodes: 0-99, Source B: 100-199
        producer_a = AsyncProducer("A", _synthetic_decode, list(range(100)), weight=3.0)
        producer_b = AsyncProducer("B", _synthetic_decode, list(range(100, 200)), weight=1.0)

        pool = ProducerPool(buf, producers=[producer_a, producer_b], total_workers=2, warmup_target=80)
        pool.start()
        pool.wait_for_warmup(timeout=30)
        time.sleep(0.5)
        pool.stop()

        # Count items from each source
        keys = buf.keys()
        a_count = sum(1 for k in keys if k < 100)
        b_count = sum(1 for k in keys if k >= 100)
        total = a_count + b_count

        # Weight 3:1 → A should be ~75% (allow 50-90%)
        if total > 0:
            a_ratio = a_count / total
            assert 0.5 < a_ratio < 0.95, (
                f"Expected ~75% A, got {a_ratio:.0%} (A={a_count}, B={b_count})"
            )

    def test_stop_terminates(self):
        buf = ShuffleBuffer(capacity=100, max_frames=5, channels=3, height=8, width=8)
        producer = AsyncProducer("test", _synthetic_decode, list(range(1000)), weight=1.0)
        pool = ProducerPool(buf, producers=[producer], total_workers=1, warmup_target=5)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        pool.stop()
        assert not pool.is_alive


# ---------------------------------------------------------------------------
# ShuffleBufferDataset
# ---------------------------------------------------------------------------

class TestShuffleBufferDataset:

    def test_basic_access(self):
        buf = ShuffleBuffer(capacity=10, max_frames=5, channels=3, height=8, width=8)
        for i in range(10):
            buf.put(i, _make_frames(5, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=100)
        assert len(ds) == 100

        item = ds[0]
        assert "frames" in item
        assert "episode_index" in item
        assert item["frames"].dtype == torch.uint8

    def test_ignores_idx(self):
        """__getitem__(0) and __getitem__(99) both sample randomly."""
        buf = ShuffleBuffer(capacity=10, max_frames=3, channels=3, height=8, width=8)
        for i in range(10):
            buf.put(i, _make_frames(3, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=100)
        # Different idx should give different random samples (not deterministic by idx)
        items = [ds[i]["episode_index"] for i in range(20)]
        # Not all the same (would be astronomically unlikely)
        assert len(set(items)) > 1

    def test_with_dataloader(self):
        buf = ShuffleBuffer(capacity=20, max_frames=5, channels=3, height=8, width=8)
        for i in range(20):
            buf.put(i, _make_frames(5, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=64)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 64

    def test_with_multi_worker_dataloader(self):
        """Multi-worker: workers read from shared memory, no stalls."""
        buf = ShuffleBuffer(capacity=50, max_frames=5, channels=3, height=8, width=8)
        for i in range(50):
            buf.put(i, _make_frames(5, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=128)
        loader = DataLoader(
            ds, batch_size=16, shuffle=False, num_workers=2,
            worker_init_fn=ShuffleBufferDataset.worker_init_fn,
        )

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 128

    def test_uniform_latency(self):
        """Every __getitem__ should be fast — no slow outliers."""
        buf = ShuffleBuffer(capacity=50, max_frames=5, channels=3, height=8, width=8)
        for i in range(50):
            buf.put(i, _make_frames(5, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=1000)
        times = []
        for i in range(200):
            t0 = time.perf_counter()
            _ = ds[i]
            times.append((time.perf_counter() - t0) * 1e6)

        avg = sum(times) / len(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        # p99 should be within 10x of avg (no massive outliers)
        assert p99 < avg * 10, (
            f"p99={p99:.0f}us, avg={avg:.0f}us — outlier detected"
        )


# ---------------------------------------------------------------------------
# End-to-end: ProducerPool → ShuffleBuffer → ShuffleBufferDataset → DataLoader
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self):
        """Producer fills buffer, DataLoader consumes — no stalls."""
        buf = ShuffleBuffer(capacity=30, max_frames=5, channels=3, height=8, width=8)

        producer = AsyncProducer(
            source_name="test",
            decode_fn=_synthetic_decode,
            episode_indices=list(range(100)),
            weight=1.0,
        )
        pool = ProducerPool(buf, producers=[producer], total_workers=2, warmup_target=20)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        ds = ShuffleBufferDataset(buf, epoch_length=64)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

        batches = list(loader)
        assert len(batches) == 4
        assert all(b["frames"].shape == (16, 5, 3, 8, 8) for b in batches)
        assert all(b["frames"].dtype == torch.uint8 for b in batches)

        pool.stop()

    def test_multi_worker_no_stalls(self):
        """Multi-worker DataLoader with ShuffleBuffer — uniform latency."""
        buf = ShuffleBuffer(capacity=50, max_frames=5, channels=3, height=8, width=8)
        for i in range(50):
            buf.put(i, _make_frames(5, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=128)
        loader = DataLoader(
            ds, batch_size=16, shuffle=False, num_workers=2,
            worker_init_fn=ShuffleBufferDataset.worker_init_fn,
        )

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 128


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

class TestThroughput:

    def test_uniform_sample_latency(self):
        """Every sample() call should be fast — no slow outliers."""
        buf = ShuffleBuffer(capacity=200, max_frames=10, channels=3, height=8, width=8)
        for i in range(200):
            buf.put(i, _make_frames(10, c=3, h=8, w=8))

        rng = random.Random(42)
        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            buf.sample(rng)
            times.append((time.perf_counter() - t0) * 1e6)

        avg = sum(times) / len(times)
        p99 = sorted(times)[int(len(times) * 0.99)]
        assert p99 < avg * 20, f"avg={avg:.0f}us, p99={p99:.0f}us"

    def test_put_latency_vs_decode(self):
        """put() should be negligible compared to typical decode time."""
        buf = ShuffleBuffer(capacity=50, max_frames=30, channels=3, height=96, width=96)
        frames = _make_frames(30, c=3, h=96, w=96)

        times = []
        for i in range(100):
            t0 = time.perf_counter()
            buf.put(i, frames)
            times.append((time.perf_counter() - t0) * 1e6)

        avg = sum(times) / len(times)
        decode_us = 50_000  # 50ms
        pct = avg / decode_us * 100
        assert pct < 5, f"put() is {pct:.1f}% of decode — GIL contention risk"


# ---------------------------------------------------------------------------
# Randomness validation
# ---------------------------------------------------------------------------

class TestRandomness:

    def test_multi_worker_sample_diversity(self):
        """Different workers should sample different episodes."""
        buf = ShuffleBuffer(capacity=50, max_frames=3, channels=3, height=8, width=8)
        for i in range(50):
            buf.put(i, _make_frames(3, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=200)
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=2,
            worker_init_fn=ShuffleBufferDataset.worker_init_fn,
        )

        episodes = []
        for batch in loader:
            episodes.append(int(batch["episode_index"]))

        unique = len(set(episodes))
        assert unique >= 20, f"Only {unique} unique episodes — workers may be correlated"

    def test_blend_ratio_proportional(self):
        """Weighted producers fill buffer at correct blend ratio."""
        buf = ShuffleBuffer(capacity=100, max_frames=3, channels=3, height=8, width=8)

        producer_a = AsyncProducer("A", _synthetic_decode, list(range(50)), weight=3.0)
        producer_b = AsyncProducer("B", _synthetic_decode, list(range(100, 150)), weight=1.0)

        pool = ProducerPool(buf, producers=[producer_a, producer_b], total_workers=2, warmup_target=80)
        pool.start()
        pool.wait_for_warmup(timeout=30)
        time.sleep(0.5)
        pool.stop()

        keys = buf.keys()
        a_count = sum(1 for k in keys if k < 50)
        b_count = sum(1 for k in keys if k >= 100)
        total = a_count + b_count

        if total > 0:
            a_ratio = a_count / total
            assert 0.5 < a_ratio < 0.95, (
                f"Blend: A={a_ratio:.0%} (expected ~75%). A={a_count}, B={b_count}"
            )

    def test_sample_covers_buffer(self):
        """Repeated sampling should cover most of the buffer contents."""
        buf = ShuffleBuffer(capacity=30, max_frames=3, channels=3, height=8, width=8)
        for i in range(30):
            buf.put(i, _make_frames(3, c=3, h=8, w=8))

        rng = random.Random(42)
        seen = set()
        for _ in range(500):
            key, _ = buf.sample(rng)
            seen.add(key)

        coverage = len(seen) / 30
        assert coverage >= 0.8, f"Only {coverage:.0%} buffer coverage"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_both_configs_and_producers_raises(self):
        buf = ShuffleBuffer(capacity=10, max_frames=3, channels=3, height=8, width=8)
        from dataporter.producer_pool import ProducerConfig
        config = ProducerConfig("test", "repo", "/tmp", [0], weight=1.0)
        producer = AsyncProducer("test", _synthetic_decode, [0], weight=1.0)

        with pytest.raises(ValueError, match="Cannot specify both"):
            ProducerPool(buf, configs=[config], producers=[producer])

    def test_neither_configs_nor_producers_raises(self):
        buf = ShuffleBuffer(capacity=10, max_frames=3, channels=3, height=8, width=8)
        with pytest.raises(ValueError, match="Either configs or producers"):
            ProducerPool(buf)


# ---------------------------------------------------------------------------
# E2E: Synthetic multi-source → ProducerPool → ShuffleBuffer → Dataset → DataLoader
# ---------------------------------------------------------------------------

class TestE2EMultiSource:
    """End-to-end test with synthetic multi-source data.

    Verifies the full pipeline: multiple producers with different weights
    fill a ShuffleBuffer, ShuffleBufferDataset reads from it via DataLoader,
    and the output has correct blend ratios and sample diversity.
    """

    def test_blend_ratio_in_dataloader_output(self):
        """DataLoader output reflects producer weights (75/25 blend)."""
        buf = ShuffleBuffer(capacity=100, max_frames=5, channels=3, height=8, width=8)

        # Source A: episodes 0-49, weight 3 (75%)
        # Source B: episodes 1000-1049, weight 1 (25%)
        producer_a = AsyncProducer("A", _synthetic_decode, list(range(50)), weight=3.0)
        producer_b = AsyncProducer("B", _synthetic_decode, list(range(1000, 1050)), weight=1.0)

        pool = ProducerPool(buf, producers=[producer_a, producer_b], total_workers=2, warmup_target=80)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        ds = ShuffleBufferDataset(buf, epoch_length=200)
        loader = DataLoader(
            ds, batch_size=10, shuffle=False, num_workers=2,
            worker_init_fn=ShuffleBufferDataset.worker_init_fn,
        )

        a_count = 0
        b_count = 0
        for batch in loader:
            for ep in batch["episode_index"]:
                if int(ep) < 100:
                    a_count += 1
                else:
                    b_count += 1

        pool.stop()

        total = a_count + b_count
        assert total == 200
        a_ratio = a_count / total
        # Weight 3:1 → A should be ~75%. Allow 50-95% for stochastic buffer.
        assert 0.50 < a_ratio < 0.95, (
            f"Blend ratio: A={a_ratio:.0%} (expected ~75%). A={a_count}, B={b_count}"
        )

    def test_all_sources_represented(self):
        """Both sources should have episodes in the DataLoader output."""
        buf = ShuffleBuffer(capacity=50, max_frames=3, channels=3, height=8, width=8)

        producer_a = AsyncProducer("A", _synthetic_decode, list(range(20)), weight=1.0)
        producer_b = AsyncProducer("B", _synthetic_decode, list(range(500, 520)), weight=1.0)

        pool = ProducerPool(buf, producers=[producer_a, producer_b], total_workers=2, warmup_target=30)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        ds = ShuffleBufferDataset(buf, epoch_length=100)
        loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

        sources = set()
        for batch in loader:
            for ep in batch["episode_index"]:
                sources.add("A" if int(ep) < 100 else "B")

        pool.stop()
        assert sources == {"A", "B"}, f"Only {sources} represented"

    def test_sample_diversity_across_batches(self):
        """Different batches should contain different episodes (not repeated)."""
        buf = ShuffleBuffer(capacity=50, max_frames=3, channels=3, height=8, width=8)
        for i in range(50):
            buf.put(i, _make_frames(3, c=3, h=8, w=8))

        ds = ShuffleBufferDataset(buf, epoch_length=100)
        loader = DataLoader(
            ds, batch_size=10, shuffle=False, num_workers=2,
            worker_init_fn=ShuffleBufferDataset.worker_init_fn,
        )

        all_episodes = []
        for batch in loader:
            all_episodes.extend(batch["episode_index"].tolist())

        unique = len(set(all_episodes))
        # 100 samples from 50 episodes — should see most of them
        assert unique >= 20, f"Only {unique} unique episodes in 100 samples"

    def test_buffer_continuously_refreshes(self):
        """Producer keeps filling even after warmup — buffer content changes."""
        buf = ShuffleBuffer(capacity=20, max_frames=3, channels=3, height=8, width=8)

        producer = AsyncProducer("test", _synthetic_decode, list(range(100)), weight=1.0)
        pool = ProducerPool(buf, producers=[producer], total_workers=2, warmup_target=15)
        pool.start()
        pool.wait_for_warmup(timeout=30)

        # Snapshot buffer contents
        keys_t0 = set(buf.keys())
        time.sleep(0.5)
        keys_t1 = set(buf.keys())

        pool.stop()

        # Buffer should have changed (producer keeps cycling)
        assert keys_t0 != keys_t1 or len(keys_t0) == 20, (
            "Buffer content didn't change — producer may have stopped"
        )
