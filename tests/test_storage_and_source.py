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
        s = ShardStorage(tmp_path, refresh_interval=0.01, max_shards=5)
        assert s.shard_count == 5
        assert not (tmp_path / "shard_000000.parquet").exists()

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
        # Fallback result should be cached
        assert storage.get(42) == {"text": "fallback_42"}


# ---------------------------------------------------------------------------
# PrefetchedSource — with background producers
# ---------------------------------------------------------------------------

class TestPrefetchedSourceWithProducers:

    def test_background_fill(self):
        storage = MemoryStorage(capacity=100)

        def producer():
            for i in range(50):
                yield i, {"value": i}

        source = PrefetchedSource(storage, producers=[producer])
        source.start()
        time.sleep(0.5)
        source.stop()

        assert len(storage) > 0
        assert storage.get(0) == {"value": 0}

    def test_multiple_producers(self):
        storage = MemoryStorage(capacity=200)

        def producer_a():
            for i in range(50):
                yield i, {"src": "a", "idx": i}

        def producer_b():
            for i in range(50, 100):
                yield i, {"src": "b", "idx": i}

        source = PrefetchedSource(storage, producers=[producer_a, producer_b])
        source.start()
        time.sleep(0.5)
        source.stop()

        assert storage.get(0)["src"] == "a"
        assert storage.get(50)["src"] == "b"

    def test_capacity_eviction(self):
        storage = MemoryStorage(capacity=10)

        def producer():
            for i in range(50):
                yield i, {"value": i}
                time.sleep(0.01)

        source = PrefetchedSource(storage, producers=[producer])
        source.start()
        time.sleep(0.3)
        source.stop()

        # Should never exceed capacity
        assert len(storage) <= 10

    def test_stop_terminates_producers(self):
        storage = MemoryStorage()

        def slow_producer():
            for i in range(10000):
                yield i, {"value": i}
                time.sleep(0.1)

        source = PrefetchedSource(storage, producers=[slow_producer])
        source.start()
        time.sleep(0.2)
        source.stop()
        # Should stop quickly, not wait for all 10k items
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

    def test_with_background_producer_and_dataloader(self):
        """Background producer fills storage while DataLoader consumes."""
        storage = MemoryStorage(capacity=200)

        def producer():
            for i in range(100):
                yield i, {"value": float(i)}

        source = PrefetchedSource(storage, producers=[producer])
        source.start()
        time.sleep(0.3)  # let producer fill

        def transform(src, idx):
            item = src[idx]
            return {"value": torch.tensor([item["value"]])}

        ds = TransformableDataset(source, transform)
        loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)

        batches = list(loader)
        assert len(batches) > 0
        source.stop()
