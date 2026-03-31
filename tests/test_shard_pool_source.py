"""Tests for ShardPoolSource: pool-based shard sampling without LRU cache."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.shard_pool_source import ShardPoolSource
from dataporter.storage import ShardStorage
from dataporter.transformable_dataset import TransformableDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_shard(path: Path, texts: list[str]):
    schema = pa.schema([("text", pa.string())])
    table = pa.table({"text": texts}, schema=schema)
    pq.write_table(table, str(path), compression="zstd")


def _write_shards(tmp_path: Path, n: int = 10, docs_per_shard: int = 20):
    """Create n shards with unique text per (shard, row)."""
    for s in range(n):
        _write_shard(
            tmp_path / f"shard_{s:06d}.parquet",
            [f"s{s}_r{r}" for r in range(docs_per_shard)],
        )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBasic:

    def test_len_matches_storage(self, tmp_path):
        _write_shards(tmp_path, n=5, docs_per_shard=20)
        src = ShardPoolSource(tmp_path, pool_size=2)
        assert len(src) == 100

    def test_getitem_returns_text(self, tmp_path):
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=2)
        item = src[0]
        assert "text" in item
        assert isinstance(item["text"], str)

    def test_all_items_returned(self, tmp_path):
        """Reading len(src) items should see all rows exactly once."""
        _write_shards(tmp_path, n=5, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=3)
        n = len(src)

        texts = set()
        for i in range(n):
            texts.add(src[i]["text"])

        assert len(texts) == n, f"Expected {n} unique texts, got {len(texts)}"

    def test_exhaustion_raises(self, tmp_path):
        """Reading past the total should raise IndexError."""
        _write_shards(tmp_path, n=2, docs_per_shard=5)
        src = ShardPoolSource(tmp_path, pool_size=2)
        n = len(src)

        # Drain all docs
        for i in range(n):
            src[i]

        with pytest.raises(IndexError, match="exhausted"):
            src[n]

    def test_pool_size_one(self, tmp_path):
        """N=1 should work — pure sequential."""
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=1)

        texts = [src[i]["text"] for i in range(30)]
        assert len(set(texts)) == 30

    def test_pool_size_exceeds_shards(self, tmp_path):
        """pool_size > shard_count should not crash — just loads all shards."""
        _write_shards(tmp_path, n=2, docs_per_shard=5)
        src = ShardPoolSource(tmp_path, pool_size=10)

        texts = [src[i]["text"] for i in range(10)]
        assert len(set(texts)) == 10


# ---------------------------------------------------------------------------
# Worker partitioning
# ---------------------------------------------------------------------------

class TestWorkerPartitioning:

    def test_disjoint_partitions(self, tmp_path):
        """Simulated workers get disjoint shard sets."""
        _write_shards(tmp_path, n=12, docs_per_shard=5)

        all_texts: list[set[str]] = []
        for worker_id in range(4):
            src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
            # Simulate worker_info
            info = type("Info", (), {"id": worker_id, "num_workers": 4})()
            with patch("torch.utils.data.get_worker_info", return_value=info):
                src._init_worker()

            texts = set()
            try:
                for _ in range(100):  # more than enough
                    texts.add(src._next_doc())
            except IndexError:
                pass
            all_texts.append(texts)

        # No overlap between workers
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = all_texts[i] & all_texts[j]
                assert not overlap, (
                    f"Worker {i} and {j} share {len(overlap)} docs"
                )

        # Together they cover everything
        combined = set()
        for t in all_texts:
            combined.update(t)
        assert len(combined) == 60

    def test_single_worker_gets_all(self, tmp_path):
        """With num_workers=1, one worker gets all shards."""
        _write_shards(tmp_path, n=5, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=3, seed=42)
        # Default: no worker info → worker_id=0, num_workers=1
        texts = [src[i]["text"] for i in range(50)]
        assert len(set(texts)) == 50


# ---------------------------------------------------------------------------
# Shard replacement
# ---------------------------------------------------------------------------

class TestShardReplacement:

    def test_pool_refills_on_exhaustion(self, tmp_path):
        """When a shard is exhausted, the next shard from queue fills in."""
        _write_shards(tmp_path, n=5, docs_per_shard=4)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
        src._init_worker()

        initial_pool_size = len(src._pool)
        assert initial_pool_size == 2

        # Read enough to exhaust at least one shard (4 rows)
        for _ in range(8):
            src._next_doc()

        # Pool should still have entries (refilled from queue)
        assert len(src._pool) > 0

    def test_all_shards_eventually_loaded(self, tmp_path):
        """Every shard's content appears across the full epoch."""
        n_shards = 8
        _write_shards(tmp_path, n=n_shards, docs_per_shard=5)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)

        texts = set()
        for i in range(n_shards * 5):
            texts.add(src[i]["text"])

        # Check that all shards are represented
        shard_ids = set()
        for t in texts:
            shard_ids.add(t.split("_")[0])  # "s3_r2" → "s3"
        assert len(shard_ids) == n_shards


# ---------------------------------------------------------------------------
# Epoch management
# ---------------------------------------------------------------------------

class TestEpochManagement:

    def test_set_epoch_changes_order(self, tmp_path):
        _write_shards(tmp_path, n=6, docs_per_shard=5)

        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
        texts_e0 = [src[i]["text"] for i in range(30)]

        src.set_epoch(1)
        texts_e1 = [src[i]["text"] for i in range(30)]

        # Same content, different order
        assert set(texts_e0) == set(texts_e1)
        assert texts_e0 != texts_e1, "Different epoch should produce different order"


# ---------------------------------------------------------------------------
# State dict / resumption
# ---------------------------------------------------------------------------

class TestStateDict:

    def test_roundtrip(self, tmp_path):
        _write_shards(tmp_path, n=5, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=3, seed=42)
        state = src.state_dict()

        src2 = ShardPoolSource(tmp_path, pool_size=3, seed=42)
        src2.load_state_dict(state)

        assert src2._pool_size == 3
        assert src2._seed == 42
        assert len(src2) == 50

    def test_load_state_dict_freezes(self, tmp_path):
        _write_shards(tmp_path, n=3, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
        state = src.state_dict()

        src.load_state_dict(state)
        # Should be frozen after load
        assert src._storage._frozen


# ---------------------------------------------------------------------------
# Integration with TransformableDataset + DataLoader
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_with_transformable_dataset(self, tmp_path):
        """ShardPoolSource works as a TransformableDataset source."""
        _write_shards(tmp_path, n=5, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)

        ds = TransformableDataset(
            src,
            lambda source, idx: {"text": source[idx]["text"]},
        )
        assert len(ds) == 50
        item = ds[0]
        assert "text" in item

    def test_with_dataloader_no_workers(self, tmp_path):
        """Works with DataLoader(num_workers=0)."""
        _write_shards(tmp_path, n=4, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
        ds = TransformableDataset(
            src,
            lambda source, idx: {"text": source[idx]["text"]},
        )

        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        all_texts = []
        for batch in loader:
            all_texts.extend(batch["text"])

        assert len(all_texts) == 40
        assert len(set(all_texts)) == 40

    def test_with_dataloader_multiworker(self, tmp_path):
        """Works with DataLoader(num_workers=2) — workers get disjoint shards."""
        _write_shards(tmp_path, n=8, docs_per_shard=10)
        src = ShardPoolSource(tmp_path, pool_size=2, seed=42)
        ds = TransformableDataset(
            src,
            lambda source, idx: {"text": source[idx]["text"]},
        )

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
        all_texts = []
        for batch in loader:
            all_texts.extend(batch["text"])

        # Should see all 80 unique texts
        assert len(set(all_texts)) == 80


# ---------------------------------------------------------------------------
# Memory bounds
# ---------------------------------------------------------------------------

class TestMemoryBounds:

    def test_pool_bounded(self, tmp_path):
        """At most pool_size shards in memory at any time."""
        _write_shards(tmp_path, n=10, docs_per_shard=5)
        src = ShardPoolSource(tmp_path, pool_size=3, seed=42)
        src._init_worker()

        for _ in range(50):
            src._next_doc()
            assert len(src._pool) <= 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_seed_same_output(self, tmp_path):
        _write_shards(tmp_path, n=5, docs_per_shard=10)

        texts1 = [ShardPoolSource(tmp_path, pool_size=2, seed=99)[i]["text"]
                   for i in range(50)]
        texts2 = [ShardPoolSource(tmp_path, pool_size=2, seed=99)[i]["text"]
                   for i in range(50)]
        assert texts1 == texts2

    def test_different_seed_different_output(self, tmp_path):
        _write_shards(tmp_path, n=5, docs_per_shard=10)

        texts1 = [ShardPoolSource(tmp_path, pool_size=2, seed=1)[i]["text"]
                   for i in range(50)]
        texts2 = [ShardPoolSource(tmp_path, pool_size=2, seed=2)[i]["text"]
                   for i in range(50)]
        assert texts1 != texts2


# ---------------------------------------------------------------------------
# Mixing quality (sanity check vs test_mixing_quality.py benchmarks)
# ---------------------------------------------------------------------------

class TestMixingQuality:

    def test_pool3_has_short_runs(self, tmp_path):
        """N=3 pool should have short consecutive same-shard runs."""
        _write_shards(tmp_path, n=20, docs_per_shard=50)
        src = ShardPoolSource(tmp_path, pool_size=3, seed=42)

        shard_ids = []
        for i in range(1000):
            text = src[i]["text"]
            shard_ids.append(text.split("_")[0])

        # Measure run lengths
        runs = []
        current = 1
        for i in range(1, len(shard_ids)):
            if shard_ids[i] == shard_ids[i - 1]:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)

        mean_run = sum(runs) / len(runs)
        assert mean_run < 3.0, f"Mean run length {mean_run:.1f} too high for N=3"
