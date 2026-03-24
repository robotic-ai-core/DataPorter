"""Tests for BasePrefetcher internals (eviction, validation).

These components are shared by TextPrefetcher, LeRobotPrefetcher, etc.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.prefetcher import BasePrefetcher, evict_shard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_shard(path: Path, n_rows: int = 10, seq_len: int = 32, seed: int = 0):
    schema = pa.schema([("input_ids", pa.list_(pa.uint16()))])
    rng = np.random.RandomState(seed)
    rows = [rng.randint(0, 8000, seq_len).tolist() for _ in range(n_rows)]
    table = pa.table({"input_ids": rows}, schema=schema)
    pq.write_table(table, path, compression="zstd")


# ---------------------------------------------------------------------------
# evict_shard
# ---------------------------------------------------------------------------

class TestEviction:

    def _populate(self, tmp_path, n: int = 10):
        for i in range(n):
            _write_test_shard(tmp_path / f"shard_{i:06d}.parquet", n_rows=10, seed=i)

    def test_fifo_evicts_oldest(self, tmp_path):
        self._populate(tmp_path, 5)
        victim = evict_shard(tmp_path, "fifo", random.Random(42))
        assert victim is not None
        assert victim.name == "shard_000000.parquet"
        assert not victim.exists()

    def test_random_evicts_one(self, tmp_path):
        self._populate(tmp_path, 5)
        victim = evict_shard(tmp_path, "random", random.Random(42))
        after = set(tmp_path.glob("shard_*.parquet"))
        assert victim is not None
        assert len(after) == 4

    def test_stochastic_oldest_evicts_from_oldest_half(self, tmp_path):
        self._populate(tmp_path, 10)
        shards = sorted(tmp_path.glob("shard_*.parquet"))
        oldest_half = {p.name for p in shards[:5]}
        victim = evict_shard(tmp_path, "stochastic_oldest", random.Random(42))
        assert victim is not None
        assert victim.name in oldest_half

    def test_evict_empty_dir(self, tmp_path):
        assert evict_shard(tmp_path, "fifo", random.Random(42)) is None

    def test_unknown_strategy_raises(self, tmp_path):
        self._populate(tmp_path, 1)
        with pytest.raises(ValueError, match="Unknown eviction"):
            evict_shard(tmp_path, "lru", random.Random(42))


# ---------------------------------------------------------------------------
# BasePrefetcher validation
# ---------------------------------------------------------------------------

class TestBasePrefetcherValidation:

    def test_min_shards_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="min_shards"):
            BasePrefetcher(output_dir=tmp_path, min_shards=0)

    def test_max_lt_min_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_shards"):
            BasePrefetcher(output_dir=tmp_path, min_shards=10, max_shards=5)

    def test_invalid_eviction_raises(self, tmp_path):
        with pytest.raises(ValueError, match="eviction"):
            BasePrefetcher(output_dir=tmp_path, eviction="invalid")

    def test_max_shards_none_allowed(self, tmp_path):
        bp = BasePrefetcher(output_dir=tmp_path, max_shards=None)
        assert bp._max_shards is None
