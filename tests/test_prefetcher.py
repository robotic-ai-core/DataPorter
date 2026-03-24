"""Tests for ParquetPrefetcher internals (shard writing, eviction).

These components are shared by LeRobotPrefetcher and the companion pool.
Text-specific tests are in test_text_pipeline.py.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.prefetcher import (
    ParquetPrefetcher,
    _ShardWriter,
    _evict_shard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema() -> pa.Schema:
    return pa.schema([("input_ids", pa.list_(pa.uint16()))])


def _write_test_shard(path: Path, n_rows: int = 10, seq_len: int = 32, seed: int = 0):
    rng = np.random.RandomState(seed)
    schema = _make_schema()
    rows = [rng.randint(0, 8000, seq_len).tolist() for _ in range(n_rows)]
    table = pa.table({"input_ids": rows}, schema=schema)
    pq.write_table(table, path, compression="zstd")


# ---------------------------------------------------------------------------
# _ShardWriter
# ---------------------------------------------------------------------------

class TestShardWriter:

    def test_writes_shard(self, tmp_path):
        schema = _make_schema()
        path = tmp_path / "shard_000.parquet"
        writer = _ShardWriter(path, schema, row_group_size=10)
        for _ in range(25):
            writer.write_row(list(range(32)))
        writer.close()

        assert path.exists()
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_row_groups == 3
        total = sum(pf.metadata.row_group(i).num_rows for i in range(3))
        assert total == 25

    def test_empty_shard(self, tmp_path):
        schema = _make_schema()
        path = tmp_path / "shard_000.parquet"
        writer = _ShardWriter(path, schema)
        writer.close()
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_rows == 0


# ---------------------------------------------------------------------------
# _evict_shard
# ---------------------------------------------------------------------------

class TestEviction:

    def _populate(self, tmp_path, n: int = 10):
        for i in range(n):
            _write_test_shard(tmp_path / f"shard_{i:06d}.parquet", n_rows=10, seed=i)

    def test_fifo_evicts_oldest(self, tmp_path):
        self._populate(tmp_path, 5)
        victim = _evict_shard(tmp_path, "fifo", random.Random(42))
        assert victim is not None
        assert victim.name == "shard_000000.parquet"
        assert not victim.exists()

    def test_random_evicts_one(self, tmp_path):
        self._populate(tmp_path, 5)
        before = set(tmp_path.glob("shard_*.parquet"))
        victim = _evict_shard(tmp_path, "random", random.Random(42))
        after = set(tmp_path.glob("shard_*.parquet"))
        assert victim is not None
        assert len(after) == 4

    def test_stochastic_oldest_evicts_from_oldest_half(self, tmp_path):
        self._populate(tmp_path, 10)
        shards = sorted(tmp_path.glob("shard_*.parquet"))
        oldest_half = {p.name for p in shards[:5]}
        victim = _evict_shard(tmp_path, "stochastic_oldest", random.Random(42))
        assert victim is not None
        assert victim.name in oldest_half

    def test_evict_empty_dir(self, tmp_path):
        assert _evict_shard(tmp_path, "fifo", random.Random(42)) is None

    def test_unknown_strategy_raises(self, tmp_path):
        self._populate(tmp_path, 1)
        with pytest.raises(ValueError, match="Unknown eviction"):
            _evict_shard(tmp_path, "lru", random.Random(42))


# ---------------------------------------------------------------------------
# ParquetPrefetcher validation
# ---------------------------------------------------------------------------

class TestPrefetcherValidation:

    def test_empty_sources_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one source"):
            ParquetPrefetcher(sources=[], output_dir=tmp_path)

    def test_min_shards_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="min_shards"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}], output_dir=tmp_path, min_shards=0
            )

    def test_max_lt_min_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_shards"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir=tmp_path,
                min_shards=10,
                max_shards=5,
            )

    def test_invalid_eviction_raises(self, tmp_path):
        with pytest.raises(ValueError, match="eviction"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir=tmp_path,
                eviction="invalid",
            )

    def test_companion_resolver_requires_dir(self, tmp_path):
        with pytest.raises(ValueError, match="companion_dir"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir=tmp_path,
                companion_resolver=lambda doc: [],
            )
