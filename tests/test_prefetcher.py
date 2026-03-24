"""Tests for the ParquetPrefetcher and GrowableParquetDataset.

Test categories:
  1. Unit: _ShardWriter, _evict_shard
  2. Prefetcher: writes shards from mock data via _dataset_factory
  3. Eviction: shard count stays within bounds
  4. Distribution: multi-offset streams
  5. GrowableParquetDataset: reads from growing directory
  6. Rate: consumer faster/slower than producer
  7. Integration: end-to-end with DataLoader
  8. Transforms: compose, tokenize, chunk
  9. Resume: prefetcher resumes from partially filled directory
  10. Error handling: network errors, double start
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.growable_dataset import GrowableParquetDataset
from dataporter.prefetcher import (
    ParquetPrefetcher,
    _ShardWriter,
    _evict_shard,
)
from dataporter.transforms import compose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema() -> pa.Schema:
    return pa.schema([("input_ids", pa.list_(pa.uint16()))])


def _write_test_shard(
    path: Path,
    n_rows: int = 100,
    seq_len: int = 32,
    row_group_size: int = 50,
    seed: int = 0,
) -> None:
    """Write a test shard with random uint16 token IDs."""
    rng = np.random.RandomState(seed)
    schema = _make_schema()
    writer = pq.ParquetWriter(str(path), schema, compression="zstd")
    for start in range(0, n_rows, row_group_size):
        end = min(start + row_group_size, n_rows)
        rows = [rng.randint(0, 8000, seq_len).tolist() for _ in range(end - start)]
        table = pa.table({"input_ids": rows}, schema=schema)
        writer.write_table(table)
    writer.close()


def _make_text_docs(n: int = 500, word_count: int = 50) -> list[dict]:
    """Create fake text documents."""
    rng = random.Random(42)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "data", "model", "train", "test", "batch"]
    docs = []
    for i in range(n):
        text = " ".join(rng.choices(words, k=word_count))
        docs.append({"text": text, "id": i})
    return docs


def _simple_transform(doc: dict) -> list[list[int]] | None:
    """Transform: text bytes -> fixed-length chunks of uint16 token IDs."""
    text = doc.get("text", "")
    if not text.strip():
        return None
    tokens = list(text.encode("utf-8"))
    seq_len = 32
    chunks = []
    for i in range(0, len(tokens), seq_len):
        chunk = tokens[i:i + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)
    return chunks if chunks else None


class _FakeHFDataset:
    """Minimal mock that supports .skip(), .shuffle(), and iter()."""

    def __init__(self, docs: list[dict], delay: float = 0.0):
        self._docs = docs
        self._offset = 0
        self._delay = delay

    def skip(self, n: int) -> "_FakeHFDataset":
        clone = _FakeHFDataset(self._docs, self._delay)
        clone._offset = n
        return clone

    def shuffle(self, seed: int = 42, buffer_size: int = 1000) -> "_FakeHFDataset":
        rng = random.Random(seed)
        docs = self._docs.copy()
        rng.shuffle(docs)
        clone = _FakeHFDataset(docs, self._delay)
        clone._offset = self._offset
        return clone

    def __iter__(self):
        for doc in self._docs[self._offset:]:
            if self._delay > 0:
                time.sleep(self._delay)
            yield doc


def _make_factory(fake_ds):
    """Create a _dataset_factory that returns the given fake dataset."""
    def factory(src):
        return fake_ds
    return factory


def _make_multi_factory(datasets_by_name: dict[str, _FakeHFDataset]):
    """Create a _dataset_factory that returns different datasets per source name."""
    def factory(src):
        return datasets_by_name[src["dataset"]]
    return factory


# ---------------------------------------------------------------------------
# 1. Unit Tests — _ShardWriter
# ---------------------------------------------------------------------------

class TestShardWriter:

    def test_writes_shard(self, tmp_path):
        schema = _make_schema()
        path = tmp_path / "shard_000.parquet"
        writer = _ShardWriter(path, schema, row_group_size=10)
        for i in range(25):
            writer.write_row(list(range(32)))
        writer.close()

        assert path.exists()
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_row_groups == 3  # 10 + 10 + 5
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
# 2. Unit Tests — _evict_shard
# ---------------------------------------------------------------------------

class TestEviction:

    def _populate(self, tmp_path, n: int = 10):
        for i in range(n):
            _write_test_shard(tmp_path / f"shard_{i:06d}.parquet", n_rows=10, seed=i)

    def test_fifo_evicts_oldest(self, tmp_path):
        self._populate(tmp_path, 5)
        rng = random.Random(42)
        victim = _evict_shard(tmp_path, "fifo", rng)
        assert victim is not None
        assert victim.name == "shard_000000.parquet"
        assert not victim.exists()

    def test_random_evicts_one(self, tmp_path):
        self._populate(tmp_path, 5)
        rng = random.Random(42)
        before = set(tmp_path.glob("shard_*.parquet"))
        victim = _evict_shard(tmp_path, "random", rng)
        after = set(tmp_path.glob("shard_*.parquet"))
        assert victim is not None
        assert len(after) == 4
        assert victim not in after

    def test_stochastic_oldest_evicts_from_oldest_half(self, tmp_path):
        self._populate(tmp_path, 10)
        rng = random.Random(42)
        shards = sorted(tmp_path.glob("shard_*.parquet"))
        oldest_half = {p.name for p in shards[:5]}

        victim = _evict_shard(tmp_path, "stochastic_oldest", rng)
        assert victim is not None
        assert victim.name in oldest_half

    def test_evict_empty_dir(self, tmp_path):
        rng = random.Random(42)
        assert _evict_shard(tmp_path, "fifo", rng) is None

    def test_unknown_strategy_raises(self, tmp_path):
        self._populate(tmp_path, 1)
        with pytest.raises(ValueError, match="Unknown eviction"):
            _evict_shard(tmp_path, "lru", random.Random(42))


# ---------------------------------------------------------------------------
# 3. Prefetcher Tests (using _dataset_factory)
# ---------------------------------------------------------------------------

class TestPrefetcherBasic:

    def test_writes_shards(self, tmp_path):
        docs = _make_text_docs(200)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=2,
            max_shards=100,
            max_rows_per_shard=100,
            row_group_size=50,
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        assert prefetcher.shard_count >= 2
        for shard in tmp_path.glob("shard_*.parquet"):
            pf = pq.ParquetFile(shard)
            assert pf.metadata.num_rows > 0

    def test_wait_for_min_blocks(self, tmp_path):
        docs = _make_text_docs(500)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=3,
            max_shards=100,
            max_rows_per_shard=50,
            row_group_size=25,
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        assert prefetcher.shard_count >= 3
        prefetcher.stop()

    def test_stop_is_safe(self, tmp_path):
        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
        )
        prefetcher.stop()

    def test_validation_errors(self, tmp_path):
        with pytest.raises(ValueError, match="At least one source"):
            ParquetPrefetcher(sources=[], output_dir=tmp_path)

        with pytest.raises(ValueError, match="min_shards"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}], output_dir=tmp_path, min_shards=0
            )

        with pytest.raises(ValueError, match="max_shards"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir=tmp_path,
                min_shards=10,
                max_shards=5,
            )

        with pytest.raises(ValueError, match="eviction"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir=tmp_path,
                eviction="invalid",
            )


# ---------------------------------------------------------------------------
# 4. Eviction Tests (prefetcher-level)
# ---------------------------------------------------------------------------

class TestPrefetcherEviction:

    def test_max_shards_enforced_fifo(self, tmp_path):
        docs = _make_text_docs(2000)
        fake_ds = _FakeHFDataset(docs)

        max_shards = 5
        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=2,
            max_shards=max_shards,
            max_rows_per_shard=20,
            row_group_size=10,
            eviction="fifo",
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(2)
        prefetcher.stop()

        # Should never exceed max_shards (+ 1 tolerance for race)
        assert prefetcher.shard_count <= max_shards + 1

    def test_stochastic_oldest_eviction(self, tmp_path):
        docs = _make_text_docs(2000)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=2,
            max_shards=5,
            max_rows_per_shard=20,
            row_group_size=10,
            eviction="stochastic_oldest",
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(2)
        prefetcher.stop()

        assert prefetcher.shard_count <= 6

    def test_random_eviction(self, tmp_path):
        docs = _make_text_docs(2000)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=2,
            max_shards=5,
            max_rows_per_shard=20,
            row_group_size=10,
            eviction="random",
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(2)
        prefetcher.stop()

        assert prefetcher.shard_count <= 6


# ---------------------------------------------------------------------------
# 5. Distribution Tests
# ---------------------------------------------------------------------------

class TestDistribution:

    def test_multi_source_round_robin(self, tmp_path):
        """Two sources should both contribute data."""
        docs_a = [{"text": f"source_a doc {i}", "id": i} for i in range(200)]
        docs_b = [{"text": f"source_b doc {i}", "id": i} for i in range(200)]

        call_log = []

        def factory(src):
            call_log.append(src["dataset"])
            if src["dataset"] == "source_a":
                return _FakeHFDataset(docs_a)
            return _FakeHFDataset(docs_b)

        prefetcher = ParquetPrefetcher(
            sources=[
                {"dataset": "source_a"},
                {"dataset": "source_b"},
            ],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=2,
            max_shards=100,
            max_rows_per_shard=50,
            row_group_size=25,
            stream_shuffle_buffer=0,
            _dataset_factory=factory,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        assert "source_a" in call_log
        assert "source_b" in call_log

    def test_offset_skips_docs(self, tmp_path):
        """Offset source should skip the first N documents."""
        docs = [{"text": f"doc_{i} " * 10, "id": i} for i in range(100)]

        skip_called = [False]

        class TrackingDataset(_FakeHFDataset):
            def skip(self, n):
                skip_called[0] = True
                return super().skip(n)

        fake_ds = TrackingDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test", "offset": 50}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=1,
            max_shards=100,
            max_rows_per_shard=1000,
            row_group_size=50,
            stream_shuffle_buffer=0,
            _dataset_factory=lambda src: fake_ds,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        assert skip_called[0]


# ---------------------------------------------------------------------------
# 6. GrowableParquetDataset Tests
# ---------------------------------------------------------------------------

class TestGrowableDataset:

    def _write_shards(self, tmp_path, n_shards=3, n_rows=100, seq_len=32):
        for i in range(n_shards):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet",
                n_rows=n_rows,
                seq_len=seq_len,
                seed=i,
            )

    def test_reads_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=3, n_rows=100)
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=0.1)
        assert len(ds) == 300
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (32,)

    def test_picks_up_new_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=2, n_rows=50)
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=0.01)
        assert len(ds) == 100

        _write_test_shard(
            tmp_path / "shard_000002.parquet", n_rows=50, seed=99
        )
        time.sleep(0.02)
        assert len(ds) == 150

    def test_handles_evicted_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=3, n_rows=50)
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=0.01)
        assert len(ds) == 150

        (tmp_path / "shard_000000.parquet").unlink()
        time.sleep(0.02)
        assert len(ds) == 100

    def test_empty_dir(self, tmp_path):
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=0.01)
        assert len(ds) == 0

    def test_index_out_of_range(self, tmp_path):
        self._write_shards(tmp_path, n_shards=1, n_rows=10)
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        with pytest.raises(IndexError):
            ds[10]
        with pytest.raises(IndexError):
            ds[-1]

    def test_shard_count_property(self, tmp_path):
        self._write_shards(tmp_path, n_shards=5, n_rows=10)
        ds = GrowableParquetDataset(tmp_path)
        assert ds.shard_count == 5

    def test_seq_len_property(self, tmp_path):
        self._write_shards(tmp_path, n_shards=1, n_rows=10, seq_len=64)
        ds = GrowableParquetDataset(tmp_path)
        assert ds.seq_len == 64


# ---------------------------------------------------------------------------
# 7. Rate Tests
# ---------------------------------------------------------------------------

class TestRates:

    def test_consumer_faster_than_producer(self, tmp_path):
        """When consumer is faster, dataset recycles existing shards."""
        for i in range(3):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet", n_rows=50, seed=i
            )
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        assert len(ds) == 150

        for epoch in range(3):
            for i in range(len(ds)):
                sample = ds[i]
                assert sample["input_ids"].shape == (32,)

    def test_producer_faster_than_consumer(self, tmp_path):
        """When producer is faster, shards accumulate and dataset grows."""
        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=0.01)

        for i in range(5):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet", n_rows=20, seed=i
            )
            time.sleep(0.02)

            current_len = len(ds)
            if current_len > 0:
                sample = ds[0]
                assert "input_ids" in sample

        assert len(ds) == 100


# ---------------------------------------------------------------------------
# 8. Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_dataloader_reads_shards(self, tmp_path):
        for i in range(3):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet", n_rows=100, seq_len=32, seed=i
            )

        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

        total_samples = sum(b["input_ids"].shape[0] for b in loader)
        assert total_samples == 300

    def test_dataloader_with_workers(self, tmp_path):
        for i in range(3):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet", n_rows=100, seq_len=32, seed=i
            )

        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)

        total = sum(b["input_ids"].shape[0] for b in loader)
        assert total == 300

    def test_no_data_corruption(self, tmp_path):
        """Data read back should match what was written."""
        seq_len = 16
        rng = np.random.RandomState(42)
        expected_rows = []

        schema = _make_schema()
        for shard_idx in range(2):
            path = tmp_path / f"shard_{shard_idx:06d}.parquet"
            rows = []
            for _ in range(50):
                row = rng.randint(0, 8000, seq_len).tolist()
                rows.append(row)
                expected_rows.append(row)
            table = pa.table({"input_ids": rows}, schema=schema)
            pq.write_table(table, path, compression="zstd")

        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        assert len(ds) == 100

        for i in range(100):
            actual = ds[i]["input_ids"].tolist()
            assert actual == expected_rows[i], f"Mismatch at index {i}"

    def test_prefetcher_to_dataset_e2e(self, tmp_path):
        """Full pipeline: prefetcher writes, growable dataset reads, DataLoader consumes."""
        docs = _make_text_docs(1000, word_count=80)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=3,
            max_shards=20,
            max_rows_per_shard=200,
            row_group_size=50,
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        # Stop prefetcher before reading so no eviction occurs mid-iteration
        prefetcher.stop()

        ds = GrowableParquetDataset(tmp_path, refresh_interval_seconds=60)
        assert len(ds) > 0

        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
        batches = list(loader)
        assert len(batches) > 0
        assert all(b["input_ids"].shape[1] == 32 for b in batches)


# ---------------------------------------------------------------------------
# 9. Transform Tests
# ---------------------------------------------------------------------------

class TestTransforms:

    def test_compose_single(self):
        def t1(doc):
            return [[1, 2, 3]]
        composed = compose(t1)
        assert composed is t1

    def test_compose_multiple(self):
        def t1(doc):
            text = doc.get("text", "")
            return [list(text.encode("utf-8"))]

        def t2(doc):
            ids = doc.get("input_ids", [])
            return [ids[:4]] if len(ids) >= 4 else None

        composed = compose(t1, t2)
        result = composed({"text": "hello world"})
        assert result is not None
        assert len(result[0]) == 4

    def test_compose_none_propagation(self):
        def t1(doc):
            return None

        def t2(doc):
            return [[1, 2, 3]]

        composed = compose(t1, t2)
        assert composed({"text": ""}) is None

    def test_compose_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            compose()

    def test_simple_transform(self):
        doc = {"text": "hello world " * 10}
        result = _simple_transform(doc)
        assert result is not None
        assert all(len(chunk) == 32 for chunk in result)

    def test_simple_transform_empty(self):
        assert _simple_transform({"text": ""}) is None
        assert _simple_transform({"text": "   "}) is None


# ---------------------------------------------------------------------------
# 10. Resume Tests
# ---------------------------------------------------------------------------

class TestResume:

    def test_resume_from_existing_shards(self, tmp_path):
        for i in range(5):
            _write_test_shard(
                tmp_path / f"shard_{i:06d}.parquet", n_rows=10, seed=i
            )

        docs = _make_text_docs(100)
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=3,
            max_shards=100,
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        # min_ready should be set immediately since we have 5 >= 3 shards
        prefetcher.wait_for_min(timeout=1)
        prefetcher.stop()

        assert prefetcher.shard_count >= 5


# ---------------------------------------------------------------------------
# 11. Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_error_propagated_via_wait(self, tmp_path):
        def failing_factory(src):
            raise ConnectionError("Network error")

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=1,
            max_shards=10,
            _dataset_factory=failing_factory,
        )
        prefetcher.start()
        with pytest.raises(RuntimeError, match="Prefetcher failed"):
            prefetcher.wait_for_min(timeout=10)
        prefetcher.stop()

    def test_double_start_raises(self, tmp_path):
        # Use a slow dataset so the thread is still alive when we call start() again
        docs = _make_text_docs(1000)
        fake_ds = _FakeHFDataset(docs, delay=0.01)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=1,
            max_shards=100,
            max_rows_per_shard=100_000,
            _dataset_factory=_make_factory(fake_ds),
        )
        prefetcher.start()
        time.sleep(0.2)
        assert prefetcher.is_alive, "Thread should still be running"
        with pytest.raises(RuntimeError, match="already running"):
            prefetcher.start()
        prefetcher.stop()
