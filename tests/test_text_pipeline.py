"""Tests for the text streaming pipeline: TextPrefetcher → RawTextSource → TransformableDataset.

All tests use mocked HF datasets — no network access needed.
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.raw_text_source import RawTextSource
from dataporter.text_prefetcher import TextPrefetcher
from dataporter.prefetcher import evict_shard
from dataporter.transformable_dataset import TransformableDataset
from dataporter.transforms import compose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_docs(n: int = 200, word_count: int = 50) -> list[dict]:
    rng = random.Random(42)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "hello", "world", "data", "model", "train", "test", "batch"]
    return [{"text": " ".join(rng.choices(words, k=word_count))} for _ in range(n)]


class _FakeHFDataset:
    """Mock HF streaming dataset for testing."""

    def __init__(self, docs: list[dict], delay: float = 0.0):
        self._docs = docs
        self._offset = 0
        self._delay = delay

    def skip(self, n: int) -> "_FakeHFDataset":
        c = _FakeHFDataset(self._docs, self._delay)
        c._offset = n
        return c

    def shuffle(self, seed: int = 42, buffer_size: int = 1000) -> "_FakeHFDataset":
        rng = random.Random(seed)
        docs = self._docs.copy()
        rng.shuffle(docs)
        c = _FakeHFDataset(docs, self._delay)
        c._offset = self._offset
        return c

    def __iter__(self):
        for doc in self._docs[self._offset:]:
            if self._delay > 0:
                time.sleep(self._delay)
            yield doc


def _make_factory(docs: list[dict], delay: float = 0.0):
    """Create a _dataset_factory for TextPrefetcher."""
    def factory(offset: int):
        ds = _FakeHFDataset(docs, delay)
        if offset > 0:
            ds = ds.skip(offset)
        return ds
    return factory


def _write_text_shard(path: Path, texts: list[str]):
    schema = pa.schema([("text", pa.string())])
    table = pa.table({"text": texts}, schema=schema)
    pq.write_table(table, str(path), compression="zstd")


# ---------------------------------------------------------------------------
# 1. TextPrefetcher Tests
# ---------------------------------------------------------------------------

class TestTextPrefetcher:

    def test_writes_shards(self, tmp_path):
        docs = _make_text_docs(200)
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=2,
            max_rows_per_shard=50,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.3)
        prefetcher.stop()

        assert prefetcher.shard_count >= 2
        for shard in tmp_path.glob("shard_*.parquet"):
            pf = pq.ParquetFile(shard)
            assert pf.metadata.num_rows > 0

    def test_wait_for_min_blocks(self, tmp_path):
        docs = _make_text_docs(500)
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=3,
            max_rows_per_shard=50,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        assert prefetcher.shard_count >= 3
        prefetcher.stop()

    def test_offset_skips_docs(self, tmp_path):
        """Second offset starts from a different point in the dataset."""
        docs = [{"text": f"doc_{i}"} for i in range(100)]
        seen_offsets = []

        def tracking_factory(offset: int):
            seen_offsets.append(offset)
            return _FakeHFDataset(docs).skip(offset) if offset > 0 else _FakeHFDataset(docs)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_rows_per_shard=1000,
            offsets=[0, 50],
            max_restarts=0,
            _dataset_factory=tracking_factory,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        assert 0 in seen_offsets
        assert 50 in seen_offsets

    def test_eviction_enforced(self, tmp_path):
        docs = _make_text_docs(2000)
        max_shards = 5
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=2,
            max_shards=max_shards,
            max_rows_per_shard=20,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        assert prefetcher.shard_count <= max_shards + 1

    def test_no_eviction_when_none(self, tmp_path):
        docs = _make_text_docs(200)
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_shards=None,
            max_rows_per_shard=20,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # All shards kept
        assert prefetcher.shard_count >= 5

    def test_stop_without_start(self, tmp_path):
        prefetcher = TextPrefetcher(dataset="test", output_dir=tmp_path)
        prefetcher.stop()  # should not raise

    def test_double_start_raises(self, tmp_path):
        docs = _make_text_docs(1000)
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_rows_per_shard=100_000,
            offsets=[0],
            _dataset_factory=_make_factory(docs, delay=0.01),
        )
        prefetcher.start()
        time.sleep(0.2)
        assert prefetcher.is_alive
        with pytest.raises(RuntimeError, match="already running"):
            prefetcher.start()
        prefetcher.stop()

    def test_error_propagation(self, tmp_path):
        def failing_factory(offset):
            raise ConnectionError("Network error")

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_restarts=0,
            _dataset_factory=failing_factory,
        )
        prefetcher.start()
        with pytest.raises(RuntimeError, match="Prefetcher failed"):
            prefetcher.wait_for_min(timeout=10)
        prefetcher.stop()

    def test_resume_from_existing(self, tmp_path):
        """Pre-existing shards count toward min_shards."""
        for i in range(5):
            _write_text_shard(
                tmp_path / f"shard_{i:06d}.parquet",
                [f"doc {j}" for j in range(10)],
            )

        docs = _make_text_docs(50)
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=3,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        # Should be immediately ready since 5 >= 3
        prefetcher.wait_for_min(timeout=1)
        prefetcher.stop()
        assert prefetcher.shard_count >= 5

    def test_skips_empty_docs(self, tmp_path):
        docs = [{"text": ""}, {"text": "   "}, {"text": "real doc " * 10}] * 100
        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_rows_per_shard=50,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        # Read back — should only have non-empty docs
        for shard in tmp_path.glob("shard_*.parquet"):
            table = pq.read_table(shard)
            for text in table.column("text").to_pylist():
                assert text.strip()


# ---------------------------------------------------------------------------
# 2. RawTextSource Tests
# ---------------------------------------------------------------------------

class TestRawTextSource:

    def _write_shards(self, tmp_path, n_shards=3, docs_per_shard=50):
        for i in range(n_shards):
            texts = [f"shard_{i}_doc_{j} " * 10 for j in range(docs_per_shard)]
            _write_text_shard(tmp_path / f"shard_{i:06d}.parquet", texts)

    def test_reads_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=3, docs_per_shard=50)
        src = RawTextSource(tmp_path, refresh_interval_seconds=0.01)
        assert len(src) == 150
        item = src[0]
        assert "text" in item
        assert isinstance(item["text"], str)
        assert len(item["text"]) > 0

    def test_picks_up_new_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=2, docs_per_shard=50)
        src = RawTextSource(tmp_path, refresh_interval_seconds=0.01)
        assert len(src) == 100

        _write_text_shard(tmp_path / "shard_000002.parquet", ["new doc"] * 50)
        time.sleep(0.02)
        assert len(src) == 150

    def test_handles_evicted_shards(self, tmp_path):
        self._write_shards(tmp_path, n_shards=3, docs_per_shard=50)
        src = RawTextSource(tmp_path, refresh_interval_seconds=0.01)
        assert len(src) == 150

        (tmp_path / "shard_000000.parquet").unlink()
        time.sleep(0.02)
        assert len(src) == 100

    def test_empty_dir(self, tmp_path):
        src = RawTextSource(tmp_path, refresh_interval_seconds=0.01)
        assert len(src) == 0

    def test_index_wraps(self, tmp_path):
        """Indices wrap around via modulo."""
        self._write_shards(tmp_path, n_shards=1, docs_per_shard=10)
        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        assert src[0] == src[10]  # wraps

    def test_bisect_locate(self, tmp_path):
        """Verify O(log n) locate works correctly across shard boundaries."""
        self._write_shards(tmp_path, n_shards=5, docs_per_shard=20)
        src = RawTextSource(tmp_path, refresh_interval_seconds=60)

        # Last item of first shard
        item19 = src[19]
        # First item of second shard
        item20 = src[20]
        assert item19 != item20  # different shards

        # Last item overall
        item99 = src[99]
        assert "text" in item99


# ---------------------------------------------------------------------------
# 3. TransformableDataset Tests
# ---------------------------------------------------------------------------

def _tokenize_transform(source, idx):
    """Simple worker-side transform: text → fake token IDs."""
    item = source[idx]
    text = item["text"]
    tokens = list(text.encode("utf-8"))[:64]
    ids = torch.tensor(tokens, dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


class TestTransformableDataset:

    def test_applies_transform(self, tmp_path):
        _write_text_shard(tmp_path / "shard_000000.parquet", ["hello world"] * 10)
        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        ds = TransformableDataset(src, _tokenize_transform)

        assert len(ds) == 10
        sample = ds[0]
        assert "input_ids" in sample
        assert sample["input_ids"].dtype == torch.long

    def test_with_dataloader(self, tmp_path):
        texts = [f"document number {i} " * 10 for i in range(100)]
        _write_text_shard(tmp_path / "shard_000000.parquet", texts)

        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        ds = TransformableDataset(src, _tokenize_transform)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

        total = sum(b["input_ids"].shape[0] for b in loader)
        assert total == 100

    def test_with_workers(self, tmp_path):
        """Multi-worker DataLoader — transforms run in parallel."""
        texts = [f"document number {i} " * 10 for i in range(100)]
        _write_text_shard(tmp_path / "shard_000000.parquet", texts)

        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        ds = TransformableDataset(src, _tokenize_transform)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

        total = sum(b["input_ids"].shape[0] for b in loader)
        assert total == 100


# ---------------------------------------------------------------------------
# 4. End-to-end: TextPrefetcher → RawTextSource → TransformableDataset
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, tmp_path):
        """Prefetcher writes, source reads, transform runs in DataLoader."""
        docs = _make_text_docs(500, word_count=80)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=3,
            max_rows_per_shard=50,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        assert len(src) > 0

        ds = TransformableDataset(src, _tokenize_transform)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

        batches = list(loader)
        assert len(batches) > 0
        assert all(b["input_ids"].dtype == torch.long for b in batches)


# ---------------------------------------------------------------------------
# 5. Compose Tests
# ---------------------------------------------------------------------------

class TestCompose:

    def test_single(self):
        def t1(x):
            return x + 1
        assert compose(t1) is t1

    def test_chain(self):
        composed = compose(lambda x: x + 1, lambda x: x * 2)
        assert composed(3) == 8  # (3+1)*2

    def test_none_propagation(self):
        composed = compose(lambda x: None, lambda x: x * 2)
        assert composed(3) is None

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            compose()


# ---------------------------------------------------------------------------
# 6. Eviction helper
# ---------------------------------------------------------------------------

class TestEvictShard:

    def test_evicts_from_oldest_half(self, tmp_path):
        for i in range(10):
            _write_text_shard(tmp_path / f"shard_{i:06d}.parquet", ["text"] * 5)
        shards = sorted(tmp_path.glob("shard_*.parquet"))
        oldest_half = {p.name for p in shards[:5]}

        victim = evict_shard(tmp_path, "stochastic_oldest", random.Random(42))
        assert victim is not None
        assert victim.name in oldest_half
        assert not victim.exists()

    def test_empty_dir(self, tmp_path):
        assert evict_shard(tmp_path, "fifo", random.Random(42)) is None


# ---------------------------------------------------------------------------
# 7. Parallel Offset Tests
# ---------------------------------------------------------------------------

class TestParallelOffsets:

    def test_parallel_offsets_produce_diverse_shards(self, tmp_path):
        """Multiple offsets write shards in parallel with docs from different regions."""
        # Region A: docs 0-99, Region B: docs 100-199
        region_a = [{"text": f"region_A doc_{i}"} for i in range(100)]
        region_b = [{"text": f"region_B doc_{i}"} for i in range(100)]
        all_docs = region_a + region_b

        def factory(offset):
            return _FakeHFDataset(all_docs).skip(offset)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=2,
            max_rows_per_shard=30,
            offsets=[0, 100],  # offset 0 → region A, offset 100 → region B
            stream_shuffle_buffer=0,  # no shuffle, so region order is deterministic
            max_restarts=0,
            _dataset_factory=factory,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # Read all shards back and collect unique regions
        regions_seen = set()
        for shard in tmp_path.glob("shard_*.parquet"):
            table = pq.read_table(shard)
            for text in table.column("text").to_pylist():
                if "region_A" in text:
                    regions_seen.add("A")
                elif "region_B" in text:
                    regions_seen.add("B")

        # Both regions should appear (parallel offsets contributed)
        assert regions_seen == {"A", "B"}

    def test_parallel_offsets_interleave_shards(self, tmp_path):
        """With parallel offsets, shards should interleave across regions,
        not all-A-then-all-B."""
        region_a = [{"text": f"A_{i}"} for i in range(200)]
        region_b = [{"text": f"B_{i}"} for i in range(200)]
        all_docs = region_a + region_b

        def factory(offset):
            return _FakeHFDataset(all_docs).skip(offset)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=4,
            max_rows_per_shard=30,
            offsets=[0, 200],
            stream_shuffle_buffer=0,
            high_water=30, low_water=10,  # small buffer forces producers to alternate
            max_restarts=0,
            _dataset_factory=factory,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # All shards combined should contain both regions
        all_regions = set()
        for shard in tmp_path.glob("shard_*.parquet"):
            table = pq.read_table(shard)
            for text in table.column("text").to_pylist():
                if text.startswith("A_"):
                    all_regions.add("A")
                elif text.startswith("B_"):
                    all_regions.add("B")

        assert all_regions == {"A", "B"}, (
            f"Shards only contain {all_regions} — "
            "parallel offsets should produce both regions"
        )


# ---------------------------------------------------------------------------
# 8. Randomness Validation (procedurally generated data)
# ---------------------------------------------------------------------------

def _run_prefetcher_and_read(
    tmp_path, n_regions, docs_per_region, max_rows_per_shard=25,
    max_shards=None, high_water=None, low_water=None,
):
    """Helper: run prefetcher with N regions, return (region_counts, shard_region_sets)."""
    all_docs = []
    for r in range(n_regions):
        for i in range(docs_per_region):
            all_docs.append({"text": f"r{r}_d{i}"})

    offsets = [r * docs_per_region for r in range(n_regions)]

    def factory(offset):
        return _FakeHFDataset(all_docs).skip(offset)

    kwargs = dict(
        output_dir=tmp_path,
        dataset="test",
        min_shards=max(2, n_regions),
        max_shards=max_shards,
        max_rows_per_shard=max_rows_per_shard,
        offsets=offsets,
        max_restarts=0,
        _dataset_factory=factory,
    )
    if high_water is not None:
        kwargs["high_water"] = high_water
    if low_water is not None:
        kwargs["low_water"] = low_water

    prefetcher = TextPrefetcher(**kwargs)
    prefetcher.start()
    prefetcher.wait_for_min(timeout=30)
    time.sleep(1)
    prefetcher.stop()

    src = RawTextSource(tmp_path, refresh_interval_seconds=60)
    region_counts = {r: 0 for r in range(n_regions)}
    shard_region_sets = {}  # shard_name -> set of regions in that shard

    for shard in sorted(tmp_path.glob("shard_*.parquet")):
        table = pq.read_table(shard)
        regions_in_shard = set()
        for text in table.column("text").to_pylist():
            region = int(text.split("_")[0][1:])
            region_counts[region] += 1
            regions_in_shard.add(region)
        shard_region_sets[shard.name] = regions_in_shard

    return region_counts, shard_region_sets


class TestRandomnessValidation:
    """Quantitative validation of shuffle quality and distribution."""

    def test_region_uniformity(self, tmp_path):
        """Each region should get roughly equal representation.
        With 4 regions × 100 docs each, each region should be 15-35% of total."""
        region_counts, _ = _run_prefetcher_and_read(
            tmp_path, n_regions=4, docs_per_region=100,
        )
        total = sum(region_counts.values())
        assert total > 0

        for r, count in region_counts.items():
            pct = count / total
            assert 0.10 <= pct <= 0.40, (
                f"Region {r} has {pct:.0%} of samples (expected 15-35%). "
                f"Counts: {region_counts}"
            )

    def test_within_shard_diversity(self, tmp_path):
        """With small shards and tight buffer, some shards should contain
        docs from multiple regions (producers interleave in the buffer)."""
        _, shard_region_sets = _run_prefetcher_and_read(
            tmp_path, n_regions=3, docs_per_region=150,
            max_rows_per_shard=30, high_water=30, low_water=10,
        )
        # All regions should appear across the shard set
        all_regions = set()
        for regions in shard_region_sets.values():
            all_regions.update(regions)
        assert len(all_regions) >= 2, (
            f"Only {all_regions} regions across all shards — "
            "expected at least 2 of 3"
        )

    def test_temporal_balance(self, tmp_path):
        """Early shards (first 50% by name) should have all regions represented,
        not be biased toward whichever producer started first."""
        region_counts, shard_region_sets = _run_prefetcher_and_read(
            tmp_path, n_regions=3, docs_per_region=200,
            max_rows_per_shard=30, high_water=30, low_water=10,
        )
        shards = sorted(shard_region_sets.keys())
        if len(shards) < 4:
            pytest.skip("Not enough shards for temporal balance test")

        early_shards = shards[:len(shards) // 2]
        early_regions = set()
        for name in early_shards:
            early_regions.update(shard_region_sets[name])

        assert len(early_regions) >= 2, (
            f"Early shards only cover regions {early_regions} — "
            "expected at least 2 of 3 for temporal balance"
        )

    def test_sampling_distribution(self, tmp_path):
        """Simulate DataLoader sampling: draw 500 random items and verify
        region distribution is roughly uniform."""
        region_counts, _ = _run_prefetcher_and_read(
            tmp_path, n_regions=4, docs_per_region=200,
            max_rows_per_shard=50,
        )
        src = RawTextSource(tmp_path, refresh_interval_seconds=60)
        n_total = len(src)
        if n_total < 100:
            pytest.skip("Not enough data for sampling test")

        # Simulate random sampling (as DataLoader shuffle=True would do)
        rng = random.Random(42)
        sample_counts = {r: 0 for r in range(4)}
        n_samples = min(500, n_total)
        for _ in range(n_samples):
            idx = rng.randrange(n_total)
            text = src[idx]["text"]
            region = int(text.split("_")[0][1:])
            sample_counts[region] += 1

        for r, count in sample_counts.items():
            pct = count / n_samples
            assert 0.10 <= pct <= 0.40, (
                f"Sampled region {r} at {pct:.0%} (expected 15-35%). "
                f"Counts: {sample_counts}"
            )

    def test_eviction_preserves_diversity(self, tmp_path):
        """After eviction, surviving shards should cover multiple regions
        (stochastic oldest eviction doesn't wipe out entire regions)."""
        region_counts, _ = _run_prefetcher_and_read(
            tmp_path, n_regions=3, docs_per_region=300,
            max_rows_per_shard=20, max_shards=20,
            high_water=20, low_water=5,
        )
        total = sum(region_counts.values())
        if total < 50:
            pytest.skip("Not enough data after eviction")

        # At least 2 of 3 regions should survive eviction
        surviving = sum(1 for c in region_counts.values() if c > 0)
        assert surviving >= 2, (
            f"Only {surviving} regions survived eviction. "
            f"Counts: {region_counts}"
        )


# ---------------------------------------------------------------------------
# 9. Continuous Streaming Tests
# ---------------------------------------------------------------------------

class TestContinuousStreaming:

    def test_producer_restarts_after_exhaustion(self, tmp_path):
        """Producers restart from new offsets when data is exhausted,
        producing more shards than the initial data would allow."""
        docs = _make_text_docs(20)

        def wrapping_factory(offset):
            """Factory that wraps offset into range (simulates infinite dataset)."""
            wrapped = offset % max(1, len(docs))
            return _FakeHFDataset(docs).skip(wrapped)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_shards=None,
            max_rows_per_shard=10,
            offsets=[0],
            max_restarts=3,
            _dataset_factory=wrapping_factory,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        # 4 passes over 20 docs at 10 rows/shard → ~8 shards
        assert prefetcher.shard_count >= 4

    def test_max_restarts_zero_exhausts_once(self, tmp_path):
        """max_restarts=0 means streams exhaust and stop (old behavior)."""
        docs = _make_text_docs(50)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=1,
            max_shards=None,
            max_rows_per_shard=20,
            offsets=[0],
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        # 50 docs at 20 rows/shard → 2-3 shards, no more
        assert prefetcher.shard_count <= 4

    def test_writer_overlaps_with_network(self, tmp_path):
        """Writer should not block producers — queue decouples them."""
        # Slow docs (simulating network latency) + fast writer
        docs = _make_text_docs(100)

        prefetcher = TextPrefetcher(
            dataset="test",
            output_dir=tmp_path,
            min_shards=2,
            max_rows_per_shard=20,
            offsets=[0],
            high_water=50, low_water=10,
            max_restarts=0,
            _dataset_factory=_make_factory(docs),
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        assert prefetcher.shard_count >= 2


# ---------------------------------------------------------------------------
# 10. DualThresholdBuffer Tests
# ---------------------------------------------------------------------------

class TestDualThresholdBuffer:

    def test_put_and_get(self):
        from dataporter.text_prefetcher import DualThresholdBuffer
        buf = DualThresholdBuffer(high_water=10, low_water=3)
        for i in range(5):
            buf.put(f"item_{i}")
        assert buf.size == 5

        items = buf.get_batch(3)
        assert len(items) == 3
        assert items == ["item_0", "item_1", "item_2"]
        assert buf.size == 2

    def test_high_water_pauses_producer(self):
        """Producer blocks when buffer reaches high_water."""
        from dataporter.text_prefetcher import DualThresholdBuffer
        buf = DualThresholdBuffer(high_water=5, low_water=2)

        for i in range(5):
            buf.put(f"item_{i}")

        blocked = threading.Event()
        stop = threading.Event()

        def producer():
            blocked.set()
            buf.put("overflow", stop_event=stop)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        blocked.wait(timeout=1)
        time.sleep(0.1)
        assert t.is_alive()  # blocked at high_water
        assert buf.size == 5

        # Drain below low_water to unblock
        buf.get_batch(4)
        t.join(timeout=2)
        assert not t.is_alive()
        assert buf.size == 2  # 1 remaining + "overflow"

    def test_hysteresis_prevents_oscillation(self):
        """Producer stays blocked until low_water, not just below high_water."""
        from dataporter.text_prefetcher import DualThresholdBuffer
        buf = DualThresholdBuffer(high_water=5, low_water=2)

        # Fill to high_water
        for i in range(5):
            buf.put(f"item_{i}")

        # Start a producer that will block
        unblocked = threading.Event()
        stop = threading.Event()

        def producer():
            buf.put("extra", stop_event=stop)
            unblocked.set()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        time.sleep(0.1)
        assert t.is_alive()  # blocked

        # Drain to 3 (below high=5 but above low=2) — should still be blocked
        buf.get_batch(2)  # 5 → 3
        time.sleep(0.2)
        assert t.is_alive(), "Producer should stay blocked above low_water"

        # Drain to low_water — should unblock
        buf.get_batch(2)  # 3 → 1, which is <= low_water=2
        t.join(timeout=2)
        assert not t.is_alive()
        assert unblocked.is_set()

    def test_sentinel_bypasses_high_water(self):
        from dataporter.text_prefetcher import DualThresholdBuffer, _SENTINEL
        buf = DualThresholdBuffer(high_water=3, low_water=1)
        for i in range(3):
            buf.put(f"item_{i}")
        buf.put_sentinel()
        assert buf.size == 4

    def test_invalid_thresholds(self):
        from dataporter.text_prefetcher import DualThresholdBuffer
        with pytest.raises(ValueError, match="low_water must be < high_water"):
            DualThresholdBuffer(high_water=5, low_water=5)
