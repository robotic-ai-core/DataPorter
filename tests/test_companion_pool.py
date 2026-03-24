"""Tests for CompanionPool and companion-aware prefetching.

Tests:
  1. CompanionPool: submit, is_ready, wait_ready, evict, shutdown
  2. Atomic eviction: shard + companions deleted together
  3. Manifest persistence: survives restart
  4. Prefetcher with companions: end-to-end text + companion co-download
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.prefetcher import (
    CompanionPool,
    CompanionRef,
    ParquetPrefetcher,
    _evict_shard,
    _read_manifest,
    _write_manifest,
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


def _make_companion_file(path: Path, content: bytes = b"fake video data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _slow_download(remote: str, local_path: Path):
    """Simulated slow download (for testing async behavior)."""
    time.sleep(0.1)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(f"downloaded from {remote}")


def _instant_download(remote: str, local_path: Path):
    """Instant download for testing."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(f"downloaded from {remote}")


def _failing_download(remote: str, local_path: Path):
    raise ConnectionError(f"Failed to download {remote}")


# ---------------------------------------------------------------------------
# 1. CompanionPool Tests
# ---------------------------------------------------------------------------

class TestCompanionPool:

    def test_submit_and_ready(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=2, download_fn=_instant_download)
        refs = [
            CompanionRef(remote="http://example.com/video1.mp4", local="video1.mp4"),
            CompanionRef(remote="http://example.com/video2.mp4", local="video2.mp4"),
        ]
        pool.submit("shard_000000.parquet", refs)
        pool.wait_ready("shard_000000.parquet", timeout=10)

        assert pool.is_ready("shard_000000.parquet")
        assert (tmp_path / "video1.mp4").exists()
        assert (tmp_path / "video2.mp4").exists()
        pool.shutdown()

    def test_no_companions_always_ready(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        assert pool.is_ready("shard_nonexistent.parquet")
        pool.shutdown()

    def test_evict_deletes_files(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_instant_download)
        refs = [CompanionRef(remote="src", local="video.mp4")]
        pool.submit("shard_000.parquet", refs)
        pool.wait_ready("shard_000.parquet", timeout=10)
        assert (tmp_path / "video.mp4").exists()

        deleted = pool.evict("shard_000.parquet")
        assert len(deleted) == 1
        assert not (tmp_path / "video.mp4").exists()
        pool.shutdown()

    def test_evict_nonexistent_shard(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        deleted = pool.evict("nonexistent.parquet")
        assert deleted == []
        pool.shutdown()

    def test_nested_companion_dirs(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=2, download_fn=_instant_download)
        refs = [
            CompanionRef(
                remote="src",
                local="videos/chunk-000/obs.laptop/episode_000.mp4",
            ),
        ]
        pool.submit("shard_000.parquet", refs)
        pool.wait_ready("shard_000.parquet", timeout=10)

        expected = tmp_path / "videos/chunk-000/obs.laptop/episode_000.mp4"
        assert expected.exists()
        pool.shutdown()

    def test_slow_download_not_ready_immediately(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_slow_download)
        refs = [CompanionRef(remote="src", local="video.mp4")]
        pool.submit("shard_000.parquet", refs)

        # Not ready immediately
        assert not pool.is_ready("shard_000.parquet")

        # But ready after waiting
        pool.wait_ready("shard_000.parquet", timeout=10)
        assert pool.is_ready("shard_000.parquet")
        pool.shutdown()

    def test_failed_download_not_ready(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_failing_download)
        refs = [CompanionRef(remote="src", local="video.mp4")]
        pool.submit("shard_000.parquet", refs)

        result = pool.wait_ready("shard_000.parquet", timeout=10)
        assert not result
        assert not pool.is_ready("shard_000.parquet")
        pool.shutdown()

    def test_get_companion_paths(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        refs = [
            CompanionRef(remote="a", local="v1.mp4"),
            CompanionRef(remote="b", local="v2.mp4"),
        ]
        pool.submit("shard.parquet", refs)
        paths = pool.get_companion_paths("shard.parquet")
        assert len(paths) == 2
        assert tmp_path / "v1.mp4" in paths
        assert tmp_path / "v2.mp4" in paths
        pool.shutdown()


# ---------------------------------------------------------------------------
# 2. Manifest Tests
# ---------------------------------------------------------------------------

class TestManifest:

    def test_write_and_read(self, tmp_path):
        shard_path = tmp_path / "shard_000000.parquet"
        shard_path.touch()  # create empty file
        companions = ["videos/ep0.mp4", "videos/ep0_depth.mp4"]
        _write_manifest(shard_path, companions)

        result = _read_manifest(shard_path)
        assert result == companions

    def test_read_missing_manifest(self, tmp_path):
        shard_path = tmp_path / "shard_000000.parquet"
        assert _read_manifest(shard_path) == []

    def test_empty_companions_no_manifest(self, tmp_path):
        shard_path = tmp_path / "shard_000000.parquet"
        shard_path.touch()
        _write_manifest(shard_path, [])
        # No manifest written for empty list
        assert not shard_path.with_suffix(".companions.json").exists()


# ---------------------------------------------------------------------------
# 3. Atomic Eviction Tests
# ---------------------------------------------------------------------------

class TestAtomicEviction:

    def test_evict_shard_with_companions(self, tmp_path):
        """Evicting a shard also deletes its companion files via manifest."""
        shard_dir = tmp_path / "shards"
        companion_dir = tmp_path / "companions"
        shard_dir.mkdir()
        companion_dir.mkdir()

        # Write shard + manifest + companion file
        shard_path = shard_dir / "shard_000000.parquet"
        _write_test_shard(shard_path)
        comp_path = companion_dir / "videos" / "ep0.mp4"
        _make_companion_file(comp_path)
        _write_manifest(shard_path, ["videos/ep0.mp4"])

        victim = _evict_shard(
            shard_dir, "fifo", random.Random(42),
            companion_dir=companion_dir,
        )
        assert victim is not None
        assert not shard_path.exists()
        assert not comp_path.exists()
        assert not shard_path.with_suffix(".companions.json").exists()

    def test_evict_shard_without_companions(self, tmp_path):
        """Evicting a shard without companions just deletes the shard."""
        _write_test_shard(tmp_path / "shard_000000.parquet")

        victim = _evict_shard(tmp_path, "fifo", random.Random(42))
        assert victim is not None
        assert not (tmp_path / "shard_000000.parquet").exists()

    def test_evict_with_companion_pool(self, tmp_path):
        """Eviction via CompanionPool also cleans up."""
        shard_dir = tmp_path / "shards"
        comp_dir = tmp_path / "companions"
        shard_dir.mkdir()

        pool = CompanionPool(comp_dir, download_fn=_instant_download)
        refs = [CompanionRef(remote="src", local="video.mp4")]
        pool.submit("shard_000000.parquet", refs)
        pool.wait_ready("shard_000000.parquet", timeout=10)

        _write_test_shard(shard_dir / "shard_000000.parquet")

        victim = _evict_shard(
            shard_dir, "fifo", random.Random(42),
            companion_pool=pool,
            companion_dir=comp_dir,
        )
        assert victim is not None
        assert not (comp_dir / "video.mp4").exists()
        pool.shutdown()


# ---------------------------------------------------------------------------
# 4. Prefetcher + Companion Integration
# ---------------------------------------------------------------------------

class _FakeHFDataset:
    def __init__(self, docs):
        self._docs = docs
        self._offset = 0

    def skip(self, n):
        c = _FakeHFDataset(self._docs)
        c._offset = n
        return c

    def shuffle(self, seed=42, buffer_size=1000):
        rng = random.Random(seed)
        d = self._docs.copy()
        rng.shuffle(d)
        c = _FakeHFDataset(d)
        c._offset = self._offset
        return c

    def __iter__(self):
        for doc in self._docs[self._offset:]:
            yield doc


def _simple_transform(doc):
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


def _image_resolver(doc):
    """Extract image companion refs from a doc."""
    image_path = doc.get("image_path")
    if image_path:
        return [CompanionRef(remote=image_path, local=image_path)]
    return []


class TestPrefetcherWithCompanions:

    def test_text_plus_image_companions(self, tmp_path):
        """Prefetcher writes text shards and co-downloads image companions."""
        # Create fake "remote" images
        remote_dir = tmp_path / "remote_images"
        remote_dir.mkdir()
        docs = []
        for i in range(100):
            img_name = f"img_{i:04d}.jpg"
            (remote_dir / img_name).write_bytes(b"fake image " + str(i).encode())
            docs.append({
                "text": f"caption for image {i} " * 10,
                "image_path": str(remote_dir / img_name),
            })

        companion_dir = tmp_path / "companions"
        output_dir = tmp_path / "shards"
        fake_ds = _FakeHFDataset(docs)

        def resolver(doc):
            path = doc.get("image_path", "")
            if path:
                name = Path(path).name
                return [CompanionRef(remote=path, local=name)]
            return []

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=output_dir,
            transform=_simple_transform,
            companion_resolver=resolver,
            companion_dir=companion_dir,
            companion_workers=2,
            min_shards=2,
            max_shards=50,
            max_rows_per_shard=50,
            row_group_size=25,
            _dataset_factory=lambda src: fake_ds,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # Shards were written
        assert prefetcher.shard_count >= 2

        # Companions were downloaded
        companion_files = list(companion_dir.glob("*.jpg"))
        assert len(companion_files) > 0

        # Manifests were written
        manifests = list(output_dir.glob("*.companions.json"))
        assert len(manifests) > 0

    def test_no_companion_resolver_no_pool(self, tmp_path):
        """Without companion_resolver, no CompanionPool is created."""
        docs = [{"text": f"doc {i} " * 10} for i in range(50)]
        fake_ds = _FakeHFDataset(docs)

        prefetcher = ParquetPrefetcher(
            sources=[{"dataset": "test"}],
            output_dir=tmp_path,
            transform=_simple_transform,
            min_shards=1,
            max_shards=50,
            max_rows_per_shard=100,
            _dataset_factory=lambda src: fake_ds,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        assert prefetcher._companion_pool is None
        assert prefetcher.shard_count >= 1

    def test_companion_resolver_requires_dir(self):
        with pytest.raises(ValueError, match="companion_dir is required"):
            ParquetPrefetcher(
                sources=[{"dataset": "x"}],
                output_dir="/tmp/test",
                companion_resolver=lambda doc: [],
                # companion_dir omitted
            )

    def test_ready_shard_count_with_companions(self, tmp_path):
        """ready_shard_count only counts shards with all companions downloaded."""
        output_dir = tmp_path / "shards"
        companion_dir = tmp_path / "companions"
        output_dir.mkdir()

        pool = CompanionPool(companion_dir, download_fn=_slow_download)

        # Write 2 shards
        _write_test_shard(output_dir / "shard_000000.parquet")
        _write_test_shard(output_dir / "shard_000001.parquet")

        # Submit slow companions for shard 0 only
        pool.submit("shard_000000.parquet", [
            CompanionRef(remote="src", local="video0.mp4"),
        ])
        # shard_000001 has no companions -> always ready

        # shard_000000 not ready yet (slow download)
        from dataporter.prefetcher import _count_ready_shards
        # shard_000001 is ready (no companions), shard_000000 not yet
        ready = _count_ready_shards(output_dir, pool)
        assert ready == 1  # only shard_001 is ready

        pool.wait_ready("shard_000000.parquet", timeout=10)
        ready = _count_ready_shards(output_dir, pool)
        assert ready == 2
        pool.shutdown()
