"""Tests for CompanionPool and companion-aware eviction.

CompanionPool is used by LeRobotPrefetcher for video co-download.
Integration with LeRobotPrefetcher is tested in test_lerobot_prefetcher.py.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.prefetcher import (
    CompanionPool,
    CompanionRef,
    evict_shard,
    _read_manifest,
    _write_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_shard(path: Path, n_rows: int = 10):
    schema = pa.schema([("input_ids", pa.list_(pa.uint16()))])
    rng = np.random.RandomState(0)
    rows = [rng.randint(0, 8000, 32).tolist() for _ in range(n_rows)]
    table = pa.table({"input_ids": rows}, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _make_companion_file(path: Path, content: bytes = b"fake video data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _slow_download(remote: str, local_path: Path):
    time.sleep(0.1)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(f"downloaded from {remote}")


def _instant_download(remote: str, local_path: Path):
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
            CompanionRef(remote="src1", local="video1.mp4"),
            CompanionRef(remote="src2", local="video2.mp4"),
        ]
        pool.submit("shard_000000.parquet", refs)
        pool.wait_ready("shard_000000.parquet", timeout=10)

        assert pool.is_ready("shard_000000.parquet")
        assert (tmp_path / "video1.mp4").exists()
        assert (tmp_path / "video2.mp4").exists()
        pool.shutdown()

    def test_no_companions_always_ready(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        assert pool.is_ready("nonexistent.parquet")
        pool.shutdown()

    def test_evict_deletes_files(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_instant_download)
        pool.submit("shard.parquet", [CompanionRef(remote="src", local="video.mp4")])
        pool.wait_ready("shard.parquet", timeout=10)
        assert (tmp_path / "video.mp4").exists()

        deleted = pool.evict("shard.parquet")
        assert len(deleted) == 1
        assert not (tmp_path / "video.mp4").exists()
        pool.shutdown()

    def test_evict_nonexistent_shard(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        assert pool.evict("nonexistent.parquet") == []
        pool.shutdown()

    def test_nested_companion_dirs(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=2, download_fn=_instant_download)
        pool.submit("shard.parquet", [
            CompanionRef(remote="src", local="videos/chunk-000/obs/ep_000.mp4"),
        ])
        pool.wait_ready("shard.parquet", timeout=10)

        assert (tmp_path / "videos/chunk-000/obs/ep_000.mp4").exists()
        pool.shutdown()

    def test_slow_download_not_ready_immediately(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_slow_download)
        pool.submit("shard.parquet", [CompanionRef(remote="src", local="video.mp4")])

        assert not pool.is_ready("shard.parquet")
        pool.wait_ready("shard.parquet", timeout=10)
        assert pool.is_ready("shard.parquet")
        pool.shutdown()

    def test_failed_download_not_ready(self, tmp_path):
        pool = CompanionPool(tmp_path, max_workers=1, download_fn=_failing_download)
        pool.submit("shard.parquet", [CompanionRef(remote="src", local="video.mp4")])

        assert not pool.wait_ready("shard.parquet", timeout=10)
        assert not pool.is_ready("shard.parquet")
        pool.shutdown()

    def test_get_companion_paths(self, tmp_path):
        pool = CompanionPool(tmp_path, download_fn=_instant_download)
        pool.submit("shard.parquet", [
            CompanionRef(remote="a", local="v1.mp4"),
            CompanionRef(remote="b", local="v2.mp4"),
        ])
        paths = pool.get_companion_paths("shard.parquet")
        assert len(paths) == 2
        pool.shutdown()


# ---------------------------------------------------------------------------
# 2. Manifest Tests
# ---------------------------------------------------------------------------

class TestManifest:

    def test_write_and_read(self, tmp_path):
        shard_path = tmp_path / "shard_000000.parquet"
        shard_path.touch()
        companions = ["videos/ep0.mp4", "videos/ep0_depth.mp4"]
        _write_manifest(shard_path, companions)
        assert _read_manifest(shard_path) == companions

    def test_read_missing_manifest(self, tmp_path):
        assert _read_manifest(tmp_path / "shard_000000.parquet") == []

    def test_empty_companions_no_manifest(self, tmp_path):
        shard_path = tmp_path / "shard_000000.parquet"
        shard_path.touch()
        _write_manifest(shard_path, [])
        assert not shard_path.with_suffix(".companions.json").exists()


# ---------------------------------------------------------------------------
# 3. Atomic Eviction Tests
# ---------------------------------------------------------------------------

class TestAtomicEviction:

    def test_evict_shard_with_companions(self, tmp_path):
        shard_dir = tmp_path / "shards"
        companion_dir = tmp_path / "companions"
        shard_dir.mkdir()
        companion_dir.mkdir()

        shard_path = shard_dir / "shard_000000.parquet"
        _write_test_shard(shard_path)
        comp_path = companion_dir / "videos" / "ep0.mp4"
        _make_companion_file(comp_path)
        _write_manifest(shard_path, ["videos/ep0.mp4"])

        victim = evict_shard(
            shard_dir, "fifo", random.Random(42), companion_dir=companion_dir
        )
        assert victim is not None
        assert not shard_path.exists()
        assert not comp_path.exists()
        assert not shard_path.with_suffix(".companions.json").exists()

    def test_evict_shard_without_companions(self, tmp_path):
        _write_test_shard(tmp_path / "shard_000000.parquet")
        victim = evict_shard(tmp_path, "fifo", random.Random(42))
        assert victim is not None
        assert not (tmp_path / "shard_000000.parquet").exists()

    def test_evict_with_companion_pool(self, tmp_path):
        shard_dir = tmp_path / "shards"
        comp_dir = tmp_path / "companions"
        shard_dir.mkdir()

        pool = CompanionPool(comp_dir, download_fn=_instant_download)
        pool.submit("shard_000000.parquet", [
            CompanionRef(remote="src", local="video.mp4"),
        ])
        pool.wait_ready("shard_000000.parquet", timeout=10)

        _write_test_shard(shard_dir / "shard_000000.parquet")

        victim = evict_shard(
            shard_dir, "fifo", random.Random(42),
            companion_pool=pool, companion_dir=comp_dir,
        )
        assert victim is not None
        assert not (comp_dir / "video.mp4").exists()
        pool.shutdown()
