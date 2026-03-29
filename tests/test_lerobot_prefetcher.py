"""Tests for LeRobotPrefetcher.

All tests use mocked metadata and snapshot_download — no network access needed.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.lerobot_prefetcher import LeRobotPrefetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_META = {
    "codebase_version": "v2.1",
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "fps": 30,
    "total_episodes": 10,
    "total_frames": 3000,
    "total_tasks": 1,
    "total_chunks": 1,
    "chunks_size": 1000,
    "robot_type": "pusht",
    "features": {
        "observation.image": {"dtype": "video", "shape": [3, 96, 96]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    },
}


def _fake_meta_loader(repo_id: str, cache_dir: Path) -> dict:
    meta_dir = cache_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))
    return FAKE_META


def _populate_fake_dataset(source_dir: Path, n_episodes: int = 10):
    """Create a fake LeRobot-style dataset on disk."""
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
    ])

    rng = np.random.RandomState(42)
    chunks_size = FAKE_META["chunks_size"]

    # Write meta
    meta_dir = source_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))

    for ep_idx in range(n_episodes):
        chunk = ep_idx // chunks_size
        n_frames = 300

        # Write Parquet
        parquet_dir = source_dir / f"data/chunk-{chunk:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "observation.state": [[float(rng.randn()), float(rng.randn())] for _ in range(n_frames)],
            "action": [[float(rng.randn()), float(rng.randn())] for _ in range(n_frames)],
            "timestamp": [i / 30.0 for i in range(n_frames)],
            "episode_index": [ep_idx] * n_frames,
            "frame_index": list(range(n_frames)),
        }, schema=schema)
        pq.write_table(table, parquet_dir / f"episode_{ep_idx:06d}.parquet")

        # Write video
        for vid_key in ["observation.image"]:
            video_dir = source_dir / f"videos/chunk-{chunk:03d}/{vid_key}"
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / f"episode_{ep_idx:06d}.mp4").write_bytes(
                b"fake video " + str(ep_idx).encode()
            )


def _make_fake_snapshot_fn(source_dir: Path):
    """Create a mock snapshot_download that copies from source_dir.

    Respects allow_patterns if provided (simple glob matching).
    """
    import fnmatch

    def snapshot_fn(repo_id, cache_dir, allow_patterns=None, ignore_patterns=None):
        cache_dir = Path(cache_dir)
        for src_file in source_dir.rglob("*"):
            if not src_file.is_file():
                continue
            rel = str(src_file.relative_to(source_dir))

            # Filter by allow_patterns
            if allow_patterns is not None:
                if not any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    continue

            # Filter by ignore_patterns
            if ignore_patterns is not None:
                if any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                    continue

            dst = cache_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

    return snapshot_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLeRobotPrefetcher:

    def test_auto_detects_video_keys(self, tmp_path):
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=3)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            min_shards=1,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )

        assert prefetcher._get_video_keys() == ["observation.image"]

    def test_bulk_downloads_all_episodes(self, tmp_path):
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=5)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=[0, 1, 2, 3, 4],
            min_shards=2,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # Parquet files downloaded
        parquets = list(cache_dir.rglob("episode_*.parquet"))
        assert len(parquets) == 5

        # Video files downloaded
        videos = list(cache_dir.rglob("*.mp4"))
        assert len(videos) == 5

        # Each episode has both parquet + video
        for pq_file in parquets:
            ep_name = pq_file.stem
            video_files = list(cache_dir.rglob(f"{ep_name}.mp4"))
            assert len(video_files) >= 1, f"Missing video for {ep_name}"

    def test_specific_episodes(self, tmp_path):
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=10)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=[2, 5, 7],
            min_shards=1,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        parquets = list(cache_dir.rglob("episode_*.parquet"))
        ep_indices = {int(p.stem.split("_")[1]) for p in parquets}
        assert ep_indices == {2, 5, 7}

    def test_episode_paths(self, tmp_path):
        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=tmp_path,
            _meta_loader=_fake_meta_loader,
        )

        assert prefetcher._episode_parquet_path(0) == \
            "data/chunk-000/episode_000000.parquet"
        assert prefetcher._episode_parquet_path(42) == \
            "data/chunk-000/episode_000042.parquet"
        assert prefetcher._episode_parquet_path(1500) == \
            "data/chunk-001/episode_001500.parquet"

    def test_episode_patterns(self, tmp_path):
        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=tmp_path,
            _meta_loader=_fake_meta_loader,
        )

        patterns = prefetcher._episode_patterns([0, 1])
        assert "data/chunk-000/episode_000000.parquet" in patterns
        assert "data/chunk-000/episode_000001.parquet" in patterns
        # Video patterns
        assert any("observation.image" in p and "000000.mp4" in p for p in patterns)
        assert any("observation.image" in p and "000001.mp4" in p for p in patterns)

    def test_no_video_keys(self, tmp_path):
        meta_no_video = {**FAKE_META, "video_path": None, "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        }}

        def loader(repo_id, cache_dir):
            meta_dir = cache_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps(meta_no_video))
            return meta_no_video

        prefetcher = LeRobotPrefetcher(
            repo_id="test/no-video",
            cache_dir=tmp_path,
            _meta_loader=loader,
        )

        assert prefetcher._get_video_keys() == []
        assert prefetcher._episode_patterns([0]) == [
            "data/chunk-000/episode_000000.parquet",
        ]

    def test_shard_count(self, tmp_path):
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=3)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=[0, 1, 2],
            min_shards=3,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        assert prefetcher.shard_count == 3

    def test_download_all_when_no_indices(self, tmp_path):
        """Without episode_indices, downloads everything."""
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=5)

        # Update meta to reflect 5 episodes
        meta = {**FAKE_META, "total_episodes": 5}

        def loader(repo_id, cache_dir):
            meta_dir = cache_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps(meta))
            return meta

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            min_shards=1,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        parquets = list(cache_dir.rglob("episode_*.parquet"))
        assert len(parquets) == 5

    def test_snapshot_fn_receives_patterns(self, tmp_path):
        """Verify that specific episodes use allow_patterns."""
        calls = []

        def tracking_snapshot(repo_id, cache_dir, allow_patterns=None, ignore_patterns=None):
            calls.append({
                "repo_id": repo_id,
                "allow_patterns": allow_patterns,
            })
            # Create dummy files so shard_count works
            cache_dir = Path(cache_dir)
            for p in (allow_patterns or []):
                if p.endswith(".parquet"):
                    f = cache_dir / p
                    f.parent.mkdir(parents=True, exist_ok=True)
                    schema = pa.schema([("x", pa.int64())])
                    pq.write_table(pa.table({"x": [1]}, schema=schema), f)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=tmp_path,
            episode_indices=[0, 3],
            min_shards=1,
            _snapshot_fn=tracking_snapshot,
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        assert len(calls) == 1
        patterns = calls[0]["allow_patterns"]
        assert "meta/*" in patterns
        assert "data/chunk-000/episode_000000.parquet" in patterns
        assert "data/chunk-000/episode_000003.parquet" in patterns

    def test_snapshot_fn_none_patterns_for_all(self, tmp_path):
        """Without episode_indices, snapshot_download gets no allow_patterns."""
        calls = []

        def tracking_snapshot(repo_id, cache_dir, allow_patterns=None, ignore_patterns=None):
            calls.append({"allow_patterns": allow_patterns})

        meta = {**FAKE_META, "total_episodes": 3}

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=tmp_path,
            min_shards=1,
            _snapshot_fn=tracking_snapshot,
            _meta_loader=lambda r, d: meta,
        )
        prefetcher.start()
        time.sleep(0.5)
        prefetcher.stop()

        assert len(calls) == 1
        # No patterns = download everything
        assert calls[0]["allow_patterns"] is None
