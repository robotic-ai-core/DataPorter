"""Tests for LeRobotPrefetcher.

All tests use mocked metadata and downloads — no network access needed.
"""

from __future__ import annotations

import json
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


def _fake_meta_loader(repo_id: str, output_dir: Path) -> dict:
    """Return fake metadata without touching HF."""
    # Write meta/info.json so other code can find it
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))
    return FAKE_META


def _make_fake_download_fn(source_dir: Path):
    """Create a download function that copies from source_dir instead of HF Hub.

    Expects remote format: "repo_id::relative_path[::revision]"
    """

    def download_fn(remote: str, local_path: Path):
        parts = remote.split("::")
        rel_path = parts[1]
        src = source_dir / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(src, local_path)

    return download_fn


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

    for ep_idx in range(n_episodes):
        chunk = ep_idx // chunks_size

        # Write Parquet
        parquet_dir = source_dir / f"data/chunk-{chunk:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        n_frames = 300

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLeRobotPrefetcher:

    def test_auto_detects_video_keys(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=3)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=output_dir,
            min_shards=1,
            max_shards=100,
            _download_fn=_make_fake_download_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )

        assert prefetcher._get_video_keys() == ["observation.image"]

    def test_downloads_episodes(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=5)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=output_dir,
            episode_indices=[0, 1, 2, 3, 4],
            min_shards=2,
            max_shards=100,
            companion_workers=2,
            _download_fn=_make_fake_download_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        # Parquet files downloaded
        parquets = list(output_dir.rglob("episode_*.parquet"))
        assert len(parquets) >= 2

        # Video files downloaded
        videos = list(output_dir.rglob("*.mp4"))
        assert len(videos) >= 2

        # Each episode has both parquet + video
        for pq_file in parquets:
            ep_name = pq_file.stem  # e.g., "episode_000000"
            video_files = list(output_dir.rglob(f"{ep_name}.mp4"))
            assert len(video_files) >= 1, f"Missing video for {ep_name}"

    def test_specific_episodes(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=10)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=output_dir,
            episode_indices=[2, 5, 7],
            min_shards=1,
            max_shards=100,
            _download_fn=_make_fake_download_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(1)
        prefetcher.stop()

        parquets = list(output_dir.rglob("episode_*.parquet"))
        ep_indices = {int(p.stem.split("_")[1]) for p in parquets}
        assert ep_indices == {2, 5, 7}

    def test_episode_paths(self, tmp_path):
        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=tmp_path,
            _meta_loader=_fake_meta_loader,
        )

        assert prefetcher._episode_parquet_path(0) == \
            "data/chunk-000/episode_000000.parquet"
        assert prefetcher._episode_parquet_path(42) == \
            "data/chunk-000/episode_000042.parquet"
        assert prefetcher._episode_parquet_path(1500) == \
            "data/chunk-001/episode_001500.parquet"

    def test_video_companion_refs(self, tmp_path):
        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=tmp_path,
            _meta_loader=_fake_meta_loader,
        )

        refs = prefetcher._episode_companion_refs(42)
        assert len(refs) == 1  # one video key
        assert refs[0].local == "videos/chunk-000/observation.image/episode_000042.mp4"
        assert "test/dataset" in refs[0].remote

    def test_no_video_keys(self, tmp_path):
        """Dataset without video features should have no companions."""
        meta_no_video = {**FAKE_META, "video_path": None, "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        }}

        def loader(repo_id, output_dir):
            meta_dir = output_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps(meta_no_video))
            return meta_no_video

        prefetcher = LeRobotPrefetcher(
            repo_id="test/no-video",
            output_dir=tmp_path,
            _meta_loader=loader,
        )

        assert prefetcher._get_video_keys() == []
        assert prefetcher._episode_companion_refs(0) == []

    def test_shard_count(self, tmp_path):
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=3)

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=output_dir,
            min_shards=3,
            max_shards=100,
            _download_fn=_make_fake_download_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        prefetcher.stop()

        assert prefetcher.shard_count == 3

    def test_download_failure_skips_episode(self, tmp_path):
        """If a download fails, the episode is skipped, not fatal."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        _populate_fake_dataset(source_dir, n_episodes=5)

        # Delete episode 2 to simulate download failure
        (source_dir / "data/chunk-000/episode_000002.parquet").unlink()

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            output_dir=output_dir,
            episode_indices=[0, 1, 2, 3, 4],
            min_shards=1,
            max_shards=100,
            _download_fn=_make_fake_download_fn(source_dir),
            _meta_loader=_fake_meta_loader,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # 4 episodes downloaded (episode 2 failed)
        parquets = list(output_dir.rglob("episode_*.parquet"))
        assert len(parquets) == 4
        ep_indices = {int(p.stem.split("_")[1]) for p in parquets}
        assert 2 not in ep_indices
