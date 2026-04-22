"""Tests for LeRobotShardSource.

Covers the "live, lazy view" contract:
- Construction is O(1): info.json only, no per-episode parsing.
- Per-episode metadata loads on first access, cached thereafter.
- list_ready_episodes reflects current disk state (parquet + every
  video file must be present).
- Row access uses LRU, survives pickle.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# ---------------------------------------------------------------------------
# Fixture helpers — synthetic v2.1-layout LeRobot dataset on disk.
# ---------------------------------------------------------------------------


INFO_WITH_VIDEO = {
    "codebase_version": "v2.1",
    "data_path": (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ),
    "video_path": (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    ),
    "fps": 30,
    "total_episodes": 10,
    "total_frames": 3000,
    "total_tasks": 2,
    "total_chunks": 1,
    "chunks_size": 1000,
    "robot_type": "synthetic",
    "features": {
        "observation.image": {"dtype": "video", "shape": [3, 96, 96]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    },
}


INFO_NO_VIDEO = {
    **INFO_WITH_VIDEO,
    "video_path": None,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    },
}


def _write_meta(root: Path, info: dict, n_episodes: int) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info))
    # episodes.jsonl — one line per episode
    with (meta / "episodes.jsonl").open("w") as f:
        for i in range(n_episodes):
            f.write(json.dumps({
                "episode_index": i,
                "length": 100 + i,   # varied so frame-count tests matter
                "tasks": ["push_t"],
            }) + "\n")
    # tasks.jsonl
    with (meta / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": "push_t"}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "idle"}) + "\n")


def _write_episode_parquet(
    root: Path, ep_idx: int, chunks_size: int = 1000, n_rows: int = 100,
) -> None:
    chunk = ep_idx // chunks_size
    dir_ = root / f"data/chunk-{chunk:03d}"
    dir_.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
    ])
    rng = np.random.default_rng(ep_idx)
    table = pa.table({
        "observation.state": [
            [float(rng.standard_normal()), float(rng.standard_normal())]
            for _ in range(n_rows)
        ],
        "action": [
            [float(rng.standard_normal()), float(rng.standard_normal())]
            for _ in range(n_rows)
        ],
        "timestamp": [i / 30.0 for i in range(n_rows)],
        "episode_index": [ep_idx] * n_rows,
        "frame_index": list(range(n_rows)),
    }, schema=schema)
    pq.write_table(table, dir_ / f"episode_{ep_idx:06d}.parquet")


def _write_episode_videos(
    root: Path, ep_idx: int, video_keys: list[str],
    chunks_size: int = 1000,
) -> None:
    chunk = ep_idx // chunks_size
    for vk in video_keys:
        d = root / f"videos/chunk-{chunk:03d}/{vk}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"episode_{ep_idx:06d}.mp4").write_bytes(
            b"fake video " + str(ep_idx).encode()
        )


@pytest.fixture
def video_dataset(tmp_path):
    """Full v2.1 layout with 5 episodes, all parquet+video present."""
    root = tmp_path / "full_videos"
    _write_meta(root, INFO_WITH_VIDEO, n_episodes=5)
    for ep in range(5):
        _write_episode_parquet(root, ep)
        _write_episode_videos(root, ep, ["observation.image"])
    return root


@pytest.fixture
def partial_dataset(tmp_path):
    """5 parquets present, but only 3 have videos (eps 0, 1, 2).
    Eps 3 and 4 have parquet only — should be excluded from
    list_ready_episodes.
    """
    root = tmp_path / "partial"
    _write_meta(root, INFO_WITH_VIDEO, n_episodes=5)
    for ep in range(5):
        _write_episode_parquet(root, ep)
    for ep in range(3):
        _write_episode_videos(root, ep, ["observation.image"])
    return root


# ---------------------------------------------------------------------------
# Construction + global metadata
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_init_loads_info_not_episodes(self, video_dataset):
        """Construction should NOT parse episodes.jsonl.  Verify by
        deleting the file after init and checking that construction was
        still successful while per-episode accessors later fail.
        """
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        # Global metadata is accessible immediately.
        assert src.fps == 30
        assert src.total_episodes == 10
        assert src.chunks_size == 1000
        assert src.video_keys == ["observation.image"]

        # Delete episodes.jsonl — construction succeeded without it.
        (video_dataset / "meta" / "episodes.jsonl").unlink()
        # Per-episode metadata will fail on first access, not at
        # construction.  Confirms laziness.
        with pytest.raises(FileNotFoundError, match="episodes.jsonl"):
            src.episode_frame_count(0)

    def test_init_missing_info_raises_clear_error(self, tmp_path):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="meta/info.json"):
            LeRobotShardSource(empty)

    def test_init_nonexistent_root_raises(self, tmp_path):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        with pytest.raises(ValueError, match="not a directory"):
            LeRobotShardSource(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Path templates + per-episode metadata
# ---------------------------------------------------------------------------


class TestEpisodeAccessors:
    def test_episode_parquet_path_matches_template(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        assert src.episode_parquet_path(3) == (
            video_dataset / "data/chunk-000/episode_000003.parquet"
        )

    def test_episode_video_path_matches_template(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        assert src.episode_video_path(3, "observation.image") == (
            video_dataset
            / "videos/chunk-000/observation.image/episode_000003.mp4"
        )

    def test_video_path_raises_on_no_video_dataset(self, tmp_path):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        root = tmp_path / "no_video"
        _write_meta(root, INFO_NO_VIDEO, n_episodes=3)
        for ep in range(3):
            _write_episode_parquet(root, ep)
        src = LeRobotShardSource(root)
        assert src.video_keys == []
        with pytest.raises(RuntimeError, match="no 'video_path' template"):
            src.episode_video_path(0, "any")

    def test_episode_frame_count_uses_episodes_jsonl(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        # episodes.jsonl wrote length=100+ep_idx
        assert src.episode_frame_count(0) == 100
        assert src.episode_frame_count(4) == 104

    def test_tasks_returns_mapping(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        assert src.tasks() == {0: "push_t", 1: "idle"}

    def test_tasks_cached_across_calls(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        t1 = src.tasks()
        # Delete the file; second call should hit cache.
        (video_dataset / "meta" / "tasks.jsonl").unlink()
        t2 = src.tasks()
        assert t1 is t2     # same cached dict instance


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_list_ready_episodes_full(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        assert src.list_ready_episodes() == [0, 1, 2, 3, 4]

    def test_list_ready_excludes_missing_videos(self, partial_dataset):
        """The prefetcher vs consumer race: eps with parquet but no
        video must NOT appear as ready.
        """
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(partial_dataset)
        # Eps 3 and 4 have parquet but no video → excluded.
        assert src.list_ready_episodes() == [0, 1, 2]

    def test_is_episode_ready(self, partial_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(partial_dataset)
        assert src.is_episode_ready(0) is True
        assert src.is_episode_ready(3) is False   # missing video
        assert src.is_episode_ready(99) is False  # missing parquet

    def test_list_ready_no_video_dataset_uses_parquet_only(self, tmp_path):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        root = tmp_path / "no_video"
        _write_meta(root, INFO_NO_VIDEO, n_episodes=3)
        for ep in range(3):
            _write_episode_parquet(root, ep)
        src = LeRobotShardSource(root)
        assert src.list_ready_episodes() == [0, 1, 2]

    def test_list_ready_reflects_live_disk_state(self, video_dataset):
        """Core contract: the source is a LIVE view.  Adding a new
        episode after construction must appear in list_ready_episodes.
        """
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        assert len(src.list_ready_episodes()) == 5

        # Admit a new episode to disk while the source is live.
        _write_episode_parquet(video_dataset, 7)
        _write_episode_videos(
            video_dataset, 7, ["observation.image"],
        )
        assert 7 in src.list_ready_episodes()


# ---------------------------------------------------------------------------
# Row loading + LRU
# ---------------------------------------------------------------------------


class TestRowAccess:
    def test_load_episode_rows_returns_pyarrow_table(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        table = src.load_episode_rows(0)
        assert table.num_rows == 100
        assert "action" in table.column_names
        assert "timestamp" in table.column_names

    def test_load_episode_rows_caches(self, video_dataset):
        """Second call must hit the cache (no disk read)."""
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        table1 = src.load_episode_rows(0)
        # Delete the file — second call should still return from cache.
        src.episode_parquet_path(0).unlink()
        table2 = src.load_episode_rows(0)
        assert table1 is table2

    def test_load_episode_rows_lru_evicts(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset, rows_cache_maxsize=2)
        src.load_episode_rows(0)
        src.load_episode_rows(1)
        src.load_episode_rows(2)     # evicts ep 0
        # Deleting ep 0's parquet would break a cache-miss reload.
        src.episode_parquet_path(0).unlink()
        with pytest.raises(FileNotFoundError):
            src.load_episode_rows(0)

    def test_load_episode_row_dict(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        row = src.load_episode_row_dict(0, 5)
        assert row["frame_index"] == 5
        assert row["episode_index"] == 0

    def test_load_episode_window(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        window = src.load_episode_window(0, [1, 3, 5])
        assert window.num_rows == 3
        assert window.column("frame_index").to_pylist() == [1, 3, 5]

    def test_load_episode_rows_missing_file_raises(self, partial_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(partial_dataset)
        src.episode_parquet_path(0).unlink()  # now missing
        with pytest.raises(FileNotFoundError, match="not on disk"):
            src.load_episode_rows(0)


# ---------------------------------------------------------------------------
# Pickle compatibility (spawn child will unpickle)
# ---------------------------------------------------------------------------


class TestPickling:
    def test_pickle_round_trip_preserves_metadata(self, video_dataset):
        from dataporter.lerobot_shard_source import LeRobotShardSource

        src = LeRobotShardSource(video_dataset)
        src.tasks()                        # populate cache
        src.load_episode_rows(0)           # populate LRU
        src.episode_frame_count(0)         # populate episode_lengths

        blob = pickle.dumps(src)
        restored = pickle.loads(blob)
        assert restored.root == src.root
        assert restored.fps == 30
        assert restored.tasks() == src.tasks()
        assert restored.episode_frame_count(0) == 100
        # Row cache dropped on pickle, will warm back up on first use.
        assert len(restored._rows_cache) == 0
        t = restored.load_episode_rows(0)
        assert t.num_rows == 100
