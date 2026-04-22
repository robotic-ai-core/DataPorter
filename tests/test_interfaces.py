"""Lock in that reference implementations satisfy their public Protocols.

These tests catch drift: if someone renames a method on
``LeRobotShardSource`` or ``LeRobotPrefetcher`` without updating the
Protocol (or vice versa), ``isinstance`` against the
``@runtime_checkable`` Protocol fails here before it fails in a
downstream consumer.

Protocols are the contract new modalities implement; reference
implementations prove the contract is satisfiable as-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.interfaces import (
    EpisodicPrefetcher,
    EpisodicSource,
    ProducerConfigProtocol,
    TemporalEpisodicSource,
)


# ---------------------------------------------------------------------------
# Fixture: minimal on-disk LeRobot v2.1 layout
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_lerobot_root(tmp_path):
    root = tmp_path / "tiny"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 40,
        "chunks_size": 1000,
        "features": {
            "observation.image": {"dtype": "video", "shape": [3, 32, 32]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": i, "length": 20, "tasks": ["t"]})
            for i in range(2)
        ) + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "t"}) + "\n"
    )
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    for ep in range(2):
        pq.write_table(
            pa.table({
                "frame_index": list(range(20)),
                "episode_index": [ep] * 20,
                "task_index": [0] * 20,
            }),
            data_dir / f"episode_{ep:06d}.parquet",
        )
    # Video files — empty stubs; readiness checks look for existence only.
    for ep in range(2):
        vd = root / "videos/chunk-000/observation.image"
        vd.mkdir(parents=True, exist_ok=True)
        (vd / f"episode_{ep:06d}.mp4").write_bytes(b"stub")
    return root


# ---------------------------------------------------------------------------
# EpisodicSource / TemporalEpisodicSource
# ---------------------------------------------------------------------------


class TestLeRobotShardSourceSatisfiesProtocols:

    def test_satisfies_episodic_source(self, tiny_lerobot_root):
        from dataporter import LeRobotShardSource
        src = LeRobotShardSource(tiny_lerobot_root)
        assert isinstance(src, EpisodicSource)

    def test_satisfies_temporal_episodic_source(self, tiny_lerobot_root):
        """LeRobotShardSource must expose the modality-neutral
        ``media_keys`` and ``episode_media_path`` names so Protocol-
        typed code can use it without knowing it's a video dataset.
        """
        from dataporter import LeRobotShardSource
        src = LeRobotShardSource(tiny_lerobot_root)
        assert isinstance(src, TemporalEpisodicSource)
        # media_keys aliases video_keys.
        assert src.media_keys == src.video_keys
        # episode_media_path resolves to the same file as episode_video_path.
        for ep in range(2):
            for mk in src.media_keys:
                assert src.episode_media_path(ep, mk) == (
                    src.episode_video_path(ep, mk)
                )

    def test_minimal_tabular_impl_satisfies_episodic_source(self):
        """A hand-rolled class with only the narrow surface satisfies
        the Protocol — proves Protocols don't force inheritance.

        Represents a hypothetical non-temporal modality (e.g. tabular
        timeseries episodes) where ``media_keys`` isn't meaningful.
        """
        class _TabularEpisodes:
            root = Path("/tmp/tabular")
            total_episodes = 3

            def episode_frame_count(self, raw_ep: int) -> int:
                return 10

            def load_episode_row_torch(self, raw_ep, frame_idx):
                return {"frame_index": frame_idx}

            def load_episode_window_torch(self, raw_ep, frame_indices):
                return {"frame_index": list(frame_indices)}

            def list_ready_episodes(self) -> list[int]:
                return [0, 1, 2]

            def tasks(self) -> dict[int, str]:
                return {}

        impl = _TabularEpisodes()
        assert isinstance(impl, EpisodicSource)
        # But NOT the temporal variant (no fps / media_keys).
        assert not isinstance(impl, TemporalEpisodicSource)


# ---------------------------------------------------------------------------
# EpisodicPrefetcher
# ---------------------------------------------------------------------------


class TestLeRobotPrefetcherSatisfiesProtocol:

    def test_satisfies_episodic_prefetcher(self):
        """The real LeRobotPrefetcher satisfies the narrow Protocol the
        consumer's ``refresh()`` polls."""
        from dataporter import LeRobotPrefetcher
        # A minimally-configured prefetcher; don't actually start it.
        pf = LeRobotPrefetcher(
            repo_id="lerobot/pusht",
            cache_dir="/tmp/_iface_test_cache",
            _snapshot_fn=lambda *a, **kw: None,
            _meta_loader=lambda *a, **kw: {},
        )
        assert isinstance(pf, EpisodicPrefetcher)


# ---------------------------------------------------------------------------
# ProducerConfigProtocol
# ---------------------------------------------------------------------------


class TestProducerConfigSatisfiesProtocol:

    def test_satisfies_producer_config_protocol(self, tiny_lerobot_root):
        from dataporter import LeRobotShardSource
        from dataporter.producer_pool import ProducerConfig

        cfg = ProducerConfig(
            source_name="s",
            repo_id="s",
            shard_source=LeRobotShardSource(tiny_lerobot_root),
            episode_indices=[0, 1],
        )
        assert isinstance(cfg, ProducerConfigProtocol)
