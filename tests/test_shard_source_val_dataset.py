"""Unit tests for :class:`ShardSourceValDataset` — the val-path
replacement for ``Subset(FastLeRobotDataset, val_idx)``.

Checks:

- ``__len__`` / ``__getitem__`` contract matches what Lightning's
  val loop expects from a map-style dataset.
- Iteration is deterministic for a given sample-index list.
- DataLoader with ``num_workers > 0`` works (picklable end-to-end).
- Delta-timestamp windowing flows through correctly.
- The dataset iterates a subset of frames, not the whole episode
  range — val's job is to be strict about which frames are val.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

import torch
from torch.utils.data import DataLoader

from dataporter.lerobot_shard_source import LeRobotShardSource
from dataporter.shard_source_val_dataset import ShardSourceValDataset
from test_shard_source_pool_e2e import _make_dataset


# ---------------------------------------------------------------------------
# Basic map-style contract
# ---------------------------------------------------------------------------


class TestMapStyleContract:

    def test_length_matches_sample_indices(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0, 1], total_episodes=2, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        pairs = [(0, i) for i in range(5)] + [(1, i) for i in range(5)]
        ds = ShardSourceValDataset(src, pairs)
        assert len(ds) == 10

    def test_getitem_returns_mapped_sample(self, tmp_path):
        """Indexing at ``i`` returns the sample at
        ``sample_indices[i]`` — ``frame_index`` reflects the
        frame-within-episode, not the global dataset index."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        pairs = [(0, 3), (0, 7)]
        ds = ShardSourceValDataset(src, pairs)
        first = ds[0]
        assert int(first["frame_index"]) == 3
        second = ds[1]
        assert int(second["frame_index"]) == 7

    def test_iteration_is_deterministic(self, tmp_path):
        """Two passes over the same dataset produce identical frame
        indices — critical for val-loss reproducibility."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0, 1], total_episodes=2, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        pairs = [(0, i) for i in range(10)] + [(1, i) for i in range(10)]
        ds = ShardSourceValDataset(src, pairs)
        pass1 = [int(ds[i]["frame_index"]) for i in range(len(ds))]
        pass2 = [int(ds[i]["frame_index"]) for i in range(len(ds))]
        assert pass1 == pass2


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:

    def test_single_worker_dataloader(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0, 1], total_episodes=2, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        pairs = [(0, i) for i in range(10)] + [(1, i) for i in range(10)]
        ds = ShardSourceValDataset(src, pairs)
        loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
        batches = list(loader)
        total = sum(b["frame_index"].shape[0] for b in batches)
        assert total == 20

    def test_multi_worker_dataloader_is_picklable(self, tmp_path):
        """With ``num_workers > 0`` the dataset has to survive the fork
        — the shard source's pickle-drop-cache trick is how."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0, 1], total_episodes=2, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        pairs = [(0, i) for i in range(10)] + [(1, i) for i in range(10)]
        ds = ShardSourceValDataset(src, pairs)
        loader = DataLoader(
            ds, batch_size=4, num_workers=2, shuffle=False,
        )
        # Just exercise the loader — assertion is "no exception during
        # fork + first batch".
        batches = list(loader)
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# Delta windowing + sample shape
# ---------------------------------------------------------------------------


class TestDeltaWindowing:

    def test_delta_returns_windowed_video(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, fps=30,
            n_frames_per_ep=15,
        )
        src = LeRobotShardSource(root)

        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]
        pairs = [(0, 5)]
        ds = ShardSourceValDataset(
            src, pairs, delta_timestamps={"observation.image": deltas},
        )
        item = ds[0]
        assert item["observation.image"].shape[0] == 3
        assert item["observation.image_is_pad"].shape == torch.Size([3])

    def test_no_delta_single_frame_shape(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        ds = ShardSourceValDataset(src, [(0, 3)])
        item = ds[0]
        # Matches SampleReader's single-frame output: [1, C, H, W].
        assert item["observation.image"].shape == (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Consistency with train-side reader
# ---------------------------------------------------------------------------


class TestTrainValConsistency:

    def test_same_sample_from_val_and_reader(self, tmp_path):
        """``ShardSourceValDataset`` and a bare :class:`SampleReader`
        built with the same config MUST produce identical samples at
        the same ``(raw_ep, frame_in_ep)``.  This is the invariant
        that unifies train/val."""
        from dataporter.sample_reader import SampleReader

        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=[0], total_episodes=1, n_frames_per_ep=10,
        )
        src = LeRobotShardSource(root)

        deltas = [-1.0 / 30.0, 0.0, 1.0 / 30.0]
        ds = ShardSourceValDataset(
            src, [(0, 5)], delta_timestamps={"observation.image": deltas},
        )
        reader = SampleReader(
            src, delta_timestamps={"observation.image": deltas},
        )

        from_ds = ds[0]
        from_reader = reader.read(0, 5)

        assert torch.equal(
            from_ds["observation.image"], from_reader["observation.image"],
        )
        assert torch.equal(
            from_ds["observation.image_is_pad"],
            from_reader["observation.image_is_pad"],
        )
        assert int(from_ds["frame_index"]) == int(from_reader["frame_index"])
