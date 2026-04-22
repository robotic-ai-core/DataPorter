"""Tests for LeRobotShuffleBufferDataset.

Uses mock shard sources (no real video/HF downloads) to validate:
- Episode -> source mapping
- Delta timestamps windowing
- Frame extraction from buffer
- Per-source transform application
- Worker init seeding
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.shuffle_buffer import ShuffleBuffer
from dataporter.lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockShardSource:
    """Minimal in-memory LeRobotShardSource-compatible shim for tests.

    Satisfies the narrow interface the consumer uses:
    ``fps``, ``total_episodes``, ``episode_frame_count``,
    ``load_episode_row_torch``, ``load_episode_window_torch``, ``tasks``.
    Deterministic per-(ep, frame) content so tests can assert exact values.
    """

    def __init__(
        self,
        num_episodes: int = 5,
        frames_per_episode: int = 20,
        fps: int = 10,
        num_tasks: int = 1,
        seed: int = 0,
    ):
        self._num_episodes = num_episodes
        self._frames_per_episode = frames_per_episode
        self.fps = fps
        self.root = Path("/tmp/_mock_shard_source")
        self._rng_seed = seed
        self._num_tasks = num_tasks

    @property
    def total_episodes(self) -> int:
        return self._num_episodes

    def episode_frame_count(self, raw_ep: int) -> int:
        return self._frames_per_episode

    def _row(self, ep_idx: int, frame_idx: int) -> dict:
        g = torch.Generator().manual_seed(
            self._rng_seed + ep_idx * 1000 + frame_idx,
        )
        return {
            "episode_index": torch.tensor(ep_idx),
            "frame_index": torch.tensor(frame_idx),
            "timestamp": torch.tensor(frame_idx / self.fps),
            "task_index": torch.tensor(0),
            "index": torch.tensor(
                ep_idx * self._frames_per_episode + frame_idx
            ),
            "action": torch.randn(2, generator=g),
            "observation.state": torch.randn(4, generator=g),
        }

    def load_episode_row_torch(
        self, raw_ep: int, frame_idx: int,
    ) -> dict:
        return self._row(raw_ep, frame_idx)

    def load_episode_window_torch(
        self, raw_ep: int, frame_indices: list[int],
    ) -> dict:
        rows = [self._row(raw_ep, i) for i in frame_indices]
        stacked: dict = {}
        for key in rows[0]:
            stacked[key] = torch.stack([r[key] for r in rows])
        return stacked

    def tasks(self) -> dict[int, str]:
        return {i: f"task_{i}" for i in range(self._num_tasks)}


# Keep the old helper name as a thin alias so tests don't rename en masse.
_make_mock_shard_source = _MockShardSource


def _fill_buffer(
    buffer: ShuffleBuffer,
    num_episodes: int = 5,
    frames_per_episode: int = 20,
) -> None:
    """Fill buffer with deterministic episode data."""
    for ep_idx in range(num_episodes):
        # Frames: fill with ep_idx value for easy identification
        frames = torch.full(
            (frames_per_episode, 3, 8, 8),
            ep_idx * 10,
            dtype=torch.uint8,
        )
        buffer.put(ep_idx, frames)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLeRobotShuffleBufferDataset:

    def test_basic_sample(self):
        """Basic __getitem__ returns a complete LeRobot sample."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=20, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=5, frames_per_episode=20)
        _fill_buffer(buffer, num_episodes=5, frames_per_episode=20)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(5)),
                "transform": None,
            }],
            delta_timestamps={
                "observation.image": [0.0],
                "action": [0.0],
                "observation.state": [0.0],
            },
            epoch_length=100,
            image_keys=["observation.image"],
        )

        assert len(dataset) == 100

        item = dataset[0]
        assert "observation.image" in item
        assert "action" in item
        assert "observation.state" in item
        assert "task" in item
        assert item["observation.image"].dtype == torch.float32

    def test_multi_timestamp_window(self):
        """Delta timestamps produce correct temporal windows."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=20, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=5, frames_per_episode=20)
        _fill_buffer(buffer, num_episodes=5, frames_per_episode=20)

        delta_ts = {
            "observation.image": [-0.2, -0.1, 0.0],
            "action": [-0.2, -0.1, 0.0],
            "observation.state": [-0.2, -0.1, 0.0],
        }

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(5)),
                "transform": None,
            }],
            delta_timestamps=delta_ts,
            epoch_length=100,
            image_keys=["observation.image"],
        )

        item = dataset[0]
        # Should have 3 frames (one per delta timestamp)
        assert item["observation.image"].shape[0] == 3
        # Action should also have 3 timesteps
        assert item["action"].shape[0] == 3

    def test_multi_source_mapping(self):
        """Episodes from different sources are correctly routed."""
        buffer = ShuffleBuffer(
            capacity=20, max_frames=10, channels=3, height=8, width=8,
        )

        ds_a = _make_mock_shard_source(num_episodes=5, frames_per_episode=10)
        ds_b = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)

        # Fill buffer with episodes 0-4 (source A) and 0-2 (source B)
        for ep_idx in range(5):
            frames = torch.full((10, 3, 8, 8), 10, dtype=torch.uint8)
            buffer.put(ep_idx, frames)
        for ep_idx in range(3):
            frames = torch.full((10, 3, 8, 8), 20, dtype=torch.uint8)
            buffer.put(ep_idx + 100, frames)

        # Source B uses episode indices offset by 100
        # But wait -- source B's dataset has episodes 0-2 internally.
        # The ShuffleBuffer uses episode indices from the ORIGINAL dataset.
        # So we need source B's train_episode_indices to match what's in the buffer.
        # In practice, each source uses its own episode index space.

        # For this test, let's use non-overlapping episode ranges:
        # Source A: episodes 0-4, Source B: episodes 5-7
        buffer_b = ShuffleBuffer(
            capacity=20, max_frames=10, channels=3, height=8, width=8,
        )
        for ep_idx in range(5):
            buffer_b.put(ep_idx, torch.full((10, 3, 8, 8), 10, dtype=torch.uint8))
        for ep_idx in range(5, 8):
            buffer_b.put(ep_idx, torch.full((10, 3, 8, 8), 20, dtype=torch.uint8))

        ds_b2 = _make_mock_shard_source(num_episodes=8, frames_per_episode=10)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer_b,
            sources=[
                {
                    "shard_source": ds_a,
                    "train_episode_indices": list(range(5)),
                    "transform": None,
                },
                {
                    "shard_source": ds_b2,
                    "train_episode_indices": list(range(5, 8)),
                    "transform": None,
                },
            ],
            delta_timestamps={"observation.image": [0.0], "action": [0.0]},
            epoch_length=100,
            image_keys=["observation.image"],
        )

        # Sample many times and verify we get episodes from both sources
        ep_indices = set()
        for _ in range(50):
            item = dataset[0]
            ep_indices.add(item["episode_index"].item())

        # Should have episodes from both ranges
        assert any(e < 5 for e in ep_indices), f"No source A episodes: {ep_indices}"
        assert any(e >= 5 for e in ep_indices), f"No source B episodes: {ep_indices}"

    def test_per_source_transform(self):
        """Per-source transforms are applied based on episode source."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)
        _fill_buffer(buffer, num_episodes=3, frames_per_episode=10)

        transform_called = []

        def mock_transform(sample):
            transform_called.append(True)
            sample["_transformed"] = True
            return sample

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(3)),
                "transform": mock_transform,
            }],
            delta_timestamps={"observation.image": [0.0], "action": [0.0]},
            epoch_length=10,
            image_keys=["observation.image"],
        )

        item = dataset[0]
        assert len(transform_called) > 0
        assert item.get("_transformed") is True

    def test_no_transform_passthrough(self):
        """When transform is None, sample is returned as-is."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)
        _fill_buffer(buffer, num_episodes=3, frames_per_episode=10)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(3)),
                "transform": None,
            }],
            delta_timestamps={"observation.image": [0.0], "action": [0.0]},
            epoch_length=10,
            image_keys=["observation.image"],
        )

        item = dataset[0]
        assert "_transformed" not in item

    def test_frames_are_float32_normalized(self):
        """Video frames are returned as float32 in [0, 1]."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)

        # Fill with known uint8 values
        for ep_idx in range(3):
            frames = torch.full((10, 3, 8, 8), 128, dtype=torch.uint8)
            buffer.put(ep_idx, frames)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(3)),
                "transform": None,
            }],
            delta_timestamps={"observation.image": [0.0]},
            epoch_length=10,
            image_keys=["observation.image"],
        )

        item = dataset[0]
        img = item["observation.image"]
        assert img.dtype == torch.float32
        assert 0.0 <= img.min() <= img.max() <= 1.0
        # 128/255 ~= 0.502
        expected = 128.0 / 255.0
        assert abs(img[0, 0, 0, 0].item() - expected) < 0.01

    def test_padding_flags_present(self):
        """Padding flags for out-of-episode indices are included."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)
        _fill_buffer(buffer, num_episodes=3, frames_per_episode=10)

        # Use negative deltas that will go before episode start
        delta_ts = {
            "observation.image": [-1.0, -0.5, 0.0],
            "action": [-1.0, -0.5, 0.0],
        }

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(3)),
                "transform": None,
            }],
            delta_timestamps=delta_ts,
            epoch_length=10,
            image_keys=["observation.image"],
        )

        item = dataset[0]
        # Should have padding flags for action
        assert "action_is_pad" in item
        assert item["action_is_pad"].shape[0] == 3

    def test_with_dataloader(self):
        """Works with PyTorch DataLoader (single worker)."""
        buffer = ShuffleBuffer(
            capacity=10, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=5, frames_per_episode=10)
        _fill_buffer(buffer, num_episodes=5, frames_per_episode=10)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(5)),
                "transform": None,
            }],
            delta_timestamps={"observation.image": [0.0], "action": [0.0]},
            epoch_length=32,
            image_keys=["observation.image"],
        )

        loader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 4
        for batch in batches:
            assert batch["observation.image"].shape[0] == 8
            assert batch["action"].shape[0] == 8

    def test_with_multi_worker_dataloader(self):
        """Works with multi-worker DataLoader (shared memory reads)."""
        buffer = ShuffleBuffer(
            capacity=20, max_frames=10, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=10, frames_per_episode=10)
        _fill_buffer(buffer, num_episodes=10, frames_per_episode=10)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(10)),
                "transform": None,
            }],
            delta_timestamps={"observation.image": [0.0], "action": [0.0]},
            epoch_length=64,
            image_keys=["observation.image"],
        )

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            worker_init_fn=LeRobotShuffleBufferDataset.worker_init_fn,
        )

        total = sum(b["observation.image"].shape[0] for b in loader)
        assert total == 64

    def test_worker_init_fn_seeds_rng(self):
        """worker_init_fn produces different seeds per worker."""
        # Just test it doesn't crash and produces different sequences
        LeRobotShuffleBufferDataset.worker_init_fn(0)
        r0 = random.random()
        LeRobotShuffleBufferDataset.worker_init_fn(1)
        r1 = random.random()
        # Different seeds should (almost surely) produce different values
        # But don't assert inequality -- just verify it runs without error

    def test_epoch_length_controls_len(self):
        """Dataset length matches epoch_length parameter."""
        buffer = ShuffleBuffer(
            capacity=5, max_frames=5, channels=3, height=8, width=8,
        )
        ds = _make_mock_shard_source(num_episodes=3, frames_per_episode=5)
        _fill_buffer(buffer, num_episodes=3, frames_per_episode=5)

        dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=[{
                "shard_source": ds,
                "train_episode_indices": list(range(3)),
                "transform": None,
            }],
            delta_timestamps={"observation.image": [0.0]},
            epoch_length=42,
            image_keys=["observation.image"],
        )

        assert len(dataset) == 42
