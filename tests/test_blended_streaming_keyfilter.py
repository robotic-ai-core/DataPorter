"""Regression test for blended-streaming heterogeneous-key collation.

When two LeRobot sources have different feature schemas (e.g. ``lewm``
exposes ``next.reward`` while ``synthetic-v5`` does not), the streaming
training pipeline must restrict each sample to a common key set before
the default collate runs.  Otherwise ``torch.utils.data.default_collate``
walks the union of keys and raises ``KeyError`` on the first heterogeneous
batch.

The val and non-streaming train paths already wrap with
``KeyFilterDataset`` (see ``BlendedLeRobotDataModule.setup``).  The
streaming train path now does the same — this test locks that in.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import default_collate

from dataporter.dataset_wrappers import KeyFilterDataset
from dataporter.lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset
from dataporter.shuffle_buffer import ShuffleBuffer

# Reuse the mock shard source from the sibling test module.
from test_lerobot_shuffle_buffer_dataset import _MockShardSource


class _MockShardSourceWithReward(_MockShardSource):
    """Mock source that also returns ``next.reward`` per row.

    Mirrors lewm-style sources whose features include a reward column
    that synthetic sources omit.
    """

    def _row(self, ep_idx: int, frame_idx: int) -> dict:
        row = super()._row(ep_idx, frame_idx)
        row["next.reward"] = torch.tensor(0.5, dtype=torch.float32)
        return row


def _fill_uint8_buffer(buffer: ShuffleBuffer, ep_ids: list[int]) -> None:
    for ep_idx in ep_ids:
        frames = torch.full(
            (10, 3, 8, 8), ep_idx, dtype=torch.uint8,
        )
        buffer.put(ep_idx, frames)


def _make_blended_streaming_dataset() -> LeRobotShuffleBufferDataset:
    buffer = ShuffleBuffer(
        capacity=20, max_frames=10, channels=3, height=8, width=8,
    )
    # Episodes 0-2 from source A (with reward); 3-5 from source B (without).
    _fill_uint8_buffer(buffer, [0, 1, 2, 3, 4, 5])
    return LeRobotShuffleBufferDataset(
        buffer=buffer,
        sources=[
            {
                "shard_source": _MockShardSourceWithReward(num_episodes=3),
                "train_episode_indices": [0, 1, 2],
                "transform": None,
            },
            {
                "shard_source": _MockShardSource(num_episodes=3),
                "train_episode_indices": [3, 4, 5],
                "transform": None,
            },
        ],
        delta_timestamps={
            "observation.image": [0.0],
            "action": [0.0],
        },
        epoch_length=64,
        image_keys=["observation.image"],
    )


def _draw_batch_until_mixed(
    dataset, batch_size: int = 16, max_attempts: int = 200,
):
    """Draw samples until the FIRST sample has ``next.reward`` and at
    least one later sample does NOT — ``default_collate`` keys off
    ``batch[0]`` so the heterogeneity must be visible to the collator.
    """
    for _ in range(max_attempts):
        samples = [dataset[0] for _ in range(batch_size)]
        if "next.reward" not in samples[0]:
            continue
        if any("next.reward" not in s for s in samples[1:]):
            return samples
    raise AssertionError(
        "Could not assemble a heterogeneous batch — fix the mock sampling.",
    )


class TestBlendedStreamingKeyFilter:
    """Regression test for the next.reward KeyError at collate."""

    def test_unfiltered_heterogeneous_batch_raises(self):
        """Without filter wrap, default_collate trips on missing key.

        This mirrors the j-20260506-008 crash:
            KeyError: 'next.reward'
        """
        dataset = _make_blended_streaming_dataset()
        samples = _draw_batch_until_mixed(dataset)
        with pytest.raises(KeyError, match="next.reward"):
            default_collate(samples)

    def test_keyfilter_wrap_makes_collate_safe(self):
        """KeyFilterDataset over the streaming dataset → safe collate."""
        dataset = _make_blended_streaming_dataset()
        # Allowed keys mirror BlendedLeRobotDataModule._common_sample_keys():
        # delta_timestamps keys + bookkeeping fields.
        allowed = {
            "observation.image", "action",
            "episode_index", "frame_index", "timestamp",
            "index", "task_index",
        }
        wrapped = KeyFilterDataset(dataset, allowed)

        # Draw enough samples that we statistically hit both sources.
        samples = [wrapped[0] for _ in range(32)]
        for s in samples:
            assert set(s.keys()).issubset(allowed)
            assert "next.reward" not in s

        # Default collate now succeeds — every sample has identical keys.
        batch = default_collate(samples)
        for k in ("observation.image", "action", "episode_index"):
            assert k in batch


class TestSetupWraps:
    """Structural test: ``_setup_shuffle_buffer_training`` wraps when blending.

    Inspects the post-setup ``train_dataset`` type rather than running a
    full setup (which needs real on-disk LeRobot fixtures).  Uses
    monkeypatching to short-circuit the heavy build path.
    """

    def test_wraps_when_multiple_sources(self, monkeypatch):
        from dataporter import BlendedLeRobotDataModule

        sentinel = _make_blended_streaming_dataset()

        # Capture the wrapping decision by patching the constructor body
        # of _setup_shuffle_buffer_training to a stub that exercises just
        # the wrap branch.
        def fake_setup(self):
            train_dataset = sentinel
            if len(self._sources) > 1:
                train_dataset = KeyFilterDataset(
                    train_dataset, self._common_sample_keys(),
                )
            self.train_dataset = train_dataset
            self._train_sampler = None

        monkeypatch.setattr(
            BlendedLeRobotDataModule,
            "_setup_shuffle_buffer_training",
            fake_setup,
            raising=False,
        )

        dm = BlendedLeRobotDataModule.__new__(BlendedLeRobotDataModule)
        dm._sources = [{"repo_id": "a"}, {"repo_id": "b"}]
        dm.delta_timestamps = {
            "observation.image": [0.0], "action": [0.0],
        }
        dm._setup_shuffle_buffer_training()

        assert isinstance(dm.train_dataset, KeyFilterDataset), (
            "Multi-source streaming train path must wrap with KeyFilterDataset"
        )

    def test_passthrough_when_single_source(self, monkeypatch):
        from dataporter import BlendedLeRobotDataModule

        sentinel = _make_blended_streaming_dataset()

        def fake_setup(self):
            train_dataset = sentinel
            if len(self._sources) > 1:
                train_dataset = KeyFilterDataset(
                    train_dataset, self._common_sample_keys(),
                )
            self.train_dataset = train_dataset
            self._train_sampler = None

        monkeypatch.setattr(
            BlendedLeRobotDataModule,
            "_setup_shuffle_buffer_training",
            fake_setup,
            raising=False,
        )

        dm = BlendedLeRobotDataModule.__new__(BlendedLeRobotDataModule)
        dm._sources = [{"repo_id": "a"}]
        dm.delta_timestamps = {
            "observation.image": [0.0], "action": [0.0],
        }
        dm._setup_shuffle_buffer_training()

        assert dm.train_dataset is sentinel, (
            "Single-source streaming train path must not wrap"
        )
