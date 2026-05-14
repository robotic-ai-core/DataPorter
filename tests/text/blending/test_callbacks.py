"""Smoke tests for MixingScheduleCallback.

Verifies the schedule math + that the callback no-ops when the target
``BlendedTextDataset`` is not on the datamodule. Both the callback and
``BlendedTextDataset`` are deprecated in Phase 3b; DeprecationWarning
emission is covered in ``test_wrapper_equivalence.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dataporter.text.blending import (
    BlendedTextDataset,
    MixingScheduleCallback,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning"
)


class _MockDataset:
    """Minimal pretrain/chat datasets — bypass spec probe."""

    def __init__(self, n: int = 4):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"idx": idx}


def _mk_blended_dataset() -> BlendedTextDataset:
    return BlendedTextDataset(
        pretrain_dataset=_MockDataset(8),
        chat_dataset=_MockDataset(4),
        sample_spec=None,
    )


class TestComputeRatio:

    def test_zero_before_start(self):
        cb = MixingScheduleCallback(
            blend_start_step=100, blend_end_step=200, chat_ratio_end=0.5,
        )
        assert cb._compute_ratio(0) == 0.0
        assert cb._compute_ratio(99) == 0.0

    def test_linear_in_window(self):
        cb = MixingScheduleCallback(
            blend_start_step=100, blend_end_step=200, chat_ratio_end=1.0,
        )
        assert cb._compute_ratio(100) == 0.0
        assert cb._compute_ratio(150) == pytest.approx(0.5)
        assert cb._compute_ratio(199) == pytest.approx(0.99, abs=0.01)

    def test_clamps_at_end(self):
        cb = MixingScheduleCallback(
            blend_start_step=100, blend_end_step=200, chat_ratio_end=0.7,
        )
        assert cb._compute_ratio(200) == 0.7
        assert cb._compute_ratio(10_000) == 0.7

    def test_validates_endpoints(self):
        with pytest.raises(ValueError, match="must be >"):
            MixingScheduleCallback(blend_start_step=200, blend_end_step=100)


class TestBatchHook:

    def test_updates_chat_ratio_in_window(self):
        cb = MixingScheduleCallback(
            blend_start_step=10, blend_end_step=20, chat_ratio_end=1.0,
        )
        bds = _mk_blended_dataset()

        trainer = MagicMock()
        trainer.global_step = 15
        trainer.datamodule.blended_dataset = bds
        pl = MagicMock()

        cb.on_train_batch_start(trainer, pl, batch=None, batch_idx=0)
        assert bds.chat_ratio == pytest.approx(0.5)

    def test_noop_when_no_blended_dataset(self):
        cb = MixingScheduleCallback(
            blend_start_step=10, blend_end_step=20, chat_ratio_end=0.5,
        )
        trainer = MagicMock()
        trainer.global_step = 100
        trainer.datamodule.blended_dataset = None  # not a BlendedTextDataset
        # Should silently return without raising.
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)

    def test_noop_when_datamodule_is_none(self):
        cb = MixingScheduleCallback(
            blend_start_step=10, blend_end_step=20, chat_ratio_end=0.5,
        )
        trainer = MagicMock()
        trainer.datamodule = None
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
