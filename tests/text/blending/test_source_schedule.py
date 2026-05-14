"""Tests for SourceScheduleCallback — schedule-by-name + curve math."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.text.blending import (
    ScheduledBlendDataset,
    SourceScheduleCallback,
)


class _StubDataset(Dataset):
    def __init__(self, tag: str, n: int = 10):
        self._tag = tag
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return {"tag": self._tag, "idx": idx}


def _w(v: float) -> torch.Tensor:
    t = torch.tensor([v], dtype=torch.float64)
    t.share_memory_()
    return t


def _mk(*specs: tuple[str, float]) -> ScheduledBlendDataset:
    return ScheduledBlendDataset(
        [(_StubDataset(name), _w(w), name) for name, w in specs]
    )


# ----------------------------------------------------------------------
# Entry validation
# ----------------------------------------------------------------------


class TestEntryValidation:
    def test_empty_schedules_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            SourceScheduleCallback(schedules=[])

    def test_neither_name_nor_idx_rejected(self):
        with pytest.raises(ValueError, match="exactly one of"):
            SourceScheduleCallback(schedules=[{
                "points": [{"step": 0, "weight": 1.0}],
            }])

    def test_both_name_and_idx_rejected(self):
        with pytest.raises(ValueError, match="exactly one of"):
            SourceScheduleCallback(schedules=[{
                "source_name": "A",
                "source_idx": 0,
                "points": [{"step": 0, "weight": 1.0}],
            }])

    def test_missing_points_rejected(self):
        with pytest.raises(ValueError, match="missing 'points'"):
            SourceScheduleCallback(schedules=[{
                "source_name": "A",
            }])

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            SourceScheduleCallback(schedules=[{
                "source_name": "",
                "points": [{"step": 0, "weight": 1.0}],
            }])

    def test_non_int_idx_rejected(self):
        with pytest.raises(TypeError, match="source_idx must be int"):
            SourceScheduleCallback(schedules=[{
                "source_idx": "0",  # str, not int
                "points": [{"step": 0, "weight": 1.0}],
            }])

    def test_bool_idx_rejected(self):
        with pytest.raises(TypeError, match="source_idx must be int"):
            SourceScheduleCallback(schedules=[{
                "source_idx": True,
                "points": [{"step": 0, "weight": 1.0}],
            }])

    def test_points_validation_smoke(self):
        # Smoke that points validation is wired up (full coverage is in
        # the legacy callback's test file).
        with pytest.raises(ValueError, match="non-empty list"):
            SourceScheduleCallback(schedules=[{
                "source_name": "A",
                "points": [],
            }])
        with pytest.raises(ValueError, match=r"fractional step must be in"):
            SourceScheduleCallback(schedules=[{
                "source_name": "A",
                "points": [{"step": 1.5, "weight": 0.5}],
            }])


# ----------------------------------------------------------------------
# Resolution by name vs by idx
# ----------------------------------------------------------------------


class TestResolution:
    def _run_one_step(self, cb, ds, step, max_steps=100):
        trainer = MagicMock()
        trainer.datamodule.scheduled_blend_dataset = ds
        trainer.global_step = step
        trainer.max_steps = max_steps
        pl = MagicMock()
        cb.on_train_batch_start(trainer, pl, batch=None, batch_idx=0)

    def test_resolution_by_name(self):
        ds = _mk(("foo", 0.0), ("bar", 0.0))
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "bar", "points": [
                {"step": 0, "weight": 1.0},
                {"step": 100, "weight": 1.0},
            ]},
        ])
        self._run_one_step(cb, ds, step=50)
        assert ds.get_weight("bar") == pytest.approx(1.0)
        assert ds.get_weight("foo") == 0.0

    def test_resolution_by_idx(self):
        ds = _mk(("foo", 0.0), ("bar", 0.0))
        cb = SourceScheduleCallback(schedules=[
            {"source_idx": 1, "points": [{"step": 0, "weight": 0.7}]},
        ])
        self._run_one_step(cb, ds, step=5)
        assert ds.get_weight(1) == pytest.approx(0.7)
        assert ds.get_weight(0) == 0.0

    def test_name_survives_source_reordering(self):
        """Reorder sources but keep schedule keyed by name — should still
        target the right source."""
        ds_a_first = _mk(("foo", 0.0), ("bar", 0.0))
        ds_b_first = _mk(("bar", 0.0), ("foo", 0.0))
        cb1 = SourceScheduleCallback(schedules=[
            {"source_name": "bar", "points": [{"step": 0, "weight": 0.9}]},
        ])
        cb2 = SourceScheduleCallback(schedules=[
            {"source_name": "bar", "points": [{"step": 0, "weight": 0.9}]},
        ])
        self._run_one_step(cb1, ds_a_first, step=10)
        self._run_one_step(cb2, ds_b_first, step=10)
        # Both target 'bar' regardless of which index it has.
        assert ds_a_first.get_weight("bar") == pytest.approx(0.9)
        assert ds_b_first.get_weight("bar") == pytest.approx(0.9)
        # And neither modified 'foo'.
        assert ds_a_first.get_weight("foo") == 0.0
        assert ds_b_first.get_weight("foo") == 0.0

    def test_unknown_name_silently_skipped(self):
        """Stale schedule entry for missing source shouldn't crash."""
        ds = _mk(("real", 0.5),)
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "missing", "points": [{"step": 0, "weight": 1.0}]},
        ])
        self._run_one_step(cb, ds, step=10)
        # Unchanged.
        assert ds.get_weight("real") == 0.5

    def test_out_of_range_idx_silently_skipped(self):
        ds = _mk(("real", 0.5),)
        cb = SourceScheduleCallback(schedules=[
            {"source_idx": 99, "points": [{"step": 0, "weight": 1.0}]},
        ])
        self._run_one_step(cb, ds, step=10)
        assert ds.get_weight("real") == 0.5


# ----------------------------------------------------------------------
# No-op when dataset attribute absent
# ----------------------------------------------------------------------


class TestNoOp:
    def test_no_dm(self):
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "x", "points": [{"step": 0, "weight": 1.0}]},
        ])
        trainer = MagicMock()
        trainer.datamodule = None
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        # No exception, no state mutation.

    def test_no_dataset_attr(self):
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "x", "points": [{"step": 0, "weight": 1.0}]},
        ])
        trainer = MagicMock()
        # MagicMock returns a MagicMock for any attribute access, but
        # isinstance() against ScheduledBlendDataset fails. So the
        # callback should treat this as "no dataset" and return.
        trainer.datamodule = MagicMock(spec=[])  # no attributes
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)

    def test_custom_dataset_attr(self):
        ds = _mk(("foo", 0.0),)
        cb = SourceScheduleCallback(
            schedules=[
                {"source_name": "foo", "points": [{"step": 0, "weight": 0.8}]},
            ],
            dataset_attr="my_blender",
        )
        trainer = MagicMock()
        trainer.datamodule = MagicMock(spec=["my_blender"])
        trainer.datamodule.my_blender = ds
        trainer.global_step = 0
        trainer.max_steps = 100
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert ds.get_weight("foo") == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Schedule math — quick coverage; deep coverage lives in legacy test file
# ----------------------------------------------------------------------


class TestScheduleMath:
    def test_three_phase_curve_by_name(self):
        ds = _mk(("web", 0.0), ("books", 0.0), ("math", 0.0))
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "web", "points": [
                {"step": 0.0, "weight": 1.0},
                {"step": 0.5, "weight": 0.5},
                {"step": 1.0, "weight": 0.1},
            ]},
            {"source_name": "books", "points": [
                {"step": 0.0, "weight": 0.0},
                {"step": 0.5, "weight": 0.3},
                {"step": 1.0, "weight": 0.5},
            ]},
            {"source_name": "math", "points": [
                {"step": 0.5, "weight": 0.0},
                {"step": 1.0, "weight": 0.4},
            ]},
        ])
        trainer = MagicMock()
        trainer.datamodule.scheduled_blend_dataset = ds
        trainer.max_steps = 10000
        # End of training
        trainer.global_step = 10000
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert ds.get_weight("web") == pytest.approx(0.1)
        assert ds.get_weight("books") == pytest.approx(0.5)
        assert ds.get_weight("math") == pytest.approx(0.4)

    def test_resolved_curves_cached_across_steps(self):
        ds = _mk(("A", 0.0),)
        cb = SourceScheduleCallback(schedules=[
            {"source_name": "A", "points": [
                {"step": 0.0, "weight": 0.0},
                {"step": 1.0, "weight": 1.0},
            ]},
        ])
        trainer = MagicMock()
        trainer.datamodule.scheduled_blend_dataset = ds
        trainer.max_steps = 100
        trainer.global_step = 0
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert cb._resolved_curves is not None
        first = cb._resolved_curves
        # Next call: cache reused.
        trainer.global_step = 50
        cb.on_train_batch_start(trainer, MagicMock(), batch=None, batch_idx=0)
        assert cb._resolved_curves is first
        # And weight should have ramped.
        assert ds.get_weight("A") == pytest.approx(0.5)
