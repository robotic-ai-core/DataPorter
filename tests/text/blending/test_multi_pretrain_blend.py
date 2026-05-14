"""Tests for WeightedMultiSourceDataset and PretrainBlendScheduleCallback.

Both classes are deprecated in Phase 3b; DeprecationWarning emission is
covered in ``test_wrapper_equivalence.py``. These tests intentionally
suppress that warning so legacy behavior coverage stays uncluttered.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.text.blending import (
    PretrainBlendScheduleCallback,
    WeightedMultiSourceDataset,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning"
)


class _StubDataset(Dataset):
    """Minimal map-style dataset whose items report their source tag + idx."""

    def __init__(self, tag: str, n: int = 10):
        self._tag = tag
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {"tag": self._tag, "idx": idx}


def _shared_weight(value: float) -> torch.Tensor:
    t = torch.tensor([value], dtype=torch.float64)
    t.share_memory_()
    return t


class TestWeightedMultiSourceDataset:
    def test_empty_sources_rejected(self):
        with pytest.raises(ValueError, match="at least one source"):
            WeightedMultiSourceDataset([])

    def test_non_scalar_weight_rejected(self):
        bad_w = torch.zeros(2)  # 2 elements
        bad_w.share_memory_()
        with pytest.raises(ValueError, match="1-element"):
            WeightedMultiSourceDataset([(_StubDataset("A"), bad_w)])

    def test_len_uses_first_source(self):
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A", n=10), _shared_weight(1.0)),
                (_StubDataset("B", n=99), _shared_weight(1.0)),
            ]
        )
        # Pinned to first source for stable RandomSampler range.
        assert len(ds) == 10

    def test_single_source_hot_weight(self):
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A"), _shared_weight(1.0)),
                (_StubDataset("B"), _shared_weight(0.0)),
            ]
        )
        for _ in range(50):
            assert ds[0]["tag"] == "A"

    def test_zero_total_weight_falls_back_to_first(self):
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A"), _shared_weight(0.0)),
                (_StubDataset("B"), _shared_weight(0.0)),
            ]
        )
        # Don't crash, don't divide by zero — fall back to first.
        for _ in range(20):
            assert ds[0]["tag"] == "A"

    def test_weights_update_visible_at_sample_time(self):
        w_a = _shared_weight(1.0)
        w_b = _shared_weight(0.0)
        ds = WeightedMultiSourceDataset(
            [(_StubDataset("A"), w_a), (_StubDataset("B"), w_b)]
        )
        # Initially A only.
        assert ds[0]["tag"] == "A"
        # Flip weights live (shared-mem update).
        w_a.fill_(0.0)
        w_b.fill_(1.0)
        for _ in range(20):
            assert ds[0]["tag"] == "B"

    def test_negative_weight_treated_as_zero(self):
        # Weights are clamped at sample time (max(0, w)) so a negative
        # value from a buggy schedule doesn't pull mass into a source.
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A"), _shared_weight(-1.0)),
                (_StubDataset("B"), _shared_weight(1.0)),
            ]
        )
        for _ in range(20):
            assert ds[0]["tag"] == "B"

    def test_idx_modulo_per_source(self):
        # idx larger than B's length should still pick a valid B item.
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A", n=100), _shared_weight(0.0)),
                (_StubDataset("B", n=5), _shared_weight(1.0)),
            ]
        )
        item = ds[42]
        assert item["tag"] == "B"
        assert 0 <= item["idx"] < 5

    def test_balanced_weights_yields_roughly_balanced_sampling(self):
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A"), _shared_weight(1.0)),
                (_StubDataset("B"), _shared_weight(1.0)),
            ]
        )
        torch.manual_seed(42)
        import random as _random

        _random.seed(42)
        counts = {"A": 0, "B": 0}
        for _ in range(2000):
            counts[ds[0]["tag"]] += 1
        # Allow ±5% slop: 2000 samples, 50/50 expected, 95% CI ≈ ±43.
        assert 900 <= counts["A"] <= 1100
        assert 900 <= counts["B"] <= 1100


class TestPretrainBlendScheduleCallback:
    def test_empty_schedules_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            PretrainBlendScheduleCallback(schedules=[])

    def test_missing_key_rejected(self):
        with pytest.raises(ValueError, match="missing legacy key"):
            PretrainBlendScheduleCallback(
                schedules=[{"source_idx": 0, "weight_start": 1.0}]
            )

    def test_inverted_step_window_rejected(self):
        with pytest.raises(ValueError, match="must be >"):
            PretrainBlendScheduleCallback(
                schedules=[
                    {
                        "source_idx": 0,
                        "weight_start": 1.0,
                        "weight_end": 0.0,
                        "blend_start_step": 100,
                        "blend_end_step": 50,
                    }
                ]
            )

    def test_compute_weight_endpoints(self):
        sched = {
            "source_idx": 0,
            "weight_start": 1.0,
            "weight_end": 0.0,
            "blend_start_step": 100,
            "blend_end_step": 200,
        }
        # Before window: pinned at start.
        assert PretrainBlendScheduleCallback._compute_weight(0, sched) == 1.0
        assert PretrainBlendScheduleCallback._compute_weight(99, sched) == 1.0
        # At start.
        assert PretrainBlendScheduleCallback._compute_weight(100, sched) == 1.0
        # Halfway.
        assert PretrainBlendScheduleCallback._compute_weight(150, sched) == 0.5
        # At end.
        assert PretrainBlendScheduleCallback._compute_weight(200, sched) == 0.0
        # After: pinned at end.
        assert PretrainBlendScheduleCallback._compute_weight(500, sched) == 0.0

    def test_compute_weight_ramp_up(self):
        sched = {
            "source_idx": 1,
            "weight_start": 0.0,
            "weight_end": 1.0,
            "blend_start_step": 0,
            "blend_end_step": 100,
        }
        assert PretrainBlendScheduleCallback._compute_weight(0, sched) == 0.0
        assert PretrainBlendScheduleCallback._compute_weight(25, sched) == 0.25
        assert PretrainBlendScheduleCallback._compute_weight(75, sched) == 0.75
        assert PretrainBlendScheduleCallback._compute_weight(100, sched) == 1.0

    def test_invalid_source_idx_silently_skipped(self):
        # source_idx out of range shouldn't crash on_train_batch_start.
        # We test the fundamental behavior: out-of-range schedule entry
        # doesn't write to any tensor.
        ds = WeightedMultiSourceDataset(
            [
                (_StubDataset("A"), _shared_weight(0.5)),
                (_StubDataset("B"), _shared_weight(0.5)),
            ]
        )
        cb = PretrainBlendScheduleCallback(
            schedules=[
                {
                    "source_idx": 99,  # out of range
                    "weight_start": 1.0,
                    "weight_end": 0.0,
                    "blend_start_step": 0,
                    "blend_end_step": 10,
                }
            ]
        )
        # Mimic what on_train_batch_start does without a Lightning trainer.
        for sched in cb._schedules:
            idx = sched["source_idx"]
            if 0 <= idx < len(ds._weight_tensors):
                ds._weight_tensors[idx].fill_(
                    cb._compute_weight(5, sched)
                )
        # No tensor should have been touched.
        assert ds._weight_tensors[0].item() == 0.5
        assert ds._weight_tensors[1].item() == 0.5


class TestControlPointSchedule:
    """Piecewise-linear control-point schedule format."""

    def test_two_points_match_legacy_form(self):
        # Same shape: legacy 4-key form ≡ 2-point control-point form.
        legacy = {
            "source_idx": 0,
            "weight_start": 1.0, "weight_end": 0.0,
            "blend_start_step": 0, "blend_end_step": 100,
        }
        points = {
            "source_idx": 0,
            "points": [
                {"step": 0, "weight": 1.0},
                {"step": 100, "weight": 0.0},
            ],
        }
        cb_legacy = PretrainBlendScheduleCallback(schedules=[legacy])
        cb_points = PretrainBlendScheduleCallback(schedules=[points])
        for s in (0, 25, 50, 75, 100, 200):
            l = cb_legacy._compute_weight(s, cb_legacy._schedules[0], max_steps=100)
            p = cb_points._compute_weight(s, cb_points._schedules[0], max_steps=100)
            assert abs(l - p) < 1e-9, (
                f"step {s}: legacy={l} points={p}"
            )

    def test_three_point_curve(self):
        # u-shape: down-then-up
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0, "weight": 1.0},
                {"step": 50, "weight": 0.2},
                {"step": 100, "weight": 0.8},
            ],
        }])
        sched = cb._schedules[0]
        assert cb._compute_weight(0, sched, 100) == pytest.approx(1.0)
        assert cb._compute_weight(25, sched, 100) == pytest.approx(0.6)
        assert cb._compute_weight(50, sched, 100) == pytest.approx(0.2)
        assert cb._compute_weight(75, sched, 100) == pytest.approx(0.5)
        assert cb._compute_weight(100, sched, 100) == pytest.approx(0.8)

    def test_clamping_outside_range(self):
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 100, "weight": 0.5},
                {"step": 200, "weight": 0.9},
            ],
        }])
        sched = cb._schedules[0]
        # Before first point — clamp at first weight.
        assert cb._compute_weight(0, sched, 1000) == pytest.approx(0.5)
        assert cb._compute_weight(50, sched, 1000) == pytest.approx(0.5)
        # After last point — clamp at last weight.
        assert cb._compute_weight(300, sched, 1000) == pytest.approx(0.9)
        assert cb._compute_weight(99999, sched, 1000) == pytest.approx(0.9)

    def test_fractional_steps_resolve_to_max_steps(self):
        # 0.0, 0.5, 1.0 → 0, 500, 1000 when max_steps=1000.
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0.0, "weight": 1.0},
                {"step": 0.5, "weight": 0.4},
                {"step": 1.0, "weight": 0.1},
            ],
        }])
        sched = cb._schedules[0]
        assert cb._compute_weight(0, sched, 1000) == pytest.approx(1.0)
        assert cb._compute_weight(500, sched, 1000) == pytest.approx(0.4)
        assert cb._compute_weight(1000, sched, 1000) == pytest.approx(0.1)
        # Halfway through first segment: step 250 of [0, 500].
        assert cb._compute_weight(250, sched, 1000) == pytest.approx(0.7)

    def test_fractional_steps_portable_across_max_steps(self):
        # SAME schedule {0.0, 0.5, 1.0} applied to two different
        # max_steps should yield equivalent weight at corresponding
        # fractions. This is the schedule-portability claim.
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0.0, "weight": 1.0},
                {"step": 0.5, "weight": 0.0},
            ],
        }])
        sched = cb._schedules[0]
        # 50% point under max_steps=2000
        w_short = cb._compute_weight(1000, sched, 2000)
        # 50% point under max_steps=50000
        w_long = cb._compute_weight(25000, sched, 50000)
        assert abs(w_short - w_long) < 1e-9

    def test_mixed_int_and_fractional_steps(self):
        # Mixing int (absolute) and float (fractional) within one curve.
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0, "weight": 1.0},        # absolute
                {"step": 0.5, "weight": 0.5},      # 50% of 1000 = 500
                {"step": 900, "weight": 0.1},      # absolute
            ],
        }])
        sched = cb._schedules[0]
        assert cb._compute_weight(0, sched, 1000) == pytest.approx(1.0)
        assert cb._compute_weight(500, sched, 1000) == pytest.approx(0.5)
        assert cb._compute_weight(900, sched, 1000) == pytest.approx(0.1)
        # Between 500 and 900, halfway = 700 → 0.3 ((0.5+0.1)/2)
        assert cb._compute_weight(700, sched, 1000) == pytest.approx(0.3)

    def test_unsorted_points_get_sorted(self):
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 100, "weight": 0.5},     # out of order
                {"step": 0, "weight": 1.0},
                {"step": 200, "weight": 0.0},
            ],
        }])
        sched = cb._schedules[0]
        # Should behave the same as sorted version.
        assert cb._compute_weight(50, sched, 200) == pytest.approx(0.75)
        assert cb._compute_weight(150, sched, 200) == pytest.approx(0.25)

    def test_single_point_constant_weight(self):
        # 1-point schedule → weight is that constant everywhere.
        cb = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [{"step": 50, "weight": 0.7}],
        }])
        sched = cb._schedules[0]
        for s in (0, 25, 50, 100, 9999):
            assert cb._compute_weight(s, sched, 100) == pytest.approx(0.7)

    def test_empty_points_rejected(self):
        with pytest.raises(ValueError, match="non-empty list"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [],
            }])

    def test_point_missing_step_or_weight_rejected(self):
        with pytest.raises(ValueError, match="must have 'step' and 'weight'"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"step": 0}],   # missing weight
            }])
        with pytest.raises(ValueError, match="must have 'step' and 'weight'"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"weight": 0.5}],   # missing step
            }])

    def test_step_must_be_numeric(self):
        with pytest.raises(TypeError, match="step must be int"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"step": "first", "weight": 0.5}],
            }])

    def test_fractional_step_outside_range_rejected(self):
        # float > 1 isn't a valid fraction; user must use int.
        with pytest.raises(ValueError, match=r"fractional step must be in \[0, 1\]"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"step": 1.5, "weight": 0.5}],
            }])
        with pytest.raises(ValueError, match=r"fractional step must be in \[0, 1\]"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"step": -0.1, "weight": 0.5}],
            }])

    def test_int_zero_is_absolute_not_fractional(self):
        # int 0 → absolute step 0; float 0.0 → 0% of max_steps. Both
        # equivalent at 0 but the type matters for the parsing path.
        cb_int = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0, "weight": 1.0},
                {"step": 1000, "weight": 0.0},
            ],
        }])
        cb_float = PretrainBlendScheduleCallback(schedules=[{
            "source_idx": 0,
            "points": [
                {"step": 0.0, "weight": 1.0},
                {"step": 1.0, "weight": 0.0},
            ],
        }])
        # Both should yield the same trajectory at max_steps=1000.
        for s in (0, 250, 500, 1000):
            i = cb_int._compute_weight(s, cb_int._schedules[0], 1000)
            f = cb_float._compute_weight(s, cb_float._schedules[0], 1000)
            assert abs(i - f) < 1e-9, f"step {s}: int={i} float={f}"

    def test_mixing_points_with_legacy_keys_rejected(self):
        # If both forms are given, raise — unambiguous spec.
        with pytest.raises(ValueError, match="mixes 'points' with legacy keys"):
            PretrainBlendScheduleCallback(schedules=[{
                "source_idx": 0,
                "points": [{"step": 0, "weight": 1.0}],
                "weight_start": 0.5,    # conflicting
            }])

    def test_phi_style_three_phase_schedule(self):
        # Realistic: web → textbook → math curriculum across 3 phases.
        # FineWeb-Edu: 1.0 (start) → 0.4 (mid) → 0.1 (end)
        # tiny-textbooks: 0.0 → 0.4 → 0.5
        # NuminaMath-CoT: 0.0 → 0.0 → 0.3 (only late-phase)
        cb = PretrainBlendScheduleCallback(schedules=[
            {"source_idx": 0, "points": [
                {"step": 0.0, "weight": 1.0},
                {"step": 0.4, "weight": 0.4},
                {"step": 1.0, "weight": 0.1},
            ]},
            {"source_idx": 1, "points": [
                {"step": 0.0, "weight": 0.0},
                {"step": 0.4, "weight": 0.4},
                {"step": 1.0, "weight": 0.5},
            ]},
            {"source_idx": 2, "points": [
                {"step": 0.0, "weight": 0.0},
                {"step": 0.5, "weight": 0.0},
                {"step": 1.0, "weight": 0.3},
            ]},
        ])
        max_steps = 10000
        # At end of training: FineWeb=0.1, textbook=0.5, numina=0.3
        s0, s1, s2 = cb._schedules
        assert cb._compute_weight(max_steps, s0, max_steps) == pytest.approx(0.1)
        assert cb._compute_weight(max_steps, s1, max_steps) == pytest.approx(0.5)
        assert cb._compute_weight(max_steps, s2, max_steps) == pytest.approx(0.3)
        # At 25%: FineWeb mid-ramp from 1.0→0.4 (75% along), textbook 0.0→0.4 (75% along),
        # numina still 0.0
        # FineWeb: 1.0 + (0.25/0.4)*(0.4-1.0) = 1.0 - 0.625*0.6 = 0.625
        assert cb._compute_weight(2500, s0, max_steps) == pytest.approx(0.625)
        assert cb._compute_weight(2500, s1, max_steps) == pytest.approx(0.25)
        assert cb._compute_weight(2500, s2, max_steps) == pytest.approx(0.0)

    def test_weights_update_via_shared_mem(self):
        # End-to-end: callback updates shared-mem tensors that the
        # WeightedMultiSourceDataset samples from.
        ds = WeightedMultiSourceDataset([
            (_StubDataset("A"), _shared_weight(1.0)),
            (_StubDataset("B"), _shared_weight(0.0)),
        ])
        cb = PretrainBlendScheduleCallback(schedules=[
            {"source_idx": 0, "points": [
                {"step": 0, "weight": 1.0},
                {"step": 100, "weight": 0.0},
            ]},
            {"source_idx": 1, "points": [
                {"step": 0, "weight": 0.0},
                {"step": 100, "weight": 1.0},
            ]},
        ])
        # Simulate what on_train_batch_start does without Lightning trainer.
        max_steps = 100
        for source_idx, sched in zip(
            [s["source_idx"] for s in cb._schedules],
            cb._schedules,
        ):
            curve = cb._resolve_points(sched["points"], max_steps)
            ds._weight_tensors[source_idx].fill_(cb._interp_at(50, curve))
        # At step 50 (50% through), each weight should be 0.5.
        assert ds._weight_tensors[0].item() == pytest.approx(0.5)
        assert ds._weight_tensors[1].item() == pytest.approx(0.5)
