"""Lightning callbacks driving the blended-text schedules.

.. deprecated:: phase-3b
   Both callbacks here are backward-compat shims. New code should use
   :class:`dataporter.text.SourceScheduleCallback`, which subsumes both
   schedule kinds via a single per-source piecewise-linear curve API
   keyed by ``source_name`` or ``source_idx``. Removal is scheduled for
   Phase 5.

Two callbacks:

- :class:`MixingScheduleCallback` linearly anneals
  :attr:`BlendedTextDataset.chat_ratio` over a step range.
- :class:`PretrainBlendScheduleCallback` updates per-source weights
  inside a :class:`WeightedMultiSourceDataset` along piecewise-linear
  curves.

Both callbacks discover their target dataset by walking
``trainer.datamodule``; if the target isn't found they silently no-op
so non-blended runs aren't affected. Because the underlying datasets
are now :class:`ScheduledBlendDataset` wrappers, the legacy write paths
(``chat_ratio`` setter, ``_weight_tensors[i].fill_(w)``) continue to
work unchanged — they're just thinner under the hood.
"""

from __future__ import annotations

import warnings

import lightning as L

from .blended_dataset import BlendedTextDataset
from .weighted_multi_source import WeightedMultiSourceDataset


# ------------------------------------------------------------------
# MixingScheduleCallback
# ------------------------------------------------------------------


class MixingScheduleCallback(L.Callback):
    """Linearly anneals the chat mixing ratio during training.

    Updates ``BlendedTextDataset.chat_ratio`` on every training batch,
    ramping from 0 to ``chat_ratio_end`` over the interval
    ``[blend_start_step, blend_end_step]``.

    Args:
        blend_start_step: Optimizer step where blending begins (ratio=0
            before this).
        blend_end_step: Optimizer step where blending reaches
            ``chat_ratio_end``.
        chat_ratio_end: Target chat ratio at *blend_end_step*.
        log_every_n_steps: How often to log the ratio to the logger.
    """

    def __init__(
        self,
        blend_start_step: int = 30_000,
        blend_end_step: int = 45_000,
        chat_ratio_end: float = 0.5,
        log_every_n_steps: int = 100,
    ):
        super().__init__()
        warnings.warn(
            "MixingScheduleCallback is deprecated; use "
            "dataporter.text.SourceScheduleCallback with a per-source "
            "schedule on the chat source. Scheduled removal: Phase 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        if blend_end_step <= blend_start_step:
            raise ValueError(
                f"blend_end_step ({blend_end_step}) must be > "
                f"blend_start_step ({blend_start_step})"
            )
        self.blend_start_step = blend_start_step
        self.blend_end_step = blend_end_step
        self.chat_ratio_end = chat_ratio_end
        self.log_every_n_steps = log_every_n_steps

    def _compute_ratio(self, step: int) -> float:
        if step < self.blend_start_step:
            return 0.0
        if step >= self.blend_end_step:
            return self.chat_ratio_end
        progress = (step - self.blend_start_step) / (
            self.blend_end_step - self.blend_start_step
        )
        return progress * self.chat_ratio_end

    def _get_blended_dataset(self, trainer: L.Trainer) -> BlendedTextDataset | None:
        """Walk the datamodule to find the BlendedTextDataset."""
        dm = trainer.datamodule
        if dm is None:
            return None
        ds = getattr(dm, "blended_dataset", None)
        if isinstance(ds, BlendedTextDataset):
            return ds
        return None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        ds = self._get_blended_dataset(trainer)
        if ds is None:
            return

        step = trainer.global_step
        ratio = self._compute_ratio(step)
        ds.chat_ratio = ratio

        if step % self.log_every_n_steps == 0:
            pl_module.log("blend/chat_ratio", ratio)


# ------------------------------------------------------------------
# PretrainBlendScheduleCallback
# ------------------------------------------------------------------


class PretrainBlendScheduleCallback(L.Callback):
    """Schedules per-source pretrain weights as piecewise-linear curves.

    Each schedule entry is a list of (step, weight) control points that
    define a curve. Linear interpolation between consecutive points;
    weights are clamped to the first/last point's value outside the
    spanned range. Sources without a schedule entry keep their initial
    weight unchanged.

    Two equivalent input forms are accepted:

    1. **Control points (preferred — supports N-phase schedules):**

        schedules:
          - source_idx: 0
            points:
              - {step: 0,    weight: 1.0}
              - {step: 0.2,  weight: 0.5}        # 20% of trainer.max_steps
              - {step: 0.5,  weight: 0.2}
              - {step: 1.0,  weight: 0.1}

       ``step`` accepts ints (absolute step number) or floats in [0, 1]
       (fraction of ``trainer.max_steps``). Mixing the two within one
       schedule is supported. Fractional steps make schedules portable
       across runs of different lengths.

    2. **Legacy 4-key form (single-ramp, kept for backward compat):**

        schedules:
          - source_idx: 0
            weight_start: 1.0
            weight_end:   0.5
            blend_start_step: 0
            blend_end_step:   5000

       Internally lowered to a 2-point control-point schedule.

    Weights are normalised by :class:`WeightedMultiSourceDataset` at
    sample time, so absolute scale doesn't matter — only relative
    magnitudes between sources at each step.

    Args:
        schedules: list of dicts. Each dict must have ``source_idx`` plus
            either ``points`` (list of {step, weight}) or the legacy
            4-key form.
        log_every_n_steps: How often to log the current weights.
    """

    def __init__(
        self,
        schedules: list[dict],
        log_every_n_steps: int = 100,
    ):
        super().__init__()
        warnings.warn(
            "PretrainBlendScheduleCallback is deprecated; use "
            "dataporter.text.SourceScheduleCallback with schedules keyed "
            "by source_name. Scheduled removal: Phase 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not schedules:
            raise ValueError("schedules must be non-empty")
        # Normalise every entry to control-point form. Points get sorted
        # by step (ints kept; fractional [0,1] resolved later against
        # trainer.max_steps).
        self._schedules = [self._normalize_entry(s) for s in schedules]
        self.log_every_n_steps = log_every_n_steps

    @staticmethod
    def _normalize_entry(sched: dict) -> dict:
        """Validate one schedule entry; lower legacy form to control points."""
        if "source_idx" not in sched:
            raise ValueError(
                f"PretrainBlendScheduleCallback schedule entry missing "
                f"required key 'source_idx': {sched}"
            )
        idx = sched["source_idx"]

        if "points" in sched:
            points = sched["points"]
            legacy_keys = {"weight_start", "weight_end",
                           "blend_start_step", "blend_end_step"}
            extra_legacy = legacy_keys & sched.keys()
            if extra_legacy:
                raise ValueError(
                    f"schedule entry mixes 'points' with legacy keys "
                    f"{sorted(extra_legacy)!r}: {sched}"
                )
            return PretrainBlendScheduleCallback._validate_points(idx, points)

        # Legacy 4-key form → 2-point control-point list.
        for required in (
            "weight_start", "weight_end",
            "blend_start_step", "blend_end_step",
        ):
            if required not in sched:
                raise ValueError(
                    f"schedule entry has no 'points' and is missing legacy "
                    f"key {required!r}: {sched}"
                )
        if sched["blend_end_step"] <= sched["blend_start_step"]:
            raise ValueError(
                f"blend_end_step ({sched['blend_end_step']}) must be > "
                f"blend_start_step ({sched['blend_start_step']})"
            )
        points = [
            {"step": sched["blend_start_step"], "weight": sched["weight_start"]},
            {"step": sched["blend_end_step"], "weight": sched["weight_end"]},
        ]
        return PretrainBlendScheduleCallback._validate_points(idx, points)

    @staticmethod
    def _validate_points(source_idx, points: list[dict]) -> dict:
        """Normalise a points list: sort, type-check, dedupe."""
        if not isinstance(points, list) or len(points) < 1:
            raise ValueError(
                f"'points' must be a non-empty list, got {points!r}"
            )
        normed: list[dict] = []
        for p in points:
            if "step" not in p or "weight" not in p:
                raise ValueError(
                    f"each control point must have 'step' and 'weight', "
                    f"got {p!r}"
                )
            step = p["step"]
            weight = float(p["weight"])
            if isinstance(step, bool) or not isinstance(step, (int, float)):
                raise TypeError(
                    f"step must be int (absolute) or float (fractional), "
                    f"got {type(step).__name__}: {step!r}"
                )
            # Distinguish by Python type: int is always absolute, float
            # is fractional in [0, 1]. Note that 0.0 and 1.0 are
            # *floats* (different from Python int 0 / 1) and ARE valid
            # fraction endpoints, so don't filter on is_integer().
            if isinstance(step, float):
                if not (0.0 <= step <= 1.0):
                    raise ValueError(
                        f"fractional step must be in [0, 1] (got {step}). "
                        f"Use int for absolute steps."
                    )
                is_fractional = True
            else:
                is_fractional = False
            normed.append({
                "step_raw": step,
                "is_fractional": is_fractional,
                "weight": weight,
            })
        # Defer the final sort to resolve-time when fractional steps know
        # their absolute value. But we can sort by raw step now if all
        # entries are the same kind (all int or all float-fractional);
        # mixed entries require the resolve-time sort.
        return {"source_idx": source_idx, "points": normed}

    @staticmethod
    def _resolve_points(
        points: list[dict], max_steps: int,
    ) -> list[tuple[int, float]]:
        """Convert (step_raw, is_fractional, weight) → sorted [(step, w), ...]."""
        resolved: list[tuple[int, float]] = []
        for p in points:
            if p["is_fractional"]:
                step_abs = int(round(float(p["step_raw"]) * max_steps))
            else:
                step_abs = int(p["step_raw"])
            resolved.append((step_abs, p["weight"]))
        resolved.sort(key=lambda t: t[0])
        # Dedupe consecutive identical steps — keep first.
        deduped: list[tuple[int, float]] = []
        for step, w in resolved:
            if deduped and deduped[-1][0] == step:
                continue
            deduped.append((step, w))
        return deduped

    @staticmethod
    def _interp_at(step: int, points: list[tuple[int, float]]) -> float:
        """Piecewise-linear interpolation; clamp at endpoints."""
        if not points:
            return 0.0
        if step <= points[0][0]:
            return float(points[0][1])
        if step >= points[-1][0]:
            return float(points[-1][1])
        # Find the segment containing step.
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= step <= x1:
                if x1 == x0:
                    return float(y0)
                t = (step - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        # Unreachable given the clamps above, but defensive.
        return float(points[-1][1])

    @classmethod
    def _compute_weight(
        cls, step: int, sched: dict, max_steps: int = 0,
    ) -> float:
        """Compute per-source weight at given step.

        ``sched`` is the normalised form (with ``points`` list of
        {step_raw, is_fractional, weight} entries). For tests that pass
        the legacy 4-key form directly, fall through to the old code path.
        """
        # Legacy direct-form support (kept for tests only).
        if "weight_start" in sched and "blend_start_step" in sched:
            if step < sched["blend_start_step"]:
                return float(sched["weight_start"])
            if step >= sched["blend_end_step"]:
                return float(sched["weight_end"])
            progress = (step - sched["blend_start_step"]) / (
                sched["blend_end_step"] - sched["blend_start_step"]
            )
            return float(
                sched["weight_start"]
                + progress * (sched["weight_end"] - sched["weight_start"])
            )
        # Normalised control-point form.
        resolved = cls._resolve_points(sched["points"], max_steps)
        return cls._interp_at(step, resolved)

    def _get_pretrain_dataset(
        self, trainer: L.Trainer,
    ) -> WeightedMultiSourceDataset | None:
        """Get the WeightedMultiSourceDataset directly from the datamodule.

        Returns the dataset stored on ``trainer.datamodule._pretrain_multi_dataset``
        when multi-source mode is active; returns ``None`` otherwise
        (e.g. legacy single-source path), which makes the callback a no-op.
        """
        dm = trainer.datamodule
        if dm is None:
            return None
        ds = getattr(dm, "_pretrain_multi_dataset", None)
        if isinstance(ds, WeightedMultiSourceDataset):
            return ds
        return None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        ds = self._get_pretrain_dataset(trainer)
        if ds is None:
            return
        step = trainer.global_step
        # Resolve fractional steps once we know trainer.max_steps. Cache
        # the per-source resolved curve to avoid recomputing every batch.
        if not hasattr(self, "_resolved_curves"):
            max_steps = max(int(trainer.max_steps or 0), 1)
            self._resolved_curves = [
                (s["source_idx"], self._resolve_points(s["points"], max_steps))
                for s in self._schedules
            ]
        for source_idx, curve in self._resolved_curves:
            if source_idx < 0 or source_idx >= len(ds._weight_tensors):
                continue
            w = self._interp_at(step, curve)
            ds._weight_tensors[source_idx].fill_(w)
        if step % self.log_every_n_steps == 0:
            for i, t in enumerate(ds._weight_tensors):
                pl_module.log(f"blend/pretrain_w{i}", float(t.item()))
