"""Lightning callback driving :class:`ScheduledBlendDataset` weights.

Per-source schedules are piecewise-linear curves over training steps,
applied at every training batch. Each schedule entry targets a source
by **name** (preferred; survives source-reordering in YAML) or by
integer **index** (kept for migration from
:class:`PretrainBlendScheduleCallback`).

Step values may be ``int`` (absolute step number) or ``float`` in
``[0, 1]`` (fraction of ``trainer.max_steps``). Mixing the two within
one schedule is supported. Fractional steps make a schedule portable
across runs of different lengths.
"""

from __future__ import annotations

import lightning as L

from .scheduled_blend import ScheduledBlendDataset


class SourceScheduleCallback(L.Callback):
    """Schedules per-source weights as piecewise-linear curves.

    Args:
        schedules: List of dicts. Each dict has exactly one of
            ``source_name: str`` or ``source_idx: int`` plus a
            ``points: list[{step, weight}]`` entry. Linear interpolation
            between consecutive points; weights are held at the first
            point's value before the first step and the last point's
            value after the last. Sources without a schedule entry keep
            their initial weight unchanged.
        log_every_n_steps: How often to log the current weights to the
            logger. Default 100.
        dataset_attr: Attribute name on ``trainer.datamodule`` holding
            the :class:`ScheduledBlendDataset`. Default
            ``"scheduled_blend_dataset"``. If not found the callback
            silently no-ops, so non-blended runs aren't affected.

    Example (3-phase curriculum, schedules keyed by name):

        schedules:
          - source_name: fineweb-edu
            points:
              - {step: 0.0, weight: 1.0}
              - {step: 0.4, weight: 0.4}
              - {step: 1.0, weight: 0.1}
          - source_name: tiny-textbooks
            points:
              - {step: 0.0, weight: 0.0}
              - {step: 0.4, weight: 0.4}
              - {step: 1.0, weight: 0.5}
    """

    def __init__(
        self,
        schedules: list[dict],
        log_every_n_steps: int = 100,
        dataset_attr: str = "scheduled_blend_dataset",
    ):
        super().__init__()
        if not schedules:
            raise ValueError("schedules must be non-empty")
        self._schedules = [self._normalize_entry(s) for s in schedules]
        self.log_every_n_steps = log_every_n_steps
        self.dataset_attr = dataset_attr
        # Resolved-curves cache (per source): list[(source_key, [(step, w), ...])].
        # Filled lazily on the first call once trainer.max_steps is known.
        self._resolved_curves: list[tuple[int | str, list[tuple[int, float]]]] | None = None

    # ---- entry validation ----

    @staticmethod
    def _normalize_entry(sched: dict) -> dict:
        """Validate a schedule entry; canonicalise to control-point form."""
        has_name = "source_name" in sched
        has_idx = "source_idx" in sched
        if has_name == has_idx:
            raise ValueError(
                f"schedule entry must have exactly one of 'source_name' or "
                f"'source_idx'; got {sched!r}"
            )
        if has_name:
            if not isinstance(sched["source_name"], str) or not sched["source_name"].strip():
                raise ValueError(
                    f"source_name must be a non-empty string; got "
                    f"{sched['source_name']!r}"
                )
            key: int | str = sched["source_name"]
        else:
            if isinstance(sched["source_idx"], bool) or not isinstance(
                sched["source_idx"], int
            ):
                raise TypeError(
                    f"source_idx must be int; got "
                    f"{type(sched['source_idx']).__name__}: "
                    f"{sched['source_idx']!r}"
                )
            key = int(sched["source_idx"])
        if "points" not in sched:
            raise ValueError(
                f"schedule entry missing 'points': {sched!r}"
            )
        return {"key": key, "points": _validate_points(sched["points"])}

    # ---- discovery ----

    def _get_dataset(self, trainer: L.Trainer) -> ScheduledBlendDataset | None:
        dm = trainer.datamodule
        if dm is None:
            return None
        ds = getattr(dm, self.dataset_attr, None)
        if isinstance(ds, ScheduledBlendDataset):
            return ds
        return None

    # ---- hot path ----

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        ds = self._get_dataset(trainer)
        if ds is None:
            return
        if self._resolved_curves is None:
            max_steps = max(int(trainer.max_steps or 0), 1)
            self._resolved_curves = [
                (s["key"], _resolve_points(s["points"], max_steps))
                for s in self._schedules
            ]
        step = trainer.global_step
        for key, curve in self._resolved_curves:
            try:
                idx = ds.resolve(key)
            except (KeyError, IndexError):
                # Schedule references a source the dataset doesn't have.
                # Skip — matches the legacy "silently skip out-of-range"
                # behavior so a stale schedule entry doesn't crash training.
                continue
            w = _interp_at(step, curve)
            ds._weight_tensors[idx].fill_(w)
        if step % self.log_every_n_steps == 0:
            for i, w in enumerate(ds.get_weights()):
                # Log by name so dashboards survive source-reordering.
                name = ds.source_names[i]
                pl_module.log(f"blend/w/{name}", float(w))


# ----------------------------------------------------------------------
# Piecewise-linear curve math
#
# Duplicated (intentionally) from the legacy PretrainBlendScheduleCallback.
# When the legacy callback is deleted in Phase 5 the legacy copy
# disappears with it.
# ----------------------------------------------------------------------


def _validate_points(points: list[dict]) -> list[dict]:
    """Type-check and canonicalise a control-points list. No sort here —
    fractional steps resolve to absolute steps at run time, so the sort
    is deferred.
    """
    if not isinstance(points, list) or len(points) < 1:
        raise ValueError(
            f"'points' must be a non-empty list, got {points!r}"
        )
    normed: list[dict] = []
    for p in points:
        if "step" not in p or "weight" not in p:
            raise ValueError(
                f"each control point must have 'step' and 'weight', got {p!r}"
            )
        step = p["step"]
        weight = float(p["weight"])
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise TypeError(
                f"step must be int (absolute) or float (fractional), "
                f"got {type(step).__name__}: {step!r}"
            )
        if isinstance(step, float):
            if not 0.0 <= step <= 1.0:
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
    return normed


def _resolve_points(
    points: list[dict], max_steps: int,
) -> list[tuple[int, float]]:
    """Resolve fractional steps against ``max_steps`` and sort."""
    resolved: list[tuple[int, float]] = []
    for p in points:
        if p["is_fractional"]:
            step_abs = int(round(float(p["step_raw"]) * max_steps))
        else:
            step_abs = int(p["step_raw"])
        resolved.append((step_abs, p["weight"]))
    resolved.sort(key=lambda t: t[0])
    deduped: list[tuple[int, float]] = []
    for step, w in resolved:
        if deduped and deduped[-1][0] == step:
            continue
        deduped.append((step, w))
    return deduped


def _interp_at(step: int, points: list[tuple[int, float]]) -> float:
    """Piecewise-linear interpolation; clamp at the endpoints."""
    if not points:
        return 0.0
    if step <= points[0][0]:
        return float(points[0][1])
    if step >= points[-1][0]:
        return float(points[-1][1])
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= step <= x1:
            if x1 == x0:
                return float(y0)
            t = (step - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(points[-1][1])
