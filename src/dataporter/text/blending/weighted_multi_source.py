"""Backward-compat wrapper: :class:`WeightedMultiSourceDataset`.

.. deprecated:: phase-3b
   Use :class:`ScheduledBlendDataset` directly. This class survives only
   to keep existing YAML ``class_path`` entries working through Phase 4.
   Removal is scheduled for Phase 5.

The historical class accepted 2-tuples ``(dataset, weight_tensor)``
without names. The wrapper synthesises names ``"source_0"``, ``"source_1"``,
... and delegates every behavior to a :class:`ScheduledBlendDataset`
underneath. ``_weight_tensors`` is exposed for the legacy
:class:`PretrainBlendScheduleCallback` which writes to it directly.
"""

from __future__ import annotations

import warnings

import torch
from torch.utils.data import Dataset

from .scheduled_blend import ScheduledBlendDataset


class WeightedMultiSourceDataset(Dataset):
    """Wrapper preserving the historical 2-tuple constructor signature.

    .. deprecated::
       Pass ``(dataset, weight_tensor, name)`` triples to
       :class:`ScheduledBlendDataset` instead — names make schedules
       survive source reordering and are required by
       :class:`SourceScheduleCallback`.
    """

    def __init__(self, sources: list[tuple[Dataset, torch.Tensor]]):
        warnings.warn(
            "WeightedMultiSourceDataset is deprecated; use "
            "dataporter.text.ScheduledBlendDataset with (dataset, "
            "weight_tensor, name) triples. Scheduled removal: Phase 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not sources:
            raise ValueError("WeightedMultiSourceDataset requires at least one source")
        triples = [
            (ds, w, f"source_{i}") for i, (ds, w) in enumerate(sources)
        ]
        self._inner = ScheduledBlendDataset(triples)

    @property
    def _datasets(self):
        # Legacy attribute name preserved for tests / debuggers that
        # poked at internals.
        return self._inner._datasets

    @property
    def _weight_tensors(self):
        # PretrainBlendScheduleCallback writes here directly.
        return self._inner._weight_tensors

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        return self._inner[idx]
