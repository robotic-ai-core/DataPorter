"""Stochastic weighted multi-source dataset for text pretraining.

A Dataset wrapper that stochastically selects from N source datasets
per ``__getitem__`` call, with relative weights backed by shared-memory
tensors so a Lightning callback (``PretrainBlendScheduleCallback``)
can mutate them mid-training without re-pickling for DataLoader
workers.

Complements DataPorter's existing
:class:`dataporter._blending.WeightedRoundRobinDispatcher`, which is
deterministic and used inside producer-pool kernels.  This class
operates at the Dataset boundary instead.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class WeightedMultiSourceDataset(Dataset):
    """Stochastically samples from N spec-compliant pretrain sources.

    Each source has its own shared-memory weight tensor so per-source
    weights can be updated mid-training by a scheduling callback.  All
    source datasets must produce SampleSpec-compliant outputs (i.e.
    already wrapped by ``pretrain_pad_adapter``) — this class is a
    pure passthrough on top of the chosen source's ``__getitem__``.

    The virtual length is taken from the *first* source so the
    DataLoader's RandomSampler has a stable index range, mirroring how
    BlendedTextDataset uses the pretrain length even when chat is
    sampled instead.

    Args:
        sources: list of (dataset, weight_tensor) tuples, where each
            ``weight_tensor`` is a 1-element shared-memory ``torch.Tensor``
            holding the (mutable) sampling weight for that source.
            Weights are normalised at sample time, so absolute scale
            doesn't matter — only relative magnitudes.
    """

    def __init__(self, sources: list[tuple[Dataset, torch.Tensor]]):
        if not sources:
            raise ValueError("WeightedMultiSourceDataset requires at least one source")
        self._datasets = [s[0] for s in sources]
        self._weight_tensors = [s[1] for s in sources]
        for w in self._weight_tensors:
            if w.numel() != 1:
                raise ValueError(
                    f"weight tensors must be 1-element, got numel={w.numel()}"
                )

    def __len__(self) -> int:
        # Virtual length pinned to the first source — matches the
        # BlendedTextDataset convention of using a single dominant
        # source's length as the sampler range.
        return len(self._datasets[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        weights = [max(0.0, t.item()) for t in self._weight_tensors]
        total = sum(weights)
        if total <= 0:
            # All weights zero — fall back to first source rather than raising;
            # transient zero-weight states can occur during a schedule's
            # leading edge before the first ramp tick.
            ds = self._datasets[0]
            return ds[idx % len(ds)]
        # Inline cumulative sample for clarity (avoids allocating numpy/torch
        # for a tiny K-way choice on the hot path).
        r = random.random() * total
        c = 0.0
        for ds, w in zip(self._datasets, weights):
            c += w
            if r < c:
                return ds[idx % len(ds)]
        # Floating-point fall-through: numerical edge case where
        # r ≈ total. Treat as last source.
        ds = self._datasets[-1]
        return ds[idx % len(ds)]
