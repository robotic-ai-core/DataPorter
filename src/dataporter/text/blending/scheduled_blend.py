"""N-way weighted-random mixer over heterogeneous datasets.

:class:`ScheduledBlendDataset` is the canonical multi-source mixing
primitive — one implementation that subsumes both the 2-way pretrain/chat
mix (formerly :class:`BlendedTextDataset`) and the N-way pretrain blend
(formerly :class:`WeightedMultiSourceDataset`).

Each source carries a 1-element shared-memory ``torch.Tensor`` holding
its mutable sampling weight; a Lightning callback
(:class:`SourceScheduleCallback`) mutates these mid-training to drive
per-source curriculum curves. Weights are normalised at sample time, so
absolute scale is irrelevant — only ratios matter.

Sources may be heterogeneous: chat datasets (on-the-fly template),
pretrain Parquet (memory-mapped), or any other Dataset. Per-source
adapter chains (drop-extras → labels → loss-mask → stamp-source-tag)
must already be applied — this class is a pure passthrough on top of
the chosen source's ``__getitem__``, then stamps ``source_idx``.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


class ScheduledBlendDataset(Dataset):
    """Stochastic N-way weighted draw over spec-compliant sources.

    Args:
        sources: Sequence of ``(dataset, weight_tensor, name)`` tuples.
            ``weight_tensor`` is a 1-element ``torch.Tensor`` holding
            the mutable sampling weight; it is :meth:`share_memory_`'d
            in the constructor so per-worker DataLoader processes see
            mutations made on rank 0. ``name`` is a short string used
            for sample tagging + schedule resolution by name (e.g.
            ``"v1a"``, ``"fineweb-edu"``, ``"no_robots"``).
        virtual_length: If set, ``__len__`` returns this. Otherwise
            falls back to the length of the first source (matching the
            historical convention so :class:`torch.utils.data.RandomSampler`
            has a stable index range even when the dominant source
            rotates).
        zero_weight_fallback: Source index to fall back to when every
            weight is zero (transient state at schedule leading edges).
            Default ``0``.

    Raises:
        ValueError: empty ``sources``; duplicate ``name`` entries;
            non-1-element weight tensor; ``zero_weight_fallback`` out
            of range.
    """

    def __init__(
        self,
        sources: Sequence[tuple[Dataset, torch.Tensor, str]],
        virtual_length: int | None = None,
        zero_weight_fallback: int = 0,
    ) -> None:
        if not sources:
            raise ValueError(
                "ScheduledBlendDataset requires at least one source"
            )
        datasets: list[Dataset] = []
        weights: list[torch.Tensor] = []
        names: list[str] = []
        seen: dict[str, int] = {}
        for i, entry in enumerate(sources):
            if len(entry) != 3:
                raise ValueError(
                    f"sources[{i}] must be a (dataset, weight_tensor, name) "
                    f"3-tuple; got tuple of length {len(entry)}"
                )
            ds, w, name = entry
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"sources[{i}].name must be a non-empty string; got {name!r}"
                )
            if w.numel() != 1:
                raise ValueError(
                    f"sources[{i}].weight_tensor must be 1-element; "
                    f"got numel={w.numel()}"
                )
            if name in seen:
                raise ValueError(
                    f"duplicate source name {name!r} at indices "
                    f"{seen[name]} and {i}; names must be unique so "
                    f"schedules-by-name can disambiguate"
                )
            # Ensure shared memory: caller may or may not have done this
            # already; idempotent.
            if not w.is_shared():
                w.share_memory_()
            datasets.append(ds)
            weights.append(w)
            names.append(name)
            seen[name] = i

        if not 0 <= zero_weight_fallback < len(datasets):
            raise ValueError(
                f"zero_weight_fallback={zero_weight_fallback} out of range "
                f"[0, {len(datasets)})"
            )

        self._datasets = datasets
        self._weight_tensors = weights
        self._names = names
        self._name_to_idx = seen
        self._zero_weight_fallback = zero_weight_fallback
        self._virtual_length = (
            int(virtual_length)
            if virtual_length is not None
            else len(datasets[0])
        )

    # ---- introspection ----

    @property
    def num_sources(self) -> int:
        return len(self._datasets)

    @property
    def source_names(self) -> list[str]:
        """Indexed source names; ``source_names[i]`` is source ``i``'s label."""
        return list(self._names)

    def resolve(self, source: int | str) -> int:
        """Resolve an int index or string name to an index. Raises on
        unknown name or out-of-range index."""
        if isinstance(source, bool):
            # bool is a subclass of int; reject explicitly to avoid
            # ``set_weight(True, 0.5)`` accidentally writing to index 1.
            raise TypeError(
                f"source must be int (index) or str (name); got bool {source!r}"
            )
        if isinstance(source, int):
            if not 0 <= source < len(self._datasets):
                raise IndexError(
                    f"source index {source} out of range "
                    f"[0, {len(self._datasets)})"
                )
            return source
        if isinstance(source, str):
            if source not in self._name_to_idx:
                raise KeyError(
                    f"unknown source name {source!r}; known: "
                    f"{sorted(self._name_to_idx)}"
                )
            return self._name_to_idx[source]
        raise TypeError(
            f"source must be int (index) or str (name); got "
            f"{type(source).__name__}: {source!r}"
        )

    # ---- weight mutation ----

    def set_weight(self, source: int | str, value: float) -> None:
        """Update one source's weight in shared memory.

        ``source`` is either the integer index or the string name. The
        change is visible to in-flight ``__getitem__`` calls in worker
        processes via the shared-memory tensor backing.
        """
        idx = self.resolve(source)
        self._weight_tensors[idx].fill_(float(value))

    def get_weight(self, source: int | str) -> float:
        idx = self.resolve(source)
        return float(self._weight_tensors[idx].item())

    def get_weights(self) -> list[float]:
        """Snapshot of all current weights, in source-index order."""
        return [float(t.item()) for t in self._weight_tensors]

    # ---- Dataset protocol ----

    def __len__(self) -> int:
        return self._virtual_length

    def __getitem__(self, idx: int) -> dict:
        chosen = self._choose_source()
        ds = self._datasets[chosen]
        sample = ds[idx % len(ds)]
        # Always copy the dict at the top level so adding source_idx
        # doesn't mutate any cache the inner dataset may hand back.
        out = dict(sample)
        # int32 scalar; default_collate produces [B] int32 downstream.
        out["source_idx"] = torch.tensor(chosen, dtype=torch.int32)
        return out

    def _choose_source(self) -> int:
        """Cumulative-sum weighted draw. Mirrors the canonical pattern
        from the historical ``WeightedMultiSourceDataset``.
        """
        weights = [max(0.0, t.item()) for t in self._weight_tensors]
        total = sum(weights)
        if total <= 0.0:
            return self._zero_weight_fallback
        r = random.random() * total
        c = 0.0
        for i, w in enumerate(weights):
            c += w
            if r < c:
                return i
        # Floating-point edge: r ≈ total after the loop. Bucket into
        # the last source.
        return len(weights) - 1
