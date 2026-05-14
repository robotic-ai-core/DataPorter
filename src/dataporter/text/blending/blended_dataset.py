"""Backward-compat wrapper: :class:`BlendedTextDataset`.

.. deprecated:: phase-3b
   Use :class:`ScheduledBlendDataset` with explicit sources and a
   :class:`SourceScheduleCallback`. This class survives only to keep
   existing YAML ``class_path`` entries working through Phase 4.
   Removal is scheduled for Phase 5.

The historical class implemented a two-way mix between a pretrain
dataset and a chat dataset, governed by a single ``chat_ratio``
scalar. The wrapper delegates to a :class:`ScheduledBlendDataset` with
sources ``"__pretrain__"`` and ``"__chat__"``; setting ``chat_ratio``
writes both weights ``[1 - r, r]``.
"""

from __future__ import annotations

import logging
import warnings

import torch
from torch.utils.data import Dataset

from dataporter.schemas import SchemaError, TextSampleSpec

from .scheduled_blend import ScheduledBlendDataset

logger = logging.getLogger(__name__)


class BlendedTextDataset(Dataset):
    """Wrapper preserving the historical 2-way pretrain/chat API.

    .. deprecated::
       Use :class:`ScheduledBlendDataset` directly with named sources
       and :class:`SourceScheduleCallback` for the schedule. The 2-way
       mix is the N=2 case of the N-way primitive.

    Args:
        pretrain_dataset: Spec-compliant dataset with source_tag="pretrain_pad".
        chat_dataset: Spec-compliant dataset with source_tag="chat_query".
        sample_spec: Contract; used to probe both sources at construction.
            If ``None``, probe is skipped.
        seed: Retained for backward compat; unused by the underlying
            :class:`ScheduledBlendDataset` (which uses the process-global
            ``random`` for source selection — matches historical behavior).
        probe_n: Number of samples per source to validate at init.
    """

    def __init__(
        self,
        pretrain_dataset: Dataset,
        chat_dataset: Dataset,
        sample_spec: TextSampleSpec | None = None,
        seed: int = 42,
        probe_n: int = 4,
    ):
        warnings.warn(
            "BlendedTextDataset is deprecated; use "
            "dataporter.text.ScheduledBlendDataset with named sources "
            "and dataporter.text.SourceScheduleCallback for the schedule. "
            "Scheduled removal: Phase 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._pretrain = pretrain_dataset
        self._chat = chat_dataset
        self._seed = seed
        self._sample_spec = sample_spec

        if sample_spec is not None and probe_n > 0:
            self._probe(sample_spec, probe_n)

        # Build the unified mixer. Start at chat_ratio=0 (pure pretrain),
        # matching legacy behavior — the schedule callback ramps the
        # chat weight up over training.
        w_pretrain = torch.tensor([1.0], dtype=torch.float64)
        w_chat = torch.tensor([0.0], dtype=torch.float64)
        w_pretrain.share_memory_()
        w_chat.share_memory_()
        self._inner = ScheduledBlendDataset(
            [
                (pretrain_dataset, w_pretrain, "__pretrain__"),
                (chat_dataset, w_chat, "__chat__"),
            ],
            virtual_length=len(pretrain_dataset),
        )

    def _probe(self, spec: TextSampleSpec, n: int) -> None:
        """Validate both sources against the spec. Raises on mismatch."""
        try:
            spec.probe_dataset(self._pretrain, "pretrain_pad", n=n)
        except SchemaError:
            logger.exception(
                "BlendedTextDataset: pretrain source failed spec probe. "
                "Ensure it is wrapped in pretrain_pad_adapter()."
            )
            raise
        try:
            spec.probe_dataset(self._chat, "chat_query", n=n)
        except SchemaError:
            logger.exception(
                "BlendedTextDataset: chat source failed spec probe. "
                "Ensure it is wrapped in chat_query_adapter()."
            )
            raise

    # ---- ratio property ----

    @property
    def chat_ratio(self) -> float:
        # In a 2-way mix with weights [1-r, r], the chat weight IS the
        # chat ratio. Reading the chat weight is the right semantic.
        return float(self._inner._weight_tensors[1].item())

    @chat_ratio.setter
    def chat_ratio(self, value: float) -> None:
        r = max(0.0, min(1.0, float(value)))
        # Write both weights so the underlying ScheduledBlendDataset
        # normalises to exactly r (chat) and 1-r (pretrain). Without
        # the pretrain write the normalisation would shift the effective
        # ratio.
        self._inner._weight_tensors[0].fill_(1.0 - r)
        self._inner._weight_tensors[1].fill_(r)

    @property
    def _chat_ratio_t(self) -> torch.Tensor:
        """Legacy attribute exposing the chat weight tensor.

        Tests assert ``.is_shared()``; mutating this directly bypasses
        the ``chat_ratio`` setter's invariant (``w_pretrain + w_chat == 1``)
        and is discouraged — write ``chat_ratio = r`` instead.
        """
        return self._inner._weight_tensors[1]

    # ---- Dataset interface ----

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        return self._inner[idx]
