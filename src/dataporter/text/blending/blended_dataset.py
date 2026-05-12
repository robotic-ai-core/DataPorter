"""Two-source blend (pretrain + chat) with mutable mixing ratio.

The :class:`MixingScheduleCallback` updates this dataset's
``chat_ratio`` over the course of training; the ratio is backed by a
shared-memory tensor so changes from the main process are visible to
DataLoader workers without re-pickling.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import Dataset

from dataporter.schemas import SchemaError, TextSampleSpec

logger = logging.getLogger(__name__)


class BlendedTextDataset(Dataset):
    """Blends two spec-compliant text datasets with a mutable mixing ratio.

    At each ``__getitem__`` call, the dataset stochastically selects
    from either the pre-training or chat dataset based on the current
    ``chat_ratio``.

    Both sources must already be wrapped by their respective adapter
    chains (``pretrain_pad_adapter`` / ``chat_query_adapter``) so they
    emit ``{input_ids, labels, loss_mask, source_tag}`` per
    :class:`TextSampleSpec`. This class is a pure passthrough over the
    adapter output — it does **not** synthesise loss_mask itself.

    The mixing ratio is backed by a **shared-memory tensor** so that
    updates from the main process (callback) are visible to persistent
    DataLoader workers.

    Args:
        pretrain_dataset: Spec-compliant dataset with source_tag="pretrain_pad".
        chat_dataset: Spec-compliant dataset with source_tag="chat_query".
        sample_spec: Contract; used to probe both sources at construction.
            If ``None``, probe is skipped.
        seed: Base seed for worker-local RNG initialisation.
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
        self._pretrain = pretrain_dataset
        self._chat = chat_dataset
        self._seed = seed
        self._sample_spec = sample_spec

        if sample_spec is not None and probe_n > 0:
            self._probe(sample_spec, probe_n)

        # Shared-memory tensor: survives pickling for spawn-based workers,
        # visible across fork-based workers without copies.
        self._chat_ratio_t = torch.tensor([0.0], dtype=torch.float64)
        self._chat_ratio_t.share_memory_()

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
        return self._chat_ratio_t.item()

    @chat_ratio.setter
    def chat_ratio(self, value: float) -> None:
        self._chat_ratio_t.fill_(max(0.0, min(1.0, float(value))))

    # ---- Dataset interface ----

    def __len__(self) -> int:
        # Virtual length = pre-training size (dominant dataset).
        # The DataLoader shuffles these indices; chat samples are injected
        # stochastically, so the effective epoch size doesn't depend on
        # len(chat).
        return len(self._pretrain)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Both sources are spec-compliant post-adapter. Pure passthrough —
        # loss_mask semantics are owned by the source (pretrain_pad marks
        # pads False; chat_query masks the query prefix).
        ratio = self._chat_ratio_t.item()
        if random.random() < ratio:
            chat_idx = random.randrange(len(self._chat))
            return self._chat[chat_idx]
        return self._pretrain[idx]
