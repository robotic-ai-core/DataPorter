"""Composable Dataset adapters for meeting the TextSampleSpec contract.

Each adapter is a tiny Dataset wrapper that does exactly one thing. Sources
that emit partial/nonconforming samples (e.g. DataPorter's TokenShuffleBufferDataset
emits extra ``length``/``key`` keys; ParquetTokenDataset doesn't emit a
``loss_mask``) pass through an adapter chain to meet the contract.

Three recipes are provided for the three known sources:

  - ``pretrain_pad_adapter``: DropExtras → AddCausalLabels → EnsureLossMask(pad_id) → StampSourceTag
  - ``chat_query_adapter``:   DropExtras → StampSourceTag (chat dataset already emits mask+labels)
  - ``val_full_adapter``:     DropExtras → AddCausalLabels → EnsureLossMask(None=all-True) → StampSourceTag

Adapters are plain classes, picklable for DataLoader workers.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import Dataset

from .text import TextSampleSpec


_DEFAULT_KEEP = frozenset({"input_ids", "labels", "loss_mask", "source_tag"})


class _AdapterBase(Dataset):
    def __init__(self, inner: Dataset):
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)


class DropExtras(_AdapterBase):
    """Drop keys outside the keep-set.

    Reason: DataPorter emits ``length``/``key``; those break default_collate when
    mixed with ChatDataset batches that don't include them.
    """

    def __init__(self, inner: Dataset, keep: Iterable[str] | None = None):
        super().__init__(inner)
        self._keep = frozenset(keep) if keep is not None else _DEFAULT_KEEP

    def __getitem__(self, idx):
        sample = self._inner[idx]
        return {k: v for k, v in sample.items() if k in self._keep}


class AddCausalLabels(_AdapterBase):
    """Set ``labels = input_ids.clone()`` if missing. Passthrough otherwise."""

    def __getitem__(self, idx):
        sample = dict(self._inner[idx])
        if "labels" not in sample:
            sample["labels"] = sample["input_ids"].clone()
        return sample


class EnsureLossMask(_AdapterBase):
    """Ensure ``loss_mask`` is present as bool.

    If missing and ``pad_token_id`` is provided: ``loss_mask = (input_ids != pad_id)``.
    If missing and ``pad_token_id is None``: ``loss_mask = all-True`` (every token counts).
    If present with non-bool dtype: cast to bool.
    """

    def __init__(self, inner: Dataset, pad_token_id: int | None):
        super().__init__(inner)
        self._pad_id = pad_token_id

    def __getitem__(self, idx):
        sample = dict(self._inner[idx])
        if "loss_mask" not in sample:
            if self._pad_id is None:
                sample["loss_mask"] = torch.ones(
                    sample["input_ids"].shape, dtype=torch.bool,
                )
            else:
                sample["loss_mask"] = sample["input_ids"] != self._pad_id
        elif sample["loss_mask"].dtype != torch.bool:
            sample["loss_mask"] = sample["loss_mask"].bool()
        return sample


class StampSourceTag(_AdapterBase):
    """Attach a static ``source_tag`` string to every sample."""

    def __init__(self, inner: Dataset, tag: str):
        super().__init__(inner)
        self._tag = tag

    def __getitem__(self, idx):
        sample = dict(self._inner[idx])
        sample["source_tag"] = self._tag
        return sample


class ValidateSpec(_AdapterBase):
    """Validate samples against a TextSampleSpec at access time.

    Two modes:
      - ``first_n`` (default): validate only the first N samples seen, then passthrough.
      - ``every``: validate every Nth sample seen.

    Useful as a belt-and-braces runtime guard in addition to the setup-time
    probe on ``BlendedTextDataset.__init__``.
    """

    def __init__(
        self,
        inner: Dataset,
        spec: TextSampleSpec,
        source: str,
        *,
        every: int | None = None,
        first_n: int = 8,
    ):
        super().__init__(inner)
        self._spec = spec
        self._source = source
        self._every = every
        self._first_n = first_n
        self._count = 0

    def __getitem__(self, idx):
        sample = self._inner[idx]
        self._count += 1
        if self._every is not None:
            if self._count % self._every == 0:
                self._spec.validate(sample, self._source)
        else:
            if self._count <= self._first_n:
                self._spec.validate(sample, self._source)
        return sample


# ----- recipes ---------------------------------------------------------------


def pretrain_pad_adapter(inner: Dataset, spec: TextSampleSpec) -> Dataset:
    """Chain for padded pretrain data (DataPorter ShuffleBuffer or ParquetTokenDataset)."""
    return StampSourceTag(
        EnsureLossMask(
            AddCausalLabels(DropExtras(inner)),
            pad_token_id=spec.pad_token_id,
        ),
        tag="pretrain_pad",
    )


def chat_query_adapter(inner: Dataset, spec: TextSampleSpec) -> Dataset:
    """Chain for ChatDataset (already emits labels + query-masked loss_mask)."""
    return StampSourceTag(DropExtras(inner), tag="chat_query")


def val_full_adapter(inner: Dataset, spec: TextSampleSpec) -> Dataset:
    """Chain for unpadded val data (_CachedValDataset, _DummyValDataset).

    Uses pad_token_id=None so EnsureLossMask synthesizes all-True mask
    regardless of token content.
    """
    return StampSourceTag(
        EnsureLossMask(
            AddCausalLabels(DropExtras(inner)),
            pad_token_id=None,
        ),
        tag="val_full",
    )
