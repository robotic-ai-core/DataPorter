"""Text sample schema.

``TextSampleSpec`` declares the per-sample contract every text training
dataset must satisfy: ``input_ids`` / ``labels`` / ``loss_mask`` /
``source_tag`` with matching dtypes and shapes plus per-source extra
invariants on ``loss_mask``.

Known sources:
  - ``pretrain_pad``: padded sequences; ``loss_mask == (input_ids != pad_id)``
  - ``chat_query``:   response-only training; some tokens masked, not all/none
  - ``val_full``:     unpadded val data; every token is a training target

Adding a new source means adding an entry to ``TextSampleSpec._INVARIANTS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

import torch

from .base import FieldSpec, Schema, SchemaError


def _check_pretrain_pad(sample: Mapping[str, Any], spec: "TextSampleSpec") -> None:
    expected = sample["input_ids"] != spec.pad_token_id
    mask = sample["loss_mask"].bool()
    if mask.shape != expected.shape:
        raise SchemaError(
            f"pretrain_pad: loss_mask shape {tuple(mask.shape)} != "
            f"(input_ids != pad_id) shape {tuple(expected.shape)}"
        )
    if not torch.equal(mask, expected):
        bad = int((mask != expected).sum().item())
        raise SchemaError(
            f"pretrain_pad: loss_mask disagrees with (input_ids != pad_id) in "
            f"{bad} position(s). Common cause: upstream set loss_mask=all-True "
            f"while the sequence contains pad tokens (the historical blended-text "
            f"bug)."
        )


def _check_chat_query(sample: Mapping[str, Any], spec: "TextSampleSpec") -> None:
    mask = sample["loss_mask"].bool()
    if not mask.any():
        raise SchemaError(
            "chat_query: loss_mask is all-False — no response tokens contribute "
            "to loss."
        )
    if mask.all():
        raise SchemaError(
            "chat_query: loss_mask is all-True — query tokens must be masked out."
        )


def _check_val_full(sample: Mapping[str, Any], spec: "TextSampleSpec") -> None:
    mask = sample["loss_mask"].bool()
    if not mask.all():
        n_false = int((~mask).sum().item())
        raise SchemaError(
            f"val_full: loss_mask has {n_false} False entries. Val data should "
            f"contribute every token to CE (no padding, no query-masking)."
        )


@dataclass(frozen=True)
class TextSampleSpec(Schema):
    """Per-sample text contract.

    Per-instance config (``pad_token_id``, ``seq_len``, …) is referenced
    by name from the FIELDS shape entries below, e.g. ``("seq_len",)``.
    """

    pad_token_id: int
    seq_len: int
    tokenizer_id: str
    vocab_size: int | None = None

    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "input_ids", "labels", "loss_mask", "source_tag",
    )
    KNOWN_SOURCES: ClassVar[frozenset[str]] = frozenset({
        "pretrain_pad", "chat_query", "val_full",
    })

    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "input_ids": FieldSpec(dtype=torch.long, shape=("seq_len",)),
        "labels": FieldSpec(dtype=torch.long, shape=("seq_len",)),
        "loss_mask": FieldSpec(dtype=torch.bool, shape=("seq_len",)),
    }

    _INVARIANTS: ClassVar[dict[str, Callable[[Mapping[str, Any], "Schema"], None]]] = {
        "pretrain_pad": _check_pretrain_pad,
        "chat_query": _check_chat_query,
        "val_full": _check_val_full,
    }

    @classmethod
    def from_tokenizer(
        cls, tokenizer: Any, *, seq_len: int, tokenizer_id: str,
    ) -> "TextSampleSpec":
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError(
                f"tokenizer {tokenizer_id!r} has neither pad_token_id nor "
                f"eos_token_id; cannot construct TextSampleSpec"
            )
        return cls(
            pad_token_id=int(pad_id),
            seq_len=int(seq_len),
            tokenizer_id=tokenizer_id,
            vocab_size=getattr(tokenizer, "vocab_size", None),
        )
