"""Smoke tests for BlendedTextDataset.

The class itself is deprecated in Phase 3b; DeprecationWarning emission
is covered in ``test_wrapper_equivalence.py``. These tests intentionally
suppress that warning so legacy behavior coverage stays uncluttered.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.schemas import SchemaError, TextSampleSpec
from dataporter.text.blending import BlendedTextDataset

pytestmark = pytest.mark.filterwarnings(
    "ignore:BlendedTextDataset is deprecated:DeprecationWarning"
)

SEQ_LEN = 8
PAD = 0
VOCAB = 100


def _spec() -> TextSampleSpec:
    return TextSampleSpec(
        pad_token_id=PAD, seq_len=SEQ_LEN,
        tokenizer_id="test", vocab_size=VOCAB,
    )


class _CompliantDataset(Dataset):
    """Spec-compliant pretrain/chat stub."""

    def __init__(self, tag: str, n: int = 8):
        self._tag = tag
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        if self._tag == "pretrain_pad":
            ids = torch.tensor(
                [1, 2, 3, 4, PAD, PAD, PAD, PAD], dtype=torch.long,
            )
            mask = ids != PAD
        else:  # chat_query
            ids = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.long)
            mask = torch.tensor([False, False, True, True, True, True, True, True])
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "loss_mask": mask,
            "source_tag": self._tag,
        }


class _BadDataset(Dataset):
    """Tag mismatch — should fail the spec probe."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        ids = torch.zeros(SEQ_LEN, dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "loss_mask": torch.ones(SEQ_LEN, dtype=torch.bool),
            "source_tag": "wrong_tag",
        }


class TestConstruction:

    def test_constructs_without_spec(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad", n=10),
            chat_dataset=_CompliantDataset("chat_query", n=5),
            sample_spec=None,
        )
        assert len(ds) == 10
        assert ds.chat_ratio == 0.0

    def test_constructs_with_spec_probe(self):
        spec = _spec()
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad", n=10),
            chat_dataset=_CompliantDataset("chat_query", n=5),
            sample_spec=spec,
            probe_n=4,
        )
        assert len(ds) == 10

    def test_spec_probe_catches_bad_source(self):
        spec = _spec()
        with pytest.raises(SchemaError):
            BlendedTextDataset(
                pretrain_dataset=_BadDataset(),
                chat_dataset=_CompliantDataset("chat_query"),
                sample_spec=spec,
            )


class TestChatRatio:

    def test_initial_ratio_zero(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        assert ds.chat_ratio == 0.0

    def test_setter_persists(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        ds.chat_ratio = 0.3
        assert ds.chat_ratio == pytest.approx(0.3)

    def test_setter_clamps_to_unit_range(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        ds.chat_ratio = -0.5
        assert ds.chat_ratio == 0.0
        ds.chat_ratio = 1.5
        assert ds.chat_ratio == 1.0


class TestStochasticSelection:

    def test_ratio_zero_yields_only_pretrain(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        ds.chat_ratio = 0.0
        random.seed(42)
        for _ in range(40):
            sample = ds[0]
            assert sample["source_tag"] == "pretrain_pad"

    def test_ratio_one_yields_only_chat(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        ds.chat_ratio = 1.0
        random.seed(42)
        for _ in range(40):
            sample = ds[0]
            assert sample["source_tag"] == "chat_query"

    def test_mid_ratio_mixes(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad", n=20),
            chat_dataset=_CompliantDataset("chat_query", n=20),
        )
        ds.chat_ratio = 0.5
        random.seed(0)
        tags = [ds[i % 20]["source_tag"] for i in range(200)]
        n_pretrain = tags.count("pretrain_pad")
        n_chat = tags.count("chat_query")
        # Within a generous tolerance for 200 trials.
        assert 60 < n_pretrain < 140
        assert 60 < n_chat < 140


class TestSharedMemory:

    def test_ratio_tensor_is_shared_memory(self):
        ds = BlendedTextDataset(
            pretrain_dataset=_CompliantDataset("pretrain_pad"),
            chat_dataset=_CompliantDataset("chat_query"),
        )
        # Shared-memory backed: the underlying tensor reports is_shared().
        assert ds._chat_ratio_t.is_shared()
