"""Smoke tests for ChatDataset (map-style preload).

Uses a fake tokenizer to avoid network/HF downloads. Focus is on the
Dataset contract — full chat-formatting correctness is covered by
``test_template.py``.

``ChatStreamDataset`` is exercised by the mamba_qat_slm-side test
``test_chat_dataset.py`` until Phase 4 swaps the imports; we only
verify the import path here.
"""

from __future__ import annotations

import pytest
import torch

from dataporter.text import ChatDataset, ChatStreamDataset


ROLES = {"query": "[Q]", "response": "[A]", "end": "[END]"}
FIELD_MAP = {"query": "question", "response": "answer"}


class _FakeTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._next_id = 2

    def _token_id(self, t: str) -> int:
        if t not in self._vocab:
            self._vocab[t] = self._next_id
            self._next_id += 1
        return self._vocab[t]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._token_id(tok) for tok in text.split()]


def _examples(n: int) -> list[dict[str, str]]:
    return [{"question": f"q {i}", "answer": f"a {i}"} for i in range(n)]


class TestChatDataset:

    def test_loads_iterable_source(self):
        ds = ChatDataset(
            source=_examples(8),
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
        )
        assert len(ds) == 8
        assert ds.seq_len == 32

    def test_returns_tensors_with_correct_dtypes(self):
        ds = ChatDataset(
            source=_examples(4),
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
        )
        sample = ds[0]
        assert sample["input_ids"].dtype == torch.long
        assert sample["labels"].dtype == torch.long
        assert sample["loss_mask"].dtype == torch.bool
        assert sample["input_ids"].shape == (32,)

    def test_max_examples_caps_dataset(self):
        ds = ChatDataset(
            source=_examples(20),
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
            max_examples=5,
        )
        assert len(ds) == 5

    def test_dataset_name_from_string_source(self):
        # When source is a string (HF dataset id), dataset_name is the last segment.
        # We can't actually load HF here, so just check the property logic via
        # an empty/iterable source and the custom path.
        ds = ChatDataset(
            source=_examples(2),
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
        )
        assert ds.dataset_name == "custom"

    def test_too_short_seq_len_raises(self):
        with pytest.raises(ValueError, match="No examples survived tokenization"):
            ChatDataset(
                source=_examples(4),
                tokenizer=_FakeTokenizer(),
                roles=ROLES,
                field_map=FIELD_MAP,
                seq_len=2,  # too small for any example
            )

    def test_drops_examples_exceeding_seq_len(self):
        # Mix of short examples and one very long one.
        examples = _examples(4) + [
            {"question": " ".join("w" + str(i) for i in range(50)),
             "answer": " ".join("r" + str(i) for i in range(50))},
        ]
        ds = ChatDataset(
            source=examples,
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
        )
        # The long example should be dropped.
        assert len(ds) == 4


class TestChatStreamDatasetImports:
    """Smoke: ``ChatStreamDataset`` is importable from the new location.

    Full behavior is covered by projects/mamba_qat_slm/tests/test_chat_dataset.py
    against autofpv.data.chat, and will move here in Phase 4.
    """

    def test_can_construct(self):
        ds = ChatStreamDataset(
            source="dummy/dataset",
            tokenizer=_FakeTokenizer(),
            roles=ROLES,
            field_map=FIELD_MAP,
            seq_len=32,
        )
        ds.set_epoch(3)
        assert ds._epoch == 3
        assert ds._seq_len == 32
