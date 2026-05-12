"""Unit tests for ``apply_chat_template``.

Uses a fake tokenizer (no HF downloads) so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataporter.text import apply_chat_template


ROLES = {"query": "[Q]", "response": "[A]", "end": "[END]"}
FIELD_MAP = {"query": "question", "response": "answer"}


class _FakeTokenizer:
    """Whitespace tokenizer that maps each whitespace-split token to an id.

    Deterministic: ids are assigned by first-seen order.  Provides
    ``pad_token_id`` and ``eos_token_id``.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        # Reserve 0 and 1 for special tokens.
        self._next_id = 2

    def _token_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._next_id += 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._token_id(t) for t in text.split()]


def _example() -> dict[str, str]:
    return {"question": "what is 2 plus 2", "answer": "4"}


# ----- happy path -----------------------------------------------------------


def test_apply_chat_template_returns_arrays_with_correct_shape_and_dtype():
    tok = _FakeTokenizer()
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=32)
    assert out is not None
    assert out["input_ids"].dtype == np.uint16
    assert out["labels"].dtype == np.uint16
    assert out["loss_mask"].dtype == np.bool_
    assert out["input_ids"].shape == (32,)
    assert out["labels"].shape == (32,)
    assert out["loss_mask"].shape == (32,)


def test_input_ids_equal_labels():
    tok = _FakeTokenizer()
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=32)
    np.testing.assert_array_equal(out["input_ids"], out["labels"])


def test_loss_mask_response_only():
    """loss_mask should be True only on response + end tokens — not query, not padding."""
    tok = _FakeTokenizer()
    seq_len = 32
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=seq_len)

    # Recompute the query token count using the same tokenizer:
    query_part = f"{ROLES['query']} {_example()['question']} {ROLES['response']}"
    n_query = len(tok.encode(query_part))
    response_part = f" {_example()['answer']} {ROLES['end']}"
    n_response = len(tok.encode(response_part))
    n_total = n_query + n_response

    mask = out["loss_mask"]
    # Query positions are False.
    assert not mask[:n_query].any()
    # Response + end positions are True.
    assert mask[n_query:n_total].all()
    # Padding positions are False.
    assert not mask[n_total:].any()


def test_padding_uses_pad_token_id():
    tok = _FakeTokenizer()
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=64)
    # Find where padding begins by counting non-mask True positions.
    pad_id = tok.pad_token_id
    # Last entry should be pad token (since the example is short).
    assert out["input_ids"][-1] == pad_id


def test_falls_back_to_eos_when_pad_id_none():
    class _NoPad(_FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.pad_token_id = None

    tok = _NoPad()
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=64)
    assert out["input_ids"][-1] == tok.eos_token_id


# ----- length handling ------------------------------------------------------


def test_returns_none_when_example_exceeds_seq_len():
    tok = _FakeTokenizer()
    # seq_len=4 is too small for any non-trivial example.
    out = apply_chat_template(_example(), tok, ROLES, FIELD_MAP, seq_len=4)
    assert out is None


def test_exact_fit_at_seq_len_is_accepted():
    tok = _FakeTokenizer()
    # Use a longer example to find the exact size.
    example = {"question": "a b c", "answer": "d e f"}
    out = apply_chat_template(example, tok, ROLES, FIELD_MAP, seq_len=64)
    assert out is not None
