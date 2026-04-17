"""Tests for TokenShuffleBuffer — put/sample, eviction, variable length,
vocab-size guard, and cross-process spawn survival.
"""

from __future__ import annotations

import multiprocessing as mp
import random

import pytest
import torch

from dataporter.token_shuffle_buffer import TokenShuffleBuffer


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_put_then_sample_roundtrips_tokens_and_mask():
    buf = TokenShuffleBuffer(capacity=4, seq_len=8, pad_token_id=0)
    tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.uint8)
    buf.put(42, tokens, mask)

    key, got_tokens, got_mask, length = buf.sample(random.Random(0))
    assert key == 42
    assert length == 5
    assert torch.equal(got_tokens, tokens)
    assert torch.equal(got_mask, mask)


def test_sample_padded_returns_fixed_shape():
    buf = TokenShuffleBuffer(capacity=4, seq_len=8, pad_token_id=99)
    buf.put(1, torch.tensor([1, 2, 3], dtype=torch.int32))
    key, tokens, mask, length = buf.sample_padded(random.Random(0))
    assert tokens.shape == (8,)
    assert mask.shape == (8,)
    assert length == 3
    assert tokens[:3].tolist() == [1, 2, 3]
    assert tokens[3:].tolist() == [99, 99, 99, 99, 99]
    assert mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]


def test_default_loss_mask_is_all_ones():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4)
    buf.put(0, torch.tensor([7, 8, 9], dtype=torch.int32))  # no mask
    _, _, mask, length = buf.sample(random.Random(0))
    assert length == 3
    assert mask.tolist() == [1, 1, 1]


def test_longer_than_seq_len_truncates():
    buf = TokenShuffleBuffer(capacity=2, seq_len=3)
    buf.put(0, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32))
    _, tokens, _, length = buf.sample(random.Random(0))
    assert length == 3
    assert tokens.tolist() == [1, 2, 3]


def test_empty_buffer_raises_on_sample():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4)
    with pytest.raises(IndexError, match="empty"):
        buf.sample(random.Random(0))


# ---------------------------------------------------------------------------
# Ring-buffer eviction
# ---------------------------------------------------------------------------


def test_eviction_reports_displaced_key():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4)
    t = torch.tensor([1, 2], dtype=torch.int32)
    assert buf.put(10, t) is None
    assert buf.put(20, t) is None
    assert buf.put(30, t) == 10   # slot 0 overwritten
    assert buf.put(40, t) == 20


def test_eviction_zeroes_stale_mask_region():
    """After a long sequence, a shorter one in the same slot must not
    leak the older loss_mask past its true length."""
    buf = TokenShuffleBuffer(capacity=1, seq_len=8)
    buf.put(0, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32))  # len 6
    buf.put(1, torch.tensor([9, 9], dtype=torch.int32))              # len 2
    _, _, mask, length = buf.sample_padded(random.Random(0))
    assert length == 2
    assert mask.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Vocab-size guard (tokenizer misconfiguration)
# ---------------------------------------------------------------------------


def test_vocab_size_guard_rejects_out_of_range_tokens():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4, vocab_size=100)
    with pytest.raises(ValueError, match="vocab"):
        buf.put(0, torch.tensor([50, 99, 100], dtype=torch.int32))


def test_vocab_size_guard_accepts_in_range_tokens():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4, vocab_size=100)
    buf.put(0, torch.tensor([0, 50, 99], dtype=torch.int32))
    _, tokens, _, _ = buf.sample(random.Random(0))
    assert tokens.tolist() == [0, 50, 99]


# ---------------------------------------------------------------------------
# Mask length validation
# ---------------------------------------------------------------------------


def test_loss_mask_length_mismatch_raises():
    buf = TokenShuffleBuffer(capacity=2, seq_len=4)
    with pytest.raises(ValueError, match="loss_mask length"):
        buf.put(
            0,
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.uint8),
        )


# ---------------------------------------------------------------------------
# Spawn-context survival (the failure mode ShuffleBuffer has tests for)
# ---------------------------------------------------------------------------


def _child_puts(buffer: TokenShuffleBuffer, key: int, values: list[int]) -> None:
    buffer.put(key, torch.tensor(values, dtype=torch.int32))


def _child_checks_shared(buffer: TokenShuffleBuffer, result: dict) -> None:
    result["tokens_is_shared"] = buffer._tokens.is_shared()
    result["mask_is_shared"] = buffer._loss_mask.is_shared()
    result["count_is_shared"] = buffer._count.is_shared()


class TestSpawnSurvival:
    def test_child_write_visible_to_parent(self):
        buf = TokenShuffleBuffer(capacity=4, seq_len=8)
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_child_puts, args=(buf, 77, [1, 2, 3, 4]))
        p.start()
        p.join(timeout=15)
        assert p.exitcode == 0, f"child exited with {p.exitcode}"

        assert len(buf) == 1
        key, tokens, _, length = buf.sample(random.Random(0))
        assert key == 77
        assert tokens.tolist() == [1, 2, 3, 4]
        assert length == 4

    def test_shared_memory_flags_in_child(self):
        buf = TokenShuffleBuffer(capacity=2, seq_len=4)
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result = manager.dict()
        p = ctx.Process(target=_child_checks_shared, args=(buf, result))
        p.start()
        p.join(timeout=15)
        assert p.exitcode == 0

        assert result["tokens_is_shared"], (
            "tokens.is_shared() False in child — share_memory_() didn't "
            "survive spawn pickle"
        )
        assert result["mask_is_shared"]
        assert result["count_is_shared"]


# ---------------------------------------------------------------------------
# /dev/shm pre-flight
# ---------------------------------------------------------------------------


def test_fails_fast_on_insufficient_shm():
    from collections import namedtuple
    from unittest.mock import patch

    DiskUsage = namedtuple("usage", ["total", "used", "free"])
    fake = DiskUsage(total=1_000_000, used=0, free=1_000_000)
    with patch("shutil.disk_usage", return_value=fake):
        with pytest.raises(RuntimeError, match="shared memory"):
            TokenShuffleBuffer(capacity=10_000, seq_len=2048)
