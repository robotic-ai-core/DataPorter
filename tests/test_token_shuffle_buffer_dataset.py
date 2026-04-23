"""Tests for TokenShuffleBufferDataset."""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import DataLoader

from dataporter.token_shuffle_buffer import TokenShuffleBuffer
from dataporter.token_shuffle_buffer_dataset import TokenShuffleBufferDataset


def _fill_buffer(buf: TokenShuffleBuffer, n: int) -> None:
    for i in range(n):
        tokens = torch.tensor(
            [(i + j) % 100 for j in range(5 + i % 3)], dtype=torch.int32,
        )
        buf.put(i, tokens)


# ---------------------------------------------------------------------------
# __getitem__ shape + dtype contract
# ---------------------------------------------------------------------------


def test_padded_returns_fixed_shape():
    buf = TokenShuffleBuffer(capacity=8, seq_len=16, pad_token_id=0)
    _fill_buffer(buf, 8)
    ds = TokenShuffleBufferDataset(buf, epoch_length=20, padded=True)

    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["loss_mask"].shape == (16,)
    assert item["input_ids"].dtype == torch.int64
    assert item["loss_mask"].dtype == torch.uint8
    assert item["length"].dtype == torch.int32
    assert 0 < int(item["length"]) <= 16


def test_unpadded_returns_variable_length():
    # rotation_per_samples=None: this test samples 10x from 2 puts with
    # no producer pool — the default K=1 gate would block waiting for
    # write_head to advance.  Disable the gate explicitly for direct-
    # buffer unit tests.
    buf = TokenShuffleBuffer(
        capacity=4, seq_len=16, rotation_per_samples=None,
    )
    buf.put(0, torch.tensor([1, 2, 3], dtype=torch.int32))
    buf.put(1, torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32))

    ds = TokenShuffleBufferDataset(buf, epoch_length=10, padded=False)
    # Sampling is random but length should equal the sampled seq's true length.
    for _ in range(10):
        item = ds[0]
        assert item["input_ids"].shape[0] == int(item["length"])


def test_epoch_length_reported():
    buf = TokenShuffleBuffer(capacity=4, seq_len=16)
    _fill_buffer(buf, 4)
    ds = TokenShuffleBufferDataset(buf, epoch_length=123)
    assert len(ds) == 123


def test_empty_buffer_raises():
    buf = TokenShuffleBuffer(capacity=4, seq_len=16)
    ds = TokenShuffleBufferDataset(buf, epoch_length=5)
    with pytest.raises(IndexError, match="empty"):
        ds[0]


def test_invalid_epoch_length_rejected():
    buf = TokenShuffleBuffer(capacity=4, seq_len=16)
    with pytest.raises(ValueError, match="epoch_length"):
        TokenShuffleBufferDataset(buf, epoch_length=0)


# ---------------------------------------------------------------------------
# DataLoader integration — the actual thing we care about in production
# ---------------------------------------------------------------------------


def test_default_collate_stacks_padded_tensors():
    """Padded [seq_len] outputs let the default torch collate just stack
    — no custom collate_fn required.  Verify batch shapes."""
    buf = TokenShuffleBuffer(capacity=16, seq_len=8)
    _fill_buffer(buf, 16)
    ds = TokenShuffleBufferDataset(buf, epoch_length=32, padded=True)

    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        worker_init_fn=TokenShuffleBufferDataset.worker_init_fn,
    )
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 8)
    assert batch["loss_mask"].shape == (4, 8)
    assert batch["length"].shape == (4,)


def test_multi_worker_produces_diverse_samples():
    """Without worker_init_fn, forked workers share RNG state → identical
    samples.  With it, workers produce diverse keys.  This is exactly the
    bug that bit the LeRobot path (corrs) — lock it in here."""
    buf = TokenShuffleBuffer(capacity=32, seq_len=8)
    _fill_buffer(buf, 32)
    ds = TokenShuffleBufferDataset(buf, epoch_length=64, padded=True)

    loader = DataLoader(
        ds, batch_size=4, num_workers=2,
        worker_init_fn=TokenShuffleBufferDataset.worker_init_fn,
    )
    keys_seen = []
    for batch in loader:
        keys_seen.extend(batch["key"].tolist())
    # 32 buffer items + 64 samples — expect good diversity, not all identical.
    assert len(set(keys_seen)) > 4, (
        f"Only {len(set(keys_seen))} unique keys across 64 samples — "
        "workers likely share RNG state"
    )
