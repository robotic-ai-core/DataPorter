"""Tests for the shared :class:`RotationGate` class and the text
pipeline's adoption of it — pipeline-agnostic parity coverage.

The gate is shared between ``ShuffleBuffer`` (video) and
``TokenShuffleBuffer`` (text) via composition.  These tests run
directly against the gate class AND against both buffer types to
confirm identical semantics — bugs can't hide behind pipeline-specific
adapters.
"""

from __future__ import annotations

import random
import time

import pytest
import torch

from dataporter import _rotation_gate as _rg
from dataporter._rotation_gate import RotationGate
from dataporter.shuffle_buffer import ShuffleBuffer
from dataporter.token_shuffle_buffer import TokenShuffleBuffer


# ---------------------------------------------------------------------------
# Gate-class unit tests (pipeline-agnostic)
# ---------------------------------------------------------------------------


class TestRotationGateUnit:

    def test_none_k_disables_gate(self):
        gate = RotationGate(rotation_per_samples=None)
        # wait should return immediately even when write_head is zero
        # and samples_consumed would otherwise far exceed the gap.
        for _ in range(100):
            gate.record_sample()
        # fake getter returning 0 — gate is disabled so no block.
        t0 = time.monotonic()
        gate.wait_if_consumer_too_far_ahead(
            write_head_getter=lambda: 0, capacity=4,
        )
        assert time.monotonic() - t0 < 0.1

    def test_counter_starts_zero(self):
        gate = RotationGate(rotation_per_samples=1)
        assert gate.samples_consumed == 0

    def test_record_sample_increments(self):
        gate = RotationGate(rotation_per_samples=1)
        for i in range(1, 11):
            gate.record_sample()
            assert gate.samples_consumed == i

    def test_reset_zeros_counter(self):
        gate = RotationGate(rotation_per_samples=1)
        for _ in range(5):
            gate.record_sample()
        gate.reset()
        assert gate.samples_consumed == 0

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="≥ 1 or None"):
            RotationGate(rotation_per_samples=0)
        with pytest.raises(ValueError, match="≥ 1 or None"):
            RotationGate(rotation_per_samples=-3)

    def test_wait_blocks_when_consumer_ahead(self):
        """Gate semantics: if ``samples - K*write_head > capacity*K``
        the wait blocks until the getter advances write_head.
        Tested directly on the gate with a mutable write_head holder.
        """
        original = _rg.SAMPLE_TIMEOUT_S
        _rg.SAMPLE_TIMEOUT_S = 0.5
        try:
            gate = RotationGate(rotation_per_samples=1)
            head_box = [0]

            # With K=1, capacity=2 → allowed gap is 2.
            # After 3 recorded samples and head=0, gap=3 > 2 → blocks.
            for _ in range(3):
                gate.record_sample()
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="blocked.*waiting for"):
                gate.wait_if_consumer_too_far_ahead(
                    write_head_getter=lambda: head_box[0],
                    capacity=2,
                    buffer_name="TestBuffer",
                )
            elapsed = time.monotonic() - t0
            assert 0.3 < elapsed < 2.0
        finally:
            _rg.SAMPLE_TIMEOUT_S = original


# ---------------------------------------------------------------------------
# Parity: video buffer and text buffer expose the SAME gate semantic
# ---------------------------------------------------------------------------


class TestVideoTextParity:

    def _sample_signature(self, buf, rng, n: int) -> list[int]:
        """Return the counter value after each of ``n`` samples.  Used
        to compare behavior across buffer types — both must increment
        identically because they share the same gate."""
        sig: list[int] = []
        for _ in range(n):
            if isinstance(buf, ShuffleBuffer):
                buf.sample(rng)
            else:
                buf.sample(rng)   # TokenShuffleBuffer.sample
            sig.append(int(buf._samples_consumed.value))
        return sig

    def test_both_buffers_increment_counter_on_sample(self):
        video = ShuffleBuffer(
            capacity=4, max_frames=4, channels=1, height=2, width=2,
            rotation_per_samples=None,
        )
        text = TokenShuffleBuffer(
            capacity=4, seq_len=8, pad_token_id=0, vocab_size=100,
            rotation_per_samples=None,
        )
        video.put(0, torch.zeros((4, 1, 2, 2), dtype=torch.uint8))
        text.put(0, torch.arange(4, dtype=torch.int32))
        rng_v = random.Random(0)
        rng_t = random.Random(0)

        sig_v = self._sample_signature(video, rng_v, 5)
        sig_t = self._sample_signature(text, rng_t, 5)

        assert sig_v == sig_t == [1, 2, 3, 4, 5]

    def test_both_buffers_gate_on_same_invariant(self):
        """Both buffers should block at the same consumer-sample
        count given identical K and capacity.  Verifies no
        buffer-specific drift in gate semantics."""
        original = _rg.SAMPLE_TIMEOUT_S
        _rg.SAMPLE_TIMEOUT_S = 0.5
        try:
            # capacity=2, K=1 → consumer may race 2 samples ahead.
            # 5th sample blocks.
            video = ShuffleBuffer(
                capacity=2, max_frames=2, channels=1, height=2, width=2,
                rotation_per_samples=1,
            )
            text = TokenShuffleBuffer(
                capacity=2, seq_len=4, pad_token_id=0, vocab_size=100,
                rotation_per_samples=1,
            )
            video.put(0, torch.zeros((2, 1, 2, 2), dtype=torch.uint8))
            text.put(0, torch.arange(2, dtype=torch.int32))

            for buf, label in [(video, "video"), (text, "text")]:
                rng = random.Random(0)
                # First 4 samples OK.
                for _ in range(4):
                    buf.sample(rng)
                # 5th blocks & times out.
                t0 = time.monotonic()
                with pytest.raises(RuntimeError, match="blocked.*waiting for"):
                    buf.sample(rng)
                elapsed = time.monotonic() - t0
                assert 0.3 < elapsed < 2.0, (
                    f"{label} buffer gate timing wrong: {elapsed:.2f}s"
                )
        finally:
            _rg.SAMPLE_TIMEOUT_S = original

    def test_both_buffers_none_k_disables_gate(self):
        """With K=None, both buffers allow unbounded sample draws
        even with write_head stuck at 1 — confirms the None escape
        hatch is pipeline-agnostic."""
        video = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=2, width=2,
            rotation_per_samples=None,
        )
        text = TokenShuffleBuffer(
            capacity=2, seq_len=4, pad_token_id=0, vocab_size=100,
            rotation_per_samples=None,
        )
        video.put(0, torch.zeros((2, 1, 2, 2), dtype=torch.uint8))
        text.put(0, torch.arange(2, dtype=torch.int32))
        rng_v = random.Random(0)
        rng_t = random.Random(0)
        # 100 samples >> capacity × K = 2 for each buffer — would
        # block under K=1 but passes with K=None.
        t0 = time.monotonic()
        for _ in range(100):
            video.sample(rng_v)
            text.sample(rng_t)
        assert time.monotonic() - t0 < 2.0
