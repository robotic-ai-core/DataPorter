"""Tests for the shared :class:`RotationGate` class and the two
pipelines' adoption of it — pipeline-agnostic parity coverage.

The gate is frame-count-driven: tracks actual frames written into
the buffer vs samples drawn out, blocking either side when one
races more than one buffer-worth of frames ahead of the other.  No
tuning parameter — unit-matched in/out accounting.
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

    def test_counters_start_zero(self):
        gate = RotationGate()
        assert gate.samples_consumed == 0
        assert gate.frames_produced == 0

    def test_record_put_increments_frames(self):
        gate = RotationGate()
        gate.record_put(100)
        gate.record_put(50)
        assert gate.frames_produced == 150
        assert gate.samples_consumed == 0

    def test_record_sample_increments_samples(self):
        gate = RotationGate()
        for _ in range(7):
            gate.record_sample()
        assert gate.samples_consumed == 7

    def test_reset_zeros_both_counters(self):
        gate = RotationGate()
        gate.record_put(100)
        for _ in range(5):
            gate.record_sample()
        gate.reset()
        assert gate.frames_produced == 0
        assert gate.samples_consumed == 0

    def test_disabled_gate_is_noop(self):
        """When ``enabled=False`` both waits are no-ops.  Increments
        still happen so downstream telemetry still works."""
        gate = RotationGate(enabled=False)
        # Consumer way ahead — would normally block, but gate is off.
        for _ in range(1000):
            gate.record_sample()
        t0 = time.monotonic()
        gate.wait_if_consumer_too_far_ahead(10, buffer_name="test")
        assert time.monotonic() - t0 < 0.05
        # Producer-side same.
        gate.record_put(100_000)
        assert not gate.producer_should_wait(10)

    def test_consumer_waits_blocks_until_slack_satisfied(self):
        """Consumer gate blocks while samples race ahead, returns
        when producer catches up."""
        import threading
        gate = RotationGate()
        # Put 100 frames, consume 200 → gap = 100.
        gate.record_put(100)
        for _ in range(200):
            gate.record_sample()
        # slack=50 → gap=100 > 50 → should block.
        # Pump producer in a thread.
        def pump():
            time.sleep(0.1)
            gate.record_put(100)   # gap: 100 - 200 = -100, clears threshold

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        t0 = time.monotonic()
        gate.wait_if_consumer_too_far_ahead(50, buffer_name="test")
        elapsed = time.monotonic() - t0
        t.join()
        assert 0.05 < elapsed < 1.0

    def test_consumer_timeout_raises_actionable(self):
        """Never-catching-up producer → 30s timeout (shrunk for
        test) → RuntimeError with decoder-bottleneck diagnosis."""
        original = _rg.SAMPLE_TIMEOUT_S
        _rg.SAMPLE_TIMEOUT_S = 0.3
        try:
            gate = RotationGate()
            # Consumer far ahead, producer never advances.
            for _ in range(500):
                gate.record_sample()
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="blocked"):
                gate.wait_if_consumer_too_far_ahead(
                    100, buffer_name="TestBuf",
                )
            elapsed = time.monotonic() - t0
            assert 0.2 < elapsed < 1.5
        finally:
            _rg.SAMPLE_TIMEOUT_S = original

    def test_producer_should_wait_when_frames_too_far_ahead(self):
        """Producer gate (non-blocking query): returns True when
        frames_produced - samples_consumed > slack."""
        gate = RotationGate()
        gate.record_put(300)
        # samples=0, frames=300 → 300 > slack=100 → wait.
        assert gate.producer_should_wait(100) is True
        # Drain consumer to under slack.
        for _ in range(250):
            gate.record_sample()
        # frames - samples = 300 - 250 = 50 < 100 → don't wait.
        assert gate.producer_should_wait(100) is False


# ---------------------------------------------------------------------------
# Video buffer: ShuffleBuffer uses gate with slack=capacity*max_frames
# ---------------------------------------------------------------------------


class TestShuffleBufferGate:

    def _make(self, capacity=4, max_frames=10, enabled=True):
        return ShuffleBuffer(
            capacity=capacity, max_frames=max_frames,
            channels=1, height=2, width=2, gate_enabled=enabled,
        )

    def test_put_records_actual_frame_count(self):
        buf = self._make()
        buf.put(0, torch.zeros((7, 1, 2, 2), dtype=torch.uint8))
        assert buf._gate.frames_produced == 7
        buf.put(1, torch.zeros((3, 1, 2, 2), dtype=torch.uint8))
        assert buf._gate.frames_produced == 10

    def test_put_clips_at_max_frames(self):
        """If a caller hands in more frames than ``max_frames``, the
        buffer clips before writing AND the gate only records the
        clipped count — flow balance stays accurate."""
        buf = self._make(capacity=2, max_frames=5)
        buf.put(0, torch.zeros((20, 1, 2, 2), dtype=torch.uint8))
        # Only 5 frames actually written to the slot.
        assert buf._gate.frames_produced == 5

    def test_sample_increments_samples(self):
        buf = self._make(enabled=False)   # disable gate for this unit
        buf.put(0, torch.zeros((10, 1, 2, 2), dtype=torch.uint8))
        rng = random.Random(0)
        for _ in range(5):
            buf.sample(rng)
        assert buf._gate.samples_consumed == 5

    def test_clear_resets_gate(self):
        buf = self._make(enabled=False)
        buf.put(0, torch.zeros((10, 1, 2, 2), dtype=torch.uint8))
        rng = random.Random(0)
        for _ in range(5):
            buf.sample(rng)
        buf.clear()
        assert buf._gate.frames_produced == 0
        assert buf._gate.samples_consumed == 0

    def test_frame_slack_is_capacity_times_max_frames(self):
        buf = self._make(capacity=7, max_frames=13)
        assert buf.frame_slack == 7 * 13


class TestShuffleBufferGateBlocking:
    """Consumer blocks if samples race ahead of frames_produced by
    more than ``frame_slack = capacity * max_frames``."""

    def test_consumer_blocks_when_far_ahead(self):
        original = _rg.SAMPLE_TIMEOUT_S
        _rg.SAMPLE_TIMEOUT_S = 0.4
        try:
            buf = ShuffleBuffer(
                capacity=2, max_frames=3, channels=1, height=2, width=2,
                gate_enabled=True,
            )
            # frame_slack = 2 * 3 = 6.
            # Put 5 frames, but max_frames=3 caps to 3 — frames_produced=3.
            buf.put(0, torch.zeros((3, 1, 2, 2), dtype=torch.uint8))
            rng = random.Random(0)
            # 3 frames produced; gate allows gap ≤ 6 → samples up to 9
            # pass the check (entry gap -3..5) and the 10th passes (gap=6).
            # The 11th's entry gap = 10 - 3 = 7 > 6 → blocks.
            for _ in range(10):
                buf.sample(rng)
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="blocked"):
                buf.sample(rng)
            elapsed = time.monotonic() - t0
            assert 0.2 < elapsed < 1.5
        finally:
            _rg.SAMPLE_TIMEOUT_S = original


# ---------------------------------------------------------------------------
# Text buffer: TokenShuffleBuffer uses gate with slack=capacity (1 unit/put)
# ---------------------------------------------------------------------------


class TestTokenShuffleBufferGate:

    def _make(self, capacity=4, seq_len=8, enabled=True):
        return TokenShuffleBuffer(
            capacity=capacity, seq_len=seq_len, pad_token_id=0,
            gate_enabled=enabled,
        )

    def test_put_records_one_unit_per_doc(self):
        buf = self._make()
        buf.put(0, torch.arange(4, dtype=torch.int32))
        assert buf._gate.frames_produced == 1
        buf.put(1, torch.arange(3, dtype=torch.int32))
        assert buf._gate.frames_produced == 2

    def test_sample_increments_samples(self):
        buf = self._make(enabled=False)
        buf.put(0, torch.arange(3, dtype=torch.int32))
        rng = random.Random(0)
        for _ in range(5):
            buf.sample(rng)
        assert buf._gate.samples_consumed == 5

    def test_frame_slack_is_capacity(self):
        buf = self._make(capacity=11)
        assert buf.frame_slack == 11

    def test_consumer_blocks_when_far_ahead(self):
        """Same gate semantic as video, just with per-doc unit sizing.
        capacity=2 → slack=2, so consumer can be up to 2 samples
        ahead of puts before blocking."""
        original = _rg.SAMPLE_TIMEOUT_S
        _rg.SAMPLE_TIMEOUT_S = 0.4
        try:
            buf = TokenShuffleBuffer(
                capacity=2, seq_len=4, pad_token_id=0,
                gate_enabled=True,
            )
            buf.put(0, torch.arange(2, dtype=torch.int32))
            rng = random.Random(0)
            # 1 put = 1 frame; slack=2; entry gaps -1, 0, 1, 2 pass.
            # 5th call entry gap = 3 > 2 → blocks.
            for _ in range(4):
                buf.sample(rng)
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="blocked"):
                buf.sample(rng)
            elapsed = time.monotonic() - t0
            assert 0.2 < elapsed < 1.5
        finally:
            _rg.SAMPLE_TIMEOUT_S = original


# ---------------------------------------------------------------------------
# Parity: video & text both respond identically to gate settings
# ---------------------------------------------------------------------------


class TestVideoTextParity:

    def test_disabled_gate_lets_many_samples_flow_through(self):
        """Both buffer types: with gate disabled, consumer can sample
        freely without a producer running."""
        video = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=2, width=2,
            gate_enabled=False,
        )
        text = TokenShuffleBuffer(
            capacity=2, seq_len=4, pad_token_id=0,
            gate_enabled=False,
        )
        video.put(0, torch.zeros((2, 1, 2, 2), dtype=torch.uint8))
        text.put(0, torch.arange(2, dtype=torch.int32))

        rng_v = random.Random(0)
        rng_t = random.Random(0)
        t0 = time.monotonic()
        for _ in range(200):
            video.sample(rng_v)
            text.sample(rng_t)
        assert time.monotonic() - t0 < 2.0

    def test_producer_should_wait_fires_after_one_buffer_overproduction(self):
        """Both buffers: producer_should_wait flips True when
        frames_produced exceeds samples_consumed by more than the
        buffer's own ``frame_slack``."""
        video = ShuffleBuffer(
            capacity=3, max_frames=4, channels=1, height=2, width=2,
            gate_enabled=True,
        )
        # Video slack = 3 * 4 = 12.  Put 15 frames → gate wants wait.
        video.put(0, torch.zeros((4, 1, 2, 2), dtype=torch.uint8))
        video.put(1, torch.zeros((4, 1, 2, 2), dtype=torch.uint8))
        video.put(2, torch.zeros((4, 1, 2, 2), dtype=torch.uint8))
        video.put(3, torch.zeros((4, 1, 2, 2), dtype=torch.uint8))
        # 4 puts × 4 frames = 16 frames.  slack=12.  producer should wait.
        assert video._gate.producer_should_wait(video.frame_slack) is True

        text = TokenShuffleBuffer(
            capacity=3, seq_len=4, pad_token_id=0, gate_enabled=True,
        )
        # Text slack = 3. Put 4 docs = 4 frames. producer should wait.
        for i in range(4):
            text.put(i, torch.arange(2, dtype=torch.int32))
        assert text._gate.producer_should_wait(text.frame_slack) is True
