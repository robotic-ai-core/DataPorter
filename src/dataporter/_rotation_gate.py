"""Frame-count flow-balance gate for shared-memory ring buffers.

Shared between :class:`ShuffleBuffer` (video) and
:class:`TokenShuffleBuffer` (text) via composition.  Same semantic
for both pipelines: the gate tracks **actual content in vs out** of
the buffer — no rotation ratio to tune, no variance-driven
mis-fires.

Mental model
============

- ``put(tokens)`` / ``put(frames)`` reports how many "sample units"
  were written (``n_frames`` for video episodes, 1 for one-doc-per-
  put text).
- ``sample()`` is counted as consuming one sample unit (the downstream
  consumer extracts one frame / one doc from what ``sample()``
  returns — matches the real usage pattern).
- Gate tracks two counters: ``frames_produced`` and
  ``samples_consumed``, both in the same unit.

Symmetric throttles
-------------------

Let ``gap = samples_consumed - frames_produced``.  At steady state
gap ≈ 0.  Either side going out of sync is caught:

- **Consumer racing ahead** (``gap > slack``): consumer blocks in
  ``sample()`` until producer catches up.  Surfaces decode
  bottlenecks as a deterministic-timed ``RuntimeError`` (30s) if
  production never comes.

- **Producer racing ahead** (``-gap > slack``): producer blocks in
  the pool loop until consumer drains.  Prevents runaway CPU when
  training is paused (validation, checkpointing) or simply slow.

Both use the same slack — ``capacity * max_frames`` = the buffer's
physical frame-capacity.  Symmetric semantics means
``frames_produced`` and ``samples_consumed`` stay within ±one buffer-
worth of each other; buffer contents rotate at consumer rate.

No K parameter
--------------

The pre-frame-count design used an integer K = "samples per put at
steady state."  That was (a) easy to misframe (video's per-put
frame count is ~100, text's is 1 — same K didn't fit both) and (b)
fragile under variable episode lengths.  Frame-count tracks real
production per put, so variance is absorbed by construction.

Disabling
---------

Direct-buffer tests that call ``sample()`` many times without a
producer pool would block forever on the consumer-side gate.  Pass
``enabled=False`` at :class:`RotationGate` construction to opt out.
Tests and bench scripts use this; production callers leave it
enabled.
"""

from __future__ import annotations

import multiprocessing as mp
import time


# Timeout (seconds) for the consumer-side gate.  Module-level so
# tests can patch for fast failure-path verification.
SAMPLE_TIMEOUT_S: float = 30.0


class RotationGate:
    """Owns shared flow-balance counters + consumer/producer gates.

    Args:
        enabled: When ``False``, all waits are no-ops and increments
            still happen.  Intended for direct-buffer tests / bench
            scripts that drive the buffer without a pool.

    State (process-shared):
        ``_samples_consumed``: cumulative ``sample()`` calls across
            all DataLoader workers.
        ``_frames_produced``: cumulative frame count across all
            ``put()`` calls (each put reports its ``n_frames``).
    """

    __slots__ = ("enabled", "_samples_consumed", "_frames_produced")

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        ctx = mp.get_context("spawn")
        self._samples_consumed = ctx.Value("q", 0)
        self._frames_produced = ctx.Value("q", 0)

    # ------------------------------------------------------------------
    # Hot-path accessors
    # ------------------------------------------------------------------

    @property
    def samples_consumed(self) -> int:
        """Atomic read; no lock (int64 atomic on x86)."""
        return int(self._samples_consumed.value)

    @property
    def frames_produced(self) -> int:
        return int(self._frames_produced.value)

    def record_put(self, n_frames: int) -> None:
        """Atomic increment by ``n_frames`` — called by ``buffer.put``
        after the write.  Report the real frame count the put wrote
        into the buffer (not the buffer's ``max_frames`` cap)."""
        with self._frames_produced.get_lock():
            self._frames_produced.value += int(n_frames)

    def record_sample(self) -> None:
        """Atomic increment by 1 — called by ``buffer.sample()``
        after reading.  Each sample is modeled as one unit of
        content consumed."""
        with self._samples_consumed.get_lock():
            self._samples_consumed.value += 1

    def reset(self) -> None:
        """Zero both counters.  Called from ``buffer.clear()``."""
        with self._samples_consumed.get_lock():
            self._samples_consumed.value = 0
        with self._frames_produced.get_lock():
            self._frames_produced.value = 0

    # ------------------------------------------------------------------
    # Consumer-side gate — blocks ``sample()`` caller
    # ------------------------------------------------------------------

    def wait_if_consumer_too_far_ahead(
        self,
        frame_slack: int,
        *,
        buffer_name: str = "buffer",
    ) -> None:
        """Block if the consumer has drawn ``frame_slack`` more
        samples than the producer has produced.

        Invariant: ``samples_consumed - frames_produced <= frame_slack``.

        Typical ``frame_slack = capacity * max_frames`` — one full
        buffer-worth of over-draw allowed before throttling.

        Raises ``RuntimeError`` after :data:`SAMPLE_TIMEOUT_S` of
        continuous blocking so a dead pool fails loud.
        """
        if not self.enabled:
            return
        samples = int(self._samples_consumed.value)
        frames = int(self._frames_produced.value)
        if samples - frames <= frame_slack:
            return

        wait_start = time.monotonic()
        while True:
            samples = int(self._samples_consumed.value)
            frames = int(self._frames_produced.value)
            if samples - frames <= frame_slack:
                return
            if time.monotonic() - wait_start > SAMPLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{buffer_name}.sample: blocked "
                    f">{SAMPLE_TIMEOUT_S:.0f}s — consumer is "
                    f"{samples - frames} frames ahead of production "
                    f"(samples_consumed={samples}, "
                    f"frames_produced={frames}, slack={frame_slack}). "
                    f"Pool may be dead (check logs for decode errors) "
                    f"OR decoder is much slower than sample rate — "
                    f"increase worker count or reduce batch_size."
                )
            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Producer-side gate — non-blocking query; caller drives the loop
    # ------------------------------------------------------------------

    def producer_should_wait(self, frame_slack: int) -> bool:
        """Return True if producer has put ``frame_slack`` more frames
        than the consumer has drawn.

        Intended as a fast check inside the pool's hot loop: the pool
        decides whether to ``await asyncio.sleep`` (spawn pool) or
        ``time.sleep`` (thread pool) based on this.  Producer-side
        waits are NOT bounded with a timeout — a slow consumer is
        normal (training pause, validation, etc.), not a fatal
        condition.
        """
        if not self.enabled:
            return False
        samples = int(self._samples_consumed.value)
        frames = int(self._frames_produced.value)
        return frames - samples > frame_slack
