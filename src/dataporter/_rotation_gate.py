"""Sample-gated rotation controller for shared-memory ring buffers.

Shared between :class:`ShuffleBuffer` (video) and
:class:`TokenShuffleBuffer` (text) via composition, so both pipelines
get identical rotation semantics from the same code path — no surface
area for the two to drift.

The gate couples producer put rate to consumer draw rate:

- **Producer side** (pool): when the buffer is full, wait for the
  consumer to draw ``K`` more samples before the next put.  Slow
  consumer → pool decodes at consumer rate, no wasted work.  Fast
  consumer → gate is always satisfied instantly; pool runs at native
  decode rate.
- **Consumer side** (``sample()``): when the consumer is more than
  ``capacity * K`` samples ahead of the pool's puts, ``sample()``
  blocks until the pool catches up.  Fast consumer + slow decode →
  training throughput drops to decode rate, making the bottleneck
  visible as low ``train/step_time`` (the UX signal to add decode
  workers).

Set ``rotation_per_samples=None`` to disable the gate entirely —
intended for direct-buffer tests without a producer pool.  A full
30s block raises ``RuntimeError`` so a dead pool fails loud instead
of hanging training forever.
"""

from __future__ import annotations

import multiprocessing as mp
import time


# Timeout (seconds) for the consumer-side gate.  Cross-module public
# so tests can patch it for fast failure-path verification.
SAMPLE_TIMEOUT_S: float = 30.0


class RotationGate:
    """Owns the shared counter and consumer-side gate logic.

    Args:
        rotation_per_samples: K — samples consumed per producer put at
            steady state.  ``None`` disables the gate.  Integer ≥ 1
            enables it.  0 or negative are rejected.

    State (process-shared):
        ``_samples_consumed``: ``mp.Value('q', 0)`` — cumulative
            sample count across all DataLoader workers.  Reads are
            atomic for int64 on x86 (no lock on the hot path);
            increment takes the lock.
    """

    __slots__ = ("rotation_k", "_samples_consumed")

    def __init__(self, rotation_per_samples: int | None):
        if rotation_per_samples is None:
            self.rotation_k: int | None = None
        else:
            k = int(rotation_per_samples)
            if k < 1:
                raise ValueError(
                    f"rotation_per_samples must be ≥ 1 or None, "
                    f"got {rotation_per_samples!r}"
                )
            self.rotation_k = k
        ctx = mp.get_context("spawn")
        self._samples_consumed = ctx.Value("q", 0)

    # ------------------------------------------------------------------
    # Hot-path accessors (called per sample()/put())
    # ------------------------------------------------------------------

    @property
    def samples_consumed(self) -> int:
        """Atomic read of the shared counter.  No lock — relies on
        int64 atomicity on x86."""
        return int(self._samples_consumed.value)

    def record_sample(self) -> None:
        """Atomic increment; call from ``buffer.sample()`` after the
        read completes so the producer gate only sees committed
        samples."""
        with self._samples_consumed.get_lock():
            self._samples_consumed.value += 1

    def reset(self) -> None:
        """Zero the counter.  Called from ``buffer.clear()``."""
        with self._samples_consumed.get_lock():
            self._samples_consumed.value = 0

    # ------------------------------------------------------------------
    # Consumer-side gate
    # ------------------------------------------------------------------

    def wait_if_consumer_too_far_ahead(
        self,
        write_head_getter,
        capacity: int,
        *,
        buffer_name: str = "buffer",
    ) -> None:
        """Block if the consumer has raced too far ahead of the pool's
        puts; return as soon as the invariant holds.

        Invariant: ``samples_consumed - K * write_head <= capacity * K``
        — i.e., consumer may be up to "one full buffer rotation"
        ahead.  Beyond that, spin-wait until ``write_head`` advances
        (a producer put committed).

        Args:
            write_head_getter: Callable returning the current
                ``write_head`` value (int).  Passed as a callable
                rather than a raw tensor so it's re-read on every
                iteration of the wait loop.
            capacity: Buffer capacity (max slots).
            buffer_name: Class name of the calling buffer — used for
                the timeout error message so stack traces are clear.

        Raises:
            RuntimeError: after :data:`SAMPLE_TIMEOUT_S` of continuous
                blocking.  Almost always means the producer pool is
                dead; ``RuntimeError`` with diagnostics beats a silent
                hang.
        """
        K = self.rotation_k
        if K is None:
            return
        target_gap = capacity * K
        samples = int(self._samples_consumed.value)
        puts = int(write_head_getter())
        if samples - K * puts <= target_gap:
            return

        wait_start = time.monotonic()
        while True:
            samples = int(self._samples_consumed.value)
            puts = int(write_head_getter())
            if samples - K * puts <= target_gap:
                return
            if time.monotonic() - wait_start > SAMPLE_TIMEOUT_S:
                raise RuntimeError(
                    f"{buffer_name}.sample: blocked "
                    f">{SAMPLE_TIMEOUT_S:.0f}s waiting for the producer "
                    f"pool to catch up (samples_consumed={samples}, "
                    f"write_head={puts}, rotation_per_samples={K}, "
                    f"capacity={capacity}).  Pool may be dead "
                    f"(check logs for decode/tokenize errors) OR the "
                    f"decoder is far slower than the sample rate — "
                    f"increase worker count or reduce batch_size."
                )
            time.sleep(0.001)
