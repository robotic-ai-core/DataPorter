"""Tests for ShuffleBuffer's sample-gated rotation.

Covers the new two-way sample gate:

- ``ShuffleBuffer.sample()`` increments ``_samples_consumed``; a
  ``None`` ``rotation_per_samples`` disables the gate (test escape
  hatch); an int ``K`` applies the gate.
- Producer-side gate (inside ``_run_spawn_pool``): when the buffer
  is full, pool waits for the consumer to advance K samples before
  the next put.  Results in "exactly K samples per put" at
  steady state under slow consumers.
- Consumer-side gate (inside ``ShuffleBuffer.sample``): when the
  consumer is more than ``capacity * K`` samples ahead of the
  pool's puts, sample() blocks the consumer until the pool catches
  up.  Makes a decode bottleneck visible as low training throughput
  rather than silent buffer staleness.
- Timeout path: a persistent block > 30s surfaces a RuntimeError
  with actionable diagnostics (pool may be dead).
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest
import torch

from dataporter.shuffle_buffer import ShuffleBuffer, _SAMPLE_TIMEOUT_S


# ---------------------------------------------------------------------------
# Counter increment semantics
# ---------------------------------------------------------------------------


class TestSamplesConsumedCounter:

    def test_counter_starts_at_zero(self):
        buf = ShuffleBuffer(
            capacity=4, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        assert int(buf._samples_consumed.value) == 0

    def test_sample_increments_counter(self):
        buf = ShuffleBuffer(
            capacity=4, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        # Fill a slot so sample() can route.
        buf.put(0, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))
        import random as _random
        rng = _random.Random(0)
        for i in range(1, 11):
            buf.sample(rng)
            assert int(buf._samples_consumed.value) == i

    def test_clear_resets_counter(self):
        buf = ShuffleBuffer(
            capacity=4, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        buf.put(0, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))
        import random as _random
        rng = _random.Random(0)
        for _ in range(5):
            buf.sample(rng)
        assert int(buf._samples_consumed.value) == 5
        buf.clear()
        assert int(buf._samples_consumed.value) == 0


# ---------------------------------------------------------------------------
# Gate disabled (rotation_per_samples=None)
# ---------------------------------------------------------------------------


class TestGateDisabled:

    def test_many_samples_without_pool_does_not_block(self):
        """With ``rotation_per_samples=None`` (default), direct-buffer
        tests must be able to sample freely without a pool advancing
        ``write_head``.  This is the escape hatch that keeps legacy
        buffer-only tests working.
        """
        buf = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        buf.put(0, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))
        import random as _random
        rng = _random.Random(0)
        # 200 samples against capacity=2 and only 1 put — with a gate
        # this would block.  With None, it must go through fast.
        t0 = time.monotonic()
        for _ in range(200):
            buf.sample(rng)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, (
            f"sample() with gate disabled took {elapsed:.2f}s for 200 "
            f"calls — the gate is not being bypassed"
        )


# ---------------------------------------------------------------------------
# Gate enabled (rotation_per_samples=K) — consumer blocks when ahead
# ---------------------------------------------------------------------------


class TestConsumerGate:

    def test_consumer_blocks_when_too_far_ahead(self):
        """With K=1 and capacity=2, the consumer may run up to
        ``capacity * K = 2`` samples ahead of the pool's puts (gate:
        samples - K*write_head <= capacity*K).  With one put,
        samples 1..4 pass (gaps 0..2); the 5th call finds gap=3 > 2
        and blocks until ``write_head`` advances.

        We shrink the timeout so the test runs in ~1s.
        """
        import dataporter.shuffle_buffer as sb
        original = sb._SAMPLE_TIMEOUT_S
        sb._SAMPLE_TIMEOUT_S = 0.8
        try:
            buf = ShuffleBuffer(
                capacity=2, max_frames=2, channels=1, height=4, width=4,
                rotation_per_samples=1,
            )
            # Put once — write_head = 1.
            buf.put(0, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))
            import random as _random
            rng = _random.Random(0)
            # First 4 samples OK: gap on entry is 0, 1, 2, 2
            # (atomic order).  Actually: samples starts 0, after
            # N calls samples==N.  Entry gap = samples - K*puts:
            #  call 1 entry: 0-1 = -1
            #  call 2 entry: 1-1 = 0
            #  call 3 entry: 2-1 = 1
            #  call 4 entry: 3-1 = 2  (still <= 2)
            #  call 5 entry: 4-1 = 3  (> 2 → block)
            for _ in range(4):
                buf.sample(rng)
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="blocked.*waiting for"):
                buf.sample(rng)
            elapsed = time.monotonic() - t0
            assert 0.5 < elapsed < 3.0, (
                f"timeout should fire around {sb._SAMPLE_TIMEOUT_S}s, "
                f"actual {elapsed:.2f}s"
            )
        finally:
            sb._SAMPLE_TIMEOUT_S = original

    def test_consumer_unblocks_when_pool_catches_up(self):
        """If ``write_head`` advances while the consumer is blocked,
        sample() succeeds without timing out.  Simulated by pumping
        ``write_head`` from a background thread.
        """
        import threading

        buf = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=1,
        )
        buf.put(0, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))
        import random as _random
        rng = _random.Random(0)
        # Drain to the gate threshold (4 calls — see comment in the
        # _blocks test above for the exact arithmetic).
        for _ in range(4):
            buf.sample(rng)

        # Background: after 200ms, simulate a put so write_head
        # advances and the gate opens.
        def pump():
            time.sleep(0.2)
            buf.put(1, torch.zeros((2, 1, 4, 4), dtype=torch.uint8))

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        t0 = time.monotonic()
        key, _ = buf.sample(rng)   # would have blocked ~200ms
        elapsed = time.monotonic() - t0
        t.join()
        assert 0.1 < elapsed < 2.0, (
            f"sample() should block ~200ms then unblock; took {elapsed:.2f}s"
        )


# ---------------------------------------------------------------------------
# Producer-side gate via a real pool
# ---------------------------------------------------------------------------


class TestProducerGateSteadyState:

    def test_slow_consumer_gates_pool_to_k_per_sample(self):
        """Under K=1 with a slow consumer, the pool should put
        exactly N times after N samples (within a small tolerance
        for the in-flight tasks and startup).
        """
        pytest.importorskip("imageio")
        pytest.importorskip("lerobot")

        from dataporter import LeRobotShardSource, ResizeFrames
        from dataporter.producer_pool import ProducerConfig, ProducerPool
        from test_shard_source_pool_e2e import _make_dataset

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            _make_dataset(root, ready_eps=list(range(8)), total_episodes=8)
            src = LeRobotShardSource(root)

            buf = ShuffleBuffer(
                capacity=3, max_frames=32, channels=3, height=32, width=32,
                rotation_per_samples=1,
            )
            cfg = ProducerConfig.from_source(
                source={"repo_id": "synth", "weight": 1.0},
                shard_source=src,
                iteration_episodes=list(range(8)),
                producer_transform=ResizeFrames(32, 32),
            )
            pool = ProducerPool(
                buf, configs=[cfg], total_workers=2, warmup_target=3,
            )
            pool.start()
            try:
                pool.wait_for_warmup(timeout=30.0)
                # After fill, the pool should NOT put until the
                # consumer samples.
                head_before = int(buf._write_head)
                time.sleep(1.0)
                head_idle = int(buf._write_head)
                # With K=1 and zero consumer samples, the gate should
                # hold.  Allow at most a couple in-flight finishes.
                assert head_idle - head_before <= 4, (
                    f"pool put {head_idle - head_before} items with no "
                    f"consumer samples — gate is not holding"
                )

                # Now draw 20 samples from the buffer.
                import random as _random
                rng = _random.Random(0)
                for _ in range(20):
                    buf.sample(rng)
                # Give the pool a moment to process the gate release.
                time.sleep(2.0)
                head_after = int(buf._write_head)
                # With K=1, 20 samples should unlock ~20 puts
                # (plus or minus in-flight).
                delta = head_after - head_idle
                assert 10 <= delta <= 40, (
                    f"expected ~20 puts after 20 samples with K=1; "
                    f"got {delta}"
                )
            finally:
                pool.stop()


# ---------------------------------------------------------------------------
# K validation
# ---------------------------------------------------------------------------


class TestKValidation:

    def test_k_zero_rejected(self):
        with pytest.raises(ValueError, match="must be ≥ 1 or None"):
            ShuffleBuffer(
                capacity=2, max_frames=2, channels=1, height=4, width=4,
                rotation_per_samples=0,
            )

    def test_k_negative_rejected(self):
        with pytest.raises(ValueError, match="must be ≥ 1 or None"):
            ShuffleBuffer(
                capacity=2, max_frames=2, channels=1, height=4, width=4,
                rotation_per_samples=-1,
            )

    def test_k_none_accepted(self):
        buf = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        assert buf._rotation_k is None

    def test_k_positive_int_accepted(self):
        buf = ShuffleBuffer(
            capacity=2, max_frames=2, channels=1, height=4, width=4,
            rotation_per_samples=5,
        )
        assert buf._rotation_k == 5
