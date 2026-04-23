"""Adversarial e2e tests: drive a real spawned TextProducerPool with
poisoned configs, kernel failures, and lifecycle edge cases and verify
each surfaces cleanly instead of hanging, silently corrupting, or
taking the whole producer down.

Mirrors ``test_producer_pool_adversarial.py`` for the video path.  The
bar is: "fails fast with a named invariant" OR "tolerates and recovers"
— never "silently produces wrong data" and never "child hangs until
timeout."
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataporter.text_producer_pool import (
    TextProducerConfig, TextProducerPool,
)
from dataporter.token_shuffle_buffer import TokenShuffleBuffer


# ---------------------------------------------------------------------------
# Picklable kernels (top-level so spawn can pickle them)
# ---------------------------------------------------------------------------


class ToyTokenize:
    """Deterministic toy tokenizer: ord(c) % vocab per character."""
    def __init__(self, seq_len: int = 16, vocab_size: int = 256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __call__(self, text: str):
        ids = [ord(c) % self.vocab_size for c in text[: self.seq_len]]
        if not ids:
            return None
        tokens = torch.tensor(ids, dtype=torch.int32)
        mask = torch.ones_like(tokens, dtype=torch.uint8)
        return tokens, mask


class SometimesRaises:
    """Raises on every Nth call — simulates rare row-level tokenizer bugs."""
    def __init__(self, raise_every: int, seq_len: int = 16):
        self.raise_every = raise_every
        self.seq_len = seq_len
        self._counter = 0

    def __call__(self, text: str):
        self._counter += 1
        if self._counter % self.raise_every == 0:
            raise ValueError(f"toy row error at call {self._counter}")
        ids = [ord(c) % 256 for c in text[: self.seq_len]]
        if not ids:
            return None
        tokens = torch.tensor(ids, dtype=torch.int32)
        mask = torch.ones_like(tokens, dtype=torch.uint8)
        return tokens, mask


class AlwaysRaises:
    """Raises on every call past the smoke-test — systemic failure."""
    def __init__(self):
        self._smoke_test_done = False

    def __call__(self, text: str):
        if not self._smoke_test_done:
            self._smoke_test_done = True
            return (
                torch.tensor([1, 2, 3], dtype=torch.int32),
                torch.tensor([1, 1, 1], dtype=torch.uint8),
            )
        raise RuntimeError("kernel is broken")


class SlowTokenize:
    """Sleeps per call — simulates slow tokenizer for back-pressure tests."""
    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s

    def __call__(self, text: str):
        time.sleep(self.delay_s)
        ids = [ord(c) % 256 for c in text[:16]]
        if not ids:
            return None
        return (
            torch.tensor(ids, dtype=torch.int32),
            torch.tensor([1] * len(ids), dtype=torch.uint8),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_shard(path: Path, texts: list[str]) -> None:
    pq.write_table(pa.table({"text": texts}), path)


def _seed_shards(
    shard_dir: Path, n_shards: int, rows_per_shard: int,
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        texts = [
            f"doc {s}-{r} some plausible text content here"
            for r in range(rows_per_shard)
        ]
        _write_shard(shard_dir / f"shard_{s:06d}.parquet", texts)


def _make_pool(shard_dir, kernel, capacity=16, warmup=4, threads=1,
               rescan_s=0.5) -> tuple[TokenShuffleBuffer, TextProducerPool]:
    buffer = TokenShuffleBuffer(capacity=capacity, seq_len=16, vocab_size=256)
    config = TextProducerConfig(
        source_name="adversarial",
        shard_dir=str(shard_dir),
        tokenize_fn=kernel,
        thread_workers=threads,
        shard_rescan_s=rescan_s,
    )
    pool = TextProducerPool(buffer, config, warmup_target=warmup)
    return buffer, pool


# ===========================================================================
# Group 1: mid-run failures (the 12-hour-run killers)
# ===========================================================================


class TestMidRunFailures:
    """Failure modes that would silently degrade a 12h unattended run."""

    def test_row_level_kernel_errors_do_not_kill_producer(self, tmp_path):
        """A kernel that raises on ~every 3rd row must keep the producer
        alive — pipeline should just skip the bad rows.  Current code
        uses executor.submit + per-row try/except to enable this;
        executor.map would propagate the first exception."""
        _seed_shards(tmp_path, n_shards=2, rows_per_shard=50)
        buffer, pool = _make_pool(
            tmp_path, SometimesRaises(raise_every=3), capacity=16, warmup=4,
        )
        pool.start()
        try:
            # Warmup must succeed despite ~33% row failure rate.
            pool.wait_for_warmup(timeout=30.0)
            assert len(buffer) >= 4
            # Producer still alive after warmup — row errors were tolerated.
            assert pool.is_alive
        finally:
            pool.stop()

    def test_systemic_kernel_failure_surfaces_eventually(self, tmp_path):
        """When the kernel raises on every row (past the smoke test), the
        consecutive-failure counter should trip and the producer should
        report the error rather than spinning forever."""
        _seed_shards(tmp_path, n_shards=2, rows_per_shard=500)
        buffer, pool = _make_pool(
            tmp_path, AlwaysRaises(), capacity=8, warmup=4, threads=1,
        )
        pool.start()
        try:
            # Warmup never reaches target (all rows fail after smoke).
            # wait_for_warmup should eventually raise — either a
            # child-died RuntimeError or a TimeoutError — but must not hang.
            with pytest.raises((RuntimeError, TimeoutError)):
                pool.wait_for_warmup(timeout=20.0)
        finally:
            pool.stop()

    def test_child_process_death_reflected_in_is_alive(self, tmp_path):
        """If the producer child is externally killed (SIGKILL-equivalent),
        ``is_alive`` must flip False.  This is the signal a health-check
        callback would read — the single most important flag for unattended
        runs.  Without this, workers silently serve a stale buffer forever."""
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=50)
        buffer, pool = _make_pool(tmp_path, ToyTokenize(), capacity=8, warmup=4)
        pool.start()
        try:
            pool.wait_for_warmup(timeout=20.0)
            assert pool.is_alive

            # Externally kill the child — simulates OOM-killer / hardware
            # fault / ffmpeg hang / etc.
            os.kill(pool._worker.pid, signal.SIGKILL)
            # Give the kernel a moment to reap the zombie.
            for _ in range(50):
                if not pool.is_alive:
                    break
                time.sleep(0.1)
            assert not pool.is_alive, (
                "pool.is_alive stayed True after SIGKILL — a Lightning "
                "callback reading this flag would never notice the death"
            )
        finally:
            pool.stop()


# ===========================================================================
# Group 2: lifecycle edges
# ===========================================================================


class TestLifecycleEdges:
    """Start/stop edge cases that break if guards regress."""

    def test_start_while_already_running_raises(self, tmp_path):
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=10)
        buffer, pool = _make_pool(tmp_path, ToyTokenize(), capacity=4, warmup=2)
        pool.start()
        try:
            pool.wait_for_warmup(timeout=20.0)
            with pytest.raises(RuntimeError, match="already running"):
                pool.start()
        finally:
            pool.stop()

    def test_stop_during_warmup_exits_cleanly(self, tmp_path):
        """Start a pool with a slow kernel so warmup is still in progress,
        then stop before it completes.  Must not hang."""
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=100)
        buffer, pool = _make_pool(
            tmp_path, SlowTokenize(delay_s=0.2), capacity=32, warmup=30,
        )
        pool.start()
        try:
            # Don't wait_for_warmup — kill it mid-warmup.
            time.sleep(0.5)
            pool.stop()
            assert not pool.is_alive
        except Exception:
            pool.stop()
            raise

    def test_stop_without_start_is_noop(self, tmp_path):
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=10)
        buffer, pool = _make_pool(tmp_path, ToyTokenize(), capacity=4, warmup=2)
        pool.stop()  # never started
        assert not pool.is_alive


# ===========================================================================
# Group 3: input + back-pressure
# ===========================================================================


class TestInputAndBackpressure:
    """Malformed input and fast-producer / slow-consumer scenarios."""

    def test_missing_text_column_surfaces_as_error(self, tmp_path):
        """Parquet without the configured ``text`` column should surface
        as a producer error, not a silent stall."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        # Write a shard with the wrong column name.
        pq.write_table(
            pa.table({"content": ["a", "b", "c"]}),  # wrong column
            tmp_path / "shard_000000.parquet",
        )
        buffer, pool = _make_pool(tmp_path, ToyTokenize(), capacity=4, warmup=2)
        pool.start()
        try:
            with pytest.raises((RuntimeError, TimeoutError)):
                pool.wait_for_warmup(timeout=15.0)
        finally:
            pool.stop()

    def test_saturated_buffer_waits_for_consumer_under_k_gate(self, tmp_path):
        """Under the new sample-gated rotation (default K=1), a saturated
        buffer with no consumer MUST block the producer — that's the
        deterministic contract.  Training speed becomes decode-bounded
        visibly (low train/step_time) instead of silently overfitting
        on stale buffer contents.

        Pre-flip, the text producer's 5s-timeout backpressure kept
        rotating the buffer even without a consumer — at ~0.2 puts/sec.
        That hid the decode-rate ceiling from the user.  The new
        behavior intentionally stalls the producer, forcing the user to
        notice the bottleneck.

        This test locks in both behaviors side-by-side.
        """
        _seed_shards(tmp_path, n_shards=3, rows_per_shard=200)
        # Default K=1: producer blocks once buffer fills.
        buffer, pool = _make_pool(tmp_path, ToyTokenize(), capacity=4, warmup=2)
        pool.start()
        try:
            pool.wait_for_warmup(timeout=20.0)
            # Buffer may be mid-fill; give it a moment to saturate.
            deadline = time.monotonic() + 5.0
            while (
                len(buffer) < buffer.capacity
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            assert len(buffer) == buffer.capacity, (
                f"buffer didn't reach capacity: {len(buffer)}/"
                f"{buffer.capacity}"
            )
            keys_at_fill = set(buffer.keys())

            # Without a consumer sample, rotation must NOT happen.
            time.sleep(2.0)
            assert set(buffer.keys()) == keys_at_fill, (
                "producer rotated under K=1 without any consumer "
                "samples — sample-gated rotation contract broken"
            )

            # Draw a consumer sample; the producer should be unblocked
            # and a new key should rotate in.
            import random as _random
            rng = _random.Random(0)
            buffer.sample(rng)
            deadline = time.monotonic() + 5.0
            while (
                set(buffer.keys()) == keys_at_fill
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
            assert set(buffer.keys()) != keys_at_fill, (
                "producer didn't rotate after consumer sample — "
                "backpressure gate isn't releasing"
            )

            assert pool.is_alive, "producer died while buffer saturated"
        finally:
            pool.stop()

    def test_new_shards_after_start_are_consumed(self, tmp_path):
        """Write one shard, start the producer, wait for warmup, THEN
        write more shards — the producer's _shard_iterator rescans and
        should pick them up.  This is how TextPrefetcher + TextProducerPool
        compose in the real training setup."""
        _seed_shards(tmp_path, n_shards=1, rows_per_shard=10)
        buffer, pool = _make_pool(
            tmp_path, ToyTokenize(), capacity=32, warmup=4, rescan_s=0.5,
        )
        pool.start()
        try:
            pool.wait_for_warmup(timeout=20.0)
            # Note initial doc keys (first shard writes keys 0..~9 max).
            initial_keys = set(buffer.keys())
            max_initial_key = max(initial_keys) if initial_keys else 0

            # Write a second shard with distinguishable content.
            _write_shard(
                tmp_path / "shard_000001.parquet",
                [f"fresh doc {r} from the late shard" for r in range(30)],
            )

            # Wait for producer to consume the new shard and rotate the
            # buffer.  Since the producer rescans every 0.5s and tokenizes
            # quickly, new keys should appear within a few seconds.
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                now_keys = set(buffer.keys())
                if now_keys and max(now_keys) > max_initial_key + 10:
                    break
                time.sleep(0.3)
            else:
                pytest.fail(
                    f"Producer didn't pick up shard 1 within 15s "
                    f"(max key stayed at {max_initial_key})"
                )
        finally:
            pool.stop()
