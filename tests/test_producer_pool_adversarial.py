"""Adversarial e2e tests: drive a real spawned ProducerPool child with
deliberately broken configs and verify each failure surfaces clearly.

These are the tests that would have caught the v4 bug before it hit Vast.
Each scenario poisons one specific invariant; the child must either
refuse to start or report a clear error through the parent's error queue.
The bar is "fails fast with a message that names the invariant" — not
"silently produces wrong data" and not "child hangs until warmup timeout."
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

pytest.importorskip("lerobot")

from dataporter import FastLeRobotDataset
from dataporter.producer_pool import ProducerConfig, ProducerPool
from dataporter.shuffle_buffer import ShuffleBuffer


# ---------------------------------------------------------------------------
# Shared fixture: real lerobot/pusht parent with a 20-episode subset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parent_fixture():
    K = 20
    parent = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=list(range(K)),
    )
    return parent, K


def _run_pool_briefly(configs, timeout_s: float = 20.0) -> list[str]:
    """Spawn a ProducerPool with the given configs, let it try to warm
    up, then collect every error that surfaced to the parent.

    Returns the list of messages the child sent through the error queue.
    ``wait_for_warmup`` itself drains errors and re-raises them as
    RuntimeError — we capture that message too, since that's the form
    a real caller would see in the training log.
    """
    buffer = ShuffleBuffer(
        capacity=4, max_frames=400, channels=3, height=96, width=96,
    )
    pool = ProducerPool(buffer, configs=configs, total_workers=1)
    pool.start()
    errors: list[str] = []
    try:
        try:
            pool.wait_for_warmup(timeout=timeout_s)
        except TimeoutError as e:
            errors.append(str(e))
        except RuntimeError as e:
            errors.append(str(e))
        # Drain anything left after wait_for_warmup consumed the first.
        while True:
            err = pool._drain_error_queue()
            if err is None:
                break
            errors.append(err)
        return errors
    finally:
        pool.stop()


# ---------------------------------------------------------------------------
# Adversarial scenarios
# ---------------------------------------------------------------------------


class TestAdversarialConfigs:
    """Each test poisons one invariant that the v4 investigation
    identified as a trap.  All are things that used to fail slowly or
    silently; all must now fail fast with a named invariant."""

    def test_mismatched_dataset_episodes_fails_with_clear_error(
        self, parent_fixture
    ):
        """The v4 bug: child receives fewer episodes than the Arrow
        cache actually contains.  Must surface as a size-mismatch
        RuntimeError, not a 70s timestamp crash."""
        parent, K = parent_fixture
        train = list(range(int(0.9 * K)))

        cfg = ProducerConfig(
            source_name="adversarial-mismatch",
            repo_id="lerobot/pusht",
            root=str(parent.root),
            episode_indices=train,
            dataset_episodes=train,  # WRONG: should be full 0..K-1
            arrow_cache_path=parent.arrow_cache_path,
        )
        errors = _run_pool_briefly([cfg])

        assert errors, "Child failed but produced no error message"
        joined = " ".join(errors)
        assert (
            "doesn't match the Arrow table" in joined
            or "episode_data_index covers" in joined
        ), f"Expected size-mismatch error, got: {errors}"

    def test_non_contiguous_episodes_rejected(self, parent_fixture):
        """Non-contiguous self.episodes would silently fetch wrong
        video files at decode time.  Must fail at dataset init with
        a message that names the coordinate-system invariant."""
        parent, K = parent_fixture
        # Every other episode — triggers positional/raw conflation.
        non_contig = [i for i in range(K) if i % 2 == 0]

        cfg = ProducerConfig(
            source_name="adversarial-noncontig",
            repo_id="lerobot/pusht",
            root=str(parent.root),
            episode_indices=[0, 1, 2],
            dataset_episodes=non_contig,
            arrow_cache_path=parent.arrow_cache_path,
        )
        errors = _run_pool_briefly([cfg])

        assert errors, "Child failed but produced no error message"
        joined = " ".join(errors)
        assert "contiguous from 0" in joined, (
            f"Expected contiguous-from-0 error, got: {errors}"
        )

    def test_non_zero_start_rejected(self, parent_fixture):
        """Episodes starting from non-zero (e.g. [5..15]) would cause
        get_video_file_path to fetch episode 5's file when the producer
        asks for positional index 0.  Must fail fast."""
        parent, K = parent_fixture
        shifted = list(range(5, K))

        cfg = ProducerConfig(
            source_name="adversarial-shifted",
            repo_id="lerobot/pusht",
            root=str(parent.root),
            episode_indices=[0, 1],
            dataset_episodes=shifted,
            arrow_cache_path=parent.arrow_cache_path,
        )
        errors = _run_pool_briefly([cfg])

        assert errors, "Child failed but produced no error message"
        joined = " ".join(errors)
        assert "contiguous from 0" in joined, (
            f"Expected contiguous-from-0 error, got: {errors}"
        )


# ---------------------------------------------------------------------------
# Unit-level: the invariant itself (doesn't need a ProducerPool)
# ---------------------------------------------------------------------------


def test_contiguous_from_zero_rejected_directly(parent_fixture):
    """FastLeRobotDataset directly rejects non-contiguous episodes —
    this is the single source of truth for the invariant, the e2e
    tests above just verify it's enforced through the full stack."""
    parent, _ = parent_fixture

    with pytest.raises(ValueError, match="contiguous from 0"):
        FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
            root=parent.root,
            episodes=[0, 2, 4, 6],
            arrow_cache_path=parent.arrow_cache_path,
            skip_timestamp_validation=True,
        )


def test_happy_path_passes_invariants(parent_fixture):
    """Contiguous-from-0 episodes matching the Arrow cache pass both
    the size assertion and the contiguity assertion — the safety nets
    don't accidentally reject legitimate configs."""
    parent, K = parent_fixture

    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        root=parent.root,
        episodes=list(range(K)),
        arrow_cache_path=parent.arrow_cache_path,
        skip_timestamp_validation=True,
    )
    assert len(child.episode_data_index["from"]) == K
