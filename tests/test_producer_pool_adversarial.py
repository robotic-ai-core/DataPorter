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


# ---------------------------------------------------------------------------
# Unit-level: sparse / shifted subsets are NOW supported (no more
# contiguous-from-0 assertion, coordinate translation handles it).
# ---------------------------------------------------------------------------


def test_sparse_episodes_accepted(parent_fixture):
    """Sparse subsets like [0, 2, 4, 6] — previously rejected by a
    defensive contiguous-from-0 assertion — are now supported.  The raw
    episode id is translated to a positional index for
    ``episode_data_index`` lookups while ``get_video_file_path`` still
    receives the raw id.  Locks in the decoupling fix.
    """
    parent, _ = parent_fixture

    # Use the real parquet subset path (no Arrow cache) so
    # episode_data_index matches the hf_dataset row count.
    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=[0, 2, 4, 6],
    )
    del parent  # silence unused
    assert len(child.episode_data_index["from"]) == 4
    # Raw-id → positional translation populated.
    assert child._raw_to_pos == {0: 0, 2: 1, 4: 2, 6: 3}


def test_non_zero_start_accepted():
    """Shifted-start subsets [5, 6, 7] don't need the raw id to be 0.
    Regression coverage for partial-prefetch scenarios where the first
    episode to land is not episode 0.
    """
    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=[5, 6, 7],
    )
    assert len(child.episode_data_index["from"]) == 3
    assert child._raw_to_pos == {5: 0, 6: 1, 7: 2}


def test_unknown_raw_ep_id_raises_clear_error():
    """``_ep_positional`` must fail loud when asked to translate a raw
    id that isn't in the configured subset — not silently index into
    the wrong row.
    """
    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=[0, 2, 4],
    )
    with pytest.raises(KeyError, match="not in this dataset's episode subset"):
        child._ep_positional(99)


def test_sparse_getitem_returns_correct_episode_rows():
    """End-to-end correctness: with a sparse subset, ``__getitem__``
    must route each row's raw ``episode_index`` through the
    translation so videos load from the right file and
    ``_get_query_indices`` reads the right ``episode_data_index``
    slot.  Without the fix, this would IndexError or decode wrong
    frames.
    """
    subset = [0, 2, 4]
    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=subset,
    )

    # Pick one row per selected episode and verify the returned
    # episode_index matches the subset (no silent drift to a
    # neighbouring episode).
    seen_raw_ids = set()
    for pos in range(len(subset)):
        ep_start = child.episode_data_index["from"][pos].item()
        sample = child[ep_start]
        seen_raw_ids.add(int(sample["episode_index"].item()))
    assert seen_raw_ids == set(subset), (
        f"sparse-subset __getitem__ returned episodes {seen_raw_ids}, "
        f"expected {set(subset)} — coordinate translation is wrong"
    )


def test_happy_path_passes_invariants(parent_fixture):
    """Contiguous-from-0 episodes matching the Arrow cache pass the
    size assertion — the safety nets don't accidentally reject
    legitimate configs.
    """
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
    # [0..K-1] → identity map.
    assert child._raw_to_pos == {i: i for i in range(K)}
