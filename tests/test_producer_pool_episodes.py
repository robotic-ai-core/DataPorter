"""Regression tests for ProducerConfig.dataset_episodes vs episode_indices.

Root cause of the v4 (10k-episode) timestamp-tolerance failure:
the child process received ``episodes=train_ep_indices`` (the train subset,
e.g. ``[0..0.9*N-1]``), while the parent's Arrow cache contained the full
list. This caused ``get_episode_data_index`` to mask fewer boundaries than
the Arrow table actually had, and ``check_timestamps_sync`` raised at every
unmasked ``end_ts → 0.0`` transition.

These tests lock in the fix: ``ProducerConfig.dataset_episodes`` sets the
child's ``self.episodes`` to match the Arrow cache, and ``episode_indices``
is used only for the producer's iteration work queue.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

from dataporter import FastLeRobotDataset
from dataporter.producer_pool import ProducerConfig


# ---------------------------------------------------------------------------
# Shared fixture: a small real LeRobot dataset + its Arrow cache path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parent_and_cache_path():
    """Build a parent FastLeRobotDataset over a K-episode subset and
    return (parent, arrow_cache_path, K)."""
    K = 20
    parent = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        episodes=list(range(K)),
    )
    arrow_cache_path = parent.arrow_cache_path
    assert arrow_cache_path is not None, "parent must expose arrow_cache_path"
    return parent, arrow_cache_path, K


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_kwargs_uses_dataset_episodes_when_set(parent_and_cache_path):
    """dataset_kwargs() returns the full list, not the iteration subset."""
    _, arrow_cache_path, K = parent_and_cache_path
    train = list(range(int(0.9 * K)))
    full = list(range(K))

    cfg = ProducerConfig(
        source_name="pusht",
        repo_id="lerobot/pusht",
        root="/tmp/ignored",
        episode_indices=train,
        dataset_episodes=full,
        arrow_cache_path=arrow_cache_path,
    )
    kwargs = cfg.dataset_kwargs()
    assert kwargs["episodes"] == full
    assert kwargs["episodes"] != train


def test_dataset_kwargs_falls_back_to_episode_indices(parent_and_cache_path):
    """Legacy callers (no dataset_episodes) still work — iteration list
    doubles as the dataset list."""
    cfg = ProducerConfig(
        source_name="pusht",
        repo_id="lerobot/pusht",
        root="/tmp/ignored",
        episode_indices=[0, 1, 2, 3, 4],
        dataset_episodes=None,
    )
    kwargs = cfg.dataset_kwargs()
    assert kwargs["episodes"] == [0, 1, 2, 3, 4]


def test_child_dataset_loads_without_timestamp_error(parent_and_cache_path):
    """The fix path: child receives dataset_episodes matching the Arrow
    cache and iterates a train subset.  Must construct cleanly — no
    ``One or several timestamps unexpectedly violate the tolerance``."""
    parent, arrow_cache_path, K = parent_and_cache_path
    train = list(range(int(0.9 * K)))
    full = list(range(K))

    cfg = ProducerConfig(
        source_name="pusht",
        repo_id="lerobot/pusht",
        root=str(parent.root),
        episode_indices=train,
        dataset_episodes=full,
        arrow_cache_path=arrow_cache_path,
    )

    # Simulate what _make_child_decode_fn does on first decode call.
    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        **cfg.dataset_kwargs(),
    )

    # episode_data_index must cover every episode in the Arrow cache so
    # timestamp validation masks every real boundary.
    assert len(child.episode_data_index["from"]) == K
    # Sanity: last "to" equals total frame count.
    total_frames = len(child.hf_dataset)
    assert int(child.episode_data_index["to"][-1]) == total_frames


def test_from_source_pulls_all_fields_from_parent(parent_and_cache_path):
    """ProducerConfig.from_source consolidates the construction so
    dataset_episodes, arrow_cache_path, and root can't drift."""
    parent, arrow_cache_path, K = parent_and_cache_path
    train = list(range(int(0.9 * K)))

    source = {
        "repo_id": "lerobot/pusht",
        "weight": 2.5,
        "tolerance_s": 0.08,
    }
    cfg = ProducerConfig.from_source(
        source=source,
        full_ds=parent,
        iteration_episodes=train,
        episode_offset=37,
    )

    assert cfg.repo_id == "lerobot/pusht"
    assert cfg.weight == 2.5
    assert cfg.tolerance_s == 0.08
    assert cfg.episode_offset == 37
    assert cfg.root == str(parent.root)
    assert cfg.arrow_cache_path == arrow_cache_path
    assert cfg.episode_indices == train
    # dataset_episodes must equal what's in the parent's Arrow cache.
    assert cfg.dataset_episodes == list(parent.episodes)


def test_from_source_child_loads_cleanly(parent_and_cache_path):
    """End-to-end: from_source produces a config whose dataset_kwargs
    build a child dataset that (a) skips re-validation, (b) passes the
    size-integrity check."""
    parent, _, K = parent_and_cache_path
    train = list(range(int(0.9 * K)))
    source = {"repo_id": "lerobot/pusht", "weight": 1.0}

    cfg = ProducerConfig.from_source(
        source=source,
        full_ds=parent,
        iteration_episodes=train,
    )
    kwargs = cfg.dataset_kwargs()
    # Auto-enabled because arrow_cache_path is set.
    assert kwargs["skip_timestamp_validation"] is True

    child = FastLeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"observation.image": [0.0]},
        **kwargs,
    )
    assert len(child.episode_data_index["from"]) == K


def test_buggy_path_reproduces_timestamp_error(parent_and_cache_path):
    """Without the fix, passing the train subset as episodes= triggers
    exactly the ValueError seen on Vast.

    This locks the bug in: if someone refactors dataset_kwargs() and
    accidentally drops the dataset_episodes preference, this test fails.
    """
    parent, arrow_cache_path, K = parent_and_cache_path
    train = list(range(int(0.9 * K)))

    # Legacy construction — what the old ProducerConfig produced.
    with pytest.raises(ValueError, match="tolerance"):
        FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
            root=parent.root,
            episodes=train,
            arrow_cache_path=arrow_cache_path,
        )
