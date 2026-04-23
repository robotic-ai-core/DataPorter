"""Regression tests for the FastLeRobotDataset → LeRobotShardSource
migration on :class:`BlendedLeRobotDataModule`'s setup path.

The motivation is captured in the migration prompt at
``/tmp/dataporter_parent_shard_migration_prompt.md``: at 18k
episodes, the parent used to spend ~7 minutes in FastLeRobotDataset
construction (Arrow cache rebuild, metadata validation, parquet scan)
per run.  Shard-source-backed setup is O(1) — metadata loads lazily.

These tests guard:

- **Setup completes in bounded wall time** on a large synthetic
  dataset.  The threshold is loose (15s for 2k episodes) because CI
  boxes vary wildly; we want regression detection, not flaky CI.
  Actual timing is logged so a regression shows up as "now 42s
  instead of 2s" in test output.

- **Val dataset yields samples with the expected shape/keys** on the
  new path, and sample count matches what
  ``_load_and_split_source`` computed.

- **Multi-source configs still work** — the critical contract for
  blended training.

Runs use ``_make_dataset`` from ``test_shard_source_pool_e2e`` so
the fixture matches what the end-to-end pipeline exercises.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

from dataporter.blended_lerobot_datamodule import BlendedLeRobotDataModule
from test_shard_source_pool_e2e import _make_dataset


def _make_dm(
    roots: list, *, shuffle_buffer_capacity: int | None = None,
) -> BlendedLeRobotDataModule:
    """Construct a DataModule pointed at the given synthetic roots."""
    sources = [
        {"repo_id": f"synth_{i}", "root": str(root), "weight": 1.0,
         "prefetch": False}
        for i, root in enumerate(roots)
    ]
    return BlendedLeRobotDataModule(
        repo_id=sources,
        delta_timestamps={},
        batch_size=4,
        num_workers=0,
        shuffle_buffer_capacity=shuffle_buffer_capacity,
    )


# ---------------------------------------------------------------------------
# Wall-time assertion — the whole point of the migration.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_setup_completes_in_bounded_time_large_dataset(tmp_path, capsys):
    """Setup on a 2000-episode synthetic dataset completes in under
    15 seconds.

    Size rationale: 18k is what production hits but generating 18k
    synthetic mp4s + parquets takes minutes, dominating the test
    timing signal we care about.  2k episodes is enough that the old
    FastLeRobotDataset path would take 60+ seconds (~25ms/episode ×
    2000 = 50s of pure Arrow+metadata work), while the new shard-
    source path should finish in a few seconds regardless of episode
    count.  The 15s ceiling catches "we accidentally went back to
    the O(N) path" without being flaky on slow CI.

    If this ever fails with a number like 4s, 6s — lower the ceiling.
    If it fails with a number like 30s — the migration regressed.
    """
    # Small frame counts + tiny resolution keep mp4 generation fast;
    # the goal is to stress the setup-time cost of metadata load, not
    # the fixture cost.
    root = tmp_path / "ds"
    _make_dataset(
        root, ready_eps=list(range(2000)),
        total_episodes=2000,
        n_frames_per_ep=4,
        height=16, width=16,
    )

    dm = _make_dm([root])
    t0 = time.monotonic()
    dm.setup()
    elapsed = time.monotonic() - t0
    # Log actual — the Why this matters value isn't just pass/fail.
    print(f"\n[setup-time regression] setup(): {elapsed:.2f}s "
          f"(2000 synth eps, ceiling 15s)")
    assert elapsed < 15.0, (
        f"BlendedLeRobotDataModule.setup() took {elapsed:.1f}s on a "
        f"2000-episode synthetic dataset — regression on the "
        f"LeRobotShardSource migration.  The old FastLeRobotDataset "
        f"path would land around 50s here; shard-source should be a "
        f"few seconds at most.  See "
        f"/tmp/dataporter_parent_shard_migration_prompt.md for "
        f"context."
    )


# ---------------------------------------------------------------------------
# Val dataset shape/contract on the new path.
# ---------------------------------------------------------------------------


class TestValDatasetOnShardSource:

    def test_val_dataset_samples_have_expected_keys(self, tmp_path):
        """Sample dicts from the val loader contain the LeRobot
        contract keys: ``episode_index``, ``frame_index``,
        ``observation.image``, and a ``task`` string."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=list(range(10)), total_episodes=10,
            n_frames_per_ep=5,
        )
        dm = _make_dm([root])
        dm.setup()

        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert "episode_index" in batch
        assert "frame_index" in batch
        assert "observation.image" in batch
        # Task string: default-collate leaves strings as a list of N.
        assert "task" in batch

    def test_val_length_matches_split(self, tmp_path):
        """Val length equals the sum of val-episode frame counts —
        split_fn sends every 10th raw id to val by default, and each
        episode contributes ``n_frames_per_ep`` frames."""
        root = tmp_path / "ds"
        n_eps = 30
        frames_per_ep = 5
        _make_dataset(
            root, ready_eps=list(range(n_eps)), total_episodes=n_eps,
            n_frames_per_ep=frames_per_ep,
        )

        dm = _make_dm([root])
        dm.setup()

        # Default split_fn: raw_ep % 10 == 9 → val.  In [0, 30): 9, 19, 29.
        expected = 3 * frames_per_ep
        assert len(dm.val_dataset) == expected


# ---------------------------------------------------------------------------
# Multi-source smoke test — the other big contract.
# ---------------------------------------------------------------------------


class TestMultiSourceSetup:

    def test_two_sources_setup_and_val_load(self, tmp_path):
        """Two synthetic sources; setup succeeds and the val loader
        yields at least one batch from each source."""
        root_a = tmp_path / "ds_a"
        root_b = tmp_path / "ds_b"
        _make_dataset(root_a, ready_eps=list(range(10)),
                      total_episodes=10, n_frames_per_ep=4)
        _make_dataset(root_b, ready_eps=list(range(10)),
                      total_episodes=10, n_frames_per_ep=4)

        dm = _make_dm([root_a, root_b])
        dm.setup()

        loader = dm.val_dataloader()
        batches = list(loader)
        total = sum(b["frame_index"].shape[0] for b in batches)
        # Each source contributes 1 val episode × 4 frames = 4; two
        # sources → 8 val frames total.
        assert total == 8


# ---------------------------------------------------------------------------
# ShuffleBuffer path still works end-to-end after migration.
# ---------------------------------------------------------------------------


class TestShuffleBufferPathStillWorks:

    def test_shuffle_buffer_training_setup(self, tmp_path):
        """Smoke test: the ShuffleBuffer path — the one production
        actually exercises — still produces a train dataset that
        yields samples after migration."""
        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=list(range(10)), total_episodes=10,
            n_frames_per_ep=5,
        )
        dm = _make_dm([root], shuffle_buffer_capacity=4)
        try:
            dm.setup()
            loader = dm.train_dataloader()
            batch = next(iter(loader))
            assert "observation.image" in batch
            assert batch["episode_index"].shape[0] >= 1
        finally:
            dm.teardown()
