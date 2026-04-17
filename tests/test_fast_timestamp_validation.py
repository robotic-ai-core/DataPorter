"""Tests for FastLeRobotDataset's optimized timestamp validation.

Verifies that the fast Arrow-based extraction produces identical results
to the original torch.stack(list(...)) approach, and that validation
still catches actual timestamp errors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")

from lerobot.common.datasets.utils import check_timestamps_sync


# ---------------------------------------------------------------------------
# Unit tests for the fast extraction path
# ---------------------------------------------------------------------------

class TestFastTimestampExtraction:
    """Verify that Arrow .to_numpy() produces identical results to
    torch.stack(list(...)).numpy() for timestamp validation."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load lerobot/pusht (small, always available)."""
        from dataporter import FastLeRobotDataset
        ds = FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
        )
        return ds

    def test_timestamps_match(self, dataset):
        """Arrow extraction matches torch.stack(list(...)) exactly."""
        # Slow path (original)
        from lerobot.common.datasets.lerobot_dataset import hf_transform_to_torch
        ds_raw = dataset.hf_dataset
        ts_slow = torch.stack(list(ds_raw["timestamp"])).numpy()

        # Fast path (Arrow direct)
        ts_fast = ds_raw._data.column("timestamp").to_numpy()

        assert np.allclose(ts_slow, ts_fast, atol=1e-7), (
            f"Timestamps differ: max_diff={np.abs(ts_slow - ts_fast).max()}"
        )

    def test_episode_indices_match(self, dataset):
        """Arrow extraction matches for episode_index column."""
        ds_raw = dataset.hf_dataset
        ep_slow = torch.stack(list(ds_raw["episode_index"])).numpy()
        ep_fast = ds_raw._data.column("episode_index").to_numpy()

        assert np.array_equal(ep_slow, ep_fast)

    def test_validation_passes_with_fast_path(self, dataset):
        """check_timestamps_sync passes with Arrow-extracted data."""
        ds_raw = dataset.hf_dataset
        timestamps = ds_raw._data.column("timestamp").to_numpy()
        episode_indices = ds_raw._data.column("episode_index").to_numpy()
        ep_data_index_np = {
            k: t.numpy() for k, t in dataset.episode_data_index.items()
        }

        # Should not raise
        result = check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np,
            dataset.fps, dataset.tolerance_s,
        )
        assert result is True


class TestValidationStillCatchesErrors:
    """The fast path must still detect actual timestamp problems.

    These tests use synthetic data to verify the validation logic
    hasn't been silently bypassed.
    """

    def test_catches_gap_within_episode(self):
        """A missing frame (gap > 1/fps + tolerance) should fail."""
        fps = 10
        tolerance_s = 1e-4
        # Episode 0: 10 frames at 10Hz, but skip frame 5
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        episode_indices = np.zeros(9, dtype=np.int64)
        ep_data_index = {
            "from": np.array([0]),
            "to": np.array([9]),
        }

        with pytest.raises(ValueError, match="tolerance"):
            check_timestamps_sync(
                timestamps, episode_indices, ep_data_index,
                fps, tolerance_s,
            )

    def test_allows_episode_boundary_jump(self):
        """Timestamp jump at episode boundary should be ignored."""
        fps = 10
        tolerance_s = 1e-4
        # Episode 0: 0.0-0.4, Episode 1: 0.0-0.4
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3, 0.4])
        episode_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ep_data_index = {
            "from": np.array([0, 5]),
            "to": np.array([5, 10]),
        }

        # Should NOT raise — boundary jump is expected
        result = check_timestamps_sync(
            timestamps, episode_indices, ep_data_index,
            fps, tolerance_s,
        )
        assert result is True

    def test_catches_wrong_fps(self):
        """Timestamps at wrong FPS should fail."""
        fps = 10
        tolerance_s = 1e-4
        # Timestamps at 5Hz instead of 10Hz
        timestamps = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        episode_indices = np.zeros(5, dtype=np.int64)
        ep_data_index = {
            "from": np.array([0]),
            "to": np.array([5]),
        }

        with pytest.raises(ValueError, match="tolerance"):
            check_timestamps_sync(
                timestamps, episode_indices, ep_data_index,
                fps, tolerance_s,
            )

    def test_passes_with_float_rounding(self):
        """Timestamps with minor float rounding should pass."""
        fps = 10
        tolerance_s = 1e-4
        # Python float rounding: 7 * 0.1 = 0.7000000000000001
        timestamps = np.array([i * 0.1 for i in range(10)], dtype=np.float32)
        episode_indices = np.zeros(10, dtype=np.int64)
        ep_data_index = {
            "from": np.array([0]),
            "to": np.array([10]),
        }

        result = check_timestamps_sync(
            timestamps, episode_indices, ep_data_index,
            fps, tolerance_s,
        )
        assert result is True


class TestFastLeRobotDatasetInit:
    """Verify FastLeRobotDataset uses the fast path and produces
    correct results."""

    def test_validation_uses_arrow_not_torch_stack(self):
        """Upstream lerobot fix reads Arrow directly in __init__.

        Verify by checking that check_timestamps_sync receives numpy
        arrays (from Arrow .to_numpy()) not torch-converted float32.
        """
        from dataporter import FastLeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as _ld

        received_dtypes = []
        _orig = _ld.check_timestamps_sync

        def spy_check(timestamps, episode_indices, *args, **kwargs):
            received_dtypes.append(timestamps.dtype)
            return _orig(timestamps, episode_indices, *args, **kwargs)

        _ld.check_timestamps_sync = spy_check
        try:
            ds = FastLeRobotDataset(
                "lerobot/pusht",
                delta_timestamps={"observation.image": [0.0]},
            )
        finally:
            _ld.check_timestamps_sync = _orig

        assert len(received_dtypes) > 0
        # Arrow .to_numpy() on float32 column returns float32
        import numpy as np
        assert received_dtypes[0] == np.float32

    def test_init_still_validates(self):
        """Validation runs during init (not skipped)."""
        from dataporter import FastLeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as _ld

        validation_ran = False
        _orig = _ld.check_timestamps_sync

        def tracking_check(*args, **kwargs):
            nonlocal validation_ran
            validation_ran = True
            return _orig(*args, **kwargs)

        _ld.check_timestamps_sync = tracking_check
        try:
            ds = FastLeRobotDataset(
                "lerobot/pusht",
                delta_timestamps={"observation.image": [0.0]},
            )
        finally:
            _ld.check_timestamps_sync = _orig

        assert validation_ran, "check_timestamps_sync was never called"

    def test_arrow_cache_path_also_validates_by_default(self):
        """With arrow_cache_path but no skip flag, validation still runs."""
        from dataporter import FastLeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as _ld

        ds1 = FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
        )
        cache_path = ds1.hf_dataset.cache_files[0]["filename"]

        validation_ran = False
        _orig = _ld.check_timestamps_sync

        def tracking_check(*args, **kwargs):
            nonlocal validation_ran
            validation_ran = True
            return _orig(*args, **kwargs)

        _ld.check_timestamps_sync = tracking_check
        try:
            FastLeRobotDataset(
                "lerobot/pusht",
                delta_timestamps={"observation.image": [0.0]},
                arrow_cache_path=cache_path,
            )
        finally:
            _ld.check_timestamps_sync = _orig

        assert validation_ran

    def test_skip_timestamp_validation_skips_check(self):
        """skip_timestamp_validation=True bypasses check_timestamps_sync."""
        from dataporter import FastLeRobotDataset
        import lerobot.common.datasets.lerobot_dataset as _ld

        ds1 = FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
        )
        cache_path = ds1.hf_dataset.cache_files[0]["filename"]

        validation_ran = False
        _orig = _ld.check_timestamps_sync

        def tracking_check(*args, **kwargs):
            nonlocal validation_ran
            validation_ran = True
            return _orig(*args, **kwargs)

        _ld.check_timestamps_sync = tracking_check
        try:
            FastLeRobotDataset(
                "lerobot/pusht",
                delta_timestamps={"observation.image": [0.0]},
                arrow_cache_path=cache_path,
                skip_timestamp_validation=True,
            )
        finally:
            _ld.check_timestamps_sync = _orig

        assert not validation_ran, (
            "check_timestamps_sync ran despite skip_timestamp_validation=True"
        )

    def test_size_mismatch_raises_clear_error(self):
        """Arrow cache / episode_data_index mismatch raises RuntimeError.

        This is the size-assert guard that catches the class of bug
        where parent passes one episode list and child passes another.
        Reproduces the Vast v4 failure mode cheaply and deterministically.
        """
        from dataporter import FastLeRobotDataset

        parent = FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
            episodes=list(range(20)),
        )
        cache_path = parent.arrow_cache_path

        # Child receives only 18 episodes in self.episodes but loads the
        # parent's 20-episode Arrow cache. skip_timestamp_validation=True
        # would silently hide the bug without the size assert.
        with pytest.raises(RuntimeError, match="doesn't match the Arrow table"):
            FastLeRobotDataset(
                "lerobot/pusht",
                delta_timestamps={"observation.image": [0.0]},
                root=parent.root,
                episodes=list(range(18)),
                arrow_cache_path=cache_path,
                skip_timestamp_validation=True,
            )

    def test_hf_dataset_has_transform(self):
        """hf_dataset has set_transform applied, returns torch tensors."""
        from dataporter import FastLeRobotDataset

        ds = FastLeRobotDataset(
            "lerobot/pusht",
            delta_timestamps={"observation.image": [0.0]},
        )
        sample = ds.hf_dataset[0]
        assert isinstance(sample["timestamp"], torch.Tensor)
        assert isinstance(sample["episode_index"], torch.Tensor)
