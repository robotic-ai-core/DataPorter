"""Pipeline equivalence tests: BlendedLeRobotDataModule vs legacy standalone.

Validates that the new pipeline produces identical training data to the
legacy pipeline across:
  1. Individual sample content (same index → same tensors)
  2. Epoch coverage (all samples seen exactly once)
  3. Train/val split consistency (same episodes in each split)
  4. Long-horizon stability (shard eviction + reload doesn't corrupt data)
  5. Timestamp sync (prefetched root passes same validation as HF cache)

These tests use lerobot/pusht (206 episodes, ~2GB) as the reference dataset.
They require network access for the first run (HF download), then use cache.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from dataporter import FastLeRobotDataset, ResumableDataLoader

# Skip entire module if lerobot not available or dataset not cached
pytest.importorskip("lerobot")

# ---------------------------------------------------------------------------
# Shared constants matching both pipelines
# ---------------------------------------------------------------------------

REPO_ID = "lerobot/pusht"
CONTEXT_LENGTH = 4
AUTO_STEPS = 3
VAL_AR_STEPS = 8
TRAIN_SPLIT_RATIO = 0.9
FPS = 10
DELTA = 1.0 / FPS

TRAIN_TOTAL = CONTEXT_LENGTH + 1 + AUTO_STEPS  # 8
VAL_TOTAL = CONTEXT_LENGTH + 1 + max(AUTO_STEPS, VAL_AR_STEPS)  # 13

TRAIN_DELTA_TIMESTAMPS = {
    "observation.image": [i * DELTA for i in range(-TRAIN_TOTAL + 1, 1)],
    "observation.state": [i * DELTA for i in range(-TRAIN_TOTAL + 1, 1)],
    "action": [i * DELTA for i in range(-TRAIN_TOTAL + 1, 1)],
    "next.reward": [i * DELTA for i in range(-TRAIN_TOTAL + 1, 1)],
    "next.done": [i * DELTA for i in range(-TRAIN_TOTAL + 1, 1)],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(root=None, episodes=None, tolerance_s=1e-4):
    """Load FastLeRobotDataset with explicit parameters."""
    kwargs = {"delta_timestamps": TRAIN_DELTA_TIMESTAMPS}
    if root is not None:
        kwargs["root"] = root
    if episodes is not None:
        kwargs["episodes"] = episodes
    if tolerance_s is not None:
        kwargs["tolerance_s"] = tolerance_s
    return FastLeRobotDataset(REPO_ID, **kwargs)


def _split_episodes(dataset):
    """90/10 per-episode split matching both pipelines."""
    ep_index = dataset.episode_data_index
    num_episodes = len(ep_index["from"])
    train_episodes = int(TRAIN_SPLIT_RATIO * num_episodes)

    train_indices = []
    val_indices = []
    for ep_idx in range(num_episodes):
        start = int(ep_index["from"][ep_idx])
        end = int(ep_index["to"][ep_idx])
        if ep_idx < train_episodes:
            train_indices.extend(range(start, end))
        else:
            val_indices.extend(range(start, end))

    return train_indices, val_indices, train_episodes, num_episodes


def _prefetch_dataset(repo_id, local_dir):
    """Simulate prefetcher: snapshot_download to a local directory."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, local_dir=str(local_dir), repo_type="dataset")
    # Scan available episodes (same as BlendedLeRobotDataModule._start_prefetcher)
    import re
    episodes = sorted(
        int(re.search(r"episode_(\d+)", p.stem).group(1))
        for p in Path(local_dir).rglob("episode_*.parquet")
        if re.search(r"episode_(\d+)", p.stem)
    )
    return episodes


# ---------------------------------------------------------------------------
# Level 1: Sample equivalence
# ---------------------------------------------------------------------------

class TestSampleEquivalence:
    """Same index → same tensors from both loading paths."""

    @pytest.fixture(scope="class")
    def datasets(self, tmp_path_factory):
        """Load dataset via HF cache (legacy) and via prefetched root (new)."""
        local_dir = tmp_path_factory.mktemp("prefetch")
        episodes = _prefetch_dataset(REPO_ID, local_dir)

        ds_hf = _load_dataset()
        # New pipeline: loads from prefetched root with episodes filter
        # This is what triggers the check_timestamps_sync failure at 1e-4
        try:
            ds_prefetch = _load_dataset(
                root=str(local_dir), episodes=episodes, tolerance_s=1e-4
            )
        except ValueError as e:
            pytest.fail(
                f"Prefetched dataset fails timestamp sync at tolerance_s=1e-4. "
                f"This is the root cause of the pipeline divergence.\n{e}"
            )

        return ds_hf, ds_prefetch

    def test_same_length(self, datasets):
        ds_hf, ds_prefetch = datasets
        assert len(ds_hf) == len(ds_prefetch)

    def test_same_episodes(self, datasets):
        ds_hf, ds_prefetch = datasets
        assert ds_hf.meta.total_episodes == ds_prefetch.meta.total_episodes

    def test_same_fps(self, datasets):
        ds_hf, ds_prefetch = datasets
        assert ds_hf.fps == ds_prefetch.fps

    def test_sample_content_identical(self, datasets):
        """First 20 samples must be byte-identical between loading paths."""
        ds_hf, ds_prefetch = datasets
        diffs = []
        for idx in range(min(20, len(ds_hf))):
            s1 = ds_hf[idx]
            s2 = ds_prefetch[idx]
            for key in s1:
                if key not in s2:
                    diffs.append((idx, key, "missing in prefetch"))
                    continue
                v1, v2 = s1[key], s2[key]
                if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                    if not torch.equal(v1, v2):
                        max_diff = (v1.float() - v2.float()).abs().max().item()
                        diffs.append((idx, key, f"max_diff={max_diff:.6f}"))

        assert not diffs, f"Sample content differs:\n" + "\n".join(
            f"  idx={i} key={k}: {d}" for i, k, d in diffs[:10]
        )


# ---------------------------------------------------------------------------
# Level 2: Split and coverage equivalence
# ---------------------------------------------------------------------------

class TestSplitEquivalence:
    """Train/val splits must be identical between both pipelines."""

    @pytest.fixture(scope="class")
    def splits(self):
        ds = _load_dataset()
        return _split_episodes(ds)

    def test_train_val_no_overlap(self, splits):
        train_idx, val_idx, _, _ = splits
        overlap = set(train_idx) & set(val_idx)
        assert not overlap, f"{len(overlap)} overlapping indices"

    def test_full_coverage(self, splits):
        train_idx, val_idx, _, _ = splits
        ds = _load_dataset()
        assert len(train_idx) + len(val_idx) == len(ds)

    def test_split_ratio(self, splits):
        _, _, train_eps, total_eps = splits
        ratio = train_eps / total_eps
        assert abs(ratio - TRAIN_SPLIT_RATIO) < 0.01

    def test_train_episodes_contiguous(self, splits):
        """Train uses first 90% of episodes (by index), val uses the rest."""
        ds = _load_dataset()
        train_idx, val_idx, train_eps, _ = splits

        # All train indices should be from episodes 0..train_eps-1
        for idx in train_idx[:100]:
            sample = ds.hf_dataset[idx]
            assert sample["episode_index"] < train_eps


# ---------------------------------------------------------------------------
# Level 3: Epoch coverage
# ---------------------------------------------------------------------------

class TestEpochCoverage:
    """Every training sample is seen exactly once per epoch."""

    def test_dataloader_sees_all_samples(self):
        ds = _load_dataset()
        train_idx, _, _, _ = _split_episodes(ds)
        train_sub = Subset(ds, train_idx)

        loader = DataLoader(train_sub, batch_size=32, shuffle=True, num_workers=0)

        seen_indices = []
        for batch in loader:
            seen_indices.extend(batch["index"].tolist())

        # Every index should appear exactly once
        counts = Counter(seen_indices)
        duplicates = {idx: c for idx, c in counts.items() if c > 1}
        assert not duplicates, f"Duplicate indices: {list(duplicates.items())[:5]}"
        assert len(seen_indices) >= len(train_idx) - 32  # allow drop_last


# ---------------------------------------------------------------------------
# Level 4: Timestamp sync validation
# ---------------------------------------------------------------------------

class TestTimestampSync:
    """The core bug: prefetched root must pass timestamp validation."""

    def test_hf_cache_passes_at_default_tolerance(self):
        """Loading from HF cache at tolerance_s=1e-4 must work."""
        ds = _load_dataset(tolerance_s=1e-4)
        assert len(ds) > 0

    def test_prefetched_root_passes_at_default_tolerance(self, tmp_path):
        """Loading from prefetched root at tolerance_s=1e-4 must also work.

        This test currently FAILS — it's the root cause of the pipeline
        divergence. The fix should make this pass.
        """
        episodes = _prefetch_dataset(REPO_ID, tmp_path)
        try:
            ds = _load_dataset(
                root=str(tmp_path), episodes=episodes, tolerance_s=1e-4
            )
            assert len(ds) > 0
        except ValueError as e:
            pytest.fail(
                f"Prefetched dataset fails check_timestamps_sync at 1e-4.\n"
                f"This is the bug that forced tolerance_s=100.0 workaround.\n{e}"
            )

    def test_tolerance_does_not_change_frame_selection(self, tmp_path):
        """tolerance_s only gates an assert — frame selection is by argmin.

        Regardless of tolerance value, the same frames must be returned.
        """
        ds_tight = _load_dataset(tolerance_s=1e-4)
        ds_loose = _load_dataset(tolerance_s=100.0)

        for idx in [0, 100, 500, 1000, 5000]:
            if idx >= len(ds_tight):
                break
            s1 = ds_tight[idx]
            s2 = ds_loose[idx]
            assert torch.equal(
                s1["observation.image"], s2["observation.image"]
            ), f"Frame selection differs at idx={idx}"


# ---------------------------------------------------------------------------
# Level 5: Long-horizon shard stability
# ---------------------------------------------------------------------------

class TestShardStability:
    """Simulates long training with shard eviction and reloading."""

    def test_evicted_episode_not_in_training(self, tmp_path):
        """After evicting an episode's parquet, it shouldn't appear in samples."""
        episodes = _prefetch_dataset(REPO_ID, tmp_path)

        # Find and delete episode 5's parquet file
        ep5_files = list(tmp_path.rglob("episode_000005.parquet"))
        for f in ep5_files:
            f.unlink()

        remaining = [e for e in episodes if e != 5]

        # Loading with filtered episodes should work
        ds = _load_dataset(
            root=str(tmp_path), episodes=remaining, tolerance_s=100.0
        )

        # Episode 5 should not appear in any sample
        for idx in range(min(100, len(ds))):
            sample = ds[idx]
            assert sample["episode_index"].item() != 5

    def test_reload_after_eviction_restores_data(self, tmp_path):
        """Re-downloading an evicted episode restores access."""
        episodes = _prefetch_dataset(REPO_ID, tmp_path)

        # Save a copy of episode 0's parquet
        ep0_files = list(tmp_path.rglob("episode_000000.parquet"))
        backups = {}
        for f in ep0_files:
            backups[f] = f.read_bytes()
            f.unlink()

        # Load without episode 0
        remaining = [e for e in episodes if e != 0]
        ds_without = _load_dataset(
            root=str(tmp_path), episodes=remaining, tolerance_s=100.0
        )
        len_without = len(ds_without)

        # Restore episode 0
        for f, data in backups.items():
            f.write_bytes(data)

        # Load with all episodes — should have more samples
        ds_with = _load_dataset(
            root=str(tmp_path), episodes=episodes, tolerance_s=100.0
        )
        assert len(ds_with) > len_without

    def test_repeated_evict_reload_cycles(self, tmp_path):
        """Multiple eviction/reload cycles don't corrupt the dataset."""
        episodes = _prefetch_dataset(REPO_ID, tmp_path)

        reference_len = len(
            _load_dataset(root=str(tmp_path), episodes=episodes, tolerance_s=100.0)
        )

        for cycle in range(3):
            # Evict episode `cycle`
            for f in tmp_path.rglob(f"episode_{cycle:06d}.parquet"):
                f.unlink()

            remaining = [e for e in episodes if e != cycle]
            ds = _load_dataset(
                root=str(tmp_path), episodes=remaining, tolerance_s=100.0
            )
            assert len(ds) < reference_len

            # "Re-download" by copying from HF cache
            _prefetch_dataset(REPO_ID, tmp_path)
            ds_full = _load_dataset(
                root=str(tmp_path), episodes=episodes, tolerance_s=100.0
            )
            assert len(ds_full) == reference_len


# ---------------------------------------------------------------------------
# Level 6: Full pipeline batch-level comparison (legacy vs new)
# ---------------------------------------------------------------------------

def _build_legacy_datamodule(
    batch_size=16, num_workers=0, augmentation=None, pin_memory=False,
):
    """Build legacy pipeline matching compare_pipelines.py config."""
    from projects.protoworld.data.legacy_datamodule import LegacyLeRobotDataModule

    return LegacyLeRobotDataModule(
        repo_id=REPO_ID,
        context_length=CONTEXT_LENGTH,
        auto_steps=AUTO_STEPS,
        val_ar_steps=VAL_AR_STEPS,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_frames=False,
        cache_budget_gb=2.0,
        augmentation=augmentation or {},
        dtype_conversions=None,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def _build_new_datamodule(
    batch_size=16, num_workers=0, augmentation=None, pin_memory=False,
    tolerance_s=None, prefetch=False,
):
    """Build new pipeline matching compare_pipelines.py config.

    By default sets prefetch=False so we test using HF cache (same as legacy).
    """
    from projects.protoworld.data.datamodule import LeRobotDataModule

    # To bypass the prefetcher when prefetch=False, pass repo_id as a
    # source dict with prefetch=False. This loads from HF cache, same as
    # legacy, isolating the DataModule/DataLoader wrapper chain.
    if not prefetch:
        repo_id = [{"repo_id": REPO_ID, "weight": 1.0, "prefetch": False}]
    else:
        repo_id = REPO_ID

    return LeRobotDataModule(
        repo_id=repo_id,
        context_length=CONTEXT_LENGTH,
        auto_steps=AUTO_STEPS,
        val_ar_steps=VAL_AR_STEPS,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_frames=False,
        cache_budget_gb=2.0,
        augmentation=augmentation or {},
        dtype_conversions=None,
        pin_memory=pin_memory,
        persistent_workers=False,
        tolerance_s=tolerance_s,
    )


def _get_underlying_indices(dataset):
    """Unwrap AugmentedDataset/KeyFilterDataset to find Subset indices."""
    ds = dataset
    # Walk through wrapper layers looking for Subset (which has .indices)
    while ds is not None:
        if hasattr(ds, 'indices'):
            return list(ds.indices)
        # Try to unwrap: AugmentedDataset/KeyFilterDataset use .dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset
        else:
            break
    return None


def _compare_batches(loader1, loader2, n_batches=50, label1="legacy", label2="new"):
    """Compare first n_batches from two loaders.

    Returns:
        (match, report) where match is True if all batches identical,
        and report is a human-readable string describing the first difference.
    """
    iter1 = iter(loader1)
    iter2 = iter(loader2)

    for batch_idx in range(n_batches):
        try:
            b1 = next(iter1)
        except StopIteration:
            return False, f"Batch {batch_idx}: {label1} exhausted early"
        try:
            b2 = next(iter2)
        except StopIteration:
            return False, f"Batch {batch_idx}: {label2} exhausted early"

        # Compare all keys
        keys1 = set(b1.keys())
        keys2 = set(b2.keys())
        if keys1 != keys2:
            only1 = keys1 - keys2
            only2 = keys2 - keys1
            return False, (
                f"Batch {batch_idx}: key mismatch. "
                f"Only in {label1}: {only1}. Only in {label2}: {only2}."
            )

        for key in sorted(keys1):
            v1, v2 = b1[key], b2[key]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                if v1.shape != v2.shape:
                    return False, (
                        f"Batch {batch_idx}, key='{key}': shape mismatch "
                        f"{v1.shape} vs {v2.shape}"
                    )
                if v1.dtype != v2.dtype:
                    return False, (
                        f"Batch {batch_idx}, key='{key}': dtype mismatch "
                        f"{v1.dtype} vs {v2.dtype}"
                    )
                if not torch.equal(v1, v2):
                    diff = (v1.float() - v2.float()).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    # Find which sample in the batch diverges
                    per_sample = diff.flatten(1).max(dim=1).values
                    first_bad = (per_sample > 0).nonzero(as_tuple=True)[0]
                    first_bad_idx = first_bad[0].item() if len(first_bad) > 0 else -1
                    return False, (
                        f"Batch {batch_idx}, key='{key}': "
                        f"max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                        f"shape={v1.shape}, dtype={v1.dtype}, "
                        f"first divergent sample in batch: {first_bad_idx}"
                    )
            elif v1 is not None and v2 is not None:
                # Skip string/list comparisons (e.g. task descriptions)
                if isinstance(v1, (str, list, tuple)):
                    if v1 != v2:
                        return False, (
                            f"Batch {batch_idx}, key='{key}': "
                            f"non-tensor mismatch: {repr(v1)[:100]} vs {repr(v2)[:100]}"
                        )
                elif isinstance(v1, (int, float)):
                    if v1 != v2:
                        return False, (
                            f"Batch {batch_idx}, key='{key}': "
                            f"scalar mismatch: {v1} vs {v2}"
                        )

    return True, f"All {n_batches} batches identical"


class TestBatchEquivalence:
    """Compare actual training batches between legacy and new pipelines.

    Progressive complexity:
    Step 1: no augmentation, no shuffle, num_workers=0 (fully deterministic)
    Step 2: add shuffle with seed=42
    Step 3: add num_workers=2
    """

    @pytest.fixture(scope="class")
    def legacy_dm(self):
        dm = _build_legacy_datamodule()
        dm.setup("fit")
        yield dm

    @pytest.fixture(scope="class")
    def new_dm(self):
        dm = _build_new_datamodule(prefetch=False)
        dm.setup("fit")
        yield dm
        dm.teardown()

    # ------------------------------------------------------------------
    # Structural checks (before batch iteration)
    # ------------------------------------------------------------------

    def test_train_dataset_length(self, legacy_dm, new_dm):
        """Both pipelines must have the same number of training samples."""
        len_legacy = len(legacy_dm.train_dataset)
        len_new = len(new_dm.train_dataset)
        assert len_legacy == len_new, (
            f"Train dataset length: legacy={len_legacy}, new={len_new}"
        )

    def test_val_dataset_length(self, legacy_dm, new_dm):
        """Both pipelines must have the same number of validation samples."""
        len_legacy = len(legacy_dm.val_dataset)
        len_new = len(new_dm.val_dataset)
        assert len_legacy == len_new, (
            f"Val dataset length: legacy={len_legacy}, new={len_new}"
        )

    def test_train_indices_match(self, legacy_dm, new_dm):
        """Both pipelines must use the same Subset indices for training."""
        idx_legacy = _get_underlying_indices(legacy_dm.train_dataset)
        idx_new = _get_underlying_indices(new_dm.train_dataset)
        assert idx_legacy is not None, "Could not extract legacy indices"
        assert idx_new is not None, "Could not extract new indices"
        assert idx_legacy == idx_new, (
            f"Train indices differ. "
            f"Legacy first 10: {idx_legacy[:10]}, "
            f"New first 10: {idx_new[:10]}"
        )

    def test_val_indices_match(self, legacy_dm, new_dm):
        """Both pipelines must use the same Subset indices for validation."""
        idx_legacy = _get_underlying_indices(legacy_dm.val_dataset)
        idx_new = _get_underlying_indices(new_dm.val_dataset)
        assert idx_legacy is not None, "Could not extract legacy indices"
        assert idx_new is not None, "Could not extract new indices"
        assert idx_legacy == idx_new

    def test_raw_sample_equivalence(self, legacy_dm, new_dm):
        """Same index -> same tensors from both underlying datasets."""
        ds_l = legacy_dm.train_dataset
        ds_n = new_dm.train_dataset
        diffs = []
        for i in range(min(20, len(ds_l))):
            s_l = ds_l[i]
            s_n = ds_n[i]
            for key in s_l:
                if key not in s_n:
                    diffs.append((i, key, f"missing in new"))
                    continue
                v_l, v_n = s_l[key], s_n[key]
                if isinstance(v_l, torch.Tensor) and isinstance(v_n, torch.Tensor):
                    if not torch.equal(v_l, v_n):
                        max_diff = (v_l.float() - v_n.float()).abs().max().item()
                        diffs.append((i, key, f"max_diff={max_diff:.6f}"))
        assert not diffs, (
            "Raw sample content differs:\n"
            + "\n".join(f"  idx={i} key={k}: {d}" for i, k, d in diffs[:10])
        )

    # ------------------------------------------------------------------
    # Step 1: No shuffle, no workers (fully deterministic)
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_step1_no_shuffle_no_workers(self, legacy_dm, new_dm):
        """Fully deterministic: no shuffle, num_workers=0, no augmentation.

        Both pipelines should produce byte-identical batches in the same order.
        """
        # Create DataLoaders with shuffle=False, num_workers=0
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        loader_new = ResumableDataLoader(
            new_dm.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
        )
        assert match, f"Step 1 FAILED: {report}"

    # ------------------------------------------------------------------
    # Step 2: Add shuffle with seed=42
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_step2_shuffle_seeded(self, legacy_dm, new_dm):
        """Add shuffle with seed=42 — should still be deterministic.

        ResumableDataLoader uses ResumableSampler with seed=42 by default.
        """
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        loader_new = ResumableDataLoader(
            new_dm.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
        )
        assert match, f"Step 2 FAILED: {report}"

    # ------------------------------------------------------------------
    # Step 3: Add num_workers=2
    # ------------------------------------------------------------------

    @pytest.mark.slow
    def test_step3_shuffle_with_workers(self, legacy_dm, new_dm):
        """Add num_workers=2 — this is where worker-level non-determinism
        could cause differences.

        Note: with the same sampler seed and shuffle, the INDEX ORDER is
        deterministic. But worker assignment can cause different video
        decode timing. The DATA for each index should still be identical.
        """
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=2,
            drop_last=True,
            pin_memory=False,
            persistent_workers=False,
        )
        loader_new = ResumableDataLoader(
            new_dm.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=2,
            drop_last=True,
            pin_memory=False,
            persistent_workers=False,
        )

        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
        )
        assert match, f"Step 3 FAILED: {report}"


# ---------------------------------------------------------------------------
# Level 7: Pipeline comparison WITH prefetcher (production path)
# ---------------------------------------------------------------------------

class TestBatchEquivalenceWithPrefetcher:
    """Same as Level 6 but uses the prefetcher path for the new pipeline.

    This is the actual production configuration that causes overfitting.
    """

    @pytest.fixture(scope="class")
    def legacy_dm(self):
        dm = _build_legacy_datamodule()
        dm.setup("fit")
        yield dm

    @pytest.fixture(scope="class")
    def new_dm_prefetched(self):
        dm = _build_new_datamodule(prefetch=True, tolerance_s=100.0)
        dm.setup("fit")
        yield dm
        dm.teardown()

    def test_train_dataset_length(self, legacy_dm, new_dm_prefetched):
        len_legacy = len(legacy_dm.train_dataset)
        len_new = len(new_dm_prefetched.train_dataset)
        assert len_legacy == len_new, (
            f"Train dataset length: legacy={len_legacy}, new={len_new}"
        )

    def test_train_indices_match(self, legacy_dm, new_dm_prefetched):
        idx_legacy = _get_underlying_indices(legacy_dm.train_dataset)
        idx_new = _get_underlying_indices(new_dm_prefetched.train_dataset)
        assert idx_legacy is not None, "Could not extract legacy indices"
        assert idx_new is not None, "Could not extract new indices"
        if idx_legacy != idx_new:
            # Find first difference
            for i, (a, b) in enumerate(zip(idx_legacy, idx_new)):
                if a != b:
                    pytest.fail(
                        f"First index difference at position {i}: "
                        f"legacy={a}, new={b}. "
                        f"Legacy has {len(idx_legacy)} indices, "
                        f"new has {len(idx_new)}"
                    )
            if len(idx_legacy) != len(idx_new):
                pytest.fail(
                    f"Index lengths differ: "
                    f"legacy={len(idx_legacy)}, new={len(idx_new)}"
                )

    @pytest.mark.slow
    def test_step1_no_shuffle_no_workers_prefetched(
        self, legacy_dm, new_dm_prefetched,
    ):
        """Fully deterministic comparison with prefetched data."""
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        loader_new = ResumableDataLoader(
            new_dm_prefetched.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
        )
        assert match, f"Prefetched Step 1 FAILED: {report}"

    @pytest.mark.slow
    def test_step2_shuffle_seeded_prefetched(
        self, legacy_dm, new_dm_prefetched,
    ):
        """Shuffled comparison with prefetched data."""
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        loader_new = ResumableDataLoader(
            new_dm_prefetched.train_dataset,
            batch_size=16,
            shuffle=True,
            seed=42,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
        )
        assert match, f"Prefetched Step 2 FAILED: {report}"


# ---------------------------------------------------------------------------
# Level 7b: Root cause — rglob duplicate detection
# ---------------------------------------------------------------------------

class TestPrefetcherDuplicateDetection:
    """Test that the prefetcher's episode scanning produces deduplicated lists.

    ROOT CAUSE: BlendedLeRobotDataModule._start_prefetcher scans for episodes
    using rglob("episode_*.parquet"), which recurses into ALL subdirectories.
    If the prefetch root contains a stale nested copy (e.g., data/data/chunk-000/
    alongside data/chunk-000/), the same episode index appears multiple times.

    When this duplicate list is passed to LeRobotDataset(episodes=[0, 0, 1, 2, 2, ...]),
    get_data_file_path returns the same path for each duplicate, and
    load_dataset(parquet, data_files=[dup, dup, ...]) loads the file twice.
    This inflates the hf_dataset, causing some episodes to appear 2x per epoch.

    The overfitting behavior is explained by this 2x overrepresentation of ~40%
    of episodes, which reduces effective dataset diversity.
    """

    def test_rglob_finds_duplicates_in_nested_dirs(self, tmp_path):
        """Reproduce: rglob finds files in nested data/data/ structure."""
        # Create normal structure
        normal_dir = tmp_path / "data" / "chunk-000"
        normal_dir.mkdir(parents=True)
        (normal_dir / "episode_000000.parquet").touch()
        (normal_dir / "episode_000001.parquet").touch()

        # Create stale nested structure (happens when prefetcher runs twice
        # or snapshot_download is called with different root)
        nested_dir = tmp_path / "data" / "data" / "chunk-000"
        nested_dir.mkdir(parents=True)
        (nested_dir / "episode_000000.parquet").touch()

        # Simulate the prefetcher's scan logic
        import re
        available = sorted(
            int(re.search(r"episode_(\d+)", p.stem).group(1))
            for p in tmp_path.rglob("episode_*.parquet")
            if re.search(r"episode_(\d+)", p.stem)
        )

        # This reproduces the bug: episode 0 appears twice
        assert available == [0, 0, 1], (
            f"Expected [0, 0, 1] (showing the duplicate bug), got {available}"
        )

    def test_dedup_episodes_fixes_the_bug(self, tmp_path):
        """Deduplicating the episode list fixes the inflation."""
        normal_dir = tmp_path / "data" / "chunk-000"
        normal_dir.mkdir(parents=True)
        (normal_dir / "episode_000000.parquet").touch()
        (normal_dir / "episode_000001.parquet").touch()

        nested_dir = tmp_path / "data" / "data" / "chunk-000"
        nested_dir.mkdir(parents=True)
        (nested_dir / "episode_000000.parquet").touch()

        import re
        available = sorted(set(
            int(re.search(r"episode_(\d+)", p.stem).group(1))
            for p in tmp_path.rglob("episode_*.parquet")
            if re.search(r"episode_(\d+)", p.stem)
        ))

        assert available == [0, 1], (
            f"Dedup should give [0, 1], got {available}"
        )

    def test_actual_prefetch_dir_has_duplicates(self):
        """Check the actual /tmp/prefetch/ directory for this bug."""
        import re
        local_dir = Path("/tmp/prefetch/lerobot_pusht")
        if not local_dir.exists():
            pytest.skip("Prefetch directory not present")

        all_files = list(local_dir.rglob("episode_*.parquet"))
        available = sorted(
            int(re.search(r"episode_(\d+)", p.stem).group(1))
            for p in all_files
            if re.search(r"episode_(\d+)", p.stem)
        )
        unique = sorted(set(available))

        if len(available) != len(unique):
            # Check for nested data/data/ directory
            nested = [
                p for p in all_files
                if "data/data/" in str(p) or "data\\data\\" in str(p)
            ]
            pytest.fail(
                f"Prefetch dir has {len(available)} parquet matches but only "
                f"{len(unique)} unique episodes. "
                f"{len(available) - len(unique)} duplicates from nested dirs. "
                f"Found {len(nested)} files in nested data/data/ paths. "
                f"This is the root cause of the pipeline divergence."
            )


# ---------------------------------------------------------------------------
# Level 7c: Verify that dedup fix restores equivalence
# ---------------------------------------------------------------------------

class TestDedupFixRestoresEquivalence:
    """Confirm that deduplicating the episodes list makes the prefetched
    pipeline produce identical data to legacy.

    This is the verification that the identified root cause is the ONLY
    cause of the divergence — not a secondary symptom.
    """

    @pytest.fixture(scope="class")
    def legacy_dm(self):
        dm = _build_legacy_datamodule()
        dm.setup("fit")
        yield dm

    @pytest.fixture(scope="class")
    def new_dm_deduped(self):
        """Build new pipeline with prefetcher, then manually dedup episodes."""
        from projects.protoworld.data.datamodule import LeRobotDataModule

        dm = LeRobotDataModule(
            repo_id="lerobot/pusht",
            context_length=CONTEXT_LENGTH,
            auto_steps=AUTO_STEPS,
            val_ar_steps=VAL_AR_STEPS,
            batch_size=16,
            num_workers=0,
            cache_frames=False,
            augmentation={},
            dtype_conversions=None,
            pin_memory=False,
            persistent_workers=False,
            tolerance_s=100.0,
        )
        # Manually set up by calling the prefetcher but then deduping
        # the _available_episodes before _load_and_split_source runs
        import re

        # Start prefetcher for the source
        for source in dm._sources:
            if "root" not in source and source.get("prefetch", True):
                dm._start_prefetcher(source)
            # DEDUP FIX: remove duplicates from _available_episodes
            if "_available_episodes" in source:
                source["_available_episodes"] = sorted(
                    set(source["_available_episodes"])
                )

        # Now run the rest of setup
        dm.delta_timestamps = dm._common_delta_timestamps(dm.delta_timestamps)
        val_ts = dm._val_delta_timestamps or dm.delta_timestamps
        val_ts = dm._common_delta_timestamps(val_ts)

        full_datasets = []
        for source in dm._sources:
            train_sub, val_idx, full_ds = dm._load_and_split_source(
                source, dm.delta_timestamps
            )
            full_datasets.append((source, val_idx, full_ds))

        # Standard train path (single source, no sampler)
        train_subsets = []
        for source, val_idx, full_ds in full_datasets:
            ep_data_index = full_ds.episode_data_index
            num_episodes = len(ep_data_index["from"])
            train_episodes = int(dm.train_split_ratio * num_episodes)
            train_indices = []
            for ep_idx in range(train_episodes):
                start = int(ep_data_index["from"][ep_idx])
                end = int(ep_data_index["to"][ep_idx])
                train_indices.extend(range(start, end))
            from torch.utils.data import Subset
            train_subsets.append(Subset(full_ds, train_indices))

        dm.train_dataset = train_subsets[0]

        # Val dataset
        val_parts = []
        for source, val_idx, full_ds in full_datasets:
            val_parts.append(Subset(full_ds, val_idx))
        dm.val_dataset = val_parts[0]
        dm._train_sampler = None

        yield dm
        dm.teardown()

    def test_deduped_dataset_length_matches(self, legacy_dm, new_dm_deduped):
        """After dedup, both datasets should have the same length."""
        len_l = len(legacy_dm.train_dataset)
        len_n = len(new_dm_deduped.train_dataset)
        assert len_l == len_n, (
            f"legacy={len_l}, deduped_new={len_n}"
        )

    def test_deduped_hf_dataset_length_matches(self, legacy_dm, new_dm_deduped):
        """The underlying hf_dataset should also match after dedup."""
        ds_l = legacy_dm.train_dataset
        ds_n = new_dm_deduped.train_dataset
        # Unwrap to FastLeRobotDataset
        while hasattr(ds_l, 'dataset'):
            ds_l = ds_l.dataset
        while hasattr(ds_n, 'dataset'):
            ds_n = ds_n.dataset
        len_l = len(ds_l.hf_dataset)
        len_n = len(ds_n.hf_dataset)
        assert len_l == len_n, (
            f"hf_dataset: legacy={len_l}, deduped_new={len_n}"
        )

    @pytest.mark.slow
    def test_deduped_batches_identical(self, legacy_dm, new_dm_deduped):
        """After dedup fix, first 50 batches should be byte-identical."""
        loader_legacy = ResumableDataLoader(
            legacy_dm.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        loader_new = ResumableDataLoader(
            new_dm_deduped.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )
        match, report = _compare_batches(
            loader_legacy, loader_new, n_batches=50,
            label1="legacy", label2="deduped_new",
        )
        assert match, f"Deduped fix did NOT restore equivalence: {report}"


# ---------------------------------------------------------------------------
# Level 8: DataModule train_dataloader() comparison (production DataLoader)
# ---------------------------------------------------------------------------

class TestProductionDataLoaderEquivalence:
    """Compare the actual DataLoaders returned by train_dataloader().

    This tests the full production path including the DataModule's own
    DataLoader construction (which may add samplers, collation, etc.).
    """

    @pytest.fixture(scope="class")
    def legacy_dm(self):
        dm = _build_legacy_datamodule()
        dm.setup("fit")
        yield dm

    @pytest.fixture(scope="class")
    def new_dm(self):
        dm = _build_new_datamodule(prefetch=False)
        dm.setup("fit")
        yield dm
        dm.teardown()

    def test_sampler_type_match(self, legacy_dm, new_dm):
        """Both DataLoaders should use the same sampler type."""
        loader_l = legacy_dm.train_dataloader()
        loader_n = new_dm.train_dataloader()

        sampler_l = type(loader_l.sampler).__name__
        sampler_n = type(loader_n.sampler).__name__
        assert sampler_l == sampler_n, (
            f"Sampler type mismatch: legacy={sampler_l}, new={sampler_n}"
        )

    def test_sampler_seed_match(self, legacy_dm, new_dm):
        """Both samplers should use the same seed."""
        loader_l = legacy_dm.train_dataloader()
        loader_n = new_dm.train_dataloader()

        seed_l = getattr(loader_l.sampler, 'base_seed', None)
        seed_n = getattr(loader_n.sampler, 'base_seed', None)
        assert seed_l == seed_n, (
            f"Sampler seed mismatch: legacy={seed_l}, new={seed_n}"
        )

    def test_drop_last_match(self, legacy_dm, new_dm):
        """Both DataLoaders should have the same drop_last setting."""
        loader_l = legacy_dm.train_dataloader()
        loader_n = new_dm.train_dataloader()
        assert loader_l.drop_last == loader_n.drop_last

    def test_batch_size_match(self, legacy_dm, new_dm):
        """Both DataLoaders should have the same batch size."""
        loader_l = legacy_dm.train_dataloader()
        loader_n = new_dm.train_dataloader()
        assert loader_l.batch_size == loader_n.batch_size

    @pytest.mark.slow
    def test_production_dataloader_batches(self, legacy_dm, new_dm):
        """Compare actual batches from production DataLoaders.

        This is the most realistic test: uses train_dataloader() which
        includes the ResumableDataLoader with its default seed=42 sampler.
        """
        loader_l = legacy_dm.train_dataloader()
        loader_n = new_dm.train_dataloader()

        match, report = _compare_batches(
            loader_l, loader_n, n_batches=50,
            label1="legacy_prod", label2="new_prod",
        )
        assert match, f"Production DataLoader FAILED: {report}"
