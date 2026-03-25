"""Blended multi-source LeRobot DataModule.

Orchestrates loading, blending, and serving multiple LeRobot datasets
with weighted sampling, key intersection, per-source transforms,
background prefetching, and per-episode train/val splitting.

Designed as a base class — subclass and override ``get_train_transform()``
and ``get_val_transform()`` to add domain-specific augmentations.

Requires: lerobot, lightning, dataporter
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable

import lightning as L
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler

from .dataset_wrappers import AugmentedDataset, KeyFilterDataset
from .fast_lerobot_dataset import FastLeRobotDataset
from .resumable import ResumableDataLoader

logger = logging.getLogger(__name__)


class BlendedLeRobotDataModule(L.LightningDataModule):
    """Multi-source LeRobot DataModule with weighted blending.

    Handles:
    - Multiple data sources with weighted sampling
    - Per-episode train/val splits (no data leakage)
    - Key intersection across sources (for collation compatibility)
    - Background prefetching for remote datasets
    - Per-source transform routing
    - HF rate limit retry

    Subclass hooks (override for domain-specific behavior):
    - ``get_train_transform(source)`` — per-source training transform
    - ``get_val_transform()`` — validation transform

    Args:
        repo_id: Single repo ID string or list of source dicts with
            ``repo_id``, ``weight``, ``root``, ``tolerance_s``, etc.
        delta_timestamps: Dict mapping keys to timestamp lists for training.
        val_delta_timestamps: Dict for validation. None = same as training.
        batch_size: Training batch size.
        val_batch_size: Validation batch size (defaults to batch_size).
        num_workers: DataLoader worker count.
        prefetch_factor: Batches each worker prefetches.
        persistent_workers: Keep workers alive between epochs.
        pin_memory: Enable pinned memory for faster HtoD transfer.
        pin_memory_device: Device string for pinned memory target.
        multiprocessing_context: Multiprocessing start method.
        dtype_conversions: Dtype conversion rules for DataPorter.
        cache_frames: If True, cache decoded video frames in memory.
        cache_budget_gb: Per-worker memory budget for frame cache in GB.
        train_split_ratio: Fraction of episodes for training (default 0.9).
    """

    def __init__(
        self,
        repo_id: str | list[dict],
        delta_timestamps: dict,
        val_delta_timestamps: dict | None = None,
        batch_size: int = 32,
        val_batch_size: int | None = None,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        pin_memory_device: str | None = None,
        multiprocessing_context: str | None = None,
        dtype_conversions: list[dict] | dict | None = None,
        cache_frames: bool = False,
        cache_budget_gb: float = 2.0,
        train_split_ratio: float = 0.9,
    ):
        super().__init__()

        # Normalize to list of source dicts
        if isinstance(repo_id, str):
            self._sources = [{"repo_id": repo_id, "weight": 1.0}]
        else:
            self._sources = [{"weight": 1.0, **src} for src in repo_id]

        self.repo_id = self._sources[0]["repo_id"]  # backward compat
        self._prefetchers: list = []
        self.cache_frames = cache_frames
        self.cache_budget_gb = cache_budget_gb
        self.dtype_conversions = dtype_conversions
        self.train_split_ratio = train_split_ratio

        self.delta_timestamps = dict(delta_timestamps)
        self._val_delta_timestamps = (
            dict(val_delta_timestamps) if val_delta_timestamps else None
        )

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.multiprocessing_context = multiprocessing_context

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def get_train_transform(self, source: dict) -> Callable | None:
        """Return training transform for a source. Override in subclass."""
        return None

    def get_val_transform(self) -> Callable | None:
        """Return validation transform. Override in subclass."""
        return None

    # ------------------------------------------------------------------
    # Key intersection
    # ------------------------------------------------------------------

    def _common_delta_timestamps(self, raw_timestamps: dict) -> dict:
        """Filter delta_timestamps to keys available in ALL sources."""
        if len(self._sources) <= 1:
            return raw_timestamps

        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDatasetMetadata,
        )

        available_per_source = []
        for source in self._sources:
            kwargs = {}
            if "root" in source:
                kwargs["root"] = source["root"]
            meta = LeRobotDatasetMetadata(source["repo_id"], **kwargs)
            available = set(meta.features.keys())
            available.update(
                k.replace("_path", "") for k in available
                if k.endswith("_path")
            )
            available_per_source.append(available)

        common_keys = set.intersection(*available_per_source)
        filtered = {}
        for key, timestamps in raw_timestamps.items():
            if key in common_keys:
                filtered[key] = timestamps
            else:
                logger.info(
                    f"Dropping delta_timestamps key '{key}' — "
                    f"not available in all {len(self._sources)} sources"
                )
        return filtered

    # ------------------------------------------------------------------
    # Prefetcher
    # ------------------------------------------------------------------

    def _start_prefetcher(self, source: dict) -> None:
        """Start a background prefetcher for a remote dataset source."""
        from .lerobot_prefetcher import LeRobotPrefetcher

        repo_id = source["repo_id"]
        local_dir = Path(f"/tmp/prefetch/{repo_id.replace('/', '_')}")

        prefetcher = LeRobotPrefetcher(
            repo_id=repo_id,
            output_dir=local_dir,
            min_shards=min(50, source.get("prefetch_min_episodes", 50)),
            max_shards=source.get("prefetch_max_episodes", 10000),
            companion_workers=4,
        )
        prefetcher.start()
        prefetcher.wait_for_min()

        self._prefetchers.append(prefetcher)
        source["root"] = str(local_dir)

        available_episodes = sorted(
            int(re.search(r"episode_(\d+)", p.stem).group(1))
            for p in local_dir.rglob("episode_*.parquet")
            if re.search(r"episode_(\d+)", p.stem)
        )
        source["_available_episodes"] = available_episodes
        logger.info(
            f"Prefetcher started for {repo_id} → {local_dir} "
            f"({len(available_episodes)} episodes ready)"
        )

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _load_and_split_source(
        self, source: dict, delta_timestamps: dict,
    ) -> tuple[Subset, list[int], FastLeRobotDataset]:
        """Load a single source dataset and split into train/val."""
        kwargs = {}
        if "root" in source:
            kwargs["root"] = source["root"]
        if "tolerance_s" in source:
            kwargs["tolerance_s"] = source["tolerance_s"]
        if "_available_episodes" in source:
            kwargs["episodes"] = source["_available_episodes"]

        # Retry on HF rate limit (429)
        for attempt in range(3):
            try:
                dataset = FastLeRobotDataset(
                    source["repo_id"],
                    cache_frames=self.cache_frames,
                    cache_budget_gb=self.cache_budget_gb,
                    delta_timestamps=delta_timestamps,
                    **kwargs,
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 310
                    logger.warning(
                        f"HF rate limited loading {source['repo_id']}, "
                        f"retrying in {wait}s (attempt {attempt + 1}/3)"
                    )
                    time.sleep(wait)
                else:
                    raise

        episode_index = dataset.episode_data_index
        num_episodes = len(episode_index["from"])
        train_episodes = int(self.train_split_ratio * num_episodes)

        train_indices = []
        val_indices = []
        for ep_idx in range(num_episodes):
            start = int(episode_index["from"][ep_idx])
            end = int(episode_index["to"][ep_idx])
            if ep_idx < train_episodes:
                train_indices.extend(range(start, end))
            else:
                val_indices.extend(range(start, end))

        train_subset = Subset(dataset, train_indices)
        return train_subset, val_indices, dataset

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def teardown(self, stage: str | None = None):
        """Stop background prefetchers."""
        for p in self._prefetchers:
            p.stop()
        self._prefetchers.clear()

    @staticmethod
    def _patch_hf_rate_limiter():
        """Monkey-patch huggingface_hub to use our rate limiter globally.

        LeRobot's LeRobotDatasetMetadata calls snapshot_download internally,
        bypassing our hf_client. This patches hf_hub_download at the module
        level so ALL HF requests go through the shared token bucket.
        """
        try:
            import huggingface_hub
            import huggingface_hub.file_download
            from .hf_client import get_limiter, _retry_with_backoff

            if not getattr(huggingface_hub, '_rate_limit_patched', False):
                _original = huggingface_hub.hf_hub_download

                def _rate_limited_hf_hub_download(*args, **kwargs):
                    return _retry_with_backoff(_original, *args, **kwargs)

                huggingface_hub.hf_hub_download = _rate_limited_hf_hub_download
                # Also patch the internal module reference used by snapshot_download
                huggingface_hub.file_download.hf_hub_download = _rate_limited_hf_hub_download
                huggingface_hub._rate_limit_patched = True
                logger.info("Patched huggingface_hub.hf_hub_download with rate limiter")
        except Exception as e:
            logger.warning(f"Failed to patch HF rate limiter: {e}")

    def setup(self, stage: str | None = None):
        # Patch HF rate limiter before any HF calls
        self._patch_hf_rate_limiter()

        # Start background prefetchers for remote sources
        for source in self._sources:
            if "root" not in source and source.get("prefetch", True):
                self._start_prefetcher(source)

        # Filter to keys common across all sources
        self.delta_timestamps = self._common_delta_timestamps(
            self.delta_timestamps
        )
        val_ts = self._val_delta_timestamps or self.delta_timestamps
        val_ts = self._common_delta_timestamps(val_ts)

        # Load all sources and split each into train/val
        train_subsets = []
        train_weights = []
        full_datasets = []

        for source in self._sources:
            train_sub, val_idx, full_ds = self._load_and_split_source(
                source, self.delta_timestamps
            )

            # Per-source transform
            transform = self.get_train_transform(source)
            if transform is not None:
                train_sub = AugmentedDataset(train_sub, transform)

            train_subsets.append(train_sub)
            train_weights.append((source["weight"], len(train_sub)))
            full_datasets.append((source, val_idx, full_ds))

        # Common keys for collator when blending
        common_keys = set(self.delta_timestamps.keys())
        common_keys.update([
            "episode_index", "frame_index", "timestamp", "index",
            "task_index",
        ])

        # Filter samples to common keys when blending
        if len(train_subsets) > 1:
            train_subsets = [
                KeyFilterDataset(sub, common_keys) for sub in train_subsets
            ]

        # Combine training subsets
        if len(train_subsets) == 1:
            combined_train = train_subsets[0]
            self._train_sampler = None
        else:
            combined_train = ConcatDataset(train_subsets)
            sample_weights = []
            for weight, count in train_weights:
                sample_weights.extend([weight] * count)
            self._train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(combined_train),
                replacement=True,
            )

        self.train_dataset = combined_train

        # Build validation datasets
        val_parts = []
        for source, val_idx, full_ds in full_datasets:
            if val_ts is not self.delta_timestamps:
                kwargs = {"delta_timestamps": val_ts}
                if "root" in source:
                    kwargs["root"] = source["root"]
                if "tolerance_s" in source:
                    kwargs["tolerance_s"] = source["tolerance_s"]
                val_ds = FastLeRobotDataset(
                    source["repo_id"],
                    cache_frames=self.cache_frames,
                    cache_budget_gb=self.cache_budget_gb,
                    **kwargs,
                )
                val_parts.append(Subset(val_ds, val_idx))
            else:
                val_parts.append(Subset(full_ds, val_idx))

        # Filter val samples to common keys when blending
        if len(val_parts) > 1:
            val_parts = [
                KeyFilterDataset(vp, common_keys) for vp in val_parts
            ]

        combined_val = (
            val_parts[0] if len(val_parts) == 1
            else ConcatDataset(val_parts)
        )

        # Apply validation transform
        val_transform = self.get_val_transform()
        if val_transform is not None:
            combined_val = AugmentedDataset(combined_val, val_transform)

        self.val_dataset = combined_val

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _build_loader_kwargs(self, batch_size: int, shuffle: bool) -> dict:
        """Build common DataLoader keyword arguments."""
        from .converters import KeyBasedDtypeConverter

        kwargs: dict = {
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "shuffle": shuffle,
            "pin_memory": self.pin_memory,
            "drop_last": True,
        }

        if self.pin_memory_device is not None:
            kwargs["pin_memory_device"] = self.pin_memory_device

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                import multiprocessing as mp
                kwargs["multiprocessing_context"] = mp.get_context(
                    self.multiprocessing_context
                )

        if self.dtype_conversions is not None:
            if isinstance(self.dtype_conversions, list):
                kwargs["converter"] = KeyBasedDtypeConverter(
                    self.dtype_conversions
                )
            else:
                kwargs["converter"] = self.dtype_conversions

        return kwargs

    def train_dataloader(self):
        kwargs = self._build_loader_kwargs(self.batch_size, shuffle=True)
        if self._train_sampler is not None:
            kwargs["shuffle"] = False
            kwargs["sampler"] = self._train_sampler
        return ResumableDataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = self._build_loader_kwargs(
            self.val_batch_size, shuffle=False
        )
        return ResumableDataLoader(self.val_dataset, **kwargs)
