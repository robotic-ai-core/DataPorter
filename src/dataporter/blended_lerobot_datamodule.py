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
from .resumable import ResumableDataLoader, resolve_num_workers

logger = logging.getLogger(__name__)


def scan_available_episodes(local_dir: Path) -> list[int]:
    """Scan a prefetch directory for available episode parquet files.

    Uses ``rglob`` to find all ``episode_*.parquet`` files, extracts the
    episode index from each filename, and returns a **deduplicated**,
    sorted list of episode indices.

    Deduplication is critical: stale nested directories (e.g.
    ``data/data/chunk-000/``) can cause ``rglob`` to return the same
    episode index from multiple paths, inflating the dataset.

    Args:
        local_dir: Root directory to scan (the prefetch cache directory).

    Returns:
        Sorted list of unique episode indices found on disk.
    """
    return sorted(set(
        int(m.group(1))
        for p in local_dir.rglob("episode_*.parquet")
        if (m := re.search(r"episode_(\d+)", p.stem))
    ))


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
        frame_buffer_capacity: int | None = None,
        shuffle_buffer_capacity: int | None = None,
        train_split_ratio: float = 0.9,
        tolerance_s: float | None = None,
    ):
        super().__init__()

        # Normalize to list of source dicts
        if isinstance(repo_id, str):
            source = {"repo_id": repo_id, "weight": 1.0}
            if tolerance_s is not None:
                source["tolerance_s"] = tolerance_s
            self._sources = [source]
        else:
            # Per-source tolerance_s in dicts takes precedence over global
            self._sources = [
                {"weight": 1.0, **src}
                | ({"tolerance_s": tolerance_s} if tolerance_s is not None and "tolerance_s" not in src else {})
                for src in repo_id
            ]

        self.repo_id = self._sources[0]["repo_id"]  # backward compat
        self._prefetchers: list = []
        self.cache_frames = cache_frames
        self.cache_budget_gb = cache_budget_gb
        # Mutually exclusive: shuffle_buffer overrides frame_buffer
        if shuffle_buffer_capacity is not None:
            self.frame_buffer_capacity = None
        else:
            self.frame_buffer_capacity = frame_buffer_capacity
        self.shuffle_buffer_capacity = shuffle_buffer_capacity
        self.dtype_conversions = dtype_conversions
        self.train_split_ratio = train_split_ratio

        self.delta_timestamps = dict(delta_timestamps)
        self._val_delta_timestamps = (
            dict(val_delta_timestamps) if val_delta_timestamps else None
        )

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = resolve_num_workers(num_workers)
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.multiprocessing_context = multiprocessing_context

        self._producer_pool = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def get_train_transform(self, source: dict) -> Callable | None:
        """Return training transform for a source. Override in subclass."""
        return None

    def get_val_transform(self) -> Callable | None:
        """Return validation transform. Override in subclass."""
        return None

    def get_image_keys(self) -> list[str]:
        """Return image/video keys used in the dataset. Override in subclass."""
        return ["observation.image"]

    # ------------------------------------------------------------------
    # Key intersection
    # ------------------------------------------------------------------

    def _common_delta_timestamps(self, raw_timestamps: dict) -> dict:
        """Filter delta_timestamps to keys available in ALL sources."""
        if len(self._sources) <= 1:
            return raw_timestamps

        import tempfile
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDatasetMetadata,
        )

        available_per_source = []
        for source in self._sources:
            kwargs = {}
            if "root" in source:
                kwargs["root"] = source["root"]
            else:
                # Probe in a temp dir so metadata-only downloads don't
                # pollute the HF cache (which would leave an empty data/
                # directory that breaks LeRobotDataset's download logic).
                kwargs["root"] = Path(
                    tempfile.mkdtemp(prefix="lerobot_meta_probe_")
                )
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
            cache_dir=local_dir,
            min_shards=min(50, source.get("prefetch_min_episodes", 50)),
            max_shards=source.get("prefetch_max_episodes", 10000),
        )
        prefetcher.start()
        # 3600s timeout: bulk mode downloads all episodes in one snapshot_download
        # call before signalling ready. With 2000+ files and HF rate-limiting
        # (429 retries up to 3x with exponential backoff), downloads can take
        # 5-20 minutes on a fresh machine. 1 hour is a safe upper bound.
        wait_timeout = source.get("prefetch_wait_timeout_s", 3600.0)
        prefetcher.wait_for_min(timeout=wait_timeout)

        self._prefetchers.append(prefetcher)
        source["root"] = str(local_dir)

        available_episodes = scan_available_episodes(local_dir)
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
                # Per-source frame_buffer_capacity overrides global default
                buf_cap = source.get(
                    "frame_buffer_capacity", self.frame_buffer_capacity
                )
                dataset = FastLeRobotDataset(
                    source["repo_id"],
                    cache_frames=self.cache_frames,
                    cache_budget_gb=self.cache_budget_gb,
                    frame_buffer_capacity=buf_cap,
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
    # Shuffle buffer setup
    # ------------------------------------------------------------------

    def _setup_shuffle_buffer_training(
        self, delta_timestamps: dict, full_datasets: list,
    ) -> None:
        """Set up training dataset using ShuffleBuffer pipeline.

        Replaces the standard Subset + WeightedRandomSampler path with:
        1. ShuffleBuffer (pre-allocated shared memory)
        2. ProducerPool (background video decode, weighted scheduling)
        3. LeRobotShuffleBufferDataset (complete samples from buffer)
        """
        from .shuffle_buffer import ShuffleBuffer
        from .producer_pool import ProducerConfig, ProducerPool
        from .lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset

        # Determine max_frames and frame dimensions across all sources
        max_frames = 0
        # Probe frame shape from first source's metadata
        first_ds = full_datasets[0][2]
        vid_keys = first_ds.meta.video_keys
        if vid_keys:
            # LeRobot metadata stores shape as (H, W, C)
            shape = first_ds.meta.features[vid_keys[0]].get("shape", (96, 96, 3))
            height, width, channels = shape[0], shape[1], shape[2]
        else:
            height, width, channels = 96, 96, 3
        producers = []
        sources_for_dataset = []

        for source, val_idx, full_ds in full_datasets:
            ep_data_index = full_ds.episode_data_index
            num_episodes = len(ep_data_index["from"])
            train_episodes = int(self.train_split_ratio * num_episodes)
            train_ep_indices = list(range(train_episodes))

            # Max frames per episode across this source
            for ep_idx in train_ep_indices:
                ep_len = int(
                    ep_data_index["to"][ep_idx]
                    - ep_data_index["from"][ep_idx]
                )
                max_frames = max(max_frames, ep_len)

            # Build ProducerConfig (picklable, for spawn mode)
            config = ProducerConfig(
                source_name=source["repo_id"],
                repo_id=source["repo_id"],
                root=str(full_ds.root),
                episode_indices=train_ep_indices,
                weight=source["weight"],
                tolerance_s=source.get("tolerance_s"),
            )
            producers.append(config)

            # Build source dict for LeRobotShuffleBufferDataset
            transform = self.get_train_transform(source)
            sources_for_dataset.append({
                "dataset": full_ds,
                "train_episode_indices": train_ep_indices,
                "transform": transform,
            })

        # Create buffer + pool
        buffer = ShuffleBuffer(
            capacity=self.shuffle_buffer_capacity,
            max_frames=max_frames,
            channels=channels,
            height=height,
            width=width,
        )
        self._producer_pool = ProducerPool(
            buffer, configs=producers, total_workers=4,
        )
        self._producer_pool.start()
        logger.info(
            f"ShuffleBuffer pipeline started: capacity={self.shuffle_buffer_capacity}, "
            f"max_frames={max_frames}, sources={len(producers)}"
        )
        self._producer_pool.wait_for_warmup()

        # Create training dataset
        # epoch_length: use total training samples across all sources
        # as a reasonable epoch size
        total_train_samples = 0
        for source, val_idx, full_ds in full_datasets:
            ep_data_index = full_ds.episode_data_index
            num_episodes = len(ep_data_index["from"])
            train_episodes = int(self.train_split_ratio * num_episodes)
            for ep_idx in range(train_episodes):
                total_train_samples += int(
                    ep_data_index["to"][ep_idx]
                    - ep_data_index["from"][ep_idx]
                )

        self.train_dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=sources_for_dataset,
            delta_timestamps=delta_timestamps,
            epoch_length=total_train_samples,
            image_keys=self.get_image_keys(),
        )
        self._train_sampler = None  # No sampler needed

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def teardown(self, stage: str | None = None):
        """Stop background prefetchers and producer pool."""
        if self._producer_pool is not None:
            self._producer_pool.stop()
            self._producer_pool = None
        for p in self._prefetchers:
            p.stop()
        self._prefetchers.clear()

    def setup(self, stage: str | None = None):
        # Probe metadata FIRST (lightweight API calls) before heavy downloads
        self.delta_timestamps = self._common_delta_timestamps(
            self.delta_timestamps
        )
        val_ts = self._val_delta_timestamps or self.delta_timestamps
        val_ts = self._common_delta_timestamps(val_ts)

        # Start background prefetchers AFTER metadata probe
        for source in self._sources:
            if "root" not in source and source.get("prefetch", True):
                self._start_prefetcher(source)

        # Load all sources and split each into train/val
        full_datasets = []

        for source in self._sources:
            train_sub, val_idx, full_ds = self._load_and_split_source(
                source, self.delta_timestamps
            )
            full_datasets.append((source, val_idx, full_ds))

        # ---- Training dataset ----
        if self.shuffle_buffer_capacity is not None:
            # ShuffleBuffer path: ProducerPool decodes video in background,
            # workers only read from shared memory + HF dataset
            self._setup_shuffle_buffer_training(
                self.delta_timestamps, full_datasets,
            )
        else:
            # Standard path: Subset + WeightedRandomSampler
            train_subsets = []
            train_weights = []

            for source, val_idx, full_ds in full_datasets:
                # Rebuild train subset (same split as _load_and_split_source)
                ep_data_index = full_ds.episode_data_index
                num_episodes = len(ep_data_index["from"])
                train_episodes = int(self.train_split_ratio * num_episodes)

                train_indices = []
                for ep_idx in range(train_episodes):
                    start = int(ep_data_index["from"][ep_idx])
                    end = int(ep_data_index["to"][ep_idx])
                    train_indices.extend(range(start, end))

                train_sub = Subset(full_ds, train_indices)

                # Per-source transform
                transform = self.get_train_transform(source)
                if transform is not None:
                    train_sub = AugmentedDataset(train_sub, transform)

                train_subsets.append(train_sub)
                train_weights.append((source["weight"], len(train_sub)))

            # Common keys for collator when blending
            common_keys = set(self.delta_timestamps.keys())
            common_keys.update([
                "episode_index", "frame_index", "timestamp", "index",
                "task_index",
            ])

            # Filter samples to common keys when blending
            if len(train_subsets) > 1:
                train_subsets = [
                    KeyFilterDataset(sub, common_keys)
                    for sub in train_subsets
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

        # ---- Validation dataset (always uses old path) ----
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

        # Common keys for collator when blending
        common_keys = set(self.delta_timestamps.keys())
        common_keys.update([
            "episode_index", "frame_index", "timestamp", "index",
            "task_index",
        ])

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
        # ShuffleBuffer mode: workers need per-worker RNG seeding
        if self.shuffle_buffer_capacity is not None:
            from .lerobot_shuffle_buffer_dataset import (
                LeRobotShuffleBufferDataset,
            )
            kwargs["shuffle"] = False  # Sampling is random from buffer
            kwargs["worker_init_fn"] = (
                LeRobotShuffleBufferDataset.worker_init_fn
            )
        dl = ResumableDataLoader(self.train_dataset, **kwargs)
        # Apply pending state from checkpoint (if resuming)
        if hasattr(self, "_pending_dl_state") and self._pending_dl_state is not None:
            dl.load_state_dict(self._pending_dl_state)
            self._pending_dl_state = None
        self._train_dataloader = dl
        return dl

    def val_dataloader(self):
        kwargs = self._build_loader_kwargs(
            self.val_batch_size, shuffle=False
        )
        return ResumableDataLoader(self.val_dataset, **kwargs)

    # ------------------------------------------------------------------
    # State dict for scientific resumption
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Save DataModule state for deterministic resume.

        Saves DataLoader position and frame buffer episode lists
        (so producers can re-decode the same episodes on resume).
        """
        state = {}
        # DataLoader position
        if hasattr(self, "_train_dataloader") and self._train_dataloader is not None:
            state["dataloader"] = self._train_dataloader.state_dict()
        # Frame buffer state (episode indices to re-decode on resume)
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            ds = self.train_dataset
            # Walk through dataset hierarchy to find PrefetchedSource / storage
            source = getattr(ds, "_source", None)
            if source is not None and hasattr(source, "state_dict"):
                state["source"] = source.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore DataModule state from checkpoint."""
        self._pending_dl_state = state.get("dataloader")
        source_state = state.get("source")
        if source_state is not None:
            if hasattr(self, "train_dataset") and self.train_dataset is not None:
                source = getattr(self.train_dataset, "_source", None)
                if source is not None and hasattr(source, "load_state_dict"):
                    source.load_state_dict(source_state)
