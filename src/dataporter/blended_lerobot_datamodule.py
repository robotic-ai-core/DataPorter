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
from pathlib import Path
from typing import Callable

import lightning as L
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from .dataset_wrappers import AugmentedDataset, KeyFilterDataset
from .lerobot_shard_source import LeRobotShardSource
from .resumable import ResumableDataLoader, resolve_num_workers
from .shard_source_val_dataset import ShardSourceValDataset

logger = logging.getLogger(__name__)


def _make_default_split_fn(train_ratio: float) -> Callable[[int], bool]:
    """Build a stable modulo-based split predicate from a train ratio.

    Keeps an episode's train/val assignment constant across dataset
    growth (the anti-pattern the old "first-N" split caused: episodes
    silently migrating from val to train as more became available).

    Rounds the ratio to the nearest 10% bucket — 0.9 → ``e % 10 != 9``,
    0.8 → ``e % 5 != 4``, etc.  Logs a warning for values that don't map
    cleanly; the bug the warning flags is usually a misread config knob.
    """
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(
            f"train_split_ratio must be in (0, 1), got {train_ratio}"
        )
    # Snap to modulo-10 → modulo-2 lattice.
    val_ratio = 1.0 - train_ratio
    for bucket in (10, 5, 4, 2):
        cutoff = int(round(bucket * val_ratio))
        if cutoff >= 1 and abs(cutoff / bucket - val_ratio) < 1e-3:
            val_start = bucket - cutoff
            return lambda e, _b=bucket, _s=val_start: e % _b < _s
    logger.warning(
        f"train_split_ratio={train_ratio!r} doesn't map cleanly to a "
        f"modulo bucket; falling back to ``e % 10 != 9`` (≈90/10). "
        f"Pass an explicit split_fn=lambda e: ... to override."
    )
    return lambda e: e % 10 != 9


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


# Sentinel file written after a successful dataset load. sparkinstance's
# `hf_cache_source.skip_if_present` feature can check for this to decide
# whether to re-run rsync at job startup.
_CACHE_SENTINEL = ".dataporter_cache_complete"


def _layout_is_complete(root: Path) -> bool:
    """True iff ``root`` contains a complete LeRobot v2.1 layout.

    "Complete" means ``meta/info.json`` parses, ``meta/episodes.jsonl``
    exists, and every episode declared by ``info.json:total_episodes``
    has its parquet (and every video) on disk.  Used by
    :meth:`BlendedLeRobotDataModule._start_prefetcher` to skip the
    prefetcher when the directory is already pre-populated (rsync,
    pre-mounted volume, prior run) — avoids spending HF XET 500/5min
    rate-limit budget on etag checks for files we already have.

    Cheap to call: lazy ``LeRobotShardSource`` construction is O(1),
    `list_ready_episodes` is one ``rglob`` + per-episode ``is_file``
    check (~ms at PushT scale, hundreds of ms at 18k+ scale).
    """
    if not (root / "meta" / "info.json").is_file():
        return False
    if not (root / "meta" / "episodes.jsonl").is_file():
        return False
    try:
        from .lerobot_shard_source import LeRobotShardSource
        shard = LeRobotShardSource(root)
        declared = shard.total_episodes
        if declared <= 0:
            return False
        ready = shard.list_ready_episodes()
        # Compare as sorted lists to catch sparse-but-equal-count cases
        # (an unlikely shape but cheap to be exact about).
        return len(ready) >= declared and ready[-1] >= declared - 1
    except Exception:
        return False


def hf_cache_repo_path(repo_id: str) -> Path:
    """Return the canonical HF hub cache directory for a dataset repo_id.

    Uses ``HF_HOME`` if set, else ``~/.cache/huggingface``.

    Example:
        >>> hf_cache_repo_path("lerobot/pusht")
        PosixPath('~/.cache/huggingface/hub/datasets--lerobot--pusht')
    """
    import os
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    safe_name = f"datasets--{repo_id.replace('/', '--')}"
    return hf_home / "hub" / safe_name


def check_hf_cache_populated(repo_id: str) -> tuple[bool, str]:
    """Check if a dataset is present in the HF cache.

    Returns (is_populated, reason) where is_populated is True if the
    cache contains at least one parquet file for this repo.

    This is a pre-flight check — when it returns False on a repo with
    many files (>500), the runtime download is very likely to hit the
    HF XET 500-req/5min per-IP rate limit on Vast shared IPs.
    """
    cache_dir = hf_cache_repo_path(repo_id)
    if not cache_dir.exists():
        return False, f"cache dir does not exist: {cache_dir}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return False, f"no snapshots dir in {cache_dir}"
    parquets = list(snapshots.rglob("*.parquet"))
    if not parquets:
        return False, f"no parquet files under {snapshots}"
    return True, f"{len(parquets)} parquet files in {snapshots}"


def write_cache_sentinel(cache_dir: Path) -> None:
    """Write the DataPorter cache-complete sentinel file.

    Called after a successful dataset load so sparkinstance's
    ``hf_cache_source.skip_if_present`` can detect a healthy cache
    and skip the pre-sync rsync step on subsequent runs.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / _CACHE_SENTINEL).touch()
    except OSError as e:
        logger.debug(f"Could not write cache sentinel in {cache_dir}: {e}")


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
        producer_pool_workers: Decode worker threads inside the
            producer pool child.  Default 4.  Independent from
            ``num_workers`` (DataLoader workers in the parent
            process); raise to scale decode throughput.

    Rotation semantics are flow-balance-driven: the buffer tracks
    actual frames written vs samples drawn (one frame per sample)
    and throttles whichever side races more than one buffer-worth
    ahead of the other.  No tuning knob — the gate is correct for
    any episode length distribution by construction.  See
    :class:`RotationGate` for details.
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
        producer_transform: Callable | None = None,
        prefetch_min_episodes: int = 50,
        prefetch_min_fraction: float | None = None,
        refresh_min_new: int = 0,
        split_fn: Callable[[int], bool] | None = None,
        train_split_ratio: float = 0.9,
        tolerance_s: float | None = None,
        self_refresh_every_n_items: int | None = None,
        nominal_total_frames: int | None = None,
        producer_pool_workers: int = 4,
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
        # Optional producer-side frame transform.  Applied to decoded
        # episode tensors before they land in the ShuffleBuffer; the
        # buffer's shm allocation matches the transform's output shape.
        # Typical use: pass ``ResizeFrames(H, W)`` so the buffer stores
        # training-resolution frames instead of source-resolution
        # frames (74 GB → ~14 GB at 224→96 for capacity=2000).  Any
        # picklable callable works; exposed ``output_shape(input_shape)``
        # lets the DataModule compute the buffer shape without probing.
        self.producer_transform = producer_transform
        self.dtype_conversions = dtype_conversions
        self.train_split_ratio = train_split_ratio
        # Three knobs for the growing-set behaviour.
        # - prefetch_min_episodes: absolute floor for the setup-time gate.
        #   Default 50 = conservative "training can start with some data"
        #   (historic behaviour).  For large datasets (10k+ episodes),
        #   raise this to ~500 so epoch 1 iterates meaningful diversity
        #   instead of the same few episodes.  Small datasets hit the
        #   is_done() short-circuit inside refresh() and settle for
        #   whatever's available.
        # - prefetch_min_fraction: optional size-aware scaler.  When set,
        #   effective gate = max(prefetch_min_episodes, fraction * total).
        #   Portable configs: the same YAML works across dataset sizes.
        #   e.g. ``prefetch_min_fraction=0.1`` on an 18k dataset yields a
        #   1,800-episode setup gate; on a 200-episode dataset it yields
        #   20 (capped by the floor to max(50, 20) = 50).
        # - refresh_min_new: default delta for subsequent refresh() calls
        #   (typically driven by GrowingDatasetCallback).  0 = non-blocking.
        self.prefetch_min_episodes = int(prefetch_min_episodes)
        if prefetch_min_fraction is not None:
            if not (0.0 < prefetch_min_fraction <= 1.0):
                raise ValueError(
                    f"prefetch_min_fraction must be in (0, 1], got "
                    f"{prefetch_min_fraction!r}"
                )
        self.prefetch_min_fraction = prefetch_min_fraction
        self.refresh_min_new = int(refresh_min_new)
        # Self-refresh / pinned-length knobs — forwarded to
        # LeRobotShuffleBufferDataset.  When set together they let the
        # user drop GrowingDatasetCallback and
        # reload_dataloaders_every_n_epochs=1 entirely: each DataLoader
        # worker keeps its own admission map fresh via self.refresh()
        # every N __getitem__ calls, and Lightning sees a stable
        # __len__ so num_training_batches is correct from fit() start.
        self.self_refresh_every_n_items = (
            int(self_refresh_every_n_items)
            if self_refresh_every_n_items is not None else None
        )
        self.nominal_total_frames = (
            int(nominal_total_frames)
            if nominal_total_frames is not None else None
        )
        # Producer pool concurrency — threads inside the spawned
        # pool child that decode episodes in parallel.  Independent
        # from ``num_workers`` (DataLoader workers in the parent).
        # Raise to improve the decode-rate / sample-rate ratio when
        # decode is the bottleneck; watch CPU usage.
        self.producer_pool_workers = int(producer_pool_workers)
        # Split predicate.  Default: 90/10 by modulo-10 on raw episode id.
        # Stable across refreshes; an episode's train/val assignment never
        # changes as the admitted set grows.
        if split_fn is None:
            split_fn = _make_default_split_fn(train_split_ratio)
        self.split_fn = split_fn

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

    def _common_sample_keys(self) -> set[str]:
        """Return keys common to all samples (for collation compatibility)."""
        keys = set(self.delta_timestamps.keys())
        keys.update(["episode_index", "frame_index", "timestamp", "index", "task_index"])
        return keys

    # ------------------------------------------------------------------
    # Key intersection
    # ------------------------------------------------------------------

    def _common_delta_timestamps(self, raw_timestamps: dict) -> dict:
        """Filter delta_timestamps to keys available in ALL sources."""
        if len(self._sources) <= 1:
            return raw_timestamps

        available_per_source = []
        for source in self._sources:
            # Prefer a direct ``info.json`` read via LeRobotShardSource
            # when the source has a local root — avoids the
            # LeRobotDatasetMetadata code path that calls
            # ``get_repo_versions`` (HF API) even for local layouts.
            if "root" in source:
                shard = LeRobotShardSource(source["root"])
                available = set(shard.features.keys())
            else:
                import tempfile
                from lerobot.common.datasets.lerobot_dataset import (
                    LeRobotDatasetMetadata,
                )
                # Probe in a temp dir so metadata-only downloads don't
                # pollute the HF cache (which would leave an empty
                # data/ directory that breaks LeRobotDataset's
                # download logic).
                kwargs = {"root": Path(
                    tempfile.mkdtemp(prefix="lerobot_meta_probe_")
                )}
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

        # Clean stale nested data directories from previous runs.
        # snapshot_download can leave data/data/ alongside data/ which
        # causes load_dataset(data_dir=) to load duplicate parquets,
        # breaking episode boundaries and timestamp validation.
        stale_nested = local_dir / "data" / "data"
        if stale_nested.exists():
            import shutil
            logger.warning(
                f"Removing stale nested directory: {stale_nested}"
            )
            shutil.rmtree(stale_nested, ignore_errors=True)

        # Skip the prefetcher entirely when the local dir is already a
        # complete LeRobot v2.1 layout — sparkinstance ``hf_cache_source``
        # / pre-mounted volumes / a prior run can leave the directory
        # fully populated.  Letting ``snapshot_download`` "verify"
        # anyway hits HF's etag API which counts against the 500/5min
        # XET rate limit — the bug that prompted this guard.
        if _layout_is_complete(local_dir):
            logger.info(
                f"Prefetch dir {local_dir} is already a complete LeRobot "
                f"v2.1 layout for {repo_id}; skipping prefetcher (no HF "
                f"API calls, rate-limit safe)."
            )
            source["root"] = str(local_dir)
            from .lerobot_shard_source import LeRobotShardSource
            shard = LeRobotShardSource(local_dir)
            source["_available_episodes"] = shard.list_ready_episodes()
            return

        prefetcher = LeRobotPrefetcher(
            repo_id=repo_id,
            cache_dir=local_dir,
            min_shards=source.get(
                "prefetch_min_episodes", self.prefetch_min_episodes,
            ),
            max_shards=source.get("prefetch_max_episodes", 10000),
        )
        # Optional size-aware scaling: `prefetch_min_fraction` lifts the
        # effective setup gate to `fraction × total_episodes` when that
        # exceeds the floor.  Triggers a one-shot metadata load via the
        # prefetcher's `total_episodes` property (cheap — just reads
        # info.json from the HF hub cache or issues a meta-only
        # snapshot_download).
        if self.prefetch_min_fraction is not None:
            try:
                total = prefetcher.total_episodes
            except Exception as e:
                logger.warning(
                    f"prefetch_min_fraction requested but metadata "
                    f"load failed ({e}); falling back to "
                    f"prefetch_min_episodes={prefetcher._min_shards}"
                )
            else:
                fraction_gate = int(self.prefetch_min_fraction * total)
                if fraction_gate > prefetcher._min_shards:
                    logger.info(
                        f"prefetch_min_fraction="
                        f"{self.prefetch_min_fraction:.2f}: raising setup "
                        f"gate from {prefetcher._min_shards} to "
                        f"{fraction_gate} "
                        f"({self.prefetch_min_fraction:.0%} of {total})"
                    )
                    prefetcher._min_shards = fraction_gate
        prefetcher.start()
        # 3600s timeout: bulk mode downloads all episodes in one snapshot_download
        # call before signalling ready. With 2000+ files and HF rate-limiting
        # (429 retries up to 3x with exponential backoff), downloads can take
        # 5-20 minutes on a fresh machine. 1 hour is a safe upper bound.
        wait_timeout = source.get("prefetch_wait_timeout_s", 3600.0)
        prefetcher.wait_for_min(timeout=wait_timeout)

        self._prefetchers.append(prefetcher)
        source["root"] = str(local_dir)

        # Use the prefetcher's episode-level readiness predicate (parquet
        # AND video files present) rather than the parquet-only scan —
        # otherwise ProducerPool crashes trying to decode MP4s whose
        # download hasn't completed.
        available_episodes = prefetcher.ready_episodes()
        source["_available_episodes"] = available_episodes
        logger.info(
            f"Prefetcher started for {repo_id} → {local_dir} "
            f"({len(available_episodes)} episodes ready "
            f"with parquet+video)"
        )

    # ------------------------------------------------------------------
    # Source loading
    # ------------------------------------------------------------------

    def _resolve_source_root(self, source: dict) -> Path:
        """Return the filesystem root for a source.

        Reached only when a source dict has neither ``"root"`` nor
        ``prefetch: true`` (the prefetcher would have populated
        ``"root"`` itself in the latter case).  Always raises with an
        actionable message — DataPorter doesn't auto-resolve HF cache
        paths; the caller picks where the data lives.
        """
        if "root" in source:
            return Path(source["root"])
        repo_id = source["repo_id"]
        raise RuntimeError(
            f"Source {repo_id!r} declares neither ``root`` nor "
            f"``prefetch: true`` — DataPorter has nowhere to read the "
            f"dataset from.  Pick one:\n"
            f"  • Set ``root: /path/to/<dataset>/`` if the data is "
            f"already on disk (rsynced, mounted volume, manually "
            f"downloaded).  This is the recommended path for jobs "
            f"with sparkinstance ``hf_cache_source`` / similar pre-"
            f"staging.\n"
            f"  • Omit ``prefetch: false`` (the default ``prefetch: "
            f"true`` runs ``LeRobotPrefetcher`` to download into "
            f"``/tmp/prefetch/<repo>/``).\n"
            f"  • Set ``root`` to the HF cache snapshot path if you "
            f"want to read directly from there: "
            f"``~/.cache/huggingface/hub/datasets--<org>--<repo>/"
            f"snapshots/<rev>/``."
        )

    def _load_and_split_source(
        self, source: dict, delta_timestamps: dict,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], LeRobotShardSource]:
        """Load a single source and split into train/val frame pairs.

        Uses :class:`LeRobotShardSource` (O(1) construction, lazy
        metadata) instead of :class:`FastLeRobotDataset` (heavyweight,
        materializes ``episode_data_index`` and the HuggingFace Arrow
        cache at construction).  For an 18k-episode dataset this drops
        setup time from ~7 minutes to a few seconds.

        Returns:
            ``(train_pairs, val_pairs, shard_source)`` where each pair
            is ``(raw_ep_id, frame_in_ep)``.  Downstream paths either
            dedup pairs to raw episode ids (ShuffleBuffer path,
            episode-level decoding) or feed them straight into
            :class:`ShardSourceValDataset` (val + legacy map-style
            train).
        """
        root = (
            Path(source["root"]) if "root" in source
            else self._resolve_source_root(source)
        )
        shard = LeRobotShardSource(root)

        # Split + pair computation uses RAW episode ids — matches the
        # downstream consumer's contract.  Readiness comes from the
        # shard's disk scan (prefetcher-aware, since the prefetcher is
        # the writer for this root).
        ready_raw_eps = shard.list_ready_episodes()
        # Fall back to ``_available_episodes`` if a caller pre-populated
        # it; matches the legacy kwarg behaviour for external callers
        # still driving the prefetcher themselves.
        override = source.get("_available_episodes")
        if override is not None:
            ready_raw_eps = sorted(set(override))

        train_pairs: list[tuple[int, int]] = []
        val_pairs: list[tuple[int, int]] = []
        for raw_ep in ready_raw_eps:
            n_frames = int(shard.episode_frame_count(raw_ep))
            pairs = [(raw_ep, f) for f in range(n_frames)]
            if self.split_fn(raw_ep):
                train_pairs.extend(pairs)
            else:
                val_pairs.extend(pairs)

        return train_pairs, val_pairs, shard

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

        Consumes the ``(train_pairs, val_pairs, shard)`` shape returned
        by :meth:`_load_and_split_source` — no FastLeRobotDataset
        involvement on this path.
        """
        from .frame_transforms import probe_output_shape
        from .lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset
        from .producer_pool import ProducerConfig, ProducerPool
        from .shuffle_buffer import ShuffleBuffer

        # Probe frame shape from the first source's shard metadata.
        first_shard: LeRobotShardSource = full_datasets[0][3]
        vid_keys = first_shard.video_keys
        if vid_keys:
            # LeRobot info.json stores video shape in one of two
            # conventions across the ecosystem: ``[H, W, C]`` (real
            # hub datasets) or ``[C, H, W]`` (some synthetic fixtures
            # and legacy exports).  The ``names`` field is
            # authoritative when present; otherwise fall back to the
            # small-leading-dim heuristic (channels are 1/3/4).
            feat = first_shard.features.get(vid_keys[0], {})
            shape = list(feat.get("shape", [96, 96, 3]))
            names = feat.get("names")
            if names == ["channels", "height", "width"] or (
                names is None and shape[0] in (1, 3, 4)
            ):
                channels, source_height, source_width = (
                    int(shape[0]), int(shape[1]), int(shape[2]),
                )
            else:
                source_height, source_width, channels = (
                    int(shape[0]), int(shape[1]), int(shape[2]),
                )
        else:
            source_height, source_width, channels = 96, 96, 3

        # Producer-side transform: probe its output shape so the
        # ShuffleBuffer allocates shm at training (post-transform)
        # resolution.
        source_spatial = (channels, source_height, source_width)
        if self.producer_transform is not None:
            out_spatial = probe_output_shape(
                self.producer_transform, (1, *source_spatial),
            )
            _, channels, height, width = out_spatial
            logger.info(
                f"ShuffleBuffer: producer_transform={self.producer_transform!r} "
                f"({source_height}x{source_width} -> {height}x{width})"
            )
        else:
            height, width = source_height, source_width

        producers: list = []
        sources_for_dataset: list[dict] = []
        cumulative_offset = 0
        max_frames = 0
        total_train_samples = 0

        for source, train_pairs, _val_pairs, shard in full_datasets:
            # Unique raw episode ids across the train split for this
            # source.  Pool decodes whole episodes; the consumer samples
            # frames-within-episodes at sampling time.
            train_raw_eps = sorted({raw_ep for raw_ep, _ in train_pairs})

            # Size the ShuffleBuffer's per-slot allocation to the
            # longest episode we'll actually decode.
            for raw_ep in train_raw_eps:
                max_frames = max(
                    max_frames, int(shard.episode_frame_count(raw_ep)),
                )

            total_train_samples += len(train_pairs)

            config = ProducerConfig.from_source(
                source=source,
                shard_source=shard,
                iteration_episodes=train_raw_eps,
                episode_offset=cumulative_offset,
                producer_transform=self.producer_transform,
            )
            producers.append(config)

            transform = self.get_train_transform(source)
            sources_for_dataset.append({
                "shard_source": shard,
                "source_name": source["repo_id"],
                "train_episode_indices": train_raw_eps,
                "episode_offset": cumulative_offset,
                "transform": transform,
            })

            # Advance offset past this source's raw-id space so buffer
            # keys from different sources stay disjoint.
            cumulative_offset += shard.total_episodes + 1

        # Create buffer + pool
        buffer = ShuffleBuffer(
            capacity=self.shuffle_buffer_capacity,
            max_frames=max_frames,
            channels=channels,
            height=height,
            width=width,
        )
        self._producer_pool = ProducerPool(
            buffer, configs=producers,
            total_workers=self.producer_pool_workers,
        )
        self._producer_pool.start()
        logger.info(
            f"ShuffleBuffer pipeline started: "
            f"capacity={self.shuffle_buffer_capacity}, "
            f"max_frames={max_frames}, sources={len(producers)}"
        )
        self._producer_pool.wait_for_warmup()

        self.train_dataset = LeRobotShuffleBufferDataset(
            buffer=buffer,
            sources=sources_for_dataset,
            delta_timestamps=delta_timestamps,
            prefetchers=list(self._prefetchers),
            producer_pool=self._producer_pool,
            split_fn=self.split_fn,
            default_min_new=self.refresh_min_new,
            epoch_length=total_train_samples,
            image_keys=self.get_image_keys(),
            refresh_every_n_items=self.self_refresh_every_n_items,
            nominal_total_frames=self.nominal_total_frames,
        )
        self._train_sampler = None

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

        # Pre-flight HF cache check — warn loudly if the dataset isn't
        # cached, because the runtime download is likely to hit HF's XET
        # 500-req/5min per-IP rate limit on Vast shared IPs.
        for source in self._sources:
            if "root" in source:
                continue  # Using local root, no HF download
            repo_id = source["repo_id"]
            populated, reason = check_hf_cache_populated(repo_id)
            if not populated:
                cache_dir = hf_cache_repo_path(repo_id)
                logger.warning(
                    f"HF cache empty for {repo_id} ({reason}). "
                    f"Runtime download may hit HF XET 500-req/5min per-IP "
                    f"rate limit for repos with >500 files. "
                    f"Recommended: pre-sync the cache via "
                    f"`rsync -az <source>:{cache_dir.parent}/ {cache_dir.parent}/` "
                    f"or configure sparkinstance hf_cache_source in the job config."
                )

        # Start background prefetchers AFTER metadata probe
        for source in self._sources:
            if "root" not in source and source.get("prefetch", True):
                self._start_prefetcher(source)

        # Load all sources and split each into train/val
        full_datasets = []

        for source in self._sources:
            train_ep_idx, val_idx, full_ds = self._load_and_split_source(
                source, self.delta_timestamps
            )
            full_datasets.append((source, train_ep_idx, val_idx, full_ds))
            # Write sentinel so sparkinstance skip_if_present sees a clean cache
            write_cache_sentinel(hf_cache_repo_path(source["repo_id"]))

        # ---- Training dataset ----
        if self.shuffle_buffer_capacity is not None:
            # ShuffleBuffer path: ProducerPool decodes video in background,
            # workers only read from shared memory + HF dataset
            self._setup_shuffle_buffer_training(
                self.delta_timestamps, full_datasets,
            )
        else:
            # Legacy standard path: map-style Dataset + optional
            # WeightedRandomSampler.  Same primitive as val — each
            # source's train frames become a :class:`ShardSourceValDataset`
            # over ``train_pairs``.  No FastLeRobotDataset involvement.
            train_subsets = []
            train_weights = []

            image_keys = self.get_image_keys()
            for source, train_pairs, _val_pairs, shard in full_datasets:
                train_sub: Dataset = ShardSourceValDataset(
                    shard, train_pairs,
                    delta_timestamps=self.delta_timestamps,
                    image_keys=image_keys,
                )

                # Per-source transform
                transform = self.get_train_transform(source)
                if transform is not None:
                    train_sub = AugmentedDataset(train_sub, transform)

                train_subsets.append(train_sub)
                train_weights.append((source["weight"], len(train_sub)))

            # Filter samples to common keys when blending
            common_keys = self._common_sample_keys()
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

        # ---- Validation dataset ----
        # Backed by :class:`ShardSourceValDataset` — same primitive on
        # every path, indexed by ``(raw_ep, frame_in_ep)`` pairs that
        # come straight from :meth:`_load_and_split_source`.  Val uses
        # ``val_ts`` (either the default delta spec or a caller-
        # supplied override via ``val_delta_timestamps``).
        val_parts: list = []
        image_keys = self.get_image_keys()
        for source, _train_pairs, val_pairs, shard in full_datasets:
            val_parts.append(ShardSourceValDataset(
                shard, val_pairs,
                delta_timestamps=val_ts,
                image_keys=image_keys,
            ))

        # Filter val samples to common keys when blending
        common_keys = self._common_sample_keys()
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
