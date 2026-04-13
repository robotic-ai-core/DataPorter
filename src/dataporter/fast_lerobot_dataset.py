"""Fast LeRobot Dataset with optimized HuggingFace dataset access.

Supports two frame caching modes:
1. Per-worker LRU cache (cache_frames=True, default) — each worker caches independently
2. Shared memory buffer (frame_buffer_capacity set) — background producer fills shared
   buffer before fork, workers read zero-copy

Typical speedup: 10x faster data loading compared to base LeRobotDataset.

Requires lerobot: pip install -e external/lerobot
"""

import logging
import random
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import torch

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    raise ImportError(
        "FastLeRobotDataset requires lerobot. "
        "Install with: pip install -e external/lerobot"
    )

logger = logging.getLogger(__name__)

_check_ts_lock = threading.Lock()


def decode_episode_frames(
    video_path: Path,
    num_frames: int,
    fps: float,
    tolerance_s: float,
    video_backend: str,
) -> torch.Tensor:
    """Decode all frames for an episode as uint8 [T, C, H, W]."""
    # Local import so mock patches at the canonical location
    # (lerobot.common.datasets.video_utils.decode_video_frames) take effect.
    from lerobot.common.datasets.video_utils import decode_video_frames as _decode
    all_ts = [i / fps for i in range(num_frames)]
    all_frames = _decode(video_path, all_ts, tolerance_s, video_backend)
    if all_frames.dim() == 5:
        all_frames = all_frames.squeeze(0)
    return (all_frames * 255).to(torch.uint8)


def _make_frame_producer(
    dataset: "FastLeRobotDataset",
    seed: int = 42,
) -> callable:
    """Create a producer that decodes video frames for all episodes.

    Yields (episode_idx, frames_uint8) pairs in random order.
    Cycles through the full dataset.
    """
    def producer() -> Iterator[tuple[int, torch.Tensor]]:
        rng = random.Random(seed)
        num_episodes = len(dataset.episode_data_index["from"])
        episode_order = list(range(num_episodes))

        while True:
            rng.shuffle(episode_order)
            for ep_idx in episode_order:
                for vid_key in dataset.meta.video_keys:
                    ep_start = dataset.episode_data_index["from"][ep_idx].item()
                    ep_end = dataset.episode_data_index["to"][ep_idx].item()
                    num_frames = ep_end - ep_start

                    video_path = (
                        dataset.root
                        / dataset.meta.get_video_file_path(ep_idx, vid_key)
                    )
                    try:
                        frames_uint8 = decode_episode_frames(
                            video_path, num_frames,
                            dataset.fps, dataset.tolerance_s,
                            dataset.video_backend,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to decode ep {ep_idx}/{vid_key}: {e}")
                        break
                    yield ep_idx, frames_uint8
                    break  # Only first video key for now (PushT has one)

    return producer


class FastLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with optimized HuggingFace dataset access.

    Args:
        cache_frames: Enable per-worker LRU frame caching.
        cache_budget_gb: Max memory for LRU cache per worker.
        frame_buffer_capacity: If set, use SharedMemoryStorage with
            background producer instead of per-worker LRU cache.
            Overrides cache_frames/cache_budget_gb.
    """

    def __init__(self, *args, cache_frames: bool = False,
                 cache_budget_gb: float = 2.0,
                 frame_buffer_capacity: int | None = None,
                 return_uint8: bool = False,
                 arrow_cache_path: str | None = None,
                 **kwargs):
        # Stash before super().__init__ — load_hf_dataset() reads this.
        self._arrow_cache_path = arrow_cache_path

        # Replace check_timestamps_sync with a version that reads
        # directly from the Arrow table (0.004s) instead of using the
        # pre-extracted torch.stack(list(...)) arguments (71s for 1M rows).
        # The slow extraction on lines 487-488 still runs, but the patched
        # function ignores those arguments and re-reads from Arrow.
        import lerobot.common.datasets.lerobot_dataset as _ld
        from lerobot.common.datasets.utils import check_timestamps_sync as _real_check
        _orig = _ld.check_timestamps_sync

        def _fast_check(_timestamps, _episode_indices, _ep_data_index,
                        fps, tolerance_s, *args, **kwargs):
            # Ignore the slow pre-extracted args — read Arrow directly
            ts = self.hf_dataset._data.column("timestamp").to_numpy()
            ep = self.hf_dataset._data.column("episode_index").to_numpy()
            ep_idx = {k: t.numpy() for k, t in self.episode_data_index.items()}
            return _real_check(ts, ep, ep_idx, fps, tolerance_s, *args, **kwargs)

        with _check_ts_lock:
            _ld.check_timestamps_sync = _fast_check
            try:
                super().__init__(*args, **kwargs)
            finally:
                _ld.check_timestamps_sync = _orig
        self._cache_frames = cache_frames
        self._return_uint8 = return_uint8
        self._frame_source = None

        # Shared memory buffer mode (DEPRECATED — use ShuffleBuffer pipeline)
        if frame_buffer_capacity is not None:
            import warnings
            warnings.warn(
                "frame_buffer_capacity is deprecated. Use ShuffleBuffer + "
                "ProducerPool + ShuffleBufferDataset instead (see "
                "BlendedLeRobotDataModule with shuffle_buffer_capacity).",
                DeprecationWarning,
                stacklevel=2,
            )
            from .storage import SharedMemoryStorage
            from .prefetched_source import PrefetchedSource

            # Estimate max frames per episode
            ep_lengths = [
                int(self.episode_data_index["to"][i] - self.episode_data_index["from"][i])
                for i in range(len(self.episode_data_index["from"]))
            ]
            max_frames = max(ep_lengths) if ep_lengths else 50

            # Probe frame dimensions from dataset metadata
            vid_keys = self.meta.video_keys
            if vid_keys:
                shape = self.meta.features[vid_keys[0]].get("shape", (96, 96, 3))
                height, width, channels = shape[0], shape[1], shape[2]
            else:
                height, width, channels = 96, 96, 3

            storage = SharedMemoryStorage(
                capacity=frame_buffer_capacity,
                max_frames=max_frames,
                channels=channels,
                height=height,
                width=width,
            )
            producer = _make_frame_producer(self)

            self._frame_source = PrefetchedSource(
                storage=storage,
                producers=[producer],
                shuffle_available=False,
                fallback=self._decode_episode_fallback,
            )
            self._frame_source.start()
            logger.info(
                f"FastLeRobotDataset shared memory buffer started "
                f"(capacity={frame_buffer_capacity}, max_frames={max_frames})"
            )
            self._cache_frames = False  # Disable LRU cache

        # Per-worker LRU cache mode
        if self._cache_frames:
            self._cache_budget_bytes = int(cache_budget_gb * 1024**3)
            self._frame_cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
            self._cache_bytes_used = 0
            logger.info(
                f"FastLeRobotDataset LRU cache enabled "
                f"(budget={cache_budget_gb:.1f} GB)"
            )

        logger.info("FastLeRobotDataset initialized with optimized HF access")

    @property
    def arrow_cache_path(self) -> str | None:
        """Return the Arrow IPC cache file path, if available."""
        cache_files = getattr(self.hf_dataset, "cache_files", None)
        if cache_files and len(cache_files) > 0:
            return cache_files[0].get("filename")
        return None

    def load_hf_dataset(self):
        """Override with two optimizations:

        1. **Arrow cache short-circuit**: when ``arrow_cache_path`` is set
           (spawned ProducerPool child), loads the parent's pre-built Arrow
           IPC file directly via ``Dataset.from_file()``.  Instant — skips
           the 300s rebuild from 10k parquet files.

        2. **Memory-mapped mode**: ``keep_in_memory=False`` forces HF
           datasets to mmap the Arrow IPC cache.  Workers share physical
           pages — zero COW fragmentation (~200 MB vs ~8.6 GB per worker).
        """
        # Fast path: load pre-built Arrow cache from parent process
        arrow_path = getattr(self, "_arrow_cache_path", None)
        if arrow_path is not None:
            from datasets import Dataset
            logger.info(f"Loading Arrow cache: {arrow_path}")
            hf_dataset = Dataset.from_file(arrow_path)
        else:
            # Standard path: load from parquet with mmap
            from datasets import load_dataset

            if self.episodes is None:
                path = str(self.root / "data")
                hf_dataset = load_dataset(
                    "parquet", data_dir=path, split="train",
                    keep_in_memory=False,
                )
            else:
                files = [
                    str(self.root / self.meta.get_data_file_path(ep_idx))
                    for ep_idx in self.episodes
                ]
                hf_dataset = load_dataset(
                    "parquet", data_files=files, split="train",
                    keep_in_memory=False,
                )

        from lerobot.common.datasets.lerobot_dataset import hf_transform_to_torch
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

    def _decode_episode_fallback(self, ep_idx: int) -> torch.Tensor:
        """On-demand fallback: decode frames for an episode."""
        for vid_key in self.meta.video_keys:
            ep_start = self.episode_data_index["from"][ep_idx].item()
            ep_end = self.episode_data_index["to"][ep_idx].item()
            num_frames = ep_end - ep_start
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            return decode_episode_frames(
                video_path, num_frames,
                self.fps, self.tolerance_s, self.video_backend,
            )

    def __del__(self):
        if self._frame_source is not None:
            self._frame_source.stop()

    # ------------------------------------------------------------------
    # Optimized HF access
    # ------------------------------------------------------------------

    def _is_contiguous(self, indices: list[int]) -> bool:
        if len(indices) <= 1:
            return True
        return all(
            indices[i + 1] - indices[i] == 1
            for i in range(len(indices) - 1)
        )

    def _fast_hf_access(self, indices: list[int]) -> dict:
        if not indices:
            return {}
        unique_indices = sorted(set(indices))
        if self._is_contiguous(unique_indices):
            start, end = unique_indices[0], unique_indices[-1] + 1
            data = self.hf_dataset[start:end]
            if len(indices) != len(unique_indices):
                offset = unique_indices[0]
                local_indices = [idx - offset for idx in indices]
                return {
                    key: [val[i] for i in local_indices]
                    for key, val in data.items()
                }
            return data
        else:
            rows = [self.hf_dataset[i] for i in indices]
            return {
                key: [row[key] for row in rows]
                for key in rows[0].keys()
            }

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        for key, indices in query_indices.items():
            if key in self.meta.video_keys:
                continue
            data = self._fast_hf_access(indices)
            values = data[key]
            if isinstance(values, list):
                result[key] = torch.stack(list(values))
            else:
                result[key] = (
                    torch.stack(list(values))
                    if hasattr(values, "__iter__")
                    else values
                )
        return result

    def _get_query_timestamps(
        self, current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                indices = query_indices[key]
                data = self._fast_hf_access(indices)
                timestamps = data["timestamp"]
                if isinstance(timestamps, list):
                    query_timestamps[key] = [float(t) for t in timestamps]
                else:
                    query_timestamps[key] = (
                        torch.stack(list(timestamps)).tolist()
                    )
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    # ------------------------------------------------------------------
    # Frame serving
    # ------------------------------------------------------------------

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Query video frames from shared buffer, LRU cache, or on-demand."""

        # Mode 1: SharedMemoryStorage via PrefetchedSource
        if self._frame_source is not None:
            result = self._frame_source[ep_idx]
            if isinstance(result, dict):
                frames_uint8 = result.get("frames", result)
            else:
                frames_uint8 = result

            item = {}
            for vid_key, query_ts in query_timestamps.items():
                frame_indices = [
                    min(round(ts * self.fps), len(frames_uint8) - 1)
                    for ts in query_ts
                ]
                frames = frames_uint8[frame_indices]
                item[vid_key] = frames if self._return_uint8 else frames.to(torch.float32) / 255.0
            return item

        # Mode 2: Per-worker LRU cache
        if self._cache_frames:
            item = {}
            for vid_key, query_ts in query_timestamps.items():
                cache_key = (ep_idx, vid_key)
                if cache_key in self._frame_cache:
                    self._frame_cache.move_to_end(cache_key)
                else:
                    self._cache_episode_frames(ep_idx, vid_key)

                episode_frames = self._frame_cache[cache_key]
                frame_indices = [
                    min(round(ts * self.fps), len(episode_frames) - 1)
                    for ts in query_ts
                ]
                frames = episode_frames[frame_indices]
                item[vid_key] = frames if self._return_uint8 else frames.to(torch.float32) / 255.0
            return item

        # Mode 3: On-demand (no caching)
        return super()._query_videos(query_timestamps, ep_idx)

    def _cache_episode_frames(self, ep_idx: int, vid_key: str) -> None:
        """Decode all frames for an episode into per-worker LRU cache."""
        ep_start = self.episode_data_index["from"][ep_idx].item()
        ep_end = self.episode_data_index["to"][ep_idx].item()
        num_frames = ep_end - ep_start

        video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
        frames_uint8 = decode_episode_frames(
            video_path, num_frames,
            self.fps, self.tolerance_s, self.video_backend,
        )
        entry_bytes = frames_uint8.nelement() * frames_uint8.element_size()

        while (
            self._cache_bytes_used + entry_bytes > self._cache_budget_bytes
            and self._frame_cache
        ):
            _, evicted = self._frame_cache.popitem(last=False)
            self._cache_bytes_used -= evicted.nelement() * evicted.element_size()

        self._frame_cache[(ep_idx, vid_key)] = frames_uint8
        self._cache_bytes_used += entry_bytes
