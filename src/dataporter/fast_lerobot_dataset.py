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
from collections import OrderedDict
from typing import Iterator

import torch

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import decode_video_frames
except ImportError:
    raise ImportError(
        "FastLeRobotDataset requires lerobot. "
        "Install with: pip install -e external/lerobot"
    )

logger = logging.getLogger(__name__)


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
                    all_ts = [i / dataset.fps for i in range(num_frames)]

                    video_path = (
                        dataset.root
                        / dataset.meta.get_video_file_path(ep_idx, vid_key)
                    )
                    try:
                        all_frames = decode_video_frames(
                            video_path, all_ts,
                            dataset.tolerance_s, dataset.video_backend,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to decode ep {ep_idx}/{vid_key}: {e}")
                        break
                    if all_frames.dim() == 5:
                        all_frames = all_frames.squeeze(0)
                    # Yield single video key frames as uint8
                    yield ep_idx, (all_frames * 255).to(torch.uint8)
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
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_frames = cache_frames
        self._return_uint8 = return_uint8
        self._frame_source = None

        # Shared memory buffer mode (overrides LRU cache)
        if frame_buffer_capacity is not None:
            from .storage import SharedMemoryStorage
            from .prefetched_source import PrefetchedSource

            # Estimate max frames per episode
            ep_lengths = [
                int(self.episode_data_index["to"][i] - self.episode_data_index["from"][i])
                for i in range(len(self.episode_data_index["from"]))
            ]
            max_frames = max(ep_lengths) if ep_lengths else 50

            # Get frame dimensions from first video key metadata
            # PushT: 96x96x3
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
                use_process=True,
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

    def _decode_episode_fallback(self, ep_idx: int) -> torch.Tensor:
        """On-demand fallback: decode frames for an episode."""
        for vid_key in self.meta.video_keys:
            ep_start = self.episode_data_index["from"][ep_idx].item()
            ep_end = self.episode_data_index["to"][ep_idx].item()
            num_frames = ep_end - ep_start
            all_ts = [i / self.fps for i in range(num_frames)]
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            all_frames = decode_video_frames(
                video_path, all_ts, self.tolerance_s, self.video_backend,
            )
            if all_frames.dim() == 5:
                all_frames = all_frames.squeeze(0)
            return (all_frames * 255).to(torch.uint8)

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
        all_ts = [i / self.fps for i in range(num_frames)]

        video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
        all_frames = decode_video_frames(
            video_path, all_ts, self.tolerance_s, self.video_backend,
        )
        if all_frames.dim() == 5:
            all_frames = all_frames.squeeze(0)
        frames_uint8 = (all_frames * 255).to(torch.uint8)
        entry_bytes = frames_uint8.nelement() * frames_uint8.element_size()

        while (
            self._cache_bytes_used + entry_bytes > self._cache_budget_bytes
            and self._frame_cache
        ):
            _, evicted = self._frame_cache.popitem(last=False)
            self._cache_bytes_used -= evicted.nelement() * evicted.element_size()

        self._frame_cache[(ep_idx, vid_key)] = frames_uint8
        self._cache_bytes_used += entry_bytes
