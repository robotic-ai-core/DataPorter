"""Fast LeRobot Dataset with optimized HuggingFace dataset access.

This module provides FastLeRobotDataset, a subclass of LeRobotDataset that
optimizes HuggingFace dataset access by avoiding the slow `select()` method.

The main bottleneck in LeRobotDataset is hf_dataset.select() which has ~2.6ms
overhead per call. This class replaces it with slice-based access for contiguous
indices (0.2ms) or single-item loops for non-contiguous indices (0.5ms).

Optional frame caching decodes each episode's video ONCE and serves subsequent
requests from memory. An LRU eviction policy keeps memory within a configurable
budget. Frames are stored as uint8 (4x savings vs float32).

Typical speedup: 10x faster data loading compared to base LeRobotDataset.

Requires lerobot: pip install -e external/lerobot
"""

import logging
from collections import OrderedDict

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


class FastLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with optimized HuggingFace dataset access.

    Overrides the slow `_query_hf_dataset` and `_get_query_timestamps`
    methods to use slice-based access instead of `hf_dataset.select()`.

    When ``cache_frames=True``, uses per-worker LRU frame cache that
    decodes each episode once and serves from memory. Compatible with
    multi-worker DataLoader (each worker maintains its own cache).

    Args:
        cache_frames: Enable per-worker frame caching.
        cache_budget_gb: Max memory for frame cache per worker.
        frame_buffer_capacity: Ignored (reserved for future PrefetchedSource).
    """

    def __init__(self, *args, cache_frames: bool = False,
                 cache_budget_gb: float = 2.0,
                 frame_buffer_capacity: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_frames = cache_frames
        self._cache_budget_bytes = int(cache_budget_gb * 1024**3)
        self._frame_cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        self._cache_bytes_used = 0
        if cache_frames:
            logger.info(
                f"FastLeRobotDataset frame cache enabled "
                f"(budget={cache_budget_gb:.1f} GB)"
            )
        logger.info("FastLeRobotDataset initialized with optimized HF access")

    def _is_contiguous(self, indices: list[int]) -> bool:
        """Check if indices form a contiguous range."""
        if len(indices) <= 1:
            return True
        return all(
            indices[i + 1] - indices[i] == 1
            for i in range(len(indices) - 1)
        )

    def _fast_hf_access(self, indices: list[int]) -> dict:
        """Fast access to HuggingFace dataset rows."""
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
        """Query HuggingFace dataset with optimized access."""
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
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """Get query timestamps with optimized HF access."""
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
    # Frame caching (per-worker LRU)
    # ------------------------------------------------------------------

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Query video frames, serving from cache when enabled."""
        if not self._cache_frames:
            return super()._query_videos(query_timestamps, ep_idx)

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
            frames = episode_frames[frame_indices].to(torch.float32) / 255.0
            item[vid_key] = frames
        return item

    def _cache_episode_frames(self, ep_idx: int, vid_key: str) -> None:
        """Decode all frames for an episode and store as uint8."""
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

        # LRU eviction
        while (
            self._cache_bytes_used + entry_bytes > self._cache_budget_bytes
            and self._frame_cache
        ):
            _, evicted = self._frame_cache.popitem(last=False)
            self._cache_bytes_used -= evicted.nelement() * evicted.element_size()

        self._frame_cache[(ep_idx, vid_key)] = frames_uint8
        self._cache_bytes_used += entry_bytes
