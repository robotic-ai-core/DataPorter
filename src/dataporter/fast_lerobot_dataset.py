"""Fast LeRobot Dataset with optimized HuggingFace dataset access.

This module provides FastLeRobotDataset, a subclass of LeRobotDataset that
optimizes HuggingFace dataset access by avoiding the slow `select()` method.

The main bottleneck in LeRobotDataset is hf_dataset.select() which has ~2.6ms
overhead per call. This class replaces it with slice-based access for contiguous
indices (0.2ms) or single-item loops for non-contiguous indices (0.5ms).

Optional frame prefetching uses PrefetchedSource + MemoryStorage for background
video decode. A background thread decodes episodes ahead of training — zero
cache misses once the buffer is warm.

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
    """Create a producer function that decodes video frames for all episodes.

    Yields (episode_idx, {vid_key: frames_uint8}) pairs in random order.
    Cycles through the full dataset, re-shuffling each pass.
    """
    def producer() -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        rng = random.Random(seed)
        num_episodes = len(dataset.episode_data_index["from"])
        episode_order = list(range(num_episodes))

        while True:  # Cycle indefinitely until stopped
            rng.shuffle(episode_order)
            for ep_idx in episode_order:
                frames = {}
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
                        logger.warning(
                            f"Failed to decode episode {ep_idx}/{vid_key}: {e}"
                        )
                        break
                    if all_frames.dim() == 5:
                        all_frames = all_frames.squeeze(0)
                    frames[vid_key] = (all_frames * 255).to(torch.uint8)
                else:
                    # Only yield if all video keys decoded successfully
                    yield ep_idx, frames

    return producer


class FastLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with optimized HuggingFace dataset access.

    Overrides the slow `_query_hf_dataset` and `_get_query_timestamps`
    methods to use slice-based access instead of `hf_dataset.select()`.

    When ``cache_frames=True``, uses PrefetchedSource with MemoryStorage
    for background video decode. A producer thread decodes episodes ahead
    of training — zero cache misses once the buffer is warm.

    Args:
        cache_frames: Enable background frame prefetching.
        cache_budget_gb: Legacy parameter (ignored, kept for backward compat).
            Buffer capacity is set via ``frame_buffer_capacity``.
        frame_buffer_capacity: Max episodes in the frame buffer. None = all.
    """

    def __init__(self, *args, cache_frames: bool = False,
                 cache_budget_gb: float = 2.0,
                 frame_buffer_capacity: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_frames = cache_frames
        self._frame_source = None

        if cache_frames:
            from .storage import MemoryStorage
            from .prefetched_source import PrefetchedSource

            # Default capacity: all episodes (no eviction)
            if frame_buffer_capacity is None:
                frame_buffer_capacity = len(self.episode_data_index["from"])

            storage = MemoryStorage(capacity=frame_buffer_capacity)
            producer = _make_frame_producer(self)

            self._frame_source = PrefetchedSource(
                storage=storage,
                producers=[producer],
                shuffle_available=False,  # We use direct episode indexing
                fallback=self._decode_episode_frames,
            )
            self._frame_source.start()
            logger.info(
                f"FastLeRobotDataset frame prefetcher started "
                f"(capacity={frame_buffer_capacity} episodes)"
            )

        logger.info("FastLeRobotDataset initialized with optimized HF access")

    def _decode_episode_frames(self, ep_idx: int) -> dict[str, torch.Tensor]:
        """On-demand fallback: decode all frames for an episode."""
        frames = {}
        for vid_key in self.meta.video_keys:
            ep_start = self.episode_data_index["from"][ep_idx].item()
            ep_end = self.episode_data_index["to"][ep_idx].item()
            num_frames = ep_end - ep_start
            all_ts = [i / self.fps for i in range(num_frames)]

            video_path = self.root / self.meta.get_video_file_path(
                ep_idx, vid_key
            )
            all_frames = decode_video_frames(
                video_path, all_ts, self.tolerance_s, self.video_backend,
            )
            if all_frames.dim() == 5:
                all_frames = all_frames.squeeze(0)
            frames[vid_key] = (all_frames * 255).to(torch.uint8)
        return frames

    def __del__(self):
        if self._frame_source is not None:
            self._frame_source.stop()

    def _is_contiguous(self, indices: list[int]) -> bool:
        """Check if indices form a contiguous range."""
        if len(indices) <= 1:
            return True
        return all(
            indices[i + 1] - indices[i] == 1
            for i in range(len(indices) - 1)
        )

    def _fast_hf_access(self, indices: list[int]) -> dict:
        """Fast access to HuggingFace dataset rows.

        Uses slice for contiguous indices (10x faster than select()),
        or single-item loop for non-contiguous (5x faster than select()).
        """
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
    # Frame serving
    # ------------------------------------------------------------------

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Query video frames, serving from prefetch buffer when enabled."""
        if self._frame_source is None:
            return super()._query_videos(query_timestamps, ep_idx)

        # Get frames from prefetch buffer (hit) or fallback (on-demand decode)
        episode_frames = self._frame_source[ep_idx]

        item = {}
        for vid_key, query_ts in query_timestamps.items():
            frames_uint8 = episode_frames[vid_key]
            frame_indices = [
                min(round(ts * self.fps), len(frames_uint8) - 1)
                for ts in query_ts
            ]
            item[vid_key] = frames_uint8[frame_indices].to(torch.float32) / 255.0
        return item
