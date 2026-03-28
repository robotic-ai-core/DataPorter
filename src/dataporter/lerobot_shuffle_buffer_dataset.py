"""LeRobot-compatible Dataset backed by a ShuffleBuffer.

Extends ShuffleBufferDataset to return complete LeRobot samples:
video frames from the ShuffleBuffer + non-video data (actions, states,
rewards, done flags) from the HuggingFace dataset via delta_timestamps
windowing.

Workers never decode video. The ProducerPool background process fills the
ShuffleBuffer, and workers only call ``sample()`` (shared-memory read)
plus fast HF dataset slicing for non-video columns.

Usage::

    buffer = ShuffleBuffer(capacity=1250, ...)
    pool = ProducerPool(buffer, producers=[...])
    pool.start()
    pool.wait_for_warmup()

    dataset = LeRobotShuffleBufferDataset(
        buffer=buffer,
        sources=[{"dataset": ds, "episode_range": (0, 90), ...}],
        delta_timestamps={"action": [-0.1, 0.0], ...},
        epoch_length=5000,
        image_keys=["observation.image"],
    )
    loader = DataLoader(dataset, batch_size=256, num_workers=4)

Requires: lerobot, dataporter
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .shuffle_buffer import ShuffleBuffer


class LeRobotShuffleBufferDataset(Dataset):
    """LeRobot-compatible dataset backed by ShuffleBuffer.

    On each ``__getitem__`` call:
    1. Samples a random episode from the ShuffleBuffer (uint8 frames)
    2. Maps episode index to the correct source dataset
    3. Picks a random sample index within that episode
    4. Fetches non-video data via delta_timestamps windowing from HF dataset
    5. Extracts the corresponding frame window from the sampled frames
    6. Applies per-source transform if configured
    7. Returns a complete sample dict matching LeRobotDataset output format

    Args:
        buffer: ShuffleBuffer to sample from.
        sources: List of source dicts, each containing:
            - ``dataset``: FastLeRobotDataset instance
            - ``train_episode_indices``: list of episode indices (training split)
            - ``transform``: optional per-source transform callable
        delta_timestamps: Temporal windowing config (key -> list of delta times).
        epoch_length: Synthetic dataset length (controls epoch frequency).
        image_keys: List of video/image keys in the dataset.
        seed: Random seed for per-worker RNG.
    """

    def __init__(
        self,
        buffer: ShuffleBuffer,
        sources: list[dict],
        delta_timestamps: dict,
        epoch_length: int,
        image_keys: list[str] | None = None,
        seed: int = 42,
    ):
        self._buffer = buffer
        self._sources = sources
        self._delta_timestamps = delta_timestamps
        self._epoch_length = epoch_length
        self._image_keys = image_keys or ["observation.image"]
        self._seed = seed
        self._rng = random.Random(seed)

        # Build episode -> source lookup table
        # Each source's train_episode_indices are the episode indices from
        # the original dataset. We build a flat sorted list of
        # (episode_idx, source_idx) for O(log n) lookup.
        self._ep_to_source: dict[int, int] = {}
        for src_idx, src in enumerate(self._sources):
            for ep_idx in src["train_episode_indices"]:
                self._ep_to_source[ep_idx] = src_idx

        # Pre-compute delta_indices (frame offsets from delta_timestamps)
        # Same logic as lerobot's get_delta_indices
        self._delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps:
            # Need fps from first source (all sources should share fps)
            fps = self._sources[0]["dataset"].fps
            self._delta_indices = {
                key: [round(d * fps) for d in deltas]
                for key, deltas in delta_timestamps.items()
            }
            self._fps = fps

    def __len__(self) -> int:
        return self._epoch_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        """Return a complete LeRobot sample from the buffer.

        ``idx`` is ignored -- every call samples uniformly from the buffer.
        """
        # 1. Sample random episode from buffer
        ep_idx, frames_uint8 = self._buffer.sample(self._rng)

        # 2. Map to source dataset (retry up to 10 times on miss)
        src_idx = self._ep_to_source.get(ep_idx)
        _retries = getattr(self, "_retry_count", 0)
        if src_idx is None:
            if _retries >= 10:
                self._retry_count = 0
                raise RuntimeError(
                    f"Episode {ep_idx} not in any source after 10 retries. "
                    f"Buffer may contain stale episodes."
                )
            self._retry_count = _retries + 1
            ep_idx, frames_uint8 = self._buffer.sample(self._rng)
            src_idx = self._ep_to_source.get(ep_idx)
            if src_idx is None:
                self._retry_count = 0
                raise RuntimeError(
                    f"Episode {ep_idx} not in any source. "
                    f"Known episodes: {sorted(self._ep_to_source.keys())[:10]}..."
                )
        self._retry_count = 0

        source = self._sources[src_idx]
        dataset = source["dataset"]

        # 3. Pick random sample index within episode
        ep_data_index = dataset.episode_data_index
        ep_start = int(ep_data_index["from"][ep_idx])
        ep_end = int(ep_data_index["to"][ep_idx])
        num_frames_in_ep = ep_end - ep_start

        sample_idx = ep_start + self._rng.randint(0, num_frames_in_ep - 1)

        # 4. Fetch non-video data from HF dataset
        item = dataset.hf_dataset[sample_idx]

        # Apply delta_timestamps windowing for non-video keys
        if self._delta_indices is not None:
            query_indices = {}
            padding = {}
            for key, delta_idx in self._delta_indices.items():
                indices = [
                    max(ep_start, min(ep_end - 1, sample_idx + d))
                    for d in delta_idx
                ]
                query_indices[key] = indices
                padding[f"{key}_is_pad"] = torch.BoolTensor([
                    (sample_idx + d < ep_start) or (sample_idx + d >= ep_end)
                    for d in delta_idx
                ])

            item = {**item, **padding}

            # Fetch windowed non-video data
            query_result = dataset._query_hf_dataset(query_indices)
            for key, val in query_result.items():
                item[key] = val

        # 5. Extract video frame window from sampled frames
        frame_offset_in_ep = sample_idx - ep_start

        for vid_key in self._image_keys:
            if self._delta_indices is not None and vid_key in self._delta_indices:
                # Get frame indices relative to episode start
                frame_indices = []
                for d in self._delta_indices[vid_key]:
                    abs_idx = max(ep_start, min(ep_end - 1, sample_idx + d))
                    rel_idx = abs_idx - ep_start
                    # Clamp to available frames in buffer
                    rel_idx = min(rel_idx, len(frames_uint8) - 1)
                    frame_indices.append(rel_idx)
                item[vid_key] = frames_uint8[frame_indices].to(torch.float32) / 255.0
            else:
                # Single frame at current position
                rel_idx = min(frame_offset_in_ep, len(frames_uint8) - 1)
                item[vid_key] = (
                    frames_uint8[rel_idx].unsqueeze(0).to(torch.float32) / 255.0
                )

        # Add task string
        task_idx = item["task_index"].item() if hasattr(item["task_index"], "item") else int(item["task_index"])
        item["task"] = dataset.meta.tasks[task_idx]

        # 6. Apply per-source transform
        transform = source.get("transform")
        if transform is not None:
            item = transform(item)

        return item

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Seed per-worker RNG for diverse sampling across workers.

        Must be passed as ``worker_init_fn`` to DataLoader to avoid
        correlated sampling across forked workers.
        """
        import torch as _torch

        info = _torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "_rng"):
            import random as _random
            info.dataset._rng = _random.Random(info.seed)
