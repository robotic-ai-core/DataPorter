"""Shared-memory shuffle buffer for video frame data.

Pre-allocated tensor buffer with ``sample()``-based access. Workers never
look up by key — they always get a random item from whatever is in the
buffer. This eliminates cache misses and head-of-line blocking.

Single-writer design: only the ProducerPool process calls ``put()``.
DataLoader workers only call ``sample()`` (read-only). No locks needed
because tensor reads/writes to different slots are independent, and the
single writer serializes all ``put()`` calls.

Usage::

    buffer = ShuffleBuffer(capacity=800, max_frames=30, channels=3, height=96, width=96)
    # Producer process:
    buffer.put(episode_idx, frames_tensor)
    # Worker process:
    key, frames = buffer.sample(rng)
"""

from __future__ import annotations

import random
from typing import Any

import torch


class ShuffleBuffer:
    """Shared-memory ring buffer for video frames.

    Pre-allocates a fixed-size tensor buffer in shared memory. The
    producer fills it with ``put()``, workers read with ``sample()``.

    Args:
        capacity: Max number of items (episodes) in the buffer.
        max_frames: Max frames per episode (padded with zeros).
        channels: Number of image channels.
        height: Frame height.
        width: Frame width.
    """

    def __init__(
        self,
        capacity: int,
        max_frames: int,
        channels: int = 3,
        height: int = 96,
        width: int = 96,
    ):
        self._capacity = capacity
        self._max_frames = max_frames

        # Shared memory tensors — survive fork()
        self._buffer = torch.zeros(
            capacity, max_frames, channels, height, width, dtype=torch.uint8
        ).share_memory_()
        self._lengths = torch.zeros(capacity, dtype=torch.int32).share_memory_()
        self._keys = torch.full((capacity,), -1, dtype=torch.int64).share_memory_()
        self._write_head = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._count = torch.zeros(1, dtype=torch.int64).share_memory_()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return min(int(self._count), self._capacity)

    def __contains__(self, key: int) -> bool:
        return (self._keys == key).any().item()

    def put(self, key: int, frames: torch.Tensor) -> int | None:
        """Write frames to the next slot. Returns evicted key or None.

        SINGLE WRITER ONLY — called exclusively by ProducerPool.
        """
        if isinstance(frames, dict):
            frames = frames["frames"]

        n_frames = min(frames.shape[0], self._max_frames)
        slot = int(self._write_head) % self._capacity

        # Capture evicted key (if slot was occupied)
        evicted = None
        old_key = int(self._keys[slot])
        if old_key >= 0 and int(self._count) >= self._capacity:
            evicted = old_key

        # Write frames
        self._buffer[slot, :n_frames] = frames[:n_frames]
        if n_frames < self._max_frames:
            self._buffer[slot, n_frames:] = 0
        self._lengths[slot] = n_frames
        self._keys[slot] = key
        self._write_head[0] = int(self._write_head) + 1
        self._count[0] = min(int(self._count) + 1, self._capacity)

        return evicted

    def sample(self, rng: random.Random) -> tuple[int, torch.Tensor]:
        """Return (key, frames) for a random occupied slot.

        READ-ONLY — safe to call from any process. Raises IndexError
        if buffer is empty.
        """
        n = len(self)
        if n == 0:
            raise IndexError("ShuffleBuffer is empty")

        # Occupied slots are the most recent `n` slots
        head = int(self._write_head)
        slot = (head - n + rng.randint(0, n - 1)) % self._capacity

        key = int(self._keys[slot])
        n_frames = int(self._lengths[slot])
        frames = self._buffer[slot, :n_frames]

        return key, frames

    def keys(self) -> list[int]:
        """Return list of keys currently in the buffer."""
        n = len(self)
        head = int(self._write_head)
        result = []
        for i in range(n):
            slot = (head - n + i) % self._capacity
            k = int(self._keys[slot])
            if k >= 0:
                result.append(k)
        return result

    def clear(self) -> None:
        self._buffer.fill_(0)
        self._lengths.fill_(0)
        self._keys.fill_(-1)
        self._write_head.fill_(0)
        self._count.fill_(0)
