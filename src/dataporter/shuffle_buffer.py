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

from ._rotation_gate import RotationGate, SAMPLE_TIMEOUT_S as _SAMPLE_TIMEOUT_S


class ShuffleBuffer:
    """Shared-memory ring buffer for video frames.

    Pre-allocates a fixed-size tensor buffer in shared memory. The
    producer fills it with ``put()``, workers read with ``sample()``.

    Rotation is sample-gated (not time-gated): every ``rotation_per_samples``
    consumer samples allows exactly one producer put at steady state.

    - Slow consumer (K samples/sec < decode throughput): pool-side gate
      blocks the producer until the consumer draws K more samples.  No
      wasted decodes, no CPU burn during training stalls.
    - Fast consumer (K samples/sec > decode throughput): consumer-side
      gate blocks ``sample()`` if the consumer has raced more than
      ``capacity * K`` samples ahead of the pool's puts.  Training
      throughput drops to decode rate, making the decode bottleneck
      visible to the user as low ``train/step_time`` — the UX signal
      to increase ``total_workers``.

    Args:
        capacity: Max number of items (episodes) in the buffer.
        max_frames: Max frames per episode (padded with zeros).
        channels: Number of image channels.
        height: Frame height.
        width: Frame width.
        rotation_per_samples: K — samples consumed per producer put at
            steady state.  ``None`` (default) disables the gate and
            falls back to a light time throttle; tests and direct
            buffer uses without a pool should stick with the default.
            Set to 1 for "replace one slot per sample drawn" —
            equivalent to "no artificial throttle, natural decode-
            rate rotation."  Larger values deliberately slow rotation
            (serve K samples per decoded episode before eviction).
            Production code (e.g. ``BlendedLeRobotDataModule``)
            typically passes an explicit integer here.
    """

    def __init__(
        self,
        capacity: int,
        max_frames: int,
        channels: int,
        height: int,
        width: int,
        rotation_per_samples: int | None = None,
    ):
        self._capacity = capacity
        self._max_frames = max_frames
        # Shared rotation controller (counter + gate logic).  Same
        # instance semantic as TokenShuffleBuffer — both buffers get
        # identical rotation behavior from one code path.
        self._gate = RotationGate(rotation_per_samples)

        # Pre-flight: verify /dev/shm can hold the buffer
        buffer_bytes = capacity * max_frames * channels * height * width
        overhead_bytes = (
            capacity * 4          # _lengths (int32)
            + capacity * 8        # _keys (int64)
            + 8                   # _write_head (int64)
            + 8                   # _count (int64)
        )
        total_bytes = buffer_bytes + overhead_bytes
        self._check_shm_capacity(total_bytes)

        # Shared memory tensors — visible across fork() and spawn()
        self._buffer = torch.zeros(
            capacity, max_frames, channels, height, width, dtype=torch.uint8
        ).share_memory_()
        self._lengths = torch.zeros(capacity, dtype=torch.int32).share_memory_()
        self._keys = torch.full((capacity,), -1, dtype=torch.int64).share_memory_()
        self._write_head = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._count = torch.zeros(1, dtype=torch.int64).share_memory_()

    @staticmethod
    def _check_shm_capacity(required_bytes: int) -> None:
        """Fail fast if /dev/shm can't hold the buffer.

        Docker defaults to 64 MB /dev/shm which silently causes
        share_memory_() to fail for large buffers. On Vast.ai,
        set ``--shm-size=8g`` in the Docker run command.
        """
        import shutil
        from pathlib import Path

        shm = Path("/dev/shm")
        if not shm.exists():
            return  # non-Linux (macOS, Windows) — skip check

        usage = shutil.disk_usage(shm)
        required_gb = required_bytes / 1e9
        free_gb = usage.free / 1e9

        if required_bytes > usage.free:
            raise RuntimeError(
                f"ShuffleBuffer needs {required_gb:.1f} GB shared memory "
                f"but /dev/shm has only {free_gb:.1f} GB free. "
                f"Set --shm-size={max(2, int(required_gb * 1.5))}g in your "
                f"Docker run command (Vast.ai: set SHM size in template)."
            )

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    # Back-compat accessors — the producer pool reads these directly.
    # ``_rotation_k`` and ``_samples_consumed`` are forwarded from the
    # shared :class:`RotationGate` so calling code doesn't need to
    # change when the gate is refactored.
    # ------------------------------------------------------------------

    @property
    def _rotation_k(self) -> int | None:
        return self._gate.rotation_k

    @property
    def _samples_consumed(self):
        return self._gate._samples_consumed

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

        Safe to call from any process.  Raises ``IndexError`` if the
        buffer is empty.  Increments the shared
        ``_samples_consumed`` counter so the producer's rotation gate
        can see consumer progress.

        Consumer-side rotation gate: if the consumer has raced more
        than ``capacity * K`` samples ahead of the pool's puts
        (``K = rotation_per_samples``), this method blocks until the
        pool catches up.  Times out after
        :data:`_SAMPLE_TIMEOUT_S` and raises ``RuntimeError`` — that
        long a block almost always means the pool is dead (decode
        crash, ffmpeg hang, etc.) rather than simply slow.
        """
        n = len(self)
        if n == 0:
            raise IndexError("ShuffleBuffer is empty")

        # Consumer-side gate — delegated to the shared RotationGate.
        self._gate.wait_if_consumer_too_far_ahead(
            write_head_getter=lambda: int(self._write_head),
            capacity=self._capacity,
            buffer_name="ShuffleBuffer",
        )

        # Occupied slots are the most recent `n` slots
        head = int(self._write_head)
        slot = (head - n + rng.randint(0, n - 1)) % self._capacity

        key = int(self._keys[slot])
        n_frames = int(self._lengths[slot])
        frames = self._buffer[slot, :n_frames]

        # Publish the sample to the gate's counter AFTER the read so
        # the producer-side gate only sees committed samples.
        self._gate.record_sample()

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
        self._gate.reset()
