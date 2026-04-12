"""Tests for ShuffleBuffer shared memory across spawn context.

Verifies that share_memory_() tensors survive pickle→unpickle through
mp.get_context("spawn"), and that writes in the child process are
visible to the parent.

This catches the bug where ProducerPool uses spawn but ShuffleBuffer
was designed for fork — child writes to a detached copy, parent sees 0.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest
import torch

from dataporter.shuffle_buffer import ShuffleBuffer


# ---------------------------------------------------------------------------
# Child worker functions (must be top-level for spawn pickling)
# ---------------------------------------------------------------------------

def _child_put_one(buffer: ShuffleBuffer, key: int, value: int) -> None:
    """Write one item to the buffer from a spawned child.

    Frame shape must match buffer's (channels, height, width).
    We read these from the buffer's pre-allocated tensor.
    """
    _, c, h, w = buffer._buffer.shape[1], buffer._buffer.shape[2], buffer._buffer.shape[3], buffer._buffer.shape[4]
    frames = torch.full((1, c, h, w), value, dtype=torch.uint8)
    buffer.put(key, frames)


def _child_put_many(buffer: ShuffleBuffer, n: int) -> None:
    """Write n items to the buffer from a spawned child."""
    _, c, h, w = buffer._buffer.shape[1], buffer._buffer.shape[2], buffer._buffer.shape[3], buffer._buffer.shape[4]
    for i in range(n):
        frames = torch.full((1, c, h, w), i % 256, dtype=torch.uint8)
        buffer.put(i, frames)


def _child_verify_shared(buffer: ShuffleBuffer, result_dict: dict) -> None:
    """Check shared memory state in the child and report back."""
    result_dict["count_is_shared"] = buffer._count.is_shared()
    result_dict["keys_is_shared"] = buffer._keys.is_shared()
    result_dict["buffer_is_shared"] = buffer._buffer.is_shared()
    result_dict["count_ptr"] = buffer._count.data_ptr()
    result_dict["initial_count"] = int(buffer._count)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSharedMemorySurvivesSpawn:
    """Verify share_memory_() tensors work across spawn context."""

    def test_child_write_visible_to_parent(self):
        """Parent creates buffer, child writes, parent reads the write."""
        buf = ShuffleBuffer(
            capacity=10, max_frames=1, channels=3, height=4, width=4,
        )
        assert len(buf) == 0

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_child_put_one, args=(buf, 42, 255))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0, f"Child exited with {p.exitcode}"

        # Parent must see the write
        assert len(buf) == 1, (
            f"Parent sees len={len(buf)} after child put — "
            "shared memory didn't survive spawn"
        )

    def test_multiple_writes_visible(self):
        """Child writes 5 items, parent sees all 5."""
        buf = ShuffleBuffer(
            capacity=10, max_frames=1, channels=3, height=4, width=4,
        )

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_child_put_many, args=(buf, 5))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        assert len(buf) == 5, (
            f"Parent sees len={len(buf)}, expected 5"
        )

    def test_shared_memory_flags_in_child(self):
        """Verify is_shared() is True in the child after unpickling."""
        buf = ShuffleBuffer(
            capacity=5, max_frames=1, channels=3, height=4, width=4,
        )

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result = manager.dict()

        p = ctx.Process(target=_child_verify_shared, args=(buf, result))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        assert result["count_is_shared"], (
            "_count.is_shared() is False in child — "
            "share_memory_() didn't survive spawn pickle"
        )
        assert result["keys_is_shared"], "_keys not shared in child"
        assert result["buffer_is_shared"], "_buffer not shared in child"

    def test_data_content_correct(self):
        """Child writes known data, parent reads back correct values."""
        buf = ShuffleBuffer(
            capacity=10, max_frames=1, channels=3, height=4, width=4,
        )

        ctx = mp.get_context("spawn")
        # Write frame with pixel value 123
        p = ctx.Process(target=_child_put_one, args=(buf, 7, 123))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0

        import random
        key, frames = buf.sample(random.Random(0))
        assert key == 7
        assert frames[0, 0, 0, 0].item() == 123

    def test_large_buffer_survives_spawn(self):
        """A buffer approaching production size survives spawn.

        Production uses capacity=200, max_frames=105, 96×96×3.
        We test with a smaller but non-trivial size to verify
        share_memory_() handles multi-MB allocations.
        """
        # ~50 MB buffer (50 × 50 × 3 × 32 × 32)
        buf = ShuffleBuffer(
            capacity=50, max_frames=50, channels=3, height=32, width=32,
        )

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_child_put_one, args=(buf, 0, 42))
        p.start()
        p.join(timeout=30)
        assert p.exitcode == 0

        assert len(buf) == 1, (
            f"Large buffer: parent sees len={len(buf)} after child put"
        )


class TestDevShmCapacity:
    """/dev/shm size check — shared memory allocation can silently fail
    if /dev/shm is too small (common on Docker/Vast with default 64 MB)."""

    def test_devshm_has_enough_space(self):
        """Check that /dev/shm can hold at least 1 GB.

        Vast.ai defaults to 64 MB which is too small for production
        ShuffleBuffer. Training containers need --shm-size=8g or higher.
        """
        import shutil
        usage = shutil.disk_usage("/dev/shm")
        free_gb = usage.free / 1e9
        if free_gb < 1.0:
            pytest.skip(
                f"/dev/shm has only {free_gb:.1f} GB free — "
                f"production needs 1+ GB. Set --shm-size=8g in Docker."
            )

    def test_buffer_allocation_matches_expected_size(self):
        """Verify the buffer's actual memory footprint."""
        buf = ShuffleBuffer(
            capacity=10, max_frames=5, channels=3, height=8, width=8,
        )
        expected_bytes = 10 * 5 * 3 * 8 * 8  # uint8
        actual_bytes = buf._buffer.nelement() * buf._buffer.element_size()
        assert actual_bytes == expected_bytes
