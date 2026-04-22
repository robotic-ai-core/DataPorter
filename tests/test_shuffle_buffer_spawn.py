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

    def test_fail_fast_on_insufficient_shm(self):
        """ShuffleBuffer raises RuntimeError if /dev/shm is too small."""
        from collections import namedtuple
        from unittest.mock import patch

        # Mock /dev/shm with only 1 MB free
        DiskUsage = namedtuple("usage", ["total", "used", "free"])
        fake_usage = DiskUsage(total=1_000_000, used=0, free=1_000_000)
        with patch("shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(RuntimeError, match="shared memory"):
                ShuffleBuffer(
                    capacity=100, max_frames=50, channels=3,
                    height=96, width=96,
                )

    def test_no_error_when_shm_sufficient(self):
        """No error when /dev/shm has enough space."""
        # Small buffer — should always fit
        buf = ShuffleBuffer(
            capacity=2, max_frames=1, channels=3, height=4, width=4,
        )
        assert buf.capacity == 2


class TestVideoPathResolution:
    """Verify that _make_child_decode_fn resolves video symlinks.

    HF hub-cache stores video files as symlinks:
      videos/.../episode_000000.mp4 → ../../blobs/<hash>

    The decode function resolves these so the child process gets a
    concrete file path rather than a symlink chain. Verified locally
    that pyav handles extensionless blob paths correctly.
    """

    def test_decode_fn_resolves_video_path(self, tmp_path):
        """decode_video_frames receives a resolved (non-symlink) path.

        Builds a minimal v2.1 LeRobot layout with the video file as a
        symlink (mirroring HF hub-cache blob storage), then verifies
        the shard-source-backed decode fn resolves the symlink before
        handing the path to pyav.
        """
        import json
        from unittest.mock import patch
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        from dataporter.producer_pool import (
            _make_child_decode_fn, ProducerConfig,
        )

        # Minimal v2.1 meta/ — just enough for LeRobotShardSource to
        # construct and resolve the video path.
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 10,
            "chunks_size": 1000,
            "features": {
                "observation.image": {
                    "dtype": "video", "shape": [3, 96, 96],
                },
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        (meta / "episodes.jsonl").write_text(
            json.dumps({
                "episode_index": 0, "length": 10, "tasks": ["t"],
            }) + "\n"
        )
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "t"}) + "\n"
        )

        # Parquet stub (shard source opens it only if rows are requested;
        # decode_fn doesn't touch it, but the path must exist for
        # readiness checks if anything upstream calls them).
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"frame_index": list(range(10))}),
            data_dir / "episode_000000.parquet",
        )

        # Symlinked video file — hub-cache style.
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        real_video = blobs / "abc123"        # no extension
        real_video.write_bytes(b"fake mp4")
        vid_dir = tmp_path / "videos" / "chunk-000" / "observation.image"
        vid_dir.mkdir(parents=True)
        symlink = vid_dir / "episode_000000.mp4"
        symlink.symlink_to(real_video)

        captured_paths: list[str] = []

        def fake_decode(video_path, *args, **kwargs):
            captured_paths.append(str(video_path))
            return torch.zeros(10, 3, 96, 96)

        config = ProducerConfig(
            source_name="test",
            repo_id="test/repo",
            source_root=str(tmp_path),
            episode_indices=[0],
        )
        decode_fn = _make_child_decode_fn(config)

        with patch(
            "lerobot.common.datasets.video_utils.decode_video_frames",
            side_effect=fake_decode,
        ):
            decode_fn(0)

        assert len(captured_paths) == 1
        decoded_path = Path(captured_paths[0])
        assert not decoded_path.is_symlink(), (
            f"decode_video_frames received a symlink: {decoded_path}. "
            f"Should be resolved to avoid symlink chain issues in "
            f"spawned processes."
        )
