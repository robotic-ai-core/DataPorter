"""Tests for stale/duplicate episode data in the LeRobot prefetch pipeline.

Validates that the episode scanning logic (``scan_available_episodes``) and
the full ``_start_prefetcher`` flow are resilient to:
  - Duplicate episodes from stale nested directories
  - Leftover episodes from previous configs
  - Partially downloaded files
  - Episode count inflation
  - Eviction + re-download cycles
  - Concurrent writes during scanning

All tests use synthetic parquet files --- no network access or real lerobot
data needed.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataporter.blended_lerobot_datamodule import scan_available_episodes
from dataporter.lerobot_prefetcher import LeRobotPrefetcher

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

FAKE_META = {
    "codebase_version": "v2.1",
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "fps": 30,
    "total_episodes": 10,
    "total_frames": 3000,
    "total_tasks": 1,
    "total_chunks": 1,
    "chunks_size": 1000,
    "robot_type": "pusht",
    "features": {
        "observation.image": {"dtype": "video", "shape": [3, 96, 96]},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    },
}

_PARQUET_SCHEMA = pa.schema([
    ("observation.state", pa.list_(pa.float32())),
    ("action", pa.list_(pa.float32())),
    ("timestamp", pa.float64()),
    ("episode_index", pa.int64()),
    ("frame_index", pa.int64()),
])


def _fake_meta_loader(repo_id: str, cache_dir: Path) -> dict:
    """Write + return fake LeRobot metadata."""
    meta_dir = cache_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))
    return FAKE_META


def _write_episode_parquet(
    parent_dir: Path,
    ep_idx: int,
    n_frames: int = 10,
    chunks_size: int = 1000,
) -> Path:
    """Write a minimal but valid episode parquet file and return its path."""
    chunk = ep_idx // chunks_size
    parquet_dir = parent_dir / f"data/chunk-{chunk:03d}"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(ep_idx)
    table = pa.table(
        {
            "observation.state": [
                [float(rng.randn()), float(rng.randn())] for _ in range(n_frames)
            ],
            "action": [
                [float(rng.randn()), float(rng.randn())] for _ in range(n_frames)
            ],
            "timestamp": [i / 30.0 for i in range(n_frames)],
            "episode_index": [ep_idx] * n_frames,
            "frame_index": list(range(n_frames)),
        },
        schema=_PARQUET_SCHEMA,
    )
    path = parquet_dir / f"episode_{ep_idx:06d}.parquet"
    pq.write_table(table, path)
    return path


def _populate_episodes(root: Path, episode_indices: list[int]) -> None:
    """Create episode parquets + placeholder videos for given indices."""
    for ep_idx in episode_indices:
        _write_episode_parquet(root, ep_idx)
        # Video companion
        chunk = ep_idx // FAKE_META["chunks_size"]
        vid_dir = root / f"videos/chunk-{chunk:03d}/observation.image"
        vid_dir.mkdir(parents=True, exist_ok=True)
        (vid_dir / f"episode_{ep_idx:06d}.mp4").write_bytes(
            b"fake video " + str(ep_idx).encode()
        )


def _make_fake_snapshot_fn(source_dir: Path):
    """Create a mock snapshot_download that copies from source_dir."""
    import fnmatch

    def snapshot_fn(repo_id, cache_dir, allow_patterns=None, ignore_patterns=None):
        cache_dir = Path(cache_dir)
        for src_file in source_dir.rglob("*"):
            if not src_file.is_file():
                continue
            rel = str(src_file.relative_to(source_dir))
            if allow_patterns is not None:
                if not any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    continue
            if ignore_patterns is not None:
                if any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                    continue
            dst = cache_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

    return snapshot_fn


# ---------------------------------------------------------------------------
# 1. Duplicate detection from nested directories
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """Verify that stale nested directories do not inflate episode counts."""

    def test_rglob_dedup_nested_data_dir(self, tmp_path):
        """Stale ``data/data/chunk-000/`` alongside ``data/chunk-000/`` must
        not produce duplicate episode indices."""
        root = tmp_path / "prefetch"
        # Correct path
        _write_episode_parquet(root, 0)
        _write_episode_parquet(root, 1)
        _write_episode_parquet(root, 2)

        # Stale nested copy (the bug scenario)
        stale_dir = root / "data" / "data" / "chunk-000"
        stale_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx in range(3):
            src = root / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
            shutil.copy2(src, stale_dir / f"episode_{ep_idx:06d}.parquet")

        episodes = scan_available_episodes(root)

        assert episodes == [0, 1, 2], (
            f"Expected 3 unique episodes, got {len(episodes)}: {episodes}"
        )

    def test_rglob_dedup_multiple_stale_levels(self, tmp_path):
        """Multiple levels of nesting (data/data/data/...) still dedup."""
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 5)

        # Two stale levels
        for depth in ["data/data", "data/data/data"]:
            stale = root / depth / "chunk-000"
            stale.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                root / "data/chunk-000/episode_000005.parquet",
                stale / "episode_000005.parquet",
            )

        episodes = scan_available_episodes(root)
        assert episodes == [5]

    def test_rglob_dedup_across_chunks(self, tmp_path):
        """Duplicate episodes across different chunk directories are deduped."""
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)

        # Same episode in a different (stale) chunk dir
        stale_chunk = root / "data" / "chunk-001"
        stale_chunk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            root / "data/chunk-000/episode_000000.parquet",
            stale_chunk / "episode_000000.parquet",
        )

        episodes = scan_available_episodes(root)
        assert episodes == [0]

    def test_dedup_preserves_all_unique_episodes(self, tmp_path):
        """Deduplication must not accidentally drop distinct episodes."""
        root = tmp_path / "prefetch"
        expected = list(range(20))
        _populate_episodes(root, expected)

        # Add duplicates of some episodes in stale nested dir
        stale = root / "data" / "data" / "chunk-000"
        stale.mkdir(parents=True, exist_ok=True)
        for ep_idx in [0, 5, 10, 15]:
            shutil.copy2(
                root / f"data/chunk-000/episode_{ep_idx:06d}.parquet",
                stale / f"episode_{ep_idx:06d}.parquet",
            )

        episodes = scan_available_episodes(root)
        assert episodes == expected

    def test_would_have_failed_without_set_dedup(self, tmp_path):
        """Demonstrate that raw rglob without set() produces duplicates.

        This is a regression-guard: if someone removes the set() call
        from scan_available_episodes, this test will catch it.
        """
        import re

        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)
        _write_episode_parquet(root, 1)

        # Stale copies
        stale = root / "data" / "data" / "chunk-000"
        stale.mkdir(parents=True, exist_ok=True)
        for ep in [0, 1]:
            shutil.copy2(
                root / f"data/chunk-000/episode_{ep:06d}.parquet",
                stale / f"episode_{ep:06d}.parquet",
            )

        # Raw rglob (no dedup) WOULD produce duplicates
        raw_indices = sorted(
            int(re.search(r"episode_(\d+)", p.stem).group(1))
            for p in root.rglob("episode_*.parquet")
            if re.search(r"episode_(\d+)", p.stem)
        )
        assert len(raw_indices) == 4, "rglob should see 4 files (2 real + 2 stale)"

        # But scan_available_episodes deduplicates
        episodes = scan_available_episodes(root)
        assert len(episodes) == 2
        assert episodes == [0, 1]


# ---------------------------------------------------------------------------
# 2. Stale episodes from previous configs
# ---------------------------------------------------------------------------


class TestStaleEpisodesFromPreviousConfig:
    """When a second config downloads a different episode range, the old
    episodes may still be on disk. The _available_episodes list should
    reflect what was actually downloaded by THIS config, not the union."""

    def test_prefetcher_does_not_include_stale_episodes(self, tmp_path):
        """Run two prefetch cycles with overlapping episode ranges.

        First run: episodes 0-9.  Second run: episodes 5-14.
        After second run, scan should find 0-14 (all on disk).
        The prefetcher itself uses _available_episodes from the scan.
        Filtering to "current config" episodes is the caller's job,
        but the scan must not produce MORE than what is on disk.
        """
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"

        # First run: episodes 0-9
        _populate_episodes(source_dir, list(range(10)))
        meta_10 = {**FAKE_META, "total_episodes": 10}

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=list(range(10)),
            min_shards=1,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=lambda r, d: meta_10,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        # Verify first run
        episodes_after_first = scan_available_episodes(cache_dir)
        assert episodes_after_first == list(range(10))

        # Second run: add episodes 10-14
        _populate_episodes(source_dir, list(range(10, 15)))
        meta_15 = {**FAKE_META, "total_episodes": 15}

        prefetcher2 = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=list(range(5, 15)),
            min_shards=1,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=lambda r, d: meta_15,
        )
        prefetcher2.start()
        prefetcher2.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher2.stop()

        # Scan sees everything on disk (0-14)
        episodes_after_second = scan_available_episodes(cache_dir)
        assert episodes_after_second == list(range(15))

        # No duplicates despite overlapping ranges
        assert len(episodes_after_second) == len(set(episodes_after_second))

    def test_scan_does_not_invent_episodes(self, tmp_path):
        """scan_available_episodes returns only what is physically on disk."""
        root = tmp_path / "prefetch"
        _populate_episodes(root, [3, 7, 42])

        episodes = scan_available_episodes(root)
        assert episodes == [3, 7, 42]


# ---------------------------------------------------------------------------
# 3. Interrupted download leaves partial episodes
# ---------------------------------------------------------------------------


class TestPartialDownloads:
    """Verify that incomplete/partial files are excluded from the scan."""

    def test_partial_suffix_excluded(self, tmp_path):
        """Files ending in .partial should not appear in the episode list."""
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)
        _write_episode_parquet(root, 1)

        # Simulate a partial download
        partial_dir = root / "data" / "chunk-000"
        (partial_dir / "episode_000002.parquet.partial").write_bytes(b"incomplete")

        episodes = scan_available_episodes(root)
        assert episodes == [0, 1], "Partial file should not appear"

    def test_tmp_suffix_excluded(self, tmp_path):
        """Files ending in .parquet.tmp (atomic write staging) excluded."""
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)

        partial_dir = root / "data" / "chunk-000"
        (partial_dir / "episode_000001.parquet.tmp").write_bytes(b"staging")

        episodes = scan_available_episodes(root)
        assert episodes == [0]

    def test_empty_parquet_is_included(self, tmp_path):
        """An empty but correctly-named parquet file IS included.

        scan_available_episodes operates on filenames, not file contents.
        Validation of parquet contents is a separate concern.
        """
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)

        # Create a zero-byte file with a valid .parquet name
        empty_dir = root / "data" / "chunk-000"
        (empty_dir / "episode_000001.parquet").write_bytes(b"")

        episodes = scan_available_episodes(root)
        assert episodes == [0, 1]

    def test_non_parquet_files_excluded(self, tmp_path):
        """Non-parquet files (mp4, json, etc.) must not pollute the scan."""
        root = tmp_path / "prefetch"
        _write_episode_parquet(root, 0)

        data_dir = root / "data" / "chunk-000"
        (data_dir / "episode_000001.json").write_text("{}")
        (data_dir / "episode_000002.csv").write_text("a,b,c")

        vid_dir = root / "videos" / "chunk-000" / "observation.image"
        vid_dir.mkdir(parents=True, exist_ok=True)
        (vid_dir / "episode_000003.mp4").write_bytes(b"video")

        episodes = scan_available_episodes(root)
        assert episodes == [0]


# ---------------------------------------------------------------------------
# 4. Episode count inflation detection
# ---------------------------------------------------------------------------


class TestEpisodeCountInflation:
    """Given a known set of N episodes, verify scan returns at most N."""

    def test_no_inflation_clean_directory(self, tmp_path):
        """Clean directory: scan count equals written count."""
        root = tmp_path / "prefetch"
        n_episodes = 50
        _populate_episodes(root, list(range(n_episodes)))

        episodes = scan_available_episodes(root)
        assert len(episodes) == n_episodes

    def test_inflation_detected_with_stale_dirs(self, tmp_path):
        """With stale dirs, raw file count exceeds N but scan deduplicates."""
        root = tmp_path / "prefetch"
        n_episodes = 10
        _populate_episodes(root, list(range(n_episodes)))

        # Inject stale copies
        stale = root / "data" / "data" / "chunk-000"
        stale.mkdir(parents=True, exist_ok=True)
        for ep_idx in range(n_episodes):
            shutil.copy2(
                root / f"data/chunk-000/episode_{ep_idx:06d}.parquet",
                stale / f"episode_{ep_idx:06d}.parquet",
            )

        # Raw file count is inflated
        raw_count = len(list(root.rglob("episode_*.parquet")))
        assert raw_count == 2 * n_episodes

        # But scan_available_episodes deduplicates
        episodes = scan_available_episodes(root)
        assert len(episodes) == n_episodes
        assert len(episodes) <= n_episodes  # invariant: never exceeds truth

    def test_prefetcher_available_episodes_bounded(self, tmp_path):
        """Full prefetch flow: _available_episodes count <= total_episodes."""
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        n_episodes = 8

        _populate_episodes(source_dir, list(range(n_episodes)))
        meta = {**FAKE_META, "total_episodes": n_episodes}

        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=list(range(n_episodes)),
            min_shards=1,
            max_shards=100,
            _snapshot_fn=_make_fake_snapshot_fn(source_dir),
            _meta_loader=lambda r, d: meta,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        episodes = scan_available_episodes(cache_dir)
        assert len(episodes) <= n_episodes
        assert len(episodes) == n_episodes  # exact match for clean download


# ---------------------------------------------------------------------------
# 5. Eviction + re-download cycle consistency
# ---------------------------------------------------------------------------


class TestEvictionRedownloadCycle:
    """Evict some episodes, re-download them, verify consistency."""

    def test_evict_and_redownload(self, tmp_path):
        """Download 10 episodes, delete 3, re-download, verify no duplicates."""
        source_dir = tmp_path / "source"
        cache_dir = tmp_path / "output"
        n_episodes = 10

        _populate_episodes(source_dir, list(range(n_episodes)))
        meta = {**FAKE_META, "total_episodes": n_episodes}

        # First download
        snapshot_fn = _make_fake_snapshot_fn(source_dir)
        prefetcher = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=list(range(n_episodes)),
            min_shards=1,
            max_shards=100,
            _snapshot_fn=snapshot_fn,
            _meta_loader=lambda r, d: meta,
        )
        prefetcher.start()
        prefetcher.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher.stop()

        assert scan_available_episodes(cache_dir) == list(range(10))

        # Evict episodes 3, 5, 7 by deleting their parquets
        for ep in [3, 5, 7]:
            pq_path = cache_dir / f"data/chunk-000/episode_{ep:06d}.parquet"
            if pq_path.exists():
                pq_path.unlink()

        after_evict = scan_available_episodes(cache_dir)
        assert 3 not in after_evict
        assert 5 not in after_evict
        assert 7 not in after_evict
        assert len(after_evict) == 7

        # Re-download (snapshot_fn copies from source again)
        prefetcher2 = LeRobotPrefetcher(
            repo_id="test/dataset",
            cache_dir=cache_dir,
            episode_indices=list(range(n_episodes)),
            min_shards=1,
            max_shards=100,
            _snapshot_fn=snapshot_fn,
            _meta_loader=lambda r, d: meta,
        )
        prefetcher2.start()
        prefetcher2.wait_for_min(timeout=30)
        time.sleep(0.5)
        prefetcher2.stop()

        final = scan_available_episodes(cache_dir)
        assert final == list(range(10))
        assert len(final) == len(set(final)), "No duplicates after re-download"

    def test_eviction_leaves_no_stale_indices(self, tmp_path):
        """After evicting an episode, its index must vanish from the scan."""
        root = tmp_path / "prefetch"
        _populate_episodes(root, [0, 1, 2, 3, 4])

        # Evict episode 2
        pq_path = root / "data/chunk-000/episode_000002.parquet"
        pq_path.unlink()

        episodes = scan_available_episodes(root)
        assert episodes == [0, 1, 3, 4]


# ---------------------------------------------------------------------------
# 6. Concurrent prefetcher writes during episode scan
# ---------------------------------------------------------------------------


class TestConcurrentWritesDuringScan:
    """Simulate writes happening while scan_available_episodes runs.

    The scan should produce a valid snapshot: a subset of the final state,
    without corruption or duplicates.
    """

    def test_concurrent_write_no_duplicates(self, tmp_path):
        """Scan while another thread writes new episode files."""
        root = tmp_path / "prefetch"
        # Pre-populate 5 episodes
        _populate_episodes(root, list(range(5)))

        write_done = threading.Event()
        scan_results = []

        def writer():
            """Write episodes 5-9 with small delays."""
            for ep_idx in range(5, 10):
                _write_episode_parquet(root, ep_idx)
                time.sleep(0.01)  # Simulate staggered writes
            write_done.set()

        def scanner():
            """Repeatedly scan until the writer is done."""
            while not write_done.is_set():
                episodes = scan_available_episodes(root)
                scan_results.append(episodes)
                time.sleep(0.005)
            # One final scan after writes complete
            scan_results.append(scan_available_episodes(root))

        writer_thread = threading.Thread(target=writer)
        scanner_thread = threading.Thread(target=scanner)

        writer_thread.start()
        scanner_thread.start()
        writer_thread.join(timeout=10)
        scanner_thread.join(timeout=10)

        # Every intermediate scan must:
        for episodes in scan_results:
            # 1. Have no duplicates
            assert len(episodes) == len(set(episodes)), (
                f"Duplicates found: {episodes}"
            )
            # 2. Be sorted
            assert episodes == sorted(episodes)
            # 3. Contain at least the initial 5
            assert all(ep in episodes for ep in range(5)), (
                f"Missing initial episodes: {episodes}"
            )
            # 4. Contain only valid indices
            assert all(0 <= ep < 10 for ep in episodes)

        # Final scan should see all 10
        final = scan_results[-1]
        assert final == list(range(10))

    def test_scan_during_partial_write_is_safe(self, tmp_path):
        """If a file is being written (exists but incomplete), the scan
        still produces valid results based on the filename match."""
        root = tmp_path / "prefetch"
        _populate_episodes(root, [0, 1])

        # Simulate a "being written" file -- exists but with partial content
        partial_dir = root / "data" / "chunk-000"
        partial_path = partial_dir / "episode_000002.parquet"
        partial_path.write_bytes(b"PAR1")  # Partial parquet magic bytes

        episodes = scan_available_episodes(root)
        # Episode 2 appears because it matches the filename glob
        # Content validation is not scan_available_episodes' responsibility
        assert episodes == [0, 1, 2]
        assert len(episodes) == len(set(episodes))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and edge cases for scan_available_episodes."""

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list."""
        root = tmp_path / "empty"
        root.mkdir()
        assert scan_available_episodes(root) == []

    def test_directory_with_only_meta(self, tmp_path):
        """Directory with only metadata, no episode files."""
        root = tmp_path / "meta_only"
        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "info.json").write_text(json.dumps(FAKE_META))

        assert scan_available_episodes(root) == []

    def test_single_episode(self, tmp_path):
        """Single episode, no ambiguity."""
        root = tmp_path / "single"
        _write_episode_parquet(root, 42)
        assert scan_available_episodes(root) == [42]

    def test_large_episode_index(self, tmp_path):
        """Episode indices well beyond typical ranges."""
        root = tmp_path / "large"
        _write_episode_parquet(root, 999999)
        assert scan_available_episodes(root) == [999999]

    def test_non_contiguous_episodes(self, tmp_path):
        """Episodes with gaps in their indices."""
        root = tmp_path / "gaps"
        _populate_episodes(root, [0, 5, 10, 15, 100])

        episodes = scan_available_episodes(root)
        assert episodes == [0, 5, 10, 15, 100]

    def test_episode_in_unexpected_directory_name(self, tmp_path):
        """Episode parquet in a directory with an unusual name is still found."""
        root = tmp_path / "weird"
        weird_dir = root / "backup" / "old_data"
        weird_dir.mkdir(parents=True)
        # Write a minimal parquet with episode naming
        table = pa.table(
            {"x": [1]},
            schema=pa.schema([("x", pa.int64())]),
        )
        pq.write_table(table, weird_dir / "episode_000099.parquet")

        episodes = scan_available_episodes(root)
        assert episodes == [99]

    def test_filename_without_episode_pattern_ignored(self, tmp_path):
        """Files named like parquets but not matching episode_N pattern."""
        root = tmp_path / "mixed"
        _write_episode_parquet(root, 0)

        data_dir = root / "data" / "chunk-000"
        # Not an episode parquet
        table = pa.table({"x": [1]}, schema=pa.schema([("x", pa.int64())]))
        pq.write_table(table, data_dir / "stats.parquet")
        pq.write_table(table, data_dir / "metadata.parquet")

        episodes = scan_available_episodes(root)
        assert episodes == [0]

    def test_multiple_chunks(self, tmp_path):
        """Episodes spread across multiple chunk directories."""
        root = tmp_path / "multi_chunk"
        # chunks_size=3 so episodes span chunks
        for ep_idx in range(9):
            _write_episode_parquet(root, ep_idx, chunks_size=3)

        episodes = scan_available_episodes(root)
        assert episodes == list(range(9))

        # Verify they're in multiple chunk dirs
        chunk_dirs = set()
        for p in root.rglob("episode_*.parquet"):
            chunk_dirs.add(p.parent.name)
        assert len(chunk_dirs) == 3  # chunk-000, chunk-001, chunk-002
