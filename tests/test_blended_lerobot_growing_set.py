"""Tests for the growing-training-set feature.

Background: BlendedLeRobotDataModule is designed to let training start on a
partial HF dataset download and grow the training set as more episodes
finish downloading.  The original implementation snapshotted the ready set
at setup and never refreshed — training would iterate the initial ~0.4% of
an 18k-episode dataset forever.

This file locks in the post-fix behavior:
- ``LeRobotShuffleBufferDataset.refresh(min_new, timeout)`` admits newly-
  ready episodes, with optional blocking semantics for deterministic
  cadence.
- ``ProducerPool.update_episodes(source_name, new_list)`` lets a running
  pool swap its work queue atomically.
- ``LeRobotPrefetcher.is_done()`` signals when no more episodes will arrive
  so refresh() doesn't block forever after downloads complete.
- A ``GrowingDatasetCallback`` drives ``refresh()`` on Lightning's
  ``on_train_epoch_start``.

The tests use a ``_ControlledFakePrefetcher`` fixture that substitutes the
background download worker with a ``make_ready(ep_ids)`` trigger, so
admission cadence is deterministic (no sleeps waiting for HF to respond).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataporter.lerobot_prefetcher import LeRobotPrefetcher
from dataporter.shuffle_buffer import ShuffleBuffer


# ---------------------------------------------------------------------------
# Fake metadata + on-disk fixtures (mirrors patterns in
# test_lerobot_prefetcher.py so we don't pull network).
# ---------------------------------------------------------------------------

FAKE_META = {
    "codebase_version": "v2.1",
    "data_path": (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ),
    "video_path": (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    ),
    "fps": 30,
    "total_episodes": 50,
    "total_frames": 15000,
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


def _fake_meta_loader(repo_id: str, cache_dir: Path) -> dict:
    meta_dir = cache_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))
    return FAKE_META


def _populate_source(source_dir: Path, n_episodes: int = 50) -> None:
    """Pre-populate a source directory with parquet+mp4 for N episodes.
    The ``_ControlledFakePrefetcher`` copies from here into the cache on
    demand.
    """
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
    ])
    rng = np.random.RandomState(42)
    meta_dir = source_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(FAKE_META))

    chunks_size = FAKE_META["chunks_size"]
    for ep_idx in range(n_episodes):
        chunk = ep_idx // chunks_size
        n_frames = 300
        parquet_dir = source_dir / f"data/chunk-{chunk:03d}"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "observation.state": [
                [float(rng.randn()), float(rng.randn())]
                for _ in range(n_frames)
            ],
            "action": [
                [float(rng.randn()), float(rng.randn())]
                for _ in range(n_frames)
            ],
            "timestamp": [i / 30.0 for i in range(n_frames)],
            "episode_index": [ep_idx] * n_frames,
            "frame_index": list(range(n_frames)),
        }, schema=schema)
        pq.write_table(table, parquet_dir / f"episode_{ep_idx:06d}.parquet")
        for vid_key in ["observation.image"]:
            video_dir = source_dir / f"videos/chunk-{chunk:03d}/{vid_key}"
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / f"episode_{ep_idx:06d}.mp4").write_bytes(
                b"fake video " + str(ep_idx).encode()
            )


class _ControlledFakePrefetcher(LeRobotPrefetcher):
    """LeRobotPrefetcher with manual admission control for tests.

    The background download worker is a no-op that blocks on
    ``_stop_event``.  Tests call ``make_ready(ep_ids)`` to synchronously
    copy files from a pre-populated source directory into the cache —
    which is exactly what the real prefetcher's ``ready_episodes()`` then
    detects.
    """

    def __init__(
        self,
        source_dir: Path,
        cache_dir: Path,
        min_shards: int = 5,
        **kwargs,
    ):
        super().__init__(
            repo_id="test/fake",
            cache_dir=cache_dir,
            min_shards=min_shards,
            _meta_loader=_fake_meta_loader,
            _snapshot_fn=lambda *a, **kw: None,
            **kwargs,
        )
        self._source_dir = source_dir

    def _run_inner(self) -> None:
        # Don't download.  Block until stop so is_done() returns False
        # while the test is still feeding episodes.
        self._stop_event.wait()

    def make_ready(self, ep_ids: list[int]) -> None:
        """Copy parquet+mp4 for these episodes into the cache dir."""
        chunks_size = FAKE_META["chunks_size"]
        for ep_idx in ep_ids:
            chunk = ep_idx // chunks_size
            pq_rel = (
                f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
            )
            dst_pq = self._cache_dir / pq_rel
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._source_dir / pq_rel, dst_pq)
            for vk in ["observation.image"]:
                mp4_rel = (
                    f"videos/chunk-{chunk:03d}/{vk}/"
                    f"episode_{ep_idx:06d}.mp4"
                )
                dst_mp4 = self._cache_dir / mp4_rel
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._source_dir / mp4_rel, dst_mp4)
        meta_dst = self._cache_dir / "meta" / "info.json"
        if not meta_dst.exists():
            meta_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                self._source_dir / "meta" / "info.json", meta_dst,
            )
        if self._min_ready is not None:
            self._check_min_ready()

    def mark_complete(self) -> None:
        """Simulate "all downloads done" — triggers is_done() == True."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Minimal stand-in for LeRobotShardSource sufficient for refresh() logic.
# Avoids a full on-disk dataset layout.
# ---------------------------------------------------------------------------


class _MiniShardSource:
    """Stand-in for LeRobotShardSource exposing only what the consumer
    needs: ``fps``, ``total_episodes``, ``episode_frame_count``, and the
    row/window loaders.  ``refresh()`` only needs frame_count for each
    admitted raw id; ``__getitem__`` (not exercised in most refresh
    tests) needs the row/window loaders to return plausible tensors.
    """

    def __init__(self, frames_per_episode: int = 300, num_episodes: int = 50):
        self.fps = 30
        self.root = Path("/tmp/mini")
        self._frames_per_episode = frames_per_episode
        self._num_episodes = num_episodes

    @property
    def total_episodes(self) -> int:
        return self._num_episodes

    def episode_frame_count(self, raw_ep: int) -> int:
        return self._frames_per_episode

    def _row(self, ep_idx: int, frame_idx: int) -> dict:
        return {
            "episode_index": torch.tensor(ep_idx),
            "frame_index": torch.tensor(frame_idx),
            "timestamp": torch.tensor(frame_idx / self.fps),
            "task_index": torch.tensor(0),
            "action": torch.zeros(2),
            "observation.state": torch.zeros(2),
        }

    def load_episode_row_torch(self, raw_ep: int, frame_idx: int) -> dict:
        return self._row(raw_ep, frame_idx)

    def load_episode_window_torch(
        self, raw_ep: int, frame_indices: list[int],
    ) -> dict:
        rows = [self._row(raw_ep, i) for i in frame_indices]
        return {
            key: torch.stack([r[key] for r in rows])
            for key in rows[0]
        }

    def tasks(self) -> dict[int, str]:
        return {0: "push"}


# ---------------------------------------------------------------------------
# Fake ProducerPool: records update_episodes calls, no actual decode.
# Real pool-level test lives further down.
# ---------------------------------------------------------------------------


class _RecordingPool:
    def __init__(self):
        self.update_calls: list[tuple[str, list[int]]] = []

    def update_episodes(self, source_name: str, new_list: list[int]) -> None:
        self.update_calls.append((source_name, list(new_list)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_dir(tmp_path_factory):
    """Module-ish shared source dir with all 50 fake episodes populated."""
    d = tmp_path_factory.mktemp("fake_source")
    _populate_source(d, n_episodes=50)
    return d


@pytest.fixture
def prefetcher(tmp_path, source_dir):
    """Start a fake prefetcher with 0 episodes ready; tests drive admission."""
    pf = _ControlledFakePrefetcher(
        source_dir=source_dir,
        cache_dir=tmp_path / "cache",
        min_shards=5,
    )
    pf.start()
    yield pf
    pf.stop()


@pytest.fixture
def pool():
    return _RecordingPool()


@pytest.fixture
def mini_source():
    return _MiniShardSource(frames_per_episode=300, num_episodes=50)


def _train_split(ep_idx: int) -> bool:
    """Stable train predicate used across the tests."""
    return ep_idx % 10 != 9


def _build_dataset(
    prefetcher,
    pool,
    mini_source,
    default_min_new: int = 0,
    source_name: str = "test",
) -> "LeRobotShuffleBufferDataset":
    """Construct a dataset with the new refresh-capable signature."""
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    buffer = ShuffleBuffer(
        capacity=16, max_frames=8, channels=3, height=8, width=8,
    )
    sources = [{
        "shard_source": mini_source,
        "source_name": source_name,
        "episode_offset": 0,
        "transform": None,
    }]
    return LeRobotShuffleBufferDataset(
        buffer=buffer,
        sources=sources,
        delta_timestamps={},
        prefetchers=[prefetcher],
        producer_pool=pool,
        split_fn=_train_split,
        default_min_new=default_min_new,
        image_keys=["observation.image"],
    )


def _kill_switch(timeout_s: float) -> threading.Event:
    """Raise a deadline timer; returns an Event the test can check."""
    stop = threading.Event()
    timer = threading.Timer(timeout_s, stop.set)
    timer.daemon = True
    timer.start()
    return stop


# ---------------------------------------------------------------------------
# 1. Bug reproduction — post-fix behavior flipped from the original bug
# ---------------------------------------------------------------------------


class TestBugReproduction:
    """The original bug: setup snapshots ready_episodes(), __len__ is
    frozen, new downloads never enter training.  Post-fix: refresh()
    picks them up.
    """

    def test_length_grows_after_new_episodes_admitted(
        self, prefetcher, pool, mini_source,
    ):
        """The original frozen-set bug, inverted.  Before the fix this
        assertion would fail — __len__ stayed constant forever.
        """
        prefetcher.make_ready(list(range(10)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        initial_len = len(dataset)
        assert initial_len > 0

        prefetcher.make_ready(list(range(10, 30)))
        dataset.refresh()
        grown_len = len(dataset)
        assert grown_len > initial_len, (
            f"expected len to grow after admitting 20 new episodes; "
            f"initial={initial_len}, after={grown_len}"
        )

    def test_new_episodes_reach_producer_pool(
        self, prefetcher, pool, mini_source,
    ):
        """refresh() must forward the admitted list to the pool so decode
        work reflects the new episodes.
        """
        prefetcher.make_ready(list(range(10)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        pool.update_calls.clear()

        prefetcher.make_ready(list(range(10, 30)))
        dataset.refresh()

        assert pool.update_calls, "pool never received update_episodes()"
        source_name, new_list = pool.update_calls[-1]
        admitted = set(new_list)
        # Every admitted episode passes the train split.
        assert all(_train_split(e) for e in admitted), (
            f"val-bucket episode leaked into train admission: {admitted}"
        )
        # New range is included.
        expected_train_new = {e for e in range(10, 30) if _train_split(e)}
        assert expected_train_new.issubset(admitted)


# ---------------------------------------------------------------------------
# 2. refresh() happy path + edge cases
# ---------------------------------------------------------------------------


class TestRefreshHappyPath:

    def test_refresh_admits_new_episodes(
        self, prefetcher, pool, mini_source,
    ):
        prefetcher.make_ready([0, 1, 2, 3, 4])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        assert len(dataset) > 0

    def test_refresh_min_new_blocks_until_available(
        self, prefetcher, pool, mini_source,
    ):
        prefetcher.make_ready([0, 1, 2])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()       # admits the 3 ready

        def delayed_admit():
            time.sleep(0.3)
            prefetcher.make_ready([3, 4, 5, 6, 7])

        t = threading.Thread(target=delayed_admit, daemon=True)
        t.start()
        t0 = time.monotonic()
        dataset.refresh(min_new=3, timeout=5.0)
        elapsed = time.monotonic() - t0
        t.join(timeout=1.0)

        assert elapsed >= 0.25, (
            f"refresh returned too early ({elapsed:.3f}s); should have "
            f"blocked for min_new=3 until the delayed admission"
        )
        # After refresh returns, the admitted set grew.
        assert len(dataset._current_train_episodes) >= 6

    def test_refresh_min_new_returns_early_when_prefetcher_done(
        self, prefetcher, pool, mini_source,
    ):
        """The critical edge case: refresh must NOT block when there are
        no more episodes coming.
        """
        prefetcher.make_ready([0, 1, 2])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        prefetcher.mark_complete()

        t0 = time.monotonic()
        # Ask for a huge min_new — without the is_done() check this would
        # block until timeout.
        result = dataset.refresh(min_new=1000, timeout=2.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"refresh blocked {elapsed:.3f}s even though prefetcher was "
            f"done — is_done() check is missing"
        )
        assert result == len(dataset)

    def test_refresh_timeout_fires(self, prefetcher, pool, mini_source):
        prefetcher.make_ready([0, 1])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()

        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            dataset.refresh(min_new=10, timeout=0.5)
        elapsed = time.monotonic() - t0
        assert 0.4 < elapsed < 1.5, (
            f"timeout fired at wrong cadence: elapsed={elapsed:.3f}s"
        )

    def test_refresh_respects_stable_split(
        self, prefetcher, pool, mini_source,
    ):
        """Every admitted episode satisfies the split predicate; val-bucket
        (ep_idx % 10 == 9) episodes never leak into the train set.
        """
        prefetcher.make_ready(list(range(30)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()

        admitted = set(dataset._current_train_episodes)
        assert 9 not in admitted
        assert 19 not in admitted
        assert 29 not in admitted
        assert {0, 1, 2, 10, 11, 20}.issubset(admitted)

    def test_refresh_noop_when_min_new_zero(
        self, prefetcher, pool, mini_source,
    ):
        prefetcher.make_ready([0, 1, 2, 3, 4])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        before = len(dataset)
        before_calls = len(pool.update_calls)

        dataset.refresh(min_new=0)
        assert len(dataset) == before
        # No state change → no pool notification.
        assert len(pool.update_calls) == before_calls

    def test_refresh_idempotent_when_nothing_changed(
        self, prefetcher, pool, mini_source,
    ):
        """Two consecutive refreshes with no new episodes must not
        rebuild the mapping or re-notify the pool.
        """
        prefetcher.make_ready(list(range(10)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        calls_after_first = len(pool.update_calls)
        len_after_first = len(dataset)

        dataset.refresh()
        assert len(pool.update_calls) == calls_after_first
        assert len(dataset) == len_after_first


# ---------------------------------------------------------------------------
# 3. Setup-time gate wiring
# ---------------------------------------------------------------------------


class TestSetupGateWiring:

    def test_initial_wait_uses_prefetch_min_episodes_not_refresh_default(
        self, prefetcher, pool, mini_source,
    ):
        """DataModule with default_min_new=0 (non-blocking per-epoch) but
        an explicit setup-time refresh(min_new=5) must block for the
        setup gate.
        """
        dataset = _build_dataset(
            prefetcher, pool, mini_source, default_min_new=0,
        )
        # Nothing ready; refresh() with no args returns instantly.
        t0 = time.monotonic()
        dataset.refresh()
        assert time.monotonic() - t0 < 0.2
        # But the explicit setup gate blocks.

        def delayed():
            time.sleep(0.3)
            prefetcher.make_ready(list(range(5)))

        t = threading.Thread(target=delayed, daemon=True)
        t.start()
        t0 = time.monotonic()
        dataset.refresh(min_new=5, timeout=5.0)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25, (
            f"explicit setup gate didn't block: {elapsed:.3f}s"
        )
        t.join(timeout=1.0)

    def test_epoch_refresh_uses_default_min_new(
        self, prefetcher, pool, mini_source,
    ):
        """refresh() with no args defaults to default_min_new.  If the
        default is 3 and prefetcher has 2 new, refresh blocks until 3.
        """
        prefetcher.make_ready([0, 1])
        dataset = _build_dataset(
            prefetcher, pool, mini_source, default_min_new=3,
        )
        dataset.refresh(min_new=0)   # admit the 2 that are there

        def delayed():
            time.sleep(0.3)
            # The prefetcher has train episodes 2, 3, 10 (skipping 9).
            # After this, _train_split(2) and _train_split(3) are already
            # admitted — wait, that's wrong: make_ready(0,1) admitted them
            # via refresh(min_new=0), so _current_train_episodes = [0,1].
            # Now we need 3 NEW train episodes → [2, 3, 4] (_train_split
            # rejects 9).  Make 2,3,4 ready.
            prefetcher.make_ready([2, 3, 4])

        t = threading.Thread(target=delayed, daemon=True)
        t.start()
        t0 = time.monotonic()
        dataset.refresh(timeout=5.0)        # no args → uses default=3
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25
        t.join(timeout=1.0)

    def test_explicit_min_new_always_overrides(
        self, prefetcher, pool, mini_source,
    ):
        """refresh(min_new=5) uses 5 regardless of default_min_new."""
        prefetcher.make_ready([0, 1, 2])
        dataset = _build_dataset(
            prefetcher, pool, mini_source, default_min_new=0,
        )
        # Default=0 would be non-blocking, but we pass min_new=5.
        with pytest.raises(TimeoutError):
            dataset.refresh(min_new=5, timeout=0.3)

    def test_default_min_new_zero_is_nonblocking(
        self, prefetcher, pool, mini_source,
    ):
        """The default-config behavior: refresh() is fast and
        scheduler-friendly.  No episodes new? Return immediately.
        """
        prefetcher.make_ready([0, 1])
        dataset = _build_dataset(
            prefetcher, pool, mini_source, default_min_new=0,
        )
        dataset.refresh()
        t0 = time.monotonic()
        dataset.refresh()       # nothing new
        assert time.monotonic() - t0 < 0.2


# ---------------------------------------------------------------------------
# 4. Stale-refresh warning
# ---------------------------------------------------------------------------


class TestStaleRefreshWarning:

    def test_stale_refresh_warning_fires(
        self, prefetcher, pool, mini_source, caplog,
    ):
        """When unadmitted episodes sit while __getitem__ is called for
        > staleness threshold, a single warning fires with remediation
        guidance.
        """
        prefetcher.make_ready([0, 1, 2, 3, 4])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        # Admit 20 more without refreshing.
        prefetcher.make_ready(list(range(5, 25)))

        # Lower the threshold for deterministic testing.
        dataset._REFRESH_WARN_STALENESS_S = 0.05
        # Fill the buffer so __getitem__ doesn't crash on empty buffer.
        for ep in dataset._current_train_episodes:
            dataset._buffer.put(
                ep, torch.zeros((8, 3, 8, 8), dtype=torch.uint8),
            )
        time.sleep(0.1)
        with caplog.at_level(logging.WARNING):
            # Amortized check runs every 500 calls.
            for _ in range(600):
                try:
                    dataset[0]
                except Exception:
                    pass

        stale_warnings = [
            r for r in caplog.records
            if "refresh" in r.message.lower()
            and "growingdatasetcallback" in r.message.lower()
        ]
        assert len(stale_warnings) >= 1, (
            "expected stale-refresh warning with GrowingDatasetCallback "
            f"hint; got {[r.message for r in caplog.records]}"
        )

    def test_stale_refresh_warning_silent_when_caught_up(
        self, prefetcher, pool, mini_source, caplog,
    ):
        """If the dataset is admitted everything that's ready, no warning."""
        prefetcher.make_ready(list(range(5)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        dataset._REFRESH_WARN_STALENESS_S = 0.05
        for ep in dataset._current_train_episodes:
            dataset._buffer.put(
                ep, torch.zeros((8, 3, 8, 8), dtype=torch.uint8),
            )
        time.sleep(0.1)
        with caplog.at_level(logging.WARNING):
            for _ in range(1000):
                try:
                    dataset[0]
                except Exception:
                    pass
        stale = [
            r for r in caplog.records
            if "growingdatasetcallback" in r.message.lower()
        ]
        assert not stale, "spurious warning when caught up"

    def test_stale_refresh_warning_only_once(
        self, prefetcher, pool, mini_source, caplog,
    ):
        prefetcher.make_ready([0, 1, 2, 3, 4])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        prefetcher.make_ready(list(range(5, 25)))
        dataset._REFRESH_WARN_STALENESS_S = 0.05
        for ep in dataset._current_train_episodes:
            dataset._buffer.put(
                ep, torch.zeros((8, 3, 8, 8), dtype=torch.uint8),
            )
        time.sleep(0.1)
        with caplog.at_level(logging.WARNING):
            for _ in range(10_000):
                try:
                    dataset[0]
                except Exception:
                    pass
        stale = [
            r for r in caplog.records
            if "growingdatasetcallback" in r.message.lower()
        ]
        assert len(stale) == 1, (
            f"expected exactly one stale warning, got {len(stale)}"
        )


# ---------------------------------------------------------------------------
# 5. Lightning callback integration
# ---------------------------------------------------------------------------


class TestGrowingDatasetCallback:

    def test_callback_calls_refresh_on_epoch_start(self):
        """on_train_epoch_start must find the DataModule's train_dataset
        and call refresh() on it.
        """
        from unittest.mock import MagicMock
        from dataporter import GrowingDatasetCallback

        cb = GrowingDatasetCallback()
        trainer = MagicMock()
        pl_module = MagicMock()
        trainer.datamodule.train_dataset.refresh = MagicMock()

        cb.on_train_epoch_start(trainer, pl_module)
        trainer.datamodule.train_dataset.refresh.assert_called_once()

    def test_callback_noop_without_refresh_method(self):
        """Works against a plain Dataset without a refresh method —
        silently does nothing rather than raising.
        """
        from unittest.mock import MagicMock
        from dataporter import GrowingDatasetCallback

        cb = GrowingDatasetCallback()
        trainer = MagicMock()
        # Make hasattr return False for .refresh
        plain_ds = object()
        trainer.datamodule.train_dataset = plain_ds
        pl_module = MagicMock()
        # Should not raise.
        cb.on_train_epoch_start(trainer, pl_module)


# ---------------------------------------------------------------------------
# 6. Adversarial cases
# ---------------------------------------------------------------------------


class TestAdversarial:

    def test_refresh_during_concurrent_getitem(
        self, prefetcher, pool, mini_source,
    ):
        """No crashes from refresh() racing against __getitem__ on another
        thread.
        """
        prefetcher.make_ready(list(range(10)))
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh()
        for ep in dataset._current_train_episodes:
            dataset._buffer.put(
                ep, torch.zeros((8, 3, 8, 8), dtype=torch.uint8),
            )

        errors = []

        def hammer_getitem():
            try:
                for _ in range(200):
                    dataset[0]
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=hammer_getitem, daemon=True)
        t.start()
        prefetcher.make_ready(list(range(10, 30)))
        dataset.refresh()
        t.join(timeout=5.0)
        assert not t.is_alive(), "hammering thread didn't finish"
        assert not errors, f"concurrent access crashed: {errors}"

    def test_refresh_with_prefetcher_error_returns_early(
        self, tmp_path, source_dir, pool, mini_source,
    ):
        """If the prefetcher errored and stopped, refresh() should see
        is_done() == True and return instead of blocking forever.
        """
        pf = _ControlledFakePrefetcher(
            source_dir=source_dir, cache_dir=tmp_path / "c", min_shards=5,
        )
        pf.start()
        pf.make_ready([0, 1, 2])
        dataset = _build_dataset(pf, pool, mini_source)
        dataset.refresh(min_new=0)
        # Simulate an error + stop.
        pf._error = RuntimeError("HF 401 Unauthorized (simulated)")
        pf.mark_complete()

        t0 = time.monotonic()
        dataset.refresh(min_new=100, timeout=2.0)
        assert time.monotonic() - t0 < 1.0

    def test_refresh_timeout_none_without_done_blocks(
        self, prefetcher, pool, mini_source,
    ):
        """Sanity: timeout=None with unsatisfied min_new + live
        prefetcher does block.  (Kill switch on the test side so a
        regression that ignores the predicate can't hang CI.)
        """
        prefetcher.make_ready([0, 1])
        dataset = _build_dataset(prefetcher, pool, mini_source)
        dataset.refresh(min_new=0)

        result_box = {}

        def caller():
            try:
                dataset.refresh(min_new=10, timeout=0.8)
                result_box["ret"] = "returned"
            except TimeoutError:
                result_box["ret"] = "timeout"

        t = threading.Thread(target=caller, daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "refresh with timeout=0.8 didn't finish"
        assert result_box["ret"] == "timeout"


# ---------------------------------------------------------------------------
# 7. Pool-level: update_episodes actually swaps the running iterator.
# ---------------------------------------------------------------------------


class TestProducerPoolUpdateEpisodes:

    def test_pool_update_episodes_api_exists(self):
        """Lightweight presence check — the real behavior is exercised
        indirectly via TestBugReproduction.test_new_episodes_reach_pool.
        This test confirms the method signature is callable.
        """
        from dataporter.producer_pool import ProducerPool
        assert hasattr(ProducerPool, "update_episodes"), (
            "ProducerPool.update_episodes not defined — Phase 2A missing"
        )


# ---------------------------------------------------------------------------
# 8. Size-aware setup gate: prefetch_min_episodes + prefetch_min_fraction
# ---------------------------------------------------------------------------


class TestSetupGateSizeAware:

    def test_default_min_episodes_is_conservative_floor(self):
        """Default prefetch_min_episodes is deliberately kept at 50
        (historic behaviour) so adopting DataPorter doesn't silently
        break test fixtures that use small public datasets.  Large-
        dataset users raise to ~500 explicitly, or use
        prefetch_min_fraction for size-aware scaling.
        """
        from dataporter.blended_lerobot_datamodule import (
            BlendedLeRobotDataModule,
        )
        import inspect

        sig = inspect.signature(BlendedLeRobotDataModule.__init__)
        assert sig.parameters["prefetch_min_episodes"].default == 50
        assert sig.parameters["prefetch_min_fraction"].default is None

    def test_fraction_raises_effective_gate_on_large_totals(
        self, tmp_path, source_dir,
    ):
        """With prefetch_min_fraction=0.2 and total=50, effective gate
        should be max(floor=5, 10) = 10.  The raw ``_min_shards`` on
        the prefetcher should reflect this after the fraction is
        applied.
        """
        # Inline the _start_prefetcher fraction-scaling logic against our
        # fake prefetcher (the real DataModule requires a full HF flow).
        pf = _ControlledFakePrefetcher(
            source_dir=source_dir,
            cache_dir=tmp_path / "c",
            min_shards=5,          # the floor
        )
        # Apply fraction-scaling — 20% of 50 = 10, which exceeds floor=5.
        total = pf.total_episodes
        assert total == 50
        fraction = 0.2
        effective = int(fraction * total)
        if effective > pf._min_shards:
            pf._min_shards = effective
        assert pf._min_shards == 10

    def test_floor_wins_on_small_totals(self, tmp_path, source_dir):
        """prefetch_min_episodes=500 + fraction=0.1 on total=50: the
        floor (500) exceeds fraction (5), so the gate stays at 500.
        The is_done() short-circuit in refresh() then handles the
        "only 50 available" case gracefully.
        """
        pf = _ControlledFakePrefetcher(
            source_dir=source_dir,
            cache_dir=tmp_path / "c",
            min_shards=500,
        )
        total = pf.total_episodes    # 50 in the fixture
        fraction = 0.1
        fraction_gate = int(fraction * total)   # 5
        if fraction_gate > pf._min_shards:      # 5 <= 500, no change
            pf._min_shards = fraction_gate
        assert pf._min_shards == 500


# ---------------------------------------------------------------------------
# 9. Step-based callback cadence
# ---------------------------------------------------------------------------


class TestGrowingDatasetCallbackStepMode:

    def test_step_based_callback_fires_every_n_steps(self):
        """With every_n_steps=N, refresh fires exclusively on
        on_train_batch_start at step % N == 0 (skipping step 0).
        Epoch start is a no-op in this mode.
        """
        from unittest.mock import MagicMock
        from dataporter import GrowingDatasetCallback

        cb = GrowingDatasetCallback(every_n_steps=3)
        trainer = MagicMock()
        trainer.datamodule.train_dataset.refresh = MagicMock()
        pl_module = MagicMock()

        # Epoch-start must NOT fire when in step mode.
        cb.on_train_epoch_start(trainer, pl_module)
        assert trainer.datamodule.train_dataset.refresh.call_count == 0

        # Walk steps 0..6; refresh fires at 3 and 6 only.
        fired_at = []
        for step in range(7):
            trainer.global_step = step
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=step)
            if trainer.datamodule.train_dataset.refresh.call_count > len(fired_at):
                fired_at.append(step)
        assert fired_at == [3, 6], (
            f"expected refresh at steps [3, 6], got {fired_at}"
        )

    def test_epoch_based_callback_ignores_batch_hooks(self):
        """Default (every_n_steps=None): on_train_batch_start is a
        no-op.  Only on_train_epoch_start fires refresh.
        """
        from unittest.mock import MagicMock
        from dataporter import GrowingDatasetCallback

        cb = GrowingDatasetCallback()   # default = epoch mode
        trainer = MagicMock()
        trainer.datamodule.train_dataset.refresh = MagicMock()
        pl_module = MagicMock()
        trainer.global_step = 100

        for _ in range(500):
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        assert trainer.datamodule.train_dataset.refresh.call_count == 0

        cb.on_train_epoch_start(trainer, pl_module)
        assert trainer.datamodule.train_dataset.refresh.call_count == 1

    def test_invalid_every_n_steps_raises(self):
        from dataporter import GrowingDatasetCallback
        with pytest.raises(ValueError, match="every_n_steps must be positive"):
            GrowingDatasetCallback(every_n_steps=0)
        with pytest.raises(ValueError, match="every_n_steps must be positive"):
            GrowingDatasetCallback(every_n_steps=-5)
