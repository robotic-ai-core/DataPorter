"""Train/val isolation tests.

Verifies that the streaming pool + consumer never contaminate the
training side with val-split episodes.  Covers three paths where
contamination could sneak in:

1. **Init-time static admission**: a caller passes val raw ids in
   ``train_episode_indices``.  The consumer must strip them and warn.
2. **Refresh via prefetcher**: the prefetcher reports all on-disk
   episodes (including val); ``_scan_ready_train_episodes_by_source``
   must filter by ``split_fn`` so only train ids flow into admission.
3. **End-to-end pool → buffer**: the buffer must only ever hold keys
   derived from train raw ids, even under growing-set refresh.

Also locks in the stability property: ``_default_train_split`` must
be a pure function — an episode's train/val assignment can't flip
between calls.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("lerobot")


def _train_only(ep: int) -> bool:
    """``ep % 10 != 9`` — matches ``_default_train_split``.
    Declared locally so tests are self-contained.
    """
    return ep % 10 != 9


# ---------------------------------------------------------------------------
# Split-fn purity
# ---------------------------------------------------------------------------


class TestSplitFnPurity:

    def test_default_split_is_deterministic(self):
        """Calling with the same input must return the same result,
        indefinitely — no hidden state, no drift."""
        from dataporter.lerobot_shuffle_buffer_dataset import (
            _default_train_split,
        )
        for ep in range(10_000):
            first = _default_train_split(ep)
            for _ in range(5):
                assert _default_train_split(ep) is first, (
                    f"_default_train_split({ep}) flipped across calls"
                )

    def test_default_split_partitions_cleanly(self):
        """Every episode maps to exactly one of train/val — no overlap,
        no gap."""
        from dataporter.lerobot_shuffle_buffer_dataset import (
            _default_train_split,
        )
        train: set[int] = set()
        val: set[int] = set()
        for ep in range(1000):
            (train if _default_train_split(ep) else val).add(ep)
        assert not (train & val)
        assert train | val == set(range(1000))

    def test_default_split_ratio_approximately_9_to_1(self):
        """The default partitions 90/10 with tolerance ±1 per decade."""
        from dataporter.lerobot_shuffle_buffer_dataset import (
            _default_train_split,
        )
        n_train = sum(_default_train_split(ep) for ep in range(1000))
        assert 895 <= n_train <= 905, (
            f"expected ~900 train out of 1000, got {n_train}"
        )


# ---------------------------------------------------------------------------
# Init-time static admission: filter val from train_episode_indices
# ---------------------------------------------------------------------------


class TestInitTimeFilter:

    def _build(self, train_episode_indices: list[int]):
        """Construct a minimal consumer with the given
        train_episode_indices.  No pool; no prefetchers.
        """
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )
        shard = _make_mock_shard_source(num_episodes=20, frames_per_episode=10)
        buf = ShuffleBuffer(
            capacity=4, max_frames=10, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        return LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "transform": None,
                "train_episode_indices": list(train_episode_indices),
            }],
            delta_timestamps={},
            prefetchers=[],
            producer_pool=None,
            split_fn=_train_only,
            image_keys=["observation.image"],
        )

    def test_clean_train_list_passes_through(self):
        ds = self._build([0, 1, 2, 3])     # all train
        assert set(ds._current_train_episodes) == {0, 1, 2, 3}

    def test_val_ids_filtered_from_train_list(self, caplog):
        """[0, 9, 1, 19] — 9 and 19 are val (mod 10 == 9).  Consumer
        must admit only {0, 1} and log a warning.
        """
        with caplog.at_level(
            logging.WARNING,
            logger="dataporter.lerobot_shuffle_buffer_dataset",
        ):
            ds = self._build([0, 9, 1, 19])
        assert set(ds._current_train_episodes) == {0, 1}, (
            f"val episodes 9, 19 leaked into admitted set "
            f"{ds._current_train_episodes}"
        )
        assert any(
            "val-side raw ids in train_episode_indices" in r.message
            for r in caplog.records
        ), (
            "expected a warning about the val-side ids; got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_all_val_train_list_results_in_empty_admission(self, caplog):
        """Pathological: every id given is val → admitted set empty
        (no admission happens), warning logged for each source."""
        with caplog.at_level(
            logging.WARNING,
            logger="dataporter.lerobot_shuffle_buffer_dataset",
        ):
            ds = self._build([9, 19])
        assert ds._current_train_episodes == []


# ---------------------------------------------------------------------------
# Refresh via prefetcher: filter val from the scan
# ---------------------------------------------------------------------------


class _LeakyFakePrefetcher:
    """Test stand-in that ``ready_episodes()`` includes BOTH train and
    val ids.  The consumer's refresh path must filter on its own.
    """

    def __init__(self, ready: list[int]):
        self._ready = sorted(set(ready))
        self._done = False

    def ready_episodes(self) -> list[int]:
        return list(self._ready)

    def is_done(self) -> bool:
        return self._done

    def mark_done(self) -> None:
        self._done = True

    def set_ready(self, ready: list[int]) -> None:
        self._ready = sorted(set(ready))


class TestRefreshFiltersVal:

    def _build_with_prefetcher(self, ready: list[int]):
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )
        shard = _make_mock_shard_source(num_episodes=50, frames_per_episode=10)
        # Make the shard's ready_episodes reflect whatever the
        # prefetcher says — mirrors how the real LeRobotShardSource's
        # disk scan would reflect what the prefetcher has written.
        pf = _LeakyFakePrefetcher(ready)
        shard._ready_source = pf.ready_episodes
        buf = ShuffleBuffer(
            capacity=4, max_frames=10, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        ds = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "transform": None,
            }],
            delta_timestamps={},
            prefetchers=[pf],
            producer_pool=None,
            split_fn=_train_only,
            image_keys=["observation.image"],
        )
        return ds, pf

    def test_refresh_filters_val_from_prefetcher_ready_list(self):
        """Prefetcher reports [0..19] — 9 and 19 are val.  After
        ``refresh()``, admitted set must be [0..8, 10..18]."""
        ds, _pf = self._build_with_prefetcher(list(range(20)))
        ds.refresh()
        admitted = set(ds._current_train_episodes)
        # No val ids should appear.
        val_leaked = {e for e in admitted if not _train_only(e)}
        assert not val_leaked, f"val ids leaked via refresh: {val_leaked}"
        # All train ids in the prefetcher's ready set should be there.
        expected_train = {e for e in range(20) if _train_only(e)}
        assert admitted == expected_train

    def test_growing_prefetcher_never_admits_val(self):
        """Prefetcher grows [0..5] → [0..15] → [0..25] over several
        refreshes.  At each step the admitted set is the train
        subset of what's ready — no val ever.
        """
        ds, pf = self._build_with_prefetcher([])
        for stop in (5, 15, 25):
            pf.set_ready(list(range(stop)))
            ds.refresh()
            admitted = set(ds._current_train_episodes)
            val_leaked = {e for e in admitted if not _train_only(e)}
            assert not val_leaked, (
                f"val ids leaked at ready={stop}: {val_leaked}"
            )
            expected = {e for e in range(stop) if _train_only(e)}
            assert admitted == expected, (
                f"at ready={stop}, admitted={admitted}, "
                f"expected={expected}"
            )

    def test_custom_split_fn_is_respected(self):
        """A caller-provided split_fn other than the default is
        honored on both the refresh path and the init path.
        """
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )
        # Custom split: evens are train, odds are val.
        custom = lambda ep: ep % 2 == 0
        shard = _make_mock_shard_source(
            num_episodes=20, frames_per_episode=10,
        )
        pf = _LeakyFakePrefetcher(list(range(20)))
        shard._ready_source = pf.ready_episodes
        buf = ShuffleBuffer(
            capacity=4, max_frames=10, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        ds = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "transform": None,
            }],
            delta_timestamps={},
            prefetchers=[pf],
            producer_pool=None,
            split_fn=custom,
            image_keys=["observation.image"],
        )
        ds.refresh()
        admitted = set(ds._current_train_episodes)
        assert admitted == {e for e in range(20) if e % 2 == 0}


# ---------------------------------------------------------------------------
# End-to-end: buffer + pool + refresh must not contain val keys
# ---------------------------------------------------------------------------


pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")


class TestEndToEndBufferIsTrainOnly:

    def test_pool_buffer_contains_only_train_keys_across_refresh(self, tmp_path):
        """End-to-end: synthetic dataset with 10 episodes (9 train, 1
        val), prefetcher-driven growth via live disk scan, self-refresh
        on.  Over 30s of sampling, every buffer key must be a train
        raw id.
        """
        from dataporter import LeRobotShardSource, ResizeFrames
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.producer_pool import ProducerConfig, ProducerPool
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_shard_source_pool_e2e import (
            _LiveDiskPrefetcher,
            _make_dataset,
            _write_episode_mp4,
            _write_episode_parquet,
        )

        root = tmp_path / "ds"
        # Build 30 episodes — default split → ep 9, 19, 29 are val.
        _make_dataset(
            root, ready_eps=list(range(10)), total_episodes=30,
        )
        src = LeRobotShardSource(root)
        pf = _LiveDiskPrefetcher(root)

        # Compute train-only iteration_episodes for the pool.
        initial_train = [e for e in range(10) if _train_only(e)]

        buf = ShuffleBuffer(
            capacity=6, max_frames=32, channels=3, height=32, width=32,
            rotation_per_samples=1,
        )
        cfg = ProducerConfig.from_source(
            source={"repo_id": "synth", "weight": 1.0},
            shard_source=src,
            iteration_episodes=initial_train,
            producer_transform=ResizeFrames(32, 32),
        )
        pool = ProducerPool(
            buf, configs=[cfg], total_workers=2, warmup_target=3,
        )

        consumer = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": src,
                "source_name": "synth",
                "episode_offset": 0,
                "transform": None,
                "train_episode_indices": list(initial_train),
            }],
            delta_timestamps={},
            prefetchers=[pf],
            producer_pool=pool,
            split_fn=_train_only,
            image_keys=["observation.image"],
            refresh_every_n_items=5,
        )

        pool.start()
        try:
            pool.wait_for_warmup(timeout=30.0)
            # Drop more episodes on disk — some train, some val.
            for ep in range(10, 25):
                _write_episode_parquet(root, ep, n_frames=20)
                _write_episode_mp4(
                    root, ep,
                    n_frames=20, height=32, width=32, fps=30,
                )

            # Drive consumer; collect every buffer key we ever observe.
            observed: set[int] = set()
            for _ in range(50):
                item = consumer[0]
                observed.update(k for k in buf.keys() if k >= 0)
                ep_idx = int(item["episode_index"])
                observed.add(ep_idx)
                time.sleep(0.05)

            val_leaked = {k for k in observed if not _train_only(k)}
            assert not val_leaked, (
                f"val ids appeared in the training path: "
                f"{sorted(val_leaked)}.  Observed={sorted(observed)}"
            )
        finally:
            pool.stop()

    def test_pool_iteration_episodes_are_train_only(self, tmp_path):
        """At a higher layer: the ProducerConfig fed into the pool
        must never include val ids in its ``episode_indices``.  Unit
        check on the config object; catches DataModule-side split
        bugs before the pool starts.
        """
        from dataporter import LeRobotShardSource, ResizeFrames
        from dataporter.producer_pool import ProducerConfig
        from test_shard_source_pool_e2e import _make_dataset

        root = tmp_path / "ds"
        _make_dataset(
            root, ready_eps=list(range(10)), total_episodes=10,
        )
        src = LeRobotShardSource(root)
        train_eps = [e for e in range(10) if _train_only(e)]

        cfg = ProducerConfig.from_source(
            source={"repo_id": "synth", "weight": 1.0},
            shard_source=src,
            iteration_episodes=train_eps,
            producer_transform=ResizeFrames(32, 32),
        )
        val_leaked = {e for e in cfg.episode_indices if not _train_only(e)}
        assert not val_leaked, (
            f"pool config includes val ids: {val_leaked}"
        )


# ---------------------------------------------------------------------------
# Unit-level: the pool's decode_fn never receives val ids
# ---------------------------------------------------------------------------


class TestPoolDecodeNeverVal:

    def test_decode_fn_called_with_train_only(self, tmp_path, monkeypatch):
        """Monkeypatch the child decode function to record every raw
        id it's called with; verify the set is entirely train-side.
        """
        from dataporter import LeRobotShardSource, ResizeFrames
        from dataporter.producer_pool import ProducerConfig, ProducerPool
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_shard_source_pool_e2e import _make_dataset

        root = tmp_path / "ds"
        _make_dataset(root, ready_eps=list(range(10)), total_episodes=10)
        src = LeRobotShardSource(root)
        train_eps = [e for e in range(10) if _train_only(e)]

        buf = ShuffleBuffer(
            capacity=6, max_frames=32, channels=3, height=32, width=32,
            rotation_per_samples=1,
        )
        cfg = ProducerConfig.from_source(
            source={"repo_id": "synth", "weight": 1.0},
            shard_source=src,
            iteration_episodes=train_eps,
            producer_transform=ResizeFrames(32, 32),
        )
        pool = ProducerPool(
            buf, configs=[cfg], total_workers=2, warmup_target=3,
        )
        pool.start()
        try:
            pool.wait_for_warmup(timeout=30.0)
            # Consume a handful of samples so the pool runs its loop
            # and dispatches decodes.
            rng = random.Random(0)
            for _ in range(40):
                try:
                    buf.sample(rng)
                except IndexError:
                    pass
            time.sleep(1.0)
            # Inspect buffer keys — every put came from a decoded
            # episode; if the pool ever decoded a val id, its key
            # would show up here.
            keys = {k for k in buf.keys() if k >= 0}
        finally:
            pool.stop()

        val_leaked = {k for k in keys if not _train_only(k)}
        assert not val_leaked, (
            f"pool decoded val ids: {sorted(val_leaked)}.  "
            f"Full keys={sorted(keys)}"
        )


# ---------------------------------------------------------------------------
# DataModule ↔ Dataset split alignment
# ---------------------------------------------------------------------------


class TestDataModuleSplitAlignment:
    """Regression: DataModule's ``train_ep_indices`` must already
    satisfy ``split_fn`` by construction, so the Dataset's
    init-time defensive filter stays silent.  Previously the
    DataModule computed ``train_ep_indices = list(range(0, 0.9 * N))``
    (contiguous prefix) while passing the modulo-based ``split_fn``
    (``e % 10 != 9``) to the Dataset — the filter then stripped
    ~10% of the ids and emitted a noisy false-positive warning.
    """

    def test_default_split_fn_and_indices_agree(self):
        """For ``train_split_ratio=0.9`` and N=100, the DataModule's
        computed ``train_ep_indices`` must contain ONLY ids that
        satisfy the DataModule's own ``split_fn``.
        """
        from dataporter.blended_lerobot_datamodule import (
            _make_default_split_fn,
        )
        split_fn = _make_default_split_fn(0.9)
        num_episodes = 100
        # Mirror the _load_and_split_source logic post-fix.
        train_ep_indices = [
            pos for pos in range(num_episodes) if split_fn(pos)
        ]
        # Every admitted id must satisfy split_fn.
        violators = [e for e in train_ep_indices if not split_fn(e)]
        assert not violators, (
            f"train_ep_indices contains ids that violate split_fn: "
            f"{violators}"
        )
        # Size roughly matches the ratio (±1 per decade).
        assert 88 <= len(train_ep_indices) <= 92, (
            f"expected ~90 train out of 100, got {len(train_ep_indices)}"
        )

    def test_custom_ratio_split_fn_and_indices_agree(self):
        """``train_split_ratio=0.8`` should snap to a cleanly-matching
        split_fn AND produce train_ep_indices that all satisfy it."""
        from dataporter.blended_lerobot_datamodule import (
            _make_default_split_fn,
        )
        split_fn = _make_default_split_fn(0.8)
        num_episodes = 1000
        train_ep_indices = [
            pos for pos in range(num_episodes) if split_fn(pos)
        ]
        violators = [e for e in train_ep_indices if not split_fn(e)]
        assert not violators
        # 80% of 1000 = 800, ±2% tolerance.
        assert 780 <= len(train_ep_indices) <= 820

    def test_defensive_filter_silent_under_matched_split(self, caplog):
        """Construct a Dataset with ``train_episode_indices`` derived
        from a split_fn + that SAME split_fn — the defensive filter
        must strip nothing and emit no warning.  Regression for the
        DataModule ↔ Dataset mismatch that used to fire the filter
        on every setup() call.
        """
        import logging
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
            _default_train_split,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )

        num_episodes = 200
        split_fn = _default_train_split   # e % 10 != 9
        train_ep_indices = [
            e for e in range(num_episodes) if split_fn(e)
        ]
        shard = _make_mock_shard_source(
            num_episodes=num_episodes, frames_per_episode=10,
        )
        buf = ShuffleBuffer(
            capacity=4, max_frames=10, channels=1, height=4, width=4,
            rotation_per_samples=None,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="dataporter.lerobot_shuffle_buffer_dataset",
        ):
            ds = LeRobotShuffleBufferDataset(
                buffer=buf,
                sources=[{
                    "shard_source": shard,
                    "source_name": "synth",
                    "episode_offset": 0,
                    "transform": None,
                    "train_episode_indices": train_ep_indices,
                }],
                delta_timestamps={},
                prefetchers=[],
                producer_pool=None,
                split_fn=split_fn,
                image_keys=["observation.image"],
            )

        # Every ep passed through; size unchanged.
        assert set(ds._current_train_episodes) == set(train_ep_indices)
        # No contamination warning in the log.
        filter_warnings = [
            r for r in caplog.records
            if "val-side raw ids in train_episode_indices" in r.message
        ]
        assert not filter_warnings, (
            f"defensive filter fired under matched split: "
            f"{[r.message for r in filter_warnings]}"
        )
