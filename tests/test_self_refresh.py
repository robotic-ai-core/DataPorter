"""Tests for LeRobotShuffleBufferDataset self-refresh + pinned-len modes.

These modes let a caller drop ``GrowingDatasetCallback`` and
``reload_dataloaders_every_n_epochs=1`` entirely: each DataLoader
worker keeps its own admission map fresh via ``self.refresh()`` every
N ``__getitem__`` calls, and ``__len__`` is pinned so Lightning's
``num_training_batches`` cache stays correct.

Coverage:
- Pinned ``__len__`` ignores admission growth.
- ``refresh_every_n_items`` fires ``self.refresh()`` at the right cadence.
- Errors inside the self-refresh path don't kill ``__getitem__``.
- End-to-end: self-refresh admits new on-disk episodes without an
  external callback, and new keys reach the ShuffleBuffer.
- Back-compat: passing neither new arg preserves existing behaviour.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
import torch

# Reuse the synthetic dataset helpers and live-disk prefetcher from the
# main e2e module.
from test_shard_source_pool_e2e import (
    _LiveDiskPrefetcher,
    _build_pool_and_consumer,
    _make_dataset,
    _wait_for_keys,
    _write_episode_mp4,
    _write_episode_parquet,
)


pytest.importorskip("lerobot")


# ---------------------------------------------------------------------------
# Unit-level: pinned __len__ and self-refresh cadence (no pool involved)
# ---------------------------------------------------------------------------


class TestPinnedLen:

    def _make_consumer(self, *, nominal_total_frames=None, **kwargs):
        """Build a minimal consumer over an in-memory mock shard source
        so we can poke ``__len__`` / ``__getitem__`` without a pool.
        """
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )

        shard = _make_mock_shard_source(
            num_episodes=5, frames_per_episode=10,
        )
        buf = ShuffleBuffer(
            capacity=8, max_frames=10, channels=3, height=4, width=4,
        )
        return LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "train_episode_indices": list(range(5)),
                "transform": None,
            }],
            delta_timestamps={},
            image_keys=["observation.image"],
            nominal_total_frames=nominal_total_frames,
            **kwargs,
        )

    def test_default_len_follows_admitted(self):
        """Without ``nominal_total_frames``, __len__ tracks admission
        (historical behaviour)."""
        ds = self._make_consumer()
        assert len(ds) == 5 * 10   # 5 eps × 10 frames

    def test_pinned_len_ignores_admission(self):
        """With ``nominal_total_frames`` set, __len__ returns that
        number regardless of what's admitted.
        """
        ds = self._make_consumer(nominal_total_frames=999)
        assert len(ds) == 999
        # Shrinking the admitted set doesn't shrink __len__.
        ds._admit_by_source({"synth": [0]})
        assert len(ds) == 999
        # Growing doesn't grow it either.
        ds._admit_by_source({"synth": [0, 1, 2, 3, 4]})
        assert len(ds) == 999


# ---------------------------------------------------------------------------
# Unit-level: self-refresh cadence
# ---------------------------------------------------------------------------


class TestSelfRefreshCadence:

    def test_calls_refresh_every_n_items(self, monkeypatch):
        """``__getitem__`` triggers ``self.refresh()`` exactly when
        the counter hits a multiple of ``refresh_every_n_items``.
        """
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )

        shard = _make_mock_shard_source(
            num_episodes=3, frames_per_episode=10,
        )
        buf = ShuffleBuffer(
            capacity=8, max_frames=10, channels=3, height=4, width=4,
        )
        # Fill buffer so __getitem__ can route.
        for ep in range(3):
            buf.put(ep, torch.zeros((10, 3, 4, 4), dtype=torch.uint8))

        ds = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "train_episode_indices": [0, 1, 2],
                "transform": None,
            }],
            delta_timestamps={},
            image_keys=["observation.image"],
            refresh_every_n_items=5,
        )

        fire_count = {"n": 0}

        def fake_refresh(min_new=None, timeout=None):
            fire_count["n"] += 1
            return 0

        monkeypatch.setattr(ds, "refresh", fake_refresh)

        # Calls 1..4: no fire.  Call 5: fire 1.  Calls 6..9: no fire.
        # Call 10: fire 2.
        for _ in range(10):
            ds[0]
        assert fire_count["n"] == 2, (
            f"expected 2 refresh calls after 10 __getitem__ calls with "
            f"refresh_every_n_items=5; got {fire_count['n']}"
        )

    def test_disabled_when_arg_is_none(self, monkeypatch):
        """``refresh_every_n_items=None`` (default) must NOT trigger
        any self-refresh calls, preserving the historical
        callback-driven flow.
        """
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )

        shard = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)
        buf = ShuffleBuffer(
            capacity=8, max_frames=10, channels=3, height=4, width=4,
        )
        for ep in range(3):
            buf.put(ep, torch.zeros((10, 3, 4, 4), dtype=torch.uint8))

        ds = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "train_episode_indices": [0, 1, 2],
                "transform": None,
            }],
            delta_timestamps={},
            image_keys=["observation.image"],
            # refresh_every_n_items omitted — default None.
        )

        fires = {"n": 0}

        def fake_refresh(min_new=None, timeout=None):
            fires["n"] += 1
            return 0

        monkeypatch.setattr(ds, "refresh", fake_refresh)

        for _ in range(100):
            ds[0]
        assert fires["n"] == 0, (
            f"self-refresh fired {fires['n']} times when arg was None"
        )

    def test_refresh_exception_swallowed_and_logged(
        self, monkeypatch, caplog,
    ):
        """A transient error inside self.refresh() must not crash
        ``__getitem__`` — the training step should continue on the
        previously-admitted set."""
        from dataporter.lerobot_shuffle_buffer_dataset import (
            LeRobotShuffleBufferDataset,
        )
        from dataporter.shuffle_buffer import ShuffleBuffer
        from test_lerobot_shuffle_buffer_dataset import (
            _make_mock_shard_source,
        )

        shard = _make_mock_shard_source(num_episodes=3, frames_per_episode=10)
        buf = ShuffleBuffer(
            capacity=8, max_frames=10, channels=3, height=4, width=4,
        )
        for ep in range(3):
            buf.put(ep, torch.zeros((10, 3, 4, 4), dtype=torch.uint8))

        ds = LeRobotShuffleBufferDataset(
            buffer=buf,
            sources=[{
                "shard_source": shard,
                "source_name": "synth",
                "episode_offset": 0,
                "train_episode_indices": [0, 1, 2],
                "transform": None,
            }],
            delta_timestamps={},
            image_keys=["observation.image"],
            refresh_every_n_items=3,
        )

        def boom(min_new=None, timeout=None):
            raise RuntimeError("simulated disk scan failure")

        monkeypatch.setattr(ds, "refresh", boom)

        with caplog.at_level(
            logging.WARNING,
            logger="dataporter.lerobot_shuffle_buffer_dataset",
        ):
            # Should not raise even though every 3rd call hits boom().
            for _ in range(9):
                ds[0]

        assert any(
            "self-refresh failed" in r.message for r in caplog.records
        ), "expected a warning log for the swallowed refresh error"


# ---------------------------------------------------------------------------
# End-to-end: self-refresh + pinned len against a running pool
# ---------------------------------------------------------------------------


def test_self_refresh_admits_new_disk_episodes_without_callback(tmp_path):
    """Without any ``GrowingDatasetCallback``, a consumer with
    ``refresh_every_n_items`` set discovers on-disk episodes that
    landed after pool start — just by being iterated.

    Proves the self-refresh path end-to-end: worker counter crosses
    threshold → worker calls self.refresh() → rescan prefetcher →
    _admit_by_source → pool.update_episodes → decode → new keys in
    buffer.
    """
    root, _src, buf, pool, consumer, _pf = _build_pool_and_consumer(
        tmp_path, initial_eps=[0, 1],
    )
    # Swap in a self-refreshing consumer with the SAME underlying
    # buffer/pool/source dicts so this test is solely about the
    # consumer's self-refresh mechanic.
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    consumer = LeRobotShuffleBufferDataset(
        buffer=consumer._buffer,
        sources=consumer._sources,
        delta_timestamps={},
        prefetchers=consumer._prefetchers,
        producer_pool=consumer._producer_pool,
        split_fn=consumer._split_fn,
        image_keys=consumer._image_keys,
        refresh_every_n_items=3,
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Drop new episodes on disk; no callback fires them.
        for ep in (5, 11):
            _write_episode_parquet(root, ep, n_frames=20)
            _write_episode_mp4(
                root, ep, n_frames=20, height=32, width=32, fps=30,
            )

        # Pull samples from the consumer — every 3rd call triggers
        # self.refresh().  Within a few iterations the new episodes
        # should be admitted and the pool should start decoding them.
        for _ in range(12):
            try:
                _ = consumer[0]
            except Exception:
                # Buffer routing miss is tolerated; we're only driving
                # the counter.
                pass

        # The refresh(es) must have admitted the new eps.
        assert {5, 11} <= set(consumer._current_train_episodes), (
            f"self-refresh didn't admit new eps; admitted="
            f"{consumer._current_train_episodes}"
        )

        # Pool should pick them up and land them in the buffer.
        seen = _wait_for_keys(buf, wanted={5, 11}, timeout=30.0)
        assert seen, (
            f"new eps admitted via self-refresh but never reached the "
            f"buffer; keys={buf.keys()}"
        )
    finally:
        pool.stop()


def test_pinned_len_survives_admission_growth_end_to_end(tmp_path):
    """With ``nominal_total_frames`` pinned, ``__len__`` stays at the
    nominal value regardless of how the admitted set grows.  That's
    what lets Lightning's ``num_training_batches`` stay stable without
    ``reload_dataloaders_every_n_epochs=1``.
    """
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )

    root, _src, _buf, _pool, consumer, _pf = _build_pool_and_consumer(
        tmp_path, initial_eps=[0, 1],
    )
    pinned = 12_345
    consumer = LeRobotShuffleBufferDataset(
        buffer=consumer._buffer,
        sources=consumer._sources,
        delta_timestamps={},
        prefetchers=consumer._prefetchers,
        producer_pool=consumer._producer_pool,
        split_fn=consumer._split_fn,
        image_keys=consumer._image_keys,
        nominal_total_frames=pinned,
    )

    # Before any growth.
    assert len(consumer) == pinned

    # Simulate growth via direct admission.
    consumer._admit_by_source({"synth": [0, 1, 2, 3]})
    assert len(consumer) == pinned, (
        "pinned __len__ must not change when admission grows"
    )
    consumer._admit_by_source({"synth": [0]})
    assert len(consumer) == pinned, (
        "pinned __len__ must not change when admission shrinks"
    )
