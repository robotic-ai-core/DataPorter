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


# ===========================================================================
# Regression: Bug 1 — no-prefetcher + self_refresh must not wipe admission
# ===========================================================================


def test_self_refresh_on_local_only_source_preserves_admission(tmp_path):
    """Regression for commit-305e35a wipe bug.

    Repro: a source with a local ``root`` (no prefetcher) + train
    episodes admitted at construction via ``train_episode_indices`` +
    ``refresh_every_n_items`` set.  After N __getitem__ calls the
    consumer must still have the admitted set — the old
    ``_scan_ready_train_episodes_by_source`` enumerated
    ``self._prefetchers`` and returned ``{}`` when none were attached,
    which tripped ``_admit_by_source({})`` into wiping
    ``_current_train_episodes``.

    Under the shard-driven discovery design the scan uses
    ``shard.list_ready_episodes()`` for every source, so static sources
    return their own ready set and the wipe can no longer happen.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0, 1, 2, 3], total_episodes=4)
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=8, max_frames=32, channels=3, height=32, width=32,
    )
    # Fill buffer so __getitem__ can route.
    for ep in [0, 1, 2, 3]:
        buf.put(ep, torch.zeros((20, 3, 32, 32), dtype=torch.uint8))

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[{
            "shard_source": src,
            "source_name": "local",
            "episode_offset": 0,
            "train_episode_indices": [0, 1, 2, 3],
            "transform": None,
        }],
        delta_timestamps={},
        prefetchers=[],   # local-only source — no prefetcher
        producer_pool=None,
        split_fn=lambda ep: True,
        image_keys=["observation.image"],
        refresh_every_n_items=5,
    )

    assert set(consumer._current_train_episodes) == {0, 1, 2, 3}

    # Drive past the refresh threshold a few times.  Under the old
    # scanner this wiped _current_train_episodes to [].
    for _ in range(30):
        _ = consumer[0]

    assert set(consumer._current_train_episodes) == {0, 1, 2, 3}, (
        f"self-refresh on a no-prefetcher source wiped admission to "
        f"{consumer._current_train_episodes}"
    )


# ===========================================================================
# Regression: Bug 2 — update_episodes must not raise cross-fork
# ===========================================================================


def _call_update_episodes_in_subprocess(conn, buffer, config) -> None:
    """Run inside a spawned child: construct a ProducerPool whose
    worker handle was created in ANOTHER process, then call
    update_episodes.  Returns ``("ok", None)`` or ``("err", str)``.
    """
    from dataporter.producer_pool import ProducerPool
    try:
        pool = ProducerPool(
            buffer, configs=[config], total_workers=1, warmup_target=1,
        )
        # Without start(), self._worker is None — is_alive() was the
        # only path that would have raised the cross-fork assertion.
        # Simulate a post-start state by monkeypatching to a stub
        # Process handle whose is_alive() asserts on parent_pid.
        import multiprocessing as _mp
        import os as _os

        class _FakeProcess:
            _parent_pid = -1     # never matches real pid → would assert
            def is_alive(self):
                assert self._parent_pid == _os.getpid(), (
                    "can only test a child process"
                )
                return True

        pool._worker = _FakeProcess()
        pool.update_episodes("synth", [0, 1, 2])
        conn.send(("ok", None))
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def test_update_episodes_does_not_raise_cross_fork(tmp_path):
    """Regression: update_episodes once called ``self.is_alive`` which
    invoked ``mp.Process.is_alive()`` on a handle owned by the parent.
    Any non-owning process got ``AssertionError: can only test a child
    process``.  The fix drops the is_alive gate — queue puts survive
    fork, and if the pool is dead, put_nowait fails with a real
    exception.

    This test simulates a forked worker by spawning a subprocess that
    calls ``update_episodes`` on a pool whose ``_worker`` is a stub
    whose ``is_alive()`` always asserts.  Under the old code this
    would propagate the AssertionError out; under the fix
    update_episodes never touches is_alive and the put goes through.
    """
    import multiprocessing as mp

    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0], total_episodes=1)
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=2, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0],
        producer_transform=ResizeFrames(32, 32),
    )

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_call_update_episodes_in_subprocess,
        args=(child_conn, buf, cfg),
    )
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0, f"subprocess exited with {proc.exitcode}"
    status, msg = parent_conn.recv()
    assert status == "ok", (
        f"update_episodes raised in cross-process context: {msg}"
    )


# ===========================================================================
# Mixed config: HF-prefetched source + local-root source discover together
# ===========================================================================


def test_mixed_prefetched_and_local_sources_discover_together(tmp_path):
    """With one prefetched source and one local-only source, the scan
    must discover ready episodes for BOTH — not just the prefetched
    one.  The fix iterates self._sources (not self._prefetchers) so
    each source's own shard_source drives discovery regardless of
    whether it has an attached prefetcher.
    """
    from dataporter import LeRobotShardSource
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.shuffle_buffer import ShuffleBuffer

    root_hf = tmp_path / "hf_like"
    root_local = tmp_path / "local"
    _make_dataset(root_hf, ready_eps=[0, 1], total_episodes=10)
    _make_dataset(root_local, ready_eps=[0, 1, 2], total_episodes=3)
    shard_hf = LeRobotShardSource(root_hf)
    shard_local = LeRobotShardSource(root_local)

    # Prefetcher is a stand-in; only its is_done() is used by the
    # consumer now (shard.list_ready_episodes() drives discovery).
    prefetcher_hf = _LiveDiskPrefetcher(root_hf)

    buf = ShuffleBuffer(
        capacity=16, max_frames=32, channels=3, height=32, width=32,
    )
    # Fill buffer stubs so __getitem__ doesn't crash on route-miss.
    for k in [0, 1, 100, 101, 102]:
        buf.put(k, torch.zeros((20, 3, 32, 32), dtype=torch.uint8))

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[
            {
                "shard_source": shard_hf,
                "source_name": "hf",
                "episode_offset": 0,
                "transform": None,
            },
            {
                "shard_source": shard_local,
                "source_name": "local",
                "episode_offset": 100,
                "train_episode_indices": [0, 1, 2],   # static
                "transform": None,
            },
        ],
        delta_timestamps={},
        prefetchers=[prefetcher_hf],   # only for the HF source
        producer_pool=None,
        split_fn=lambda ep: True,
        image_keys=["observation.image"],
    )

    # Refresh should populate the HF source from disk and preserve the
    # local source's static admission — NOT wipe either.
    new_len = consumer.refresh(min_new=0)
    admitted = set(consumer._current_train_episodes)
    # HF source: raw ids 0, 1 → keys 0, 1
    assert {0, 1} <= admitted, (
        f"HF source's on-disk eps not admitted via scan; got {admitted}"
    )
    # Local source: raw ids 0, 1, 2 → keys 100, 101, 102
    assert {100, 101, 102} <= admitted, (
        f"Local source's static eps not admitted; got {admitted}"
    )
    assert new_len > 0
