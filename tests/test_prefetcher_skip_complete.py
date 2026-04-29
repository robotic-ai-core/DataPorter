"""Tests for BlendedLeRobotDataModule's skip-prefetcher-when-complete
optimization (polish (B) from the rsync-to-known-path workflow).

Motivation: when sparkinstance's ``hf_cache_source`` (or any other
external mechanism) has rsynced a fully-populated LeRobot v2.1 layout
into ``/tmp/prefetch/<repo>/`` before training starts, calling
``snapshot_download`` to "verify" the cache eats HF's XET 500-req/
5min rate limit budget for nothing.  ``_layout_is_complete`` lets
us detect that case and skip the prefetcher entirely.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

from dataporter.blended_lerobot_datamodule import (
    BlendedLeRobotDataModule,
    _layout_is_complete,
)
from test_shard_source_pool_e2e import _make_dataset


def test_layout_is_complete_recognises_full_layout(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(
        root, ready_eps=list(range(10)), total_episodes=10,
        n_frames_per_ep=4,
    )
    assert _layout_is_complete(root)


def test_layout_is_complete_rejects_partial(tmp_path):
    """Only some episodes ready — prefetcher still has work to do."""
    root = tmp_path / "ds"
    # Declared 10, only 5 ready on disk.
    _make_dataset(
        root, ready_eps=list(range(5)), total_episodes=10,
        n_frames_per_ep=4,
    )
    assert not _layout_is_complete(root)


def test_layout_is_complete_rejects_missing_meta(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(
        root, ready_eps=[0], total_episodes=1, n_frames_per_ep=4,
    )
    (root / "meta" / "info.json").unlink()
    assert not _layout_is_complete(root)


def test_layout_is_complete_rejects_missing_episodes_jsonl(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(
        root, ready_eps=[0], total_episodes=1, n_frames_per_ep=4,
    )
    (root / "meta" / "episodes.jsonl").unlink()
    assert not _layout_is_complete(root)


def test_start_prefetcher_skips_snapshot_when_complete(
    tmp_path, monkeypatch,
):
    """When ``/tmp/prefetch/<ds>/`` is fully populated, the prefetcher
    short-circuit fires: no ``LeRobotPrefetcher`` instance is
    created, no ``snapshot_download`` call happens, and ``source[
    'root']`` points at the local layout.

    ``monkeypatch`` replaces the ``/tmp/prefetch/<ds>/`` path the
    datamodule hard-codes with a path under ``tmp_path`` — keeps the
    test hermetic without touching the real ``/tmp``.
    """
    repo_id = "synth/skip-test"
    fake_prefetch_dir = tmp_path / "prefetch_root"
    fake_prefetch_dir.mkdir()
    target = fake_prefetch_dir / repo_id.replace("/", "_")
    _make_dataset(
        target, ready_eps=list(range(10)), total_episodes=10,
        n_frames_per_ep=4,
    )

    # Redirect the hard-coded ``/tmp/prefetch/...`` to our hermetic dir.
    import dataporter.blended_lerobot_datamodule as dm_mod
    real_path_cls = dm_mod.Path

    class _Path(type(real_path_cls(""))):
        def __new__(cls, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith(
                "/tmp/prefetch/",
            ):
                return real_path_cls(
                    str(fake_prefetch_dir / args[0][len("/tmp/prefetch/"):])
                )
            return real_path_cls(*args, **kwargs)

    monkeypatch.setattr(dm_mod, "Path", _Path)

    # Sabotage LeRobotPrefetcher so a real instantiation would error
    # loudly — proves the skip path is the one actually taken.
    def _exploding_prefetcher(*a, **kw):
        raise AssertionError(
            "LeRobotPrefetcher instantiated despite complete layout — "
            "skip optimization regressed."
        )
    import dataporter.lerobot_prefetcher as pf_mod
    monkeypatch.setattr(pf_mod, "LeRobotPrefetcher", _exploding_prefetcher)

    dm = BlendedLeRobotDataModule(
        repo_id=[{"repo_id": repo_id, "weight": 1.0, "prefetch": True}],
        delta_timestamps={},
        batch_size=4,
        num_workers=0,
    )

    # Calling _start_prefetcher would error if the skip path didn't fire.
    dm._start_prefetcher(dm._sources[0])
    assert dm._sources[0]["root"] == str(target)
    # The prefetchers list stays empty — no instance was registered.
    assert dm._prefetchers == []
