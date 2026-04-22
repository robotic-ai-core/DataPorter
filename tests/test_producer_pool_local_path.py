"""E2E tests for ProducerPool with a LOCAL-root source.

The bug we're chasing: when BlendedLeRobotDataModule is pointed at a
local filesystem path (``prefetch=False``), the spawned child never
fills the buffer — the generic 300s ``wait_for_warmup`` TimeoutError
fires with nothing useful in the error queue.  HF-sourced configs
work end-to-end; only the local-root path exhibits the hang.

These tests use the production dataset at
``/mnt/Data/lewm-pusht-96x96-full`` (18k episodes, v2.1 layout, 96x96
mp4s) when available.  That's the exact dataset that reproduces the
bug in production.

Passing tests equal "happy path works on a local dataset."  When the
hang is reproduced, the test either times out explicitly or surfaces a
clear error via the child's error_queue — and the instrumented child
logs show us where it got stuck.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest


pytest.importorskip("lerobot")


REPO_ID = "local/lewm-pusht-96x96-full"
LOCAL_ROOT = Path("/mnt/Data/lewm-pusht-96x96-full")


@pytest.fixture(scope="module")
def local_root() -> Path:
    if not LOCAL_ROOT.is_dir():
        pytest.skip(f"Local dataset not present at {LOCAL_ROOT}")
    # Minimal sanity check on LeRobot layout.
    for rel in ("data/chunk-000", "videos/chunk-000", "meta/info.json"):
        if not (LOCAL_ROOT / rel).exists():
            pytest.skip(
                f"{LOCAL_ROOT} doesn't look like a LeRobot v2.1 root "
                f"(missing {rel})"
            )
    return LOCAL_ROOT


def _suppress_lerobot_pull(monkeypatch):
    """Stop LeRobot's base __init__ from hitting HF snapshot_download.

    For a local-root dataset all files are already present; the base
    class's ``download_episodes`` call is a no-op that still tries to
    contact HF via ``snapshot_download`` — and in a machine with
    multiple revision caches, trips on ``SameFileError``.  Patch it to
    a no-op so construction can proceed against the local files only.
    """
    import lerobot.common.datasets.lerobot_dataset as _ld

    def _noop_pull(self, *args, **kwargs):
        return None

    monkeypatch.setattr(_ld.LeRobotDataset, "pull_from_repo", _noop_pull)


# ---------------------------------------------------------------------------
# Instrumentation pass-through: ensure child logs reach the captured
# stderr.  Lightning + pytest both suppress INFO by default; this test
# module opts into it so we can see child milestones.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_child_logging(caplog):
    """Enable INFO-level capture for the child's milestones."""
    caplog.set_level(logging.INFO, logger="dataporter.producer_pool")
    caplog.set_level(logging.INFO, logger="dataporter")
    yield


# ---------------------------------------------------------------------------
# Bug reproduction / happy-path tests
# ---------------------------------------------------------------------------


class TestLocalPathProducerStartup:
    """Confirms child can construct its FastLeRobotDataset and pass the
    smoke test when pointed at a local path.  If the child hangs, the
    test times out with a captured log trail.
    """

    def test_local_path_child_starts_and_fills_buffer(
        self, local_root, monkeypatch,
    ):
        """End-to-end: build pool against the local-path dataset, start
        it, wait for warmup.  Succeeds when the child's instrumentation
        shows FastLeRobotDataset constructed + at least one decode
        completed before the warmup deadline.
        """
        _suppress_lerobot_pull(monkeypatch)

        from dataporter import FastLeRobotDataset, ResizeFrames
        from dataporter.producer_pool import ProducerConfig, ProducerPool
        from dataporter.shuffle_buffer import ShuffleBuffer

        # Build the parent's FastLeRobotDataset — same path the
        # DataModule takes when `prefetch=False`.
        # Restrict to first 20 episodes so the test is fast.
        t0 = time.monotonic()
        parent = FastLeRobotDataset(
            REPO_ID,
            root=local_root,
            delta_timestamps={"observation.image": [0.0]},
            episodes=list(range(20)),
        )
        parent_construct_s = time.monotonic() - t0
        print(
            f"\n[test] parent FastLeRobotDataset constructed in "
            f"{parent_construct_s:.2f}s "
            f"(arrow_cache_path set? {parent.arrow_cache_path is not None})"
        )

        total_eps = len(parent.episode_data_index["from"])
        train_eps = list(range(min(5, total_eps)))

        resize = ResizeFrames(32, 32)
        buf = ShuffleBuffer(
            capacity=2, max_frames=200, channels=3, height=32, width=32,
        )

        config = ProducerConfig.from_source(
            source={"repo_id": REPO_ID, "weight": 1.0},
            full_ds=parent,
            iteration_episodes=train_eps,
            producer_transform=resize,
        )
        print(
            f"[test] ProducerConfig: arrow_cache_path="
            f"{config.arrow_cache_path!r}, "
            f"dataset_episodes_len="
            f"{len(config.dataset_episodes) if config.dataset_episodes else None}"
        )

        pool = ProducerPool(buf, configs=[config], total_workers=1)
        pool.start()
        try:
            t0 = time.monotonic()
            pool.wait_for_warmup(timeout=120.0)
            elapsed = time.monotonic() - t0
        finally:
            pool.stop()

        assert len(buf) >= 1, (
            f"pool warmed up in {elapsed:.1f}s but buffer empty — "
            f"smoke test passed but first decode never landed"
        )


@pytest.mark.slow
@pytest.mark.parametrize("n_episodes", [100, 500, 2000])
def test_local_path_scales_to_large_episode_counts(
    local_root, monkeypatch, n_episodes, capfd,
):
    """Scaling probe: measures FastLeRobotDataset construction time as a
    function of episode count.  If the child's smoke-test timeout
    (``_DECODE_TIMEOUT_S=30s``) is exceeded at realistic dataset sizes,
    this is where the production hang originates.
    """
    _suppress_lerobot_pull(monkeypatch)

    from dataporter import FastLeRobotDataset, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    eps = list(range(n_episodes))
    t0 = time.monotonic()
    parent = FastLeRobotDataset(
        REPO_ID,
        root=local_root,
        delta_timestamps={"observation.image": [0.0]},
        episodes=eps,
    )
    parent_t = time.monotonic() - t0
    print(
        f"\n[scale] n_episodes={n_episodes}: parent construction = "
        f"{parent_t:.2f}s, total_rows={len(parent.hf_dataset)}"
    )

    resize = ResizeFrames(32, 32)
    buf = ShuffleBuffer(
        capacity=2, max_frames=200, channels=3, height=32, width=32,
    )
    config = ProducerConfig.from_source(
        source={"repo_id": REPO_ID, "weight": 1.0},
        full_ds=parent,
        iteration_episodes=eps[:5],
        producer_transform=resize,
    )
    pool = ProducerPool(buf, configs=[config], total_workers=1)
    t_child = time.monotonic()
    pool.start()
    try:
        pool.wait_for_warmup(timeout=180.0)
    finally:
        pool.stop()
    child_t = time.monotonic() - t_child
    print(
        f"[scale] n_episodes={n_episodes}: "
        f"parent={parent_t:.1f}s, child_warmup={child_t:.1f}s"
    )
    """Sanity probe: for a local-root parent (episodes=None), verify
    what ProducerConfig.from_source actually serializes.  Matters for
    the hang hypothesis — if dataset_episodes is None but the parent's
    arrow cache covers all episodes, the child's size-mismatch assertion
    would fire rather than hanging.
    """
    _suppress_lerobot_pull(monkeypatch)

    from dataporter import FastLeRobotDataset
    from dataporter.producer_pool import ProducerConfig

    parent = FastLeRobotDataset(
        REPO_ID,
        root=local_root,
        delta_timestamps={"observation.image": [0.0]},
        episodes=list(range(20)),
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": REPO_ID, "weight": 1.0},
        full_ds=parent,
        iteration_episodes=[0, 1, 2],
    )

    # Diagnostics for the ticket — these surface the exact values the
    # child process is going to receive.
    print(f"\n[probe] parent.episodes = {parent.episodes!r}")
    print(f"[probe] parent.arrow_cache_path = {parent.arrow_cache_path!r}")
    print(f"[probe] cfg.dataset_episodes = {cfg.dataset_episodes!r}")
    print(f"[probe] cfg.arrow_cache_path = {cfg.arrow_cache_path!r}")
    print(f"[probe] cfg.episode_indices = {cfg.episode_indices!r}")
    print(
        f"[probe] len(parent.hf_dataset) = {len(parent.hf_dataset)}, "
        f"len(episode_data_index['from']) = "
        f"{len(parent.episode_data_index['from'])}"
    )

    # Assertions match my current understanding; if any of these are
    # wrong the probe output above tells us what actually happens.
    assert cfg.root == str(parent.root)
    # If parent loaded via data_dir (episodes=None), arrow_cache_path
    # behaviour is the key divergence — log it rather than assert.
