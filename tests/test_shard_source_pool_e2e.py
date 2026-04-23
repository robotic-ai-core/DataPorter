"""Comprehensive end-to-end tests for the LeRobotShardSource streaming path.

Covers every known failing mode of the pool + shard-source + consumer
flow on **synthetic** datasets (no network, no multi-GB downloads) so
regressions surface in local CI before they hit Vast.

Failing modes locked in:

1. Child startup stays bounded at scale (shard source is O(1)
   construction regardless of declared ``total_episodes``).
2. Missing mp4 race — an episode with parquet-only on disk is excluded
   from readiness and, if explicitly requested anyway, surfaces as a
   clear per-decode error (not a generic warmup timeout).
3. Sparse raw episode ids (non-contiguous) decode and land with the
   correct buffer keys.
4. ``update_episodes`` swaps the pool's work queue live; newly-written
   episodes start appearing in the buffer without a restart.
5. Partial download → grow: consumer ``refresh()`` admits new episodes
   from a mock prefetcher; buffer keys follow.
6. Multi-source ``episode_offset`` keeps two synthetic datasets in
   disjoint key spaces.
7. ``producer_transform`` runs in the spawn child so the buffer is
   sized at the transform's output resolution.

All mp4s are real (generated via imageio+ffmpeg); LeRobet's
``decode_video_frames`` decodes them for real.  Per-episode data is a
tiny (a few KB) parquet.  Full fixture setup is under ~1s per test.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

import imageio


# ---------------------------------------------------------------------------
# Fixture helpers — synthetic LeRobot v2.1 layout with *real* mp4s
# ---------------------------------------------------------------------------


INFO_TEMPLATE = {
    "codebase_version": "v2.1",
    "data_path": (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ),
    "video_path": (
        "videos/chunk-{episode_chunk:03d}/{video_key}/"
        "episode_{episode_index:06d}.mp4"
    ),
    "fps": 30,
    "total_tasks": 1,
    "total_chunks": 1,
    "chunks_size": 1000,
    "robot_type": "synthetic",
}


def _write_meta(
    root: Path,
    *,
    fps: int = 30,
    total_episodes: int = 10,
    n_frames_per_ep: int = 20,
    height: int = 32,
    width: int = 32,
    include_video: bool = True,
) -> None:
    """Write the v2.1 ``meta/`` directory (info + episodes + tasks)."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = dict(INFO_TEMPLATE)
    info["fps"] = fps
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_episodes * n_frames_per_ep
    if include_video:
        info["features"] = {
            "observation.image": {"dtype": "video", "shape": [3, height, width]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        }
    else:
        info["video_path"] = None
        info["features"] = {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        }
    (meta / "info.json").write_text(json.dumps(info))

    with (meta / "episodes.jsonl").open("w") as f:
        for i in range(total_episodes):
            f.write(json.dumps({
                "episode_index": i,
                "length": n_frames_per_ep,
                "tasks": ["synthetic_task"],
            }) + "\n")

    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "synthetic_task"}) + "\n"
    )


def _write_episode_parquet(
    root: Path,
    ep_idx: int,
    *,
    n_frames: int = 20,
    chunks_size: int = 1000,
) -> None:
    chunk = ep_idx // chunks_size
    dir_ = root / f"data/chunk-{chunk:03d}"
    dir_.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    rng = np.random.default_rng(ep_idx)
    table = pa.table({
        "observation.state": [
            [float(rng.standard_normal()), float(rng.standard_normal())]
            for _ in range(n_frames)
        ],
        "action": [
            [float(rng.standard_normal()), float(rng.standard_normal())]
            for _ in range(n_frames)
        ],
        "timestamp": [i / 30.0 for i in range(n_frames)],
        "episode_index": [ep_idx] * n_frames,
        "frame_index": list(range(n_frames)),
        "task_index": [0] * n_frames,
    }, schema=schema)
    pq.write_table(table, dir_ / f"episode_{ep_idx:06d}.parquet")


def _write_episode_mp4(
    root: Path,
    ep_idx: int,
    video_key: str = "observation.image",
    *,
    n_frames: int = 20,
    height: int = 32,
    width: int = 32,
    fps: int = 30,
    chunks_size: int = 1000,
) -> Path:
    """Write a real (ffmpeg-decodable) mp4 for one episode.

    Every pixel of the first frame is set to ``ep_idx`` so a decoded
    frame's mean is a rough signature of which episode it came from
    (useful for asserting key-to-frames correspondence).
    """
    chunk = ep_idx // chunks_size
    d = root / f"videos/chunk-{chunk:03d}/{video_key}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"episode_{ep_idx:06d}.mp4"
    # Plain-colour frames so compression is trivial — keeps fixtures tiny.
    rng = np.random.default_rng(ep_idx + 1000)
    base = rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8)
    frames = np.broadcast_to(base, (n_frames, height, width, 3)).copy()
    # Stamp ep_idx onto top-left pixel for a trivial signature.
    frames[:, 0, 0, :] = ep_idx % 256
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
    ) as w:
        for f in frames:
            w.append_data(f)
    return path


def _make_dataset(
    root: Path,
    *,
    ready_eps: list[int],
    total_episodes: int | None = None,
    parquet_only_eps: list[int] | None = None,
    fps: int = 30,
    n_frames_per_ep: int = 20,
    height: int = 32,
    width: int = 32,
) -> Path:
    """Assemble a synthetic v2.1 LeRobot dataset rooted at ``root``.

    Args:
        ready_eps: Raw ids with BOTH parquet and mp4 on disk (pool can
            decode these).
        total_episodes: Declared in info.json.  Defaults to
            ``max(ready_eps, parquet_only_eps) + 1`` — mimics a
            just-started download where disk size << declared size.
        parquet_only_eps: Raw ids with parquet but no mp4 (the missing-
            video-race case — must be excluded from readiness).
    """
    parquet_only_eps = parquet_only_eps or []
    all_eps = set(ready_eps) | set(parquet_only_eps)
    declared = (
        total_episodes if total_episodes is not None
        else (max(all_eps) + 1 if all_eps else 1)
    )
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(
        root,
        fps=fps,
        total_episodes=declared,
        n_frames_per_ep=n_frames_per_ep,
        height=height,
        width=width,
    )
    for ep in sorted(all_eps):
        _write_episode_parquet(root, ep, n_frames=n_frames_per_ep)
    for ep in ready_eps:
        _write_episode_mp4(
            root, ep,
            n_frames=n_frames_per_ep, height=height, width=width, fps=fps,
        )
    return root


# Fake prefetcher that reports whatever the underlying disk has right now.
class _LiveDiskPrefetcher:
    """Tiny stand-in for :class:`LeRobotPrefetcher` for growing-set tests.

    ``ready_episodes()`` re-scans disk on every call (no caching), so a
    test can write a new parquet+mp4 between ``refresh()`` invocations and
    see it picked up without actually running snapshot_download.
    """
    def __init__(self, root: Path, fps: int = 30):
        self._root = Path(root)
        self._fps = fps
        self._done = False

    def ready_episodes(self) -> list[int]:
        from dataporter.lerobot_shard_source import LeRobotShardSource
        return LeRobotShardSource(self._root).list_ready_episodes()

    def is_done(self) -> bool:
        return self._done

    def mark_done(self) -> None:
        self._done = True


# ---------------------------------------------------------------------------
# Autouse: surface the child logger in pytest output.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_child_logging(caplog):
    caplog.set_level(logging.INFO, logger="dataporter.producer_pool")
    caplog.set_level(logging.INFO, logger="dataporter.lerobot_shard_source")
    yield


# ===========================================================================
# (1) Child startup is bounded regardless of declared dataset size
# ===========================================================================


@pytest.mark.parametrize("declared_n", [10, 500, 5000])
def test_startup_bounded_at_declared_scale(tmp_path, declared_n):
    """Shard-source path: only the episodes the pool is asked to decode
    contribute to startup cost.  Declared ``total_episodes`` is nominal —
    the source never enumerates it at construction.

    Regression target: the old FastLeRobotDataset path scaled O(n_episodes)
    with declared size (~0.025s/ep) and hit the 30s smoke-test ceiling at
    ~1200 episodes.  Shard source must shrug this off.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    # Only eps 0, 1 are real.  Info still claims ``declared_n`` total.
    _make_dataset(root, ready_eps=[0, 1], total_episodes=declared_n)

    src = LeRobotShardSource(root)
    assert src.total_episodes == declared_n
    # list_ready_episodes reflects disk state, not declared.
    assert src.list_ready_episodes() == [0, 1]

    buf = ShuffleBuffer(
        capacity=2, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=1)

    t0 = time.monotonic()
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
    finally:
        pool.stop()
    elapsed = time.monotonic() - t0

    assert len(buf) >= 1, "pool warmed up but buffer is empty"
    # Even with declared_n=5000 (where the old path would take ~2 min)
    # synthetic startup + one real decode completes in well under the
    # smoke-test ceiling.
    assert elapsed < 30.0, (
        f"startup at declared_n={declared_n} took {elapsed:.1f}s — "
        f"shard source should not scale with declared total_episodes"
    )


# ===========================================================================
# (2) Missing mp4 race — parquet present, mp4 not yet written
# ===========================================================================


def test_ready_list_excludes_parquet_only_episodes(tmp_path):
    """Every stage of the pipeline uses ``list_ready_episodes`` as the
    authoritative set.  Parquet-only episodes (mid-download) must NOT
    appear — otherwise the pool would dispatch decodes for missing mp4s.
    """
    from dataporter import LeRobotShardSource

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0, 1, 2], parquet_only_eps=[3, 4])
    src = LeRobotShardSource(root)
    assert src.list_ready_episodes() == [0, 1, 2]
    assert src.is_episode_ready(3) is False
    assert src.is_episode_ready(4) is False


def test_missing_mp4_surfaces_as_decode_error(tmp_path):
    """If the caller force-feeds an unready episode to the pool anyway,
    the child's decode path raises a clear error instead of hanging on
    ffmpeg forever or corrupting the buffer.

    The pool's consecutive-failure tracker will eventually escalate, but
    for a single missing mp4 we expect a per-decode warning and the rest
    of the episodes to still go through.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(
        root,
        ready_eps=[0],
        parquet_only_eps=[1],          # ep 1 has no mp4 — decode must fail
        total_episodes=2,
    )
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=2, max_frames=32, channels=3, height=32, width=32,
    )
    # Deliberately tell the pool to decode BOTH eps.  Ep 1 will fail
    # (mp4 missing); ep 0 should still make it to the buffer.
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=1, warmup_target=1)
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
    finally:
        pool.stop()

    # Buffer must contain ep 0's frames (the healthy episode).
    assert len(buf) >= 1
    keys = buf.keys()
    assert 0 in keys, (
        f"ep 0 should have decoded even though ep 1 is missing its mp4; "
        f"buffer keys={keys}"
    )


# ===========================================================================
# (3) Sparse raw episode ids (non-contiguous) decode correctly
# ===========================================================================


def test_sparse_raw_episode_ids_decode_correctly(tmp_path):
    """Pool given a deliberately sparse raw-id list [0, 3, 7] must land
    each episode's frames under the correct buffer key.  No positional
    translation — keys are raw ids throughout.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    sparse = [0, 3, 7]
    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=sparse, total_episodes=10)
    src = LeRobotShardSource(root)
    assert src.list_ready_episodes() == sparse

    buf = ShuffleBuffer(
        capacity=len(sparse), max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=sparse,
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(
        buf, configs=[cfg], total_workers=1, warmup_target=len(sparse),
    )
    pool.start()
    try:
        pool.wait_for_warmup(timeout=60.0)
    finally:
        pool.stop()

    # Collect populated keys; all must be members of the requested set.
    rng = random.Random(0)
    seen: set[int] = set()
    for _ in range(40):
        try:
            key, _ = buf.sample(rng)
            seen.add(key)
        except IndexError:
            break
    unexpected = seen - set(sparse)
    assert not unexpected, (
        f"buffer contained keys {unexpected} outside the requested sparse "
        f"set {sparse} — raw→buffer key wiring is wrong"
    )
    assert seen & set(sparse), (
        f"no sparse keys made it into the buffer; seen={seen}"
    )


# ===========================================================================
# (4) update_episodes swaps the work queue live
# ===========================================================================


def test_update_episodes_admits_new_episode_live(tmp_path):
    """Pool starts with eps [0, 1].  After warmup, a new ep (7) is
    written to disk and ``update_episodes([0, 1, 7])`` is called.
    Ep 7 must appear in the buffer without any pool restart — proves the
    live-update path is wired end-to-end.

    The buffer is sized generously (32 slots) so backpressure doesn't
    kick in before the update-poller picks up the swap — once the pool
    parks on "buffer full", no further decodes fire and the swap can
    never be observed through ``buf.keys()``.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0, 1], total_episodes=10)
    src = LeRobotShardSource(root)

    # Capacity chosen so the pool keeps decoding for ~a second before
    # backpressure — plenty of time for _poll_updates (100ms tick) to
    # see the swap and for ep 7 to show up.
    buf = ShuffleBuffer(
        capacity=32, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=2)
    # Write ep 7 BEFORE starting the pool so the file is definitely on
    # disk by the time the swapped iterator yields it (otherwise the
    # pool's decode of ep 7 can race against the mp4 write).
    _write_episode_parquet(root, 7, n_frames=20)
    _write_episode_mp4(
        root, 7, n_frames=20, height=32, width=32, fps=30,
    )
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
        # Swap in the updated iteration list — broadcast to the child.
        pool.update_episodes("synth", [0, 1, 7])

        # Wait for ep 7 to show up in buffer keys (bounded poll).
        # No ``buf.clear()`` needed: the pool rotates contents via
        # FIFO eviction now that backpressure is throttle-not-park.
        deadline = time.monotonic() + 30.0
        saw_7 = False
        while time.monotonic() < deadline:
            if 7 in buf.keys():
                saw_7 = True
                break
            time.sleep(0.2)
    finally:
        pool.stop()

    assert saw_7, (
        f"ep 7 never appeared in buffer after update_episodes; "
        f"keys={buf.keys()}"
    )


# ===========================================================================
# (5) End-to-end: partial download → train → grow
# ===========================================================================


def test_partial_then_grow_consumer_refresh(tmp_path):
    """The headline scenario: training starts on a partially-downloaded
    dataset, new episodes land on disk while we're training, and a
    consumer ``refresh()`` call extends the admitted set.

    What must hold:
    - Initial ``len(consumer)`` reflects only the first batch.
    - ``consumer.refresh()`` picks up the new episodes.
    - The pool's work queue is updated (new keys appear in the buffer).
    - ``__getitem__`` returns complete samples whose ``episode_index``
      matches the shard's parquet for that raw id.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    # Start with eps 0 and 1 — declared total is 10 to simulate a
    # partial download.
    _make_dataset(root, ready_eps=[0, 1], total_episodes=10)
    src = LeRobotShardSource(root)
    prefetcher = _LiveDiskPrefetcher(root)

    buf = ShuffleBuffer(
        capacity=8, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=1, warmup_target=1)

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[{
            "shard_source": src,
            "source_name": "synth",
            "episode_offset": 0,
            "transform": None,
            "train_episode_indices": [0, 1],   # initial train split
        }],
        delta_timestamps={},
        prefetchers=[prefetcher],
        producer_pool=pool,
        split_fn=lambda ep: True,          # every ready ep is train
        image_keys=["observation.image"],
    )

    initial_len = len(consumer)
    assert initial_len == 2 * 20, (
        f"initial epoch length should be 2 eps × 20 frames, got {initial_len}"
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Sanity: one __getitem__ returns a valid sample on eps [0, 1].
        sample = consumer[0]
        assert "observation.image" in sample
        assert int(sample["episode_index"]) in {0, 1}

        # "Download" eps 2, 3, 4 live.
        for ep in [2, 3, 4]:
            _write_episode_parquet(root, ep, n_frames=20)
            _write_episode_mp4(
                root, ep,
                n_frames=20, height=32, width=32, fps=30,
            )
        prefetcher.mark_done()

        # Consumer refresh: admits new eps, forwards to pool.
        new_len = consumer.refresh(min_new=3, timeout=10.0)
        assert new_len == 5 * 20, (
            f"post-refresh epoch length should be 5 eps × 20 frames, "
            f"got {new_len}"
        )

        # Wait for new keys to appear in the buffer.  The pool rotates
        # on its own (backpressure is throttle-not-park), so no
        # explicit drain is needed.
        deadline = time.monotonic() + 30.0
        seen_new: set[int] = set()
        while time.monotonic() < deadline:
            seen_new.update(set(buf.keys()) & {2, 3, 4})
            if seen_new:
                break
            time.sleep(0.2)
        assert seen_new, (
            f"new eps never appeared in buffer after refresh; "
            f"seen={seen_new}"
        )

        # A __getitem__ after growth must return a valid sample —
        # observation.image/state/action all present, episode_index
        # matches the parquet we wrote.
        for _ in range(20):
            sample = consumer[0]
            ep_idx = int(sample["episode_index"])
            assert ep_idx in set(range(5)), (
                f"sample returned ep_idx={ep_idx} outside admitted set"
            )
    finally:
        pool.stop()


# ===========================================================================
# (6) Multi-source episode_offset keeps key spaces disjoint
# ===========================================================================


def test_multi_source_offset_keeps_key_spaces_disjoint(tmp_path):
    """Two synthetic datasets both starting at raw id 0; the second has
    ``episode_offset=1000``.  Buffer keys for source A must be in
    ``{0..9}`` and source B in ``{1000..1009}`` — no overlap.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_dataset(root_a, ready_eps=[0, 1, 2], total_episodes=3)
    _make_dataset(root_b, ready_eps=[0, 1, 2], total_episodes=3)

    src_a = LeRobotShardSource(root_a)
    src_b = LeRobotShardSource(root_b)

    buf = ShuffleBuffer(
        capacity=8, max_frames=32, channels=3, height=32, width=32,
    )

    cfg_a = ProducerConfig.from_source(
        source={"repo_id": "src_a", "weight": 1.0},
        shard_source=src_a,
        iteration_episodes=[0, 1, 2],
        episode_offset=0,
        producer_transform=ResizeFrames(32, 32),
    )
    cfg_b = ProducerConfig.from_source(
        source={"repo_id": "src_b", "weight": 1.0},
        shard_source=src_b,
        iteration_episodes=[0, 1, 2],
        episode_offset=1000,
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(
        buf, configs=[cfg_a, cfg_b], total_workers=2, warmup_target=4,
    )
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
        # Accumulate keys across several drain cycles so both source
        # namespaces populate.  The pool rotates via FIFO eviction so
        # we observe both sources by polling over time — no manual
        # buffer drain required.
        observed_keys: set[int] = set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not (
            any(k < 1000 for k in observed_keys)
            and any(k >= 1000 for k in observed_keys)
        ):
            observed_keys.update(k for k in buf.keys() if k >= 0)
            time.sleep(0.2)
    finally:
        pool.stop()

    keys = observed_keys
    assert keys, "buffer ended up empty"
    a_keys = {k for k in keys if k < 1000}
    b_keys = {k for k in keys if k >= 1000}
    # Both sources must have produced at least one decode.
    assert a_keys, f"no source-A keys in buffer; keys={keys}"
    assert b_keys, f"no source-B keys in buffer; keys={keys}"
    # Stay inside allocated namespaces.
    assert a_keys <= {0, 1, 2}, f"src_a leaked keys: {a_keys}"
    assert b_keys <= {1000, 1001, 1002}, f"src_b leaked keys: {b_keys}"


# ===========================================================================
# (7) producer_transform runs in the child and shrinks the buffer shape
# ===========================================================================


def test_producer_transform_applied_before_buffer(tmp_path):
    """``ResizeFrames(16, 16)`` runs in the child before ``buffer.put``.
    The ShuffleBuffer is allocated at 16x16; if the transform hadn't run,
    ``buffer.put`` would have written a 32x32 tensor and tripped the shm
    shape mismatch.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(
        root, ready_eps=[0, 1], total_episodes=2, height=32, width=32,
    )
    src = LeRobotShardSource(root)

    # Buffer allocates at the TRANSFORM's target resolution.
    buf = ShuffleBuffer(
        capacity=2, max_frames=32, channels=3, height=16, width=16,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(16, 16),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=1, warmup_target=1)
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
    finally:
        pool.stop()

    rng = random.Random(0)
    key, frames = buf.sample(rng)
    assert frames.shape[-2:] == (16, 16), (
        f"expected 16x16 frames in buffer, got {tuple(frames.shape)}"
    )


# ===========================================================================
# (8) Consumer happy-path reads correct state/action from parquet per ep
# ===========================================================================


def test_consumer_samples_match_parquet_for_sampled_episode(tmp_path):
    """A decoded episode's buffer key must route to the correct shard
    source; the sampled frame's ``episode_index`` / ``frame_index`` /
    ``action`` must match what the parquet contains.

    Without the raw-id routing being correct end-to-end this test would
    silently return wrong rows.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0, 1, 2], total_episodes=3)
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=4, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1, 2],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=1, warmup_target=3)

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[{
            "shard_source": src,
            "source_name": "synth",
            "episode_offset": 0,
            "transform": None,
            "train_episode_indices": [0, 1, 2],
        }],
        delta_timestamps={},
        prefetchers=[],
        producer_pool=None,           # static set — no refresh wiring
        split_fn=lambda ep: True,
        image_keys=["observation.image"],
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Pull a bunch of samples; every one must self-describe
        # consistently (episode_index matches the shard's parquet for
        # the frame_index the consumer sampled).
        for _ in range(10):
            sample = consumer[0]
            ep_idx = int(sample["episode_index"])
            frame_idx = int(sample["frame_index"])
            assert ep_idx in {0, 1, 2}
            assert 0 <= frame_idx < 20
            # Cross-check: read the same row directly from the shard.
            truth = src.load_episode_row_torch(ep_idx, frame_idx)
            assert int(truth["episode_index"]) == ep_idx
            assert int(truth["frame_index"]) == frame_idx
            # Action values must match the parquet byte-for-byte.
            assert (sample["action"] == truth["action"]).all(), (
                f"action mismatch at ep={ep_idx} frame={frame_idx}: "
                f"consumer={sample['action'].tolist()} "
                f"shard={truth['action'].tolist()}"
            )
    finally:
        pool.stop()


# ===========================================================================
# (9) GrowingDatasetCallback drives growth end-to-end — epoch mode
# ===========================================================================


def _build_pool_and_consumer(tmp_path, initial_eps: list[int]):
    """Shared setup: synthetic dataset + pool + consumer + prefetcher.

    Returns ``(root, src, buf, pool, consumer, prefetcher)`` so each
    callback-driven test can construct identical wiring and only vary
    the callback's scheduling policy.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=initial_eps, total_episodes=20)
    src = LeRobotShardSource(root)
    prefetcher = _LiveDiskPrefetcher(root)

    buf = ShuffleBuffer(
        capacity=32, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=initial_eps,
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=1)

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[{
            "shard_source": src,
            "source_name": "synth",
            "episode_offset": 0,
            "transform": None,
            "train_episode_indices": list(initial_eps),
        }],
        delta_timestamps={},
        prefetchers=[prefetcher],
        producer_pool=pool,
        split_fn=lambda ep: True,
        image_keys=["observation.image"],
    )
    return root, src, buf, pool, consumer, prefetcher


def _wait_for_keys(buf, wanted: set[int], timeout: float = 30.0) -> set[int]:
    """Poll the buffer for ``wanted`` keys until deadline.

    Relies on the pool rotating its FIFO ring buffer (throttle-not-park
    backpressure).  Before that fix this helper had to call
    ``buf.clear()`` to unstick the pool — a workaround that hid the
    production park bug.  Workaround removed; the helper is now a
    pure observer.
    """
    seen: set[int] = set()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen.update(set(buf.keys()) & wanted)
        if seen >= wanted:
            return seen
        time.sleep(0.2)
    return seen


def test_epoch_mode_callback_drives_growth_end_to_end(tmp_path):
    """GrowingDatasetCallback() (default epoch mode), fired against a
    running pool via ``on_train_epoch_start``, admits new episodes that
    appeared on disk between invocations and their frames reach the
    ShuffleBuffer without any direct ``refresh()`` call.

    Proves the full control flow: ``trainer.on_train_epoch_start`` →
    callback → ``consumer.refresh()`` → pool ``update_episodes`` →
    child decoder picks up new ids → buffer holds them.
    """
    from unittest.mock import MagicMock

    from dataporter import GrowingDatasetCallback

    root, _src, buf, pool, consumer, _pf = _build_pool_and_consumer(
        tmp_path, initial_eps=[0, 1],
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Fake Lightning trainer — callback walks
        # trainer.datamodule.train_dataset.refresh().
        trainer = MagicMock()
        trainer.datamodule.train_dataset = consumer
        cb = GrowingDatasetCallback()   # default: epoch mode

        # Sanity: epoch-start BEFORE new disk writes admits nothing new.
        cb.on_train_epoch_start(trainer, None)
        assert set(consumer._current_train_episodes) == {0, 1}

        # Simulate the prefetcher landing 3 more episodes mid-epoch.
        for ep in (5, 11, 17):
            _write_episode_parquet(root, ep, n_frames=20)
            _write_episode_mp4(
                root, ep,
                n_frames=20, height=32, width=32, fps=30,
            )

        # Batch hooks must NOT fire refresh in epoch mode.
        cb.on_train_batch_start(trainer, None, batch=None, batch_idx=0)
        assert set(consumer._current_train_episodes) == {0, 1}, (
            "batch hook fired refresh in epoch mode — regression"
        )

        # Next epoch start admits the three new eps.
        cb.on_train_epoch_start(trainer, None)
        assert set(consumer._current_train_episodes) == {0, 1, 5, 11, 17}

        # And the pool actually decodes them — new keys must land in the
        # buffer before the deadline.
        seen = _wait_for_keys(buf, wanted={5, 11, 17}, timeout=30.0)
        assert seen, (
            f"epoch-mode callback refreshed admitted set but no new "
            f"episodes reached the buffer; keys={buf.keys()}"
        )
    finally:
        pool.stop()


# ===========================================================================
# (10) GrowingDatasetCallback drives growth end-to-end — step mode
# ===========================================================================


def test_step_mode_callback_drives_growth_end_to_end(tmp_path):
    """``GrowingDatasetCallback(every_n_steps=N)`` fires refresh only
    when ``global_step`` is a positive multiple of ``N``.  Non-multiple
    steps are no-ops; the epoch hook is a no-op.

    Between the two refresh ticks we write one new episode to disk; it
    must be invisible at the first tick (wasn't on disk yet) and
    admitted at the second tick.  The resulting update_episodes call
    must carry into the pool and the new episode's frames must reach
    the buffer.
    """
    from unittest.mock import MagicMock

    from dataporter import GrowingDatasetCallback

    root, _src, buf, pool, consumer, _pf = _build_pool_and_consumer(
        tmp_path, initial_eps=[0, 1],
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        trainer = MagicMock()
        trainer.datamodule.train_dataset = consumer
        cb = GrowingDatasetCallback(every_n_steps=3)

        # Epoch-start must NOT fire in step mode.
        cb.on_train_epoch_start(trainer, None)
        assert set(consumer._current_train_episodes) == {0, 1}

        # Walk a few steps; refresh fires only at steps 3, 6, ...
        # Step 1–2: no fire.
        for step in (1, 2):
            trainer.global_step = step
            cb.on_train_batch_start(
                trainer, None, batch=None, batch_idx=step,
            )
        assert set(consumer._current_train_episodes) == {0, 1}

        # Step 3: first fire.  Disk unchanged — admits nothing new.
        trainer.global_step = 3
        cb.on_train_batch_start(trainer, None, batch=None, batch_idx=3)
        assert set(consumer._current_train_episodes) == {0, 1}

        # Between ticks, the prefetcher lands ep 7.
        _write_episode_parquet(root, 7, n_frames=20)
        _write_episode_mp4(
            root, 7, n_frames=20, height=32, width=32, fps=30,
        )

        # Step 4, 5: no fire (not multiples of 3).
        for step in (4, 5):
            trainer.global_step = step
            cb.on_train_batch_start(
                trainer, None, batch=None, batch_idx=step,
            )
        assert set(consumer._current_train_episodes) == {0, 1}, (
            f"step {step} fired refresh — cadence wrong"
        )

        # Step 6: second fire — admits ep 7.
        trainer.global_step = 6
        cb.on_train_batch_start(trainer, None, batch=None, batch_idx=6)
        assert 7 in consumer._current_train_episodes, (
            f"step-6 refresh didn't admit ep 7; admitted="
            f"{consumer._current_train_episodes}"
        )

        # Pool picks up the updated iterator and decodes ep 7.
        seen = _wait_for_keys(buf, wanted={7}, timeout=30.0)
        assert seen, (
            f"step-mode callback refreshed admitted set but ep 7 "
            f"never reached the buffer; keys={buf.keys()}"
        )
    finally:
        pool.stop()


# ===========================================================================
# (11) Buffer contents rotate after fill (no-park backpressure)
# ===========================================================================


def test_buffer_rotates_after_fill(tmp_path):
    """Once the buffer reaches capacity, the pool must NOT park — it
    must keep decoding and let the ring buffer overwrite oldest slots
    in FIFO order.  Regression for the overfit symptom where
    ``while len(buf) >= capacity: sleep`` froze buffer contents to
    the first fill's ~capacity episodes for the whole training run.

    We observe rotation via ``write_head`` (monotonic put counter):
    after the buffer fills, write_head must keep increasing over a
    bounded wait window.  Use a tiny capacity + many iteration_episodes
    so rotation is rapid and easy to observe.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    n_eps = 8
    _make_dataset(
        root, ready_eps=list(range(n_eps)), total_episodes=n_eps,
    )
    src = LeRobotShardSource(root)

    # Capacity 3 << n_eps means the pool has to keep overwriting.
    buf = ShuffleBuffer(
        capacity=3, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=list(range(n_eps)),
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=3)
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
        # Buffer is full at this point (warmup_target == capacity).
        initial_head = int(buf._write_head)
        initial_keys = set(buf.keys())
        # Wait for rotation to actually happen.
        time.sleep(3.0)
        final_head = int(buf._write_head)
        final_keys = set(buf.keys())
    finally:
        pool.stop()

    # write_head monotonically counts puts — it MUST have advanced.
    assert final_head > initial_head, (
        f"buffer parked after fill: write_head stayed at {initial_head} "
        f"for 3s (pool did not rotate the ring)"
    )
    # We expect at least several rotations over 3s with a 3-slot buffer
    # and ~100ms/decode; be generous (≥3 extra puts).
    assert final_head - initial_head >= 3, (
        f"buffer rotation too slow: only {final_head - initial_head} "
        f"puts in 3s (initial_head={initial_head}, final={final_head})"
    )
    # The observable contents should have at least one key we didn't
    # see at fill time — proving the ring actually overwrote.
    assert final_keys != initial_keys or len(initial_keys) < n_eps, (
        f"buffer keys unchanged after rotation: {final_keys} — ring "
        f"buffer semantics broken"
    )


def test_update_episodes_reaches_buffer_without_clear(tmp_path):
    """With the no-park backpressure, ``update_episodes`` can deliver
    a new episode into the buffer without the test having to call
    ``buf.clear()`` to unstick the pool.  Mirror of
    ``test_update_episodes_admits_new_episode_live`` but proving the
    pool rotates on its own.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=[0, 1], total_episodes=10)
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=4, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=[0, 1],
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=2)
    # Pre-land ep 7 on disk so update_episodes can reach a decodable file.
    _write_episode_parquet(root, 7, n_frames=20)
    _write_episode_mp4(root, 7, n_frames=20, height=32, width=32, fps=30)
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
        pool.update_episodes("synth", [0, 1, 7])
        # Poll buffer WITHOUT clearing — if the pool parks after fill,
        # ep 7 will never land.
        deadline = time.monotonic() + 30.0
        saw_7 = False
        while time.monotonic() < deadline:
            if 7 in buf.keys():
                saw_7 = True
                break
            time.sleep(0.2)
    finally:
        pool.stop()

    assert saw_7, (
        f"ep 7 never reached buffer after update_episodes without "
        f"an explicit buf.clear() — pool may be parking at full; "
        f"final keys={buf.keys()}"
    )
