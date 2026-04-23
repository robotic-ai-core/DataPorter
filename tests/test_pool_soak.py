"""Long-running soak test for the streaming pool.

Short-duration tests catch structural bugs (does it produce samples,
do dicts have the right keys) but not time-evolution bugs (buffer
contents freezing, diversity degrading, rate drift over sustained
load).  This module contains a single slow-marked soak test that
runs the full pipeline for ~30 seconds and asserts behavior remains
healthy throughout.

Why 30s and not longer: we sample the buffer at ~2s intervals, so a
30s window gives ~15 data points — enough to detect stepwise regime
changes (fill → park, fill → throttle-and-rotate) that can't be
distinguished in a 3-5s burst.  Longer would be better but costs
test-wall-clock disproportionately; leave the longer runs to manual
verification against a real dataset.

The test is marked ``@pytest.mark.slow`` so CI can skip it on PRs
but pick it up for nightly / pre-merge runs.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

from test_shard_source_pool_e2e import (
    _make_dataset,
)
from _pipeline_assertions import (
    assert_sample_diversity,
    run_in_dataloader,
    unbatch_dicts,
)


@pytest.mark.slow
def test_pool_soak_30s(tmp_path):
    """End-to-end soak: pool runs for ~30s, sampled throughout.

    Invariants checked over time:

    - ``write_head`` advances continuously, not just during warmup
      (catches pool-park regressions where rotation stops after
      fill).
    - Buffer key set drifts over time (catches frozen-contents
      regressions where only the first-fill episodes are ever
      visible).
    - Sample diversity over the full run covers more episodes than
      the buffer capacity (catches cases where rotation is too slow
      to materially increase diversity).
    - No exceptions from the DataLoader, workers, or pool child.

    Failure modes this catches that burst tests don't:

    - Parking regression: pool decodes during warmup, then stops.
      Burst tests that only check post-warmup state pass; soak
      detects the frozen write_head.
    - Rotation too slow: if the backpressure throttle is set too
      conservatively, the full buffer takes too long to turn over
      and diversity stays low.  Soak catches this by measuring
      end-to-end key coverage.
    - Drift under sustained load: e.g. a subtle leak where each
      refresh accumulates stale state.  Burst tests don't run long
      enough to surface this.
    """
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    # Enough episodes that the buffer (capacity=4) can't hold them
    # all — rotation is the only way to achieve diversity > 4.
    n_eps = 16
    root = tmp_path / "ds"
    _make_dataset(root, ready_eps=list(range(n_eps)), total_episodes=n_eps)
    src = LeRobotShardSource(root)

    buf = ShuffleBuffer(
        capacity=4, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=list(range(n_eps)),
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=3)

    consumer = LeRobotShuffleBufferDataset(
        buffer=buf,
        sources=[{
            "shard_source": src,
            "source_name": "synth",
            "episode_offset": 0,
            "transform": None,
            "train_episode_indices": list(range(n_eps)),
        }],
        delta_timestamps={},
        prefetchers=[],
        producer_pool=pool,
        split_fn=lambda ep: True,
        image_keys=["observation.image"],
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Sample write_head + buffer keys every ~2s across the window.
        duration_s = 30.0
        poll_s = 2.0
        samples_head: list[int] = []
        samples_keys: list[set[int]] = []
        deadline = time.monotonic() + duration_s
        samples_head.append(int(buf._write_head))
        samples_keys.append({k for k in buf.keys() if k >= 0})
        while time.monotonic() < deadline:
            time.sleep(poll_s)
            samples_head.append(int(buf._write_head))
            samples_keys.append({k for k in buf.keys() if k >= 0})

        # Invariant 1: write_head kept advancing throughout.  A
        # parked pool would show one big jump (during warmup) then
        # zero deltas.  Healthy rotation shows deltas > 0 in most
        # intervals.
        deltas = [b - a for a, b in zip(samples_head, samples_head[1:])]
        nonzero_intervals = sum(1 for d in deltas if d > 0)
        assert nonzero_intervals >= max(1, len(deltas) - 2), (
            f"pool rotated for {nonzero_intervals}/{len(deltas)} "
            f"intervals (expected almost all non-zero).  Deltas: "
            f"{deltas}.  Head snapshots: {samples_head}."
        )
        total_puts = samples_head[-1] - samples_head[0]

        # Invariant 2: key set drifts over time — the buffer is not
        # frozen on its initial fill.  Union of keys seen across the
        # run must exceed capacity.
        observed_keys: set[int] = set()
        for s in samples_keys:
            observed_keys |= s
        assert len(observed_keys) > buf.capacity, (
            f"buffer only ever held {len(observed_keys)} distinct "
            f"keys across {duration_s}s (capacity {buf.capacity}).  "
            f"Ring buffer is not rotating content through training."
        )
    finally:
        pool.stop()

    # Run a secondary consumer-level check via DataLoader to ensure
    # diversity percolates through sampling.  Small batch count: the
    # soak's main job is the buffer-level invariants above.
    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)
        with run_in_dataloader(
            consumer, num_workers=2, batch_size=4, max_batches=40,
            timeout_s=30.0,
        ) as batches:
            per_sample = unbatch_dicts(batches)
        # 40 batches × 4 samples = 160 per_sample entries.  With
        # 16 admitted episodes and a buffer rotating continuously,
        # a non-degenerate run sees at least 8 unique episodes.
        assert_sample_diversity(per_sample, min_unique_episodes=8)
    finally:
        pool.stop()

    # Sanity: we did a lot of puts.  Lower bound: ~1 put/sec at the
    # throttle rate × 30s = 30.  In practice we see much more.
    assert total_puts >= 10, (
        f"surprisingly few puts ({total_puts}) in {duration_s}s — "
        f"pool was likely parked most of the time"
    )
