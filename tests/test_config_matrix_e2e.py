"""Config-matrix parametrization for streaming-pool knob combinations.

Motivation: several production bugs (buffer park + overfit; self-refresh
wipe on local-only sources; cross-fork AssertionError in update_episodes)
shipped despite passing e2e coverage, because the tests exercised each
knob in isolation — not the specific COMBINATION the failing configs
used.  This module parametrizes the four user-facing pipeline knobs
and runs a real end-to-end iteration through a DataLoader with forked
workers against each, so any future bug that's specific to a
combination surfaces here.

Knobs covered:

- ``has_prefetcher``: HF-style prefetched source vs. local-only source
  (no prefetcher).  Local-only was the trap that wiped admission
  under self-refresh.
- ``self_refresh``: worker self-refresh cadence on vs. off.  Also
  exercises the cross-fork ``update_episodes`` path.
- ``pin_len``: nominal_total_frames set vs. unset.  Matters for
  Lightning cache semantics but also catches accidental coupling
  between pinned-len and admission-driven epoch recomputation.
- ``grown_eps``: does a new episode land on disk mid-iteration?
  Separate axis because a static dataset with self_refresh enabled
  has no admission to grow, which is a different code path than a
  live-growing dataset.

For each combination the harness:
  1. builds a pool + consumer + (optional) prefetcher,
  2. wraps the consumer in a real ``DataLoader(num_workers>0)``,
  3. drains ``max_batches`` batches,
  4. asserts the admitted set is coherent, buffer rotated, and
     (where applicable) diversity covers new admissions.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("lerobot")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")

# Reuse synthetic-dataset builders and the live-disk prefetcher stub
# from the main e2e module.
from test_shard_source_pool_e2e import (
    _LiveDiskPrefetcher,
    _make_dataset,
    _write_episode_mp4,
    _write_episode_parquet,
)
from _pipeline_assertions import (
    assert_buffer_rotates,
    assert_sample_diversity,
    run_in_dataloader,
    unbatch_dicts,
)


# ---------------------------------------------------------------------------
# Parametrize over meaningful knob combinations
# ---------------------------------------------------------------------------

# (has_prefetcher, self_refresh, pin_len, grown_eps)
CONFIG_MATRIX = [
    # Pure static local (the exp28v1 shape that tripped the wipe).
    pytest.param(False, False, False, False, id="static-local"),
    pytest.param(False, True,  False, False, id="static-local+self_refresh"),
    pytest.param(False, False, True,  False, id="static-local+pin_len"),
    pytest.param(False, True,  True,  False, id="static-local+self_refresh+pin_len"),
    # Prefetched + growing (the canonical HF use case).
    pytest.param(True,  False, False, True,  id="hf+grown"),
    pytest.param(True,  True,  False, True,  id="hf+grown+self_refresh"),
    pytest.param(True,  True,  True,  True,  id="hf+grown+self_refresh+pin_len"),
    # Prefetched, no growth during the test (static snapshot via HF).
    pytest.param(True,  False, False, False, id="hf-static"),
    pytest.param(True,  False, True,  False, id="hf-static+pin_len"),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _build(tmp_path, *, has_prefetcher, self_refresh, pin_len):
    """Assemble the pipeline per the knobs; return (pool, consumer, root)."""
    from dataporter import LeRobotShardSource, ResizeFrames
    from dataporter.lerobot_shuffle_buffer_dataset import (
        LeRobotShuffleBufferDataset,
    )
    from dataporter.producer_pool import ProducerConfig, ProducerPool
    from dataporter.shuffle_buffer import ShuffleBuffer

    root = tmp_path / "ds"
    initial_eps = [0, 1, 2, 3]
    _make_dataset(
        root, ready_eps=initial_eps, total_episodes=10,
    )
    src = LeRobotShardSource(root)
    prefetcher = _LiveDiskPrefetcher(root) if has_prefetcher else None

    buf = ShuffleBuffer(
        capacity=6, max_frames=32, channels=3, height=32, width=32,
    )
    cfg = ProducerConfig.from_source(
        source={"repo_id": "synth", "weight": 1.0},
        shard_source=src,
        iteration_episodes=initial_eps,
        producer_transform=ResizeFrames(32, 32),
    )
    pool = ProducerPool(buf, configs=[cfg], total_workers=2, warmup_target=3)

    consumer_kwargs: dict = {
        "buffer": buf,
        "sources": [{
            "shard_source": src,
            "source_name": "synth",
            "episode_offset": 0,
            "transform": None,
            "train_episode_indices": list(initial_eps),
        }],
        "delta_timestamps": {},
        "prefetchers": [prefetcher] if prefetcher is not None else [],
        "producer_pool": pool,
        "split_fn": lambda ep: True,
        "image_keys": ["observation.image"],
    }
    if self_refresh:
        consumer_kwargs["refresh_every_n_items"] = 3
    if pin_len:
        consumer_kwargs["nominal_total_frames"] = 10 * 20  # full declared

    consumer = LeRobotShuffleBufferDataset(**consumer_kwargs)
    return pool, consumer, root, buf


@pytest.mark.parametrize(
    "has_prefetcher, self_refresh, pin_len, grown_eps", CONFIG_MATRIX,
)
def test_end_to_end_config_matrix(
    tmp_path, has_prefetcher, self_refresh, pin_len, grown_eps,
):
    """Every knob combination must: start, fill the buffer, rotate
    contents, and (if growing-mode) pick up new on-disk episodes.
    No combination may crash; none may freeze content.
    """
    pool, consumer, root, buf = _build(
        tmp_path,
        has_prefetcher=has_prefetcher,
        self_refresh=self_refresh,
        pin_len=pin_len,
    )

    pool.start()
    try:
        pool.wait_for_warmup(timeout=30.0)

        # Simulate a new episode landing after warmup, if this axis
        # calls for it.
        if grown_eps:
            _write_episode_parquet(root, 5, n_frames=20)
            _write_episode_mp4(
                root, 5, n_frames=20, height=32, width=32, fps=30,
            )

        # Drive the consumer through a real DataLoader with forked
        # workers — this is where cross-fork bugs surface.
        with run_in_dataloader(
            consumer, num_workers=2, batch_size=4, max_batches=20,
            timeout_s=45.0,
        ) as batches:
            samples = unbatch_dicts(batches)

        # Admitted set survived the iteration.  For no-prefetcher
        # configs this was the wipe-bug regression point.
        assert set(consumer._current_train_episodes) >= {0, 1, 2, 3}, (
            f"admitted set shrank during iteration: "
            f"{consumer._current_train_episodes}"
        )

        # Buffer rotated — no pool-park regression.
        assert_buffer_rotates(
            buf, duration_s=2.0, min_write_head_delta=2,
        )

        # Samples cover the initial admitted set.
        assert_sample_diversity(
            samples, min_unique_episodes=3,
        )

        # If growth was simulated AND we have self_refresh (local) or
        # a prefetcher (HF-style), the new episode should have been
        # admitted.  For static configs without either, ep 5 should
        # NOT be admitted (no mechanism to discover it).
        if grown_eps and (self_refresh or has_prefetcher):
            # Allow bounded time for self-refresh to fire / prefetcher
            # ready-scan to pick it up via consumer.refresh().
            if has_prefetcher and not self_refresh:
                consumer.refresh(min_new=0)
            deadline = time.monotonic() + 10.0
            while (
                time.monotonic() < deadline
                and 5 not in consumer._current_train_episodes
            ):
                # Pull more samples to trigger self_refresh ticks.
                if self_refresh:
                    _ = consumer[0]
                else:
                    consumer.refresh(min_new=0)
                time.sleep(0.2)
            assert 5 in consumer._current_train_episodes, (
                f"ep 5 was written to disk but never admitted "
                f"(self_refresh={self_refresh}, "
                f"has_prefetcher={has_prefetcher}); "
                f"admitted={consumer._current_train_episodes}"
            )

        # pin_len contract: __len__ independent of admission size.
        if pin_len:
            assert len(consumer) == 10 * 20
    finally:
        pool.stop()
