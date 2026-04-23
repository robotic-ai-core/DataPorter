"""Shared test-infrastructure helpers for the streaming pipeline.

Exists because point-in-time state assertions (``key in buf.keys()``,
``len(consumer) == N``) don't catch the failure modes that hurt
actual training: buffer-content staleness over time, lack of sample
diversity, and cross-fork behavioral drift.

Three primitives:

- :func:`assert_buffer_rotates` — asserts the buffer's write-head
  advances by ≥ N over a bounded time window.  Catches the
  "park-after-fill" class of bugs that freezes training on a snapshot
  of the first ``capacity`` decoded episodes.

- :func:`assert_sample_diversity` — asserts a sampled batch has at
  least N unique episodes / frames.  Training correctness is
  distributional; a frozen buffer can still answer "is X in keys?"
  truthfully while yielding zero diversity over time.

- :func:`run_in_dataloader` — spins up a real ``torch.utils.data.DataLoader``
  with ``num_workers > 0`` forked workers and yields a bounded number
  of batches.  Exercises the fork boundary that hand-built consumer
  tests don't.  Any cross-fork bug (the kind that triggers
  ``AssertionError: can only test a child process`` or makes a
  worker's state go stale) surfaces here.

Rule of thumb for test authors: if you're about to paper over a
production behavior with ``buf.clear()``, ``monkeypatch.setattr``,
or a targeted sleep, STOP and ask whether the behavior is a bug.
Workarounds that ship in tests hide the failure mode in production.
"""

from __future__ import annotations

import contextlib
import time
from typing import Iterator

import torch


# ---------------------------------------------------------------------------
# Rate-based buffer assertion
# ---------------------------------------------------------------------------


def assert_buffer_rotates(
    buffer,
    *,
    duration_s: float,
    min_write_head_delta: int,
    poll_interval_s: float = 0.2,
) -> int:
    """Assert the buffer's producer keeps writing over a time window.

    ``_write_head`` is a monotone counter of total puts over the
    buffer's lifetime.  A parked pool stops calling ``put`` once the
    buffer fills, so ``write_head`` stops advancing — which is
    invisible to state-only assertions (``len(buf) == capacity`` stays
    true whether or not the pool is still decoding).

    Args:
        buffer: A :class:`ShuffleBuffer` or compatible ring buffer
            exposing ``_write_head`` as a 0-dim tensor counter.
        duration_s: How long to observe.
        min_write_head_delta: Minimum expected additional puts over
            ``duration_s``.  Tune to the expected decode rate on the
            test's dataset — overshooting produces flaky tests, too
            lax misses regressions.
        poll_interval_s: How often to sample ``write_head``.  Included
            for failure diagnostics (we log the per-interval deltas
            when the assertion fails).

    Returns:
        The observed ``write_head`` delta.

    Raises:
        AssertionError: if ``write_head`` advanced by less than
            ``min_write_head_delta`` in the window.  Diagnostics
            include per-interval deltas so the failure mode is
            obvious (stuck-at-0 vs. slow-but-moving).
    """
    initial = int(buffer._write_head)
    samples: list[int] = [initial]
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        samples.append(int(buffer._write_head))
    delta = samples[-1] - samples[0]
    if delta < min_write_head_delta:
        deltas = [b - a for a, b in zip(samples, samples[1:])]
        raise AssertionError(
            f"buffer did not rotate: write_head advanced by {delta} "
            f"(required ≥ {min_write_head_delta}) over {duration_s}s. "
            f"Per-interval deltas: {deltas}. "
            f"A zero delta means the pool parked (buffer contents "
            f"frozen); a small non-zero delta means the pool is "
            f"throttled below the required rate."
        )
    return delta


# ---------------------------------------------------------------------------
# Diversity assertion
# ---------------------------------------------------------------------------


def assert_sample_diversity(
    samples: list[dict],
    *,
    min_unique_episodes: int,
    min_unique_frames: int | None = None,
) -> tuple[int, int]:
    """Assert a batch of consumer samples covers enough distinct content.

    Training correctness is distributional: a consumer sampling from a
    frozen buffer can still produce valid-shaped dicts forever while
    yielding terrible diversity.  This checks the actual content
    spread.

    Args:
        samples: List of sample dicts from ``consumer.__getitem__``
            (or equivalent).  Each must have an ``episode_index``
            field and optionally a ``frame_index`` field.
        min_unique_episodes: Minimum number of distinct
            ``episode_index`` values required.
        min_unique_frames: Optional minimum number of distinct
            ``(episode_index, frame_index)`` pairs.  Skip if the
            consumer doesn't expose per-frame sampling.

    Returns:
        ``(n_unique_episodes, n_unique_frames)`` for use in
        diagnostic logs.
    """
    def _scalar(v):
        return int(v.item()) if hasattr(v, "item") else int(v)

    episodes: set[int] = set()
    frames: set[tuple[int, int]] = set()
    for s in samples:
        ep = _scalar(s["episode_index"])
        episodes.add(ep)
        if "frame_index" in s:
            frames.add((ep, _scalar(s["frame_index"])))

    n_ep = len(episodes)
    n_fr = len(frames)
    if n_ep < min_unique_episodes:
        raise AssertionError(
            f"low episode diversity: {n_ep} unique episodes in "
            f"{len(samples)} samples (required ≥ {min_unique_episodes}). "
            f"Observed: {sorted(episodes)[:20]}... — if all come from "
            f"a small subset, the pool may be parked on a frozen "
            f"buffer snapshot."
        )
    if min_unique_frames is not None and n_fr < min_unique_frames:
        raise AssertionError(
            f"low frame diversity: {n_fr} unique (ep, frame) pairs "
            f"in {len(samples)} samples (required ≥ {min_unique_frames})."
        )
    return n_ep, n_fr


# ---------------------------------------------------------------------------
# Real-DataLoader harness
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def run_in_dataloader(
    dataset,
    *,
    num_workers: int = 2,
    batch_size: int = 4,
    max_batches: int = 50,
    timeout_s: float = 60.0,
) -> Iterator[list]:
    """Iterate a real :class:`torch.utils.data.DataLoader` with forked
    workers against the given dataset; yield the collected batches.

    Why this exists: direct-construction tests build the consumer in
    the main process, so the fork boundary is never crossed.  Bugs
    that only surface when ``__getitem__`` runs in a forked worker
    with a partial copy of parent state (cross-process handles,
    inherited RNG, refresh-into-forked-state) are invisible to the
    single-process path.  The production code path always goes
    through ``DataLoader(num_workers>0)``; so should the hardening
    tests.

    Args:
        dataset: Any PyTorch map-style dataset.  Its class must have a
            ``worker_init_fn`` staticmethod if it needs per-worker RNG.
        num_workers: Fork this many DataLoader workers.  Must be ≥ 1
            to exercise the fork path; default 2.
        batch_size: Batch shape for the DataLoader.
        max_batches: Stop after this many batches collected (bounds
            test runtime).
        timeout_s: Outer deadline — if the DataLoader hangs for
            longer, fail rather than timing out at the pytest level.

    Yields:
        A list of batch dicts (up to ``max_batches``).  On context
        exit the DataLoader is shut down and workers are reaped.

    Example::

        with run_in_dataloader(consumer, num_workers=2) as batches:
            assert_sample_diversity(
                [_unbatch_one(b) for b in batches],
                min_unique_episodes=10,
            )
    """
    from torch.utils.data import DataLoader

    worker_init_fn = getattr(type(dataset), "worker_init_fn", None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=False,
        persistent_workers=False,
    )
    batches: list = []
    deadline = time.monotonic() + timeout_s
    it = iter(loader)
    try:
        while len(batches) < max_batches:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"DataLoader hung: got {len(batches)}/{max_batches} "
                    f"batches in {timeout_s}s"
                )
            try:
                batch = next(it)
            except StopIteration:
                break
            batches.append(batch)
        yield batches
    finally:
        # Break the iterator so workers see EOF, then drop the
        # DataLoader reference so its workers are reaped.
        del it
        del loader


def unbatch_dicts(batches: list[dict]) -> list[dict]:
    """Flatten a list of batch-dicts to a list of single-sample dicts.

    Useful when feeding DataLoader output into
    :func:`assert_sample_diversity`, which expects per-sample dicts.
    """
    out: list[dict] = []
    for batch in batches:
        # Infer batch size from any tensor leaf.
        n = None
        for v in batch.values():
            if torch.is_tensor(v):
                n = v.shape[0]
                break
        if n is None:
            continue
        for i in range(n):
            sample: dict = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    sample[k] = v[i]
                elif isinstance(v, list):
                    sample[k] = v[i]
                else:
                    sample[k] = v
            out.append(sample)
    return out
