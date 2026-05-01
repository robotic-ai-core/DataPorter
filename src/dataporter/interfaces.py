"""Public interface Protocols for the DataPorter streaming pipeline.

These structural-typing contracts (PEP 544) define the seams where new
modalities plug in.  Implementations satisfy a Protocol by having the
right method signatures and attributes — no inheritance required, no
framework buy-in.  ``@runtime_checkable`` is set on each so
``isinstance(x, EpisodicSource)`` works in tests and assertions.

## Layered pipeline

::

    Remote / Local disk
            │
            ▼
    ┌────────────────┐
    │   Prefetcher   │   writes shards / media to a cache dir
    └───────┬────────┘
            │  observes disk via list_ready_episodes
            ▼
    ┌────────────────┐   ◄── EpisodicSource / TemporalEpisodicSource
    │     Source     │       (narrow, read-only view of disk state)
    └───────┬────────┘
            │
     cheap? │ expensive? (video decode, on-the-fly tokenize)
            │                 │
            │                 ▼
            │         ┌───────────────┐
            │         │ ProducerPool  │  ◄── ProducerConfigProtocol
            │         │ (spawn child) │      (picklable per-source config)
            │         └───────┬───────┘
            │                 │
            │                 ▼
            │         ┌───────────────┐
            │         │ Shared-memory │
            │         │    buffer     │
            │         └───────┬───────┘
            │                 │
            └───────┬─────────┘
                    ▼
          ┌──────────────────┐   ◄── EpisodicPrefetcher
          │ Consumer Dataset │       (refresh() polls .ready_episodes()
          │                  │        + .is_done())
          └─────────┬────────┘
                    ▼
              DataLoader

## Contract guarantees

Protocols document the *call surface* used by the pipeline today.  They
are intentionally narrow — only methods that are actually called by the
producer pool or the consumer dataset appear here.  Implementations are
free to provide additional methods, but the pipeline will never call
anything beyond what these Protocols declare.

## Reference implementations

- :class:`LeRobotShardSource` — satisfies :class:`TemporalEpisodicSource`.
- :class:`LeRobotPrefetcher` — satisfies :class:`EpisodicPrefetcher`.
- :class:`ProducerConfig` — satisfies :class:`ProducerConfigProtocol`.

## Extending to a new modality

To add (say) an audio episodic pipeline, write:

- ``AudioShardSource`` satisfying :class:`TemporalEpisodicSource` — the
  same load / path / metadata surface as video, but ``media_keys``
  contains audio track names and ``episode_media_path`` returns the
  audio file.
- ``AudioPrefetcher`` subclassing ``BasePrefetcher`` — reuses the error
  queue, eviction, lifecycle; only overrides ``_run_inner`` (how to
  write the bytes) and ``_check_min_ready`` (what counts as ready).
- A decode function for the pool that speaks waveform instead of frames.

You do not subclass these Protocols.  You make a class whose method
signatures match, and the type checker / ``runtime_checkable`` will
accept it wherever the pipeline asks for one.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    "EpisodicSource",
    "TemporalEpisodicSource",
    "EpisodicPrefetcher",
    "ProducerConfigProtocol",
]


# ---------------------------------------------------------------------------
# Source layer
# ---------------------------------------------------------------------------


@runtime_checkable
class EpisodicSource(Protocol):
    """Read-only lazy view of an episodic dataset on local disk.

    The minimal surface both the consumer and the producer pool need to
    treat a dataset as "a collection of raw-id-addressed episodes, each
    with a row table and a frame count."  No temporal structure or media
    files are assumed — a tabular or point-cloud modality can satisfy
    this without a time dimension.

    Attributes:
        root: Directory containing the on-disk dataset layout.  Used for
            logging and for callers that need to construct secondary
            views (e.g. a prefetcher referencing the same root).
        total_episodes: Nominal episode count declared by the dataset's
            metadata.  This is the dataset's *nominal* size, not "how
            many are on disk right now" — use :meth:`list_ready_episodes`
            for the latter.

    The lazy contract: construction is O(1) in ``total_episodes``.
    Expensive per-episode parsing happens on first access and is cached
    thereafter.
    """

    root: Path
    total_episodes: int

    def episode_frame_count(self, raw_ep: int) -> int:
        """Number of frames / rows in episode ``raw_ep``.

        Cheap after first call (lazily parsed metadata is cached).
        """
        ...

    def load_episode_row_torch(
        self, raw_ep: int, frame_idx: int,
    ) -> dict[str, Any]:
        """Single row as ``{column_name: torch.Tensor | primitive}``.

        ``frame_idx`` is LOCAL to the episode (0-indexed within the
        episode's rows), not a global dataset offset.  Mirrors the
        shape of a LeRobet ``hf_dataset[global_idx]`` return.
        """
        ...

    def load_episode_window_torch(
        self, raw_ep: int, frame_indices: Sequence[int],
    ) -> dict[str, Any]:
        """Multi-row window, stacked along dim 0.

        ``frame_indices`` are LOCAL to the episode.  Used by consumers
        implementing delta-timestamp windowing so the framework doesn't
        need to know how the source stores the rows internally.
        """
        ...

    def list_ready_episodes(self) -> list[int]:
        """Sorted raw episode ids whose on-disk artefacts are all
        present right now.

        Must return quickly (hundreds of ms at 18k episodes) — callers
        poll this at epoch boundaries.  An episode is "ready" when
        *everything it needs to decode* is on disk; partial states (e.g.
        parquet present but mp4 still downloading) must be excluded.
        """
        ...

    def tasks(self) -> dict[int, str]:
        """``{task_index: task_string}`` for the dataset's task table.

        Empty dict if the dataset omits tasks entirely.
        """
        ...


@runtime_checkable
class TemporalEpisodicSource(EpisodicSource, Protocol):
    """Episodic source whose rows live along a time axis and whose
    media (video, audio, …) are stored as decodable per-episode files.

    Extends :class:`EpisodicSource` with sample rate and per-media-key
    path resolution so a producer pool can locate the file(s) it needs
    to decode.

    Attributes:
        fps: Sample rate in Hz.  Called ``fps`` for historical reasons
            even when the underlying modality is audio (sample rate)
            or IMU (update rate); any scalar "samples per second" fits.
        media_keys: Names of the decodable media channels present in
            this dataset, e.g. ``["observation.image"]`` for a
            single-camera LeRobot dataset, ``["audio.mic1", "audio.mic2"]``
            for stereo audio, ``["observation.image",
            "observation.image_depth"]`` for RGB+D.
    """

    fps: float
    media_keys: list[str]

    def episode_media_path(
        self, raw_ep: int, media_key: str,
    ) -> Path:
        """Absolute path to the media file for ``(raw_ep, media_key)``.

        The pool's decoder calls this and hands the path to ffmpeg /
        libsndfile / etc.  Must be resolved (no symlinks) for pyav
        compatibility in spawned process contexts — see
        :meth:`pathlib.Path.resolve`.
        """
        ...


# ---------------------------------------------------------------------------
# Prefetcher layer
# ---------------------------------------------------------------------------


@runtime_checkable
class EpisodicPrefetcher(Protocol):
    """Background downloader whose output the consumer's ``refresh()``
    polls.

    The consumer does not care *how* bytes are written (snapshot_download
    vs producer-thread streaming vs rsync vs local-file symlink).  It
    only needs to know what episodes are safe to iterate over right now
    and whether any more are expected to arrive.

    This Protocol is deliberately minimal.  Real prefetchers (e.g.
    :class:`LeRobotPrefetcher`) expose many more methods for lifecycle,
    eviction, and status reporting — see :class:`BasePrefetcher` for
    the base class that owns the lifecycle plumbing.
    """

    def ready_episodes(self) -> list[int]:
        """Raw episode ids that are fully on disk right now.

        Semantics match :meth:`EpisodicSource.list_ready_episodes` —
        kept separate because in principle a prefetcher can report
        readiness from its own write-tracking state without a disk
        scan, which is cheaper than a fresh rglob.  Today
        :class:`LeRobotPrefetcher` implements both with identical
        disk-scan logic and they produce the same answer; this is
        documented intentionally at
        :meth:`LeRobotPrefetcher.ready_episodes`.
        """
        ...

    def is_done(self) -> bool:
        """True when no more episodes will ever become ready.

        Lets the consumer's bounded-wait refresh semantics terminate
        early: if ``is_done()`` and the ready set has stabilized, there
        is no point waiting for more.
        """
        ...


# ---------------------------------------------------------------------------
# Producer-pool layer
# ---------------------------------------------------------------------------


@runtime_checkable
class ProducerConfigProtocol(Protocol):
    """Picklable per-source config carried into the spawn-child pool.

    The pool's base-class lifecycle machinery only needs the fields
    below; modality-specific subclasses can carry more (e.g. video
    adds ``video_backend`` and ``tolerance_s`` for ffmpeg; audio would
    add ``sample_format`` and ``channels``).

    Attributes:
        source_name: Unique routing key within a single pool.  The
            pool enforces uniqueness at construction so
            ``update_episodes(source_name, ...)`` can't broadcast to
            the wrong iterator.
        episode_indices: RAW episode ids this producer cycles over.
            The consumer's admission pipeline decides which ids land
            here; the pool does not re-filter.
        weight: Relative blend weight for multi-source dispatch.  The
            weighted round-robin dispatcher picks the next source by
            token accrual; a source with ``weight=3.0`` fires 3x as
            often as one with ``weight=1.0``.
        seed: RNG seed for episode shuffle order.  Fixed per config so
            the child's episode-iteration order is reproducible.
        episode_offset: Added to each raw id before writing to the
            shared-memory buffer.  Prevents key collisions when
            multiple sources share a raw-id space (two datasets both
            starting at episode 0).
        producer_transform: Picklable callable applied to each decoded
            payload *before* buffer.put.  None = identity.  Sizes the
            buffer at the transform's output shape rather than the
            source resolution — the standard way to route 512x512
            video through a 96x96 buffer without changing the source.
    """

    source_name: str
    episode_indices: list[int]
    weight: float
    seed: int
    episode_offset: int
    producer_transform: Optional[Callable]
