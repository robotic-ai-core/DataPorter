"""Map-style validation Dataset backed by :class:`LeRobotShardSource`.

Drop-in replacement for ``Subset(FastLeRobotDataset, val_idx)`` on
the val path.  Same ``(raw_ep, frame_in_ep) → Sample`` semantics as
the train path — both go through :class:`SampleReader` — so train/val
can't drift on sample construction.

Why it exists: before this class, :class:`BlendedLeRobotDataModule`
built ``FastLeRobotDataset`` once for discovery and a second time for
val.  Both invocations did ~25ms of work per episode (Arrow cache
rebuild, metadata validation, parquet scan); at 18k episodes that's
7+ minutes of dead-weight setup every run.  The shard source does
the same job lazily at O(1) construction cost, so val switches over
to it here.  Videos are decoded on-demand through the reader's LRU,
matching :class:`FastLeRobotDataset`'s per-worker cache behaviour on
the val path.
"""

from __future__ import annotations

from typing import Any, Sequence

from torch.utils.data import Dataset

from .lerobot_shard_source import LeRobotShardSource
from .sample_reader import SampleReader


class ShardSourceValDataset(Dataset):
    """Validation dataset indexed by a caller-supplied list of
    ``(raw_ep, frame_in_ep)`` pairs.

    Unlike :class:`LeRobotShuffleBufferDataset` (which samples
    uniformly from a producer-fed buffer), this class iterates a
    fixed, deterministic set of frames — the standard val contract.
    The caller computes which frames belong in val at DataModule
    setup time (by consulting ``split_fn`` over
    ``episode_frame_count``) and hands them in.

    Args:
        shard_source: Source of episode metadata + row/video reads.
        sample_indices: Sequence of ``(raw_episode_id, frame_in_ep)``
            tuples.  Indexing the dataset at position ``i`` returns
            the sample at ``sample_indices[i]``.
        delta_timestamps: Forwarded to the internal
            :class:`SampleReader`.  ``None`` for single-frame val.
        image_keys: Forwarded to the internal :class:`SampleReader`.
            Defaults to the reader's default.
        decode_cache_maxsize: Per-worker episode-decode LRU size.
            The default of 4 is enough for sequential-ish iteration
            where consecutive samples share an episode.

    DataLoader usage: picklable end-to-end (shard source drops its
    row cache on ``__getstate__``; reader's caches repopulate lazily
    in each worker).  Pass to :class:`torch.utils.data.DataLoader`
    exactly like a :class:`FastLeRobotDataset`.
    """

    def __init__(
        self,
        shard_source: LeRobotShardSource,
        sample_indices: Sequence[tuple[int, int]],
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_keys: list[str] | None = None,
        decode_cache_maxsize: int = 4,
    ) -> None:
        self._reader = SampleReader(
            shard_source,
            delta_timestamps=delta_timestamps,
            image_keys=image_keys,
            decode_cache_maxsize=decode_cache_maxsize,
        )
        # Normalize input to a list of (int, int) tuples so downstream
        # indexing is cheap and unambiguous.  A numpy / torch / pandas
        # sequence of pairs would "just work" as input but the stored
        # form is always Python.
        self._sample_indices: list[tuple[int, int]] = [
            (int(raw_ep), int(frame_in_ep))
            for raw_ep, frame_in_ep in sample_indices
        ]

    def __len__(self) -> int:
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw_ep, frame_in_ep = self._sample_indices[idx]
        return self._reader.read(raw_ep, frame_in_ep)

    # ------------------------------------------------------------------
    # Introspection — consumed by tests + the datamodule's sample-key
    # inference logic (which needs to know which sample-dict keys val
    # produces without constructing a batch).
    # ------------------------------------------------------------------

    @property
    def image_keys(self) -> list[str]:
        return self._reader.image_keys

    @property
    def shard_source(self) -> LeRobotShardSource:
        """The underlying shard source.  Exposed for the datamodule's
        metadata probes (video shapes, feature names).  Do not use for
        reads — go through ``__getitem__`` so decode caching applies.
        """
        return self._reader._shard
