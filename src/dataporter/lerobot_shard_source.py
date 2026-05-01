"""Live, lazy view of a LeRobot dataset on local disk.

Purpose: replace :class:`FastLeRobotDataset`'s load-everything-upfront
behaviour on the streaming path.  This module is the LeRobot analog of
:class:`ShardPoolSource` for text — the prefetcher writes episodes to
local disk, and this source reads them back incrementally.

Why add it:

- ``FastLeRobotDataset`` materializes a global ``episode_data_index``,
  ``hf_dataset``, and metadata at construction time (~0.025s/episode
  of real work → ~8 min at 18k episodes).  Its episode set is frozen:
  episodes downloaded AFTER construction are invisible.
- For streaming-growing training (core DataPorter design requirement),
  we want a dataset view whose "what's on disk" story is re-checkable
  cheaply, with per-episode work that scales with *new* episodes only.

Design principles:

1. **Global metadata (``meta/info.json`` + ``meta/tasks.jsonl``)** is
   loaded at construction — tiny, constant-size, never grows.
2. **Per-episode metadata (``meta/episodes.jsonl`` lengths)** is loaded
   lazily on first access and cached.  One parse for the life of the
   source; bounded cost.
3. **Per-episode row data (parquet)** is loaded per request with a
   small LRU.  Parquets are tiny (~100 rows each at PushT scale), so
   per-request load is a few ms.
4. **Per-episode videos** are NOT loaded here — the source only yields
   the path.  Decoding is the pool's job.
5. **Read-only.**  Downloads happen elsewhere (``LeRobotPrefetcher``);
   this view only observes.  Consequence: safe to construct in the
   spawn child, safe to have multiple concurrent readers against one
   disk layout.

The source is **picklable** (no open file handles, no locks at rest),
so it can be passed into :class:`ProducerConfig` and used in the spawn
child.

Typical consumer shape (not wired in this commit; see follow-ups)::

    source = LeRobotShardSource(root="/mnt/Data/my-dataset")
    for ep_idx in source.list_ready_episodes():
        video_path = source.episode_video_path(ep_idx, "observation.image")
        frames = decode_episode_frames(video_path, ...)
        buffer.put(ep_idx, frames)
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)

_EPISODE_IDX_RE = re.compile(r"episode_(\d+)")

_DEFAULT_DATA_PATH = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)


class LeRobotShardSource:
    """Live, lazy read-only view of a LeRobot dataset on local disk.

    Args:
        root: Directory containing the LeRobot v2.1 layout
            (``meta/info.json``, ``meta/tasks.jsonl``, ``meta/episodes.jsonl``,
            ``data/chunk-*/episode_*.parquet``, ``videos/chunk-*/<key>/episode_*.mp4``).
        rows_cache_maxsize: LRU size for cached per-episode parquet
            tables.  Default 32 — a few MB at PushT scales.

    Raises:
        FileNotFoundError: if ``root`` exists but ``meta/info.json``
            is missing.  Other metadata files are loaded lazily and
            raise on first access.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        rows_cache_maxsize: int = 32,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(
                f"LeRobotShardSource: root {self.root!r} is not a directory"
            )

        # Load global metadata.  info.json is small (<5 KB); tasks.jsonl
        # is small enough that we can load it eagerly too.
        self._info = self._load_info()
        self._tasks_cache: dict[int, str] | None = None

        # Per-episode metadata (episodes.jsonl) is loaded lazily on first
        # access — at 18k episodes this is a few MB of JSON and ~1-2s to
        # parse, which is noticeable at setup time.  Deferring lets
        # construction be O(1).
        self._episode_lengths: dict[int, int] | None = None

        # LRU of (raw_ep_id → pyarrow.Table) for the hot sampling path.
        self._rows_cache: "OrderedDict[int, pa.Table]" = OrderedDict()
        self._rows_cache_maxsize = int(rows_cache_maxsize)

    # ------------------------------------------------------------------
    # Pickle support — drop the in-memory LRU (can be rebuilt) but keep
    # the parsed global metadata caches (they're idempotent and small).
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Row cache holds pyarrow Tables; they're picklable but expensive
        # to serialise for no real win — the LRU will warm back up in the
        # child.  Drop.
        state["_rows_cache"] = OrderedDict()
        return state

    # ------------------------------------------------------------------
    # Global metadata (loaded at construction; cheap to re-read)
    # ------------------------------------------------------------------

    def _load_info(self) -> dict:
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"LeRobotShardSource: meta/info.json not found at "
                f"{info_path} — is this a LeRobot v2.1 root?"
            )
        return json.loads(info_path.read_text())

    @property
    def fps(self) -> int:
        return int(self._info["fps"])

    @property
    def chunks_size(self) -> int:
        return int(self._info.get("chunks_size", 1000))

    @property
    def total_episodes(self) -> int:
        """Total episodes declared in info.json.

        This is the dataset's *nominal* size, not "how many are on disk
        right now".  Use :meth:`list_ready_episodes` for that.
        """
        return int(self._info.get("total_episodes", 0))

    @property
    def total_frames(self) -> int:
        return int(self._info.get("total_frames", 0))

    @property
    def features(self) -> dict[str, Any]:
        return self._info.get("features", {})

    @property
    def video_keys(self) -> list[str]:
        return [
            k for k, feat in self.features.items()
            if feat.get("dtype") == "video"
        ]

    # ``media_keys`` / ``episode_media_path`` are modality-neutral
    # aliases for ``video_keys`` / ``episode_video_path``.  They exist
    # so this class satisfies the :class:`TemporalEpisodicSource`
    # Protocol without forcing the video-specific name into the
    # interface contract.  An audio or point-cloud source would expose
    # the same two names on top of domain-specific accessors.
    @property
    def media_keys(self) -> list[str]:
        return self.video_keys

    def episode_media_path(self, raw_ep: int, media_key: str) -> Path:
        return self.episode_video_path(raw_ep, media_key)

    @property
    def data_path_template(self) -> str:
        return self._info.get("data_path", _DEFAULT_DATA_PATH)

    @property
    def video_path_template(self) -> str | None:
        """Returns the video-path format string, or ``None`` for
        video-less datasets.
        """
        return self._info.get("video_path")

    def tasks(self) -> dict[int, str]:
        """Load (and cache) the ``meta/tasks.jsonl`` mapping.

        Returns ``{task_index: task_string}``.  Empty dict if
        ``tasks.jsonl`` is missing — some datasets omit tasks entirely.
        """
        if self._tasks_cache is not None:
            return self._tasks_cache
        tasks_path = self.root / "meta" / "tasks.jsonl"
        result: dict[int, str] = {}
        if tasks_path.is_file():
            with tasks_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    result[int(entry["task_index"])] = entry["task"]
        self._tasks_cache = result
        return result

    # ------------------------------------------------------------------
    # Per-episode metadata
    # ------------------------------------------------------------------

    def _load_episode_lengths(self) -> dict[int, int]:
        """Parse ``meta/episodes.jsonl`` once; cache the result.

        At 18k episodes this file is ~a few MB of JSON and parsing takes
        ~1-2s on a typical CPU — noticeable but bounded.  Cached on the
        instance so subsequent accesses are O(1).
        """
        if self._episode_lengths is not None:
            return self._episode_lengths
        eps_path = self.root / "meta" / "episodes.jsonl"
        if not eps_path.is_file():
            raise FileNotFoundError(
                f"LeRobotShardSource: meta/episodes.jsonl not found at "
                f"{eps_path}"
            )
        logger.info(
            f"LeRobotShardSource: parsing {eps_path} (one-time cost)..."
        )
        result: dict[int, int] = {}
        with eps_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                result[int(entry["episode_index"])] = int(entry["length"])
        self._episode_lengths = result
        return result

    def episode_frame_count(self, raw_ep: int) -> int:
        """Number of frames in the given episode.

        Triggers a lazy load of ``episodes.jsonl`` on first call, then
        O(1) per subsequent call.
        """
        return self._load_episode_lengths()[int(raw_ep)]

    def episode_parquet_path(self, raw_ep: int) -> Path:
        chunk = int(raw_ep) // self.chunks_size
        rel = self.data_path_template.format(
            episode_chunk=chunk, episode_index=int(raw_ep),
        )
        return self.root / rel

    def episode_video_path(self, raw_ep: int, video_key: str) -> Path:
        template = self.video_path_template
        if template is None:
            raise RuntimeError(
                f"LeRobotShardSource({self.root}): info.json has no "
                f"'video_path' template — dataset has no videos"
            )
        chunk = int(raw_ep) // self.chunks_size
        rel = template.format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=int(raw_ep),
        )
        return self.root / rel

    # ------------------------------------------------------------------
    # Readiness (on-disk presence checks)
    # ------------------------------------------------------------------

    def is_episode_ready(self, raw_ep: int) -> bool:
        """True iff the episode's parquet AND every video file exist."""
        if not self.episode_parquet_path(raw_ep).is_file():
            return False
        for vk in self.video_keys:
            if not self.episode_video_path(raw_ep, vk).is_file():
                return False
        return True

    def list_ready_episodes(self) -> list[int]:
        """Sorted list of raw episode ids whose parquet + videos are all
        on disk.

        Cost: one ``rglob("episode_*.parquet")`` + per-parquet stat for
        each video file.  On the 18k-episode local dataset this takes
        ~500ms — cheap enough to call every refresh.

        Mirrors :meth:`LeRobotPrefetcher.ready_episodes` semantically; the
        duplication is intentional for now (the prefetcher owns writes,
        this owns reads).
        """
        parquet_eps: set[int] = set()
        for p in self.root.rglob("episode_*.parquet"):
            m = _EPISODE_IDX_RE.search(p.stem)
            if m:
                parquet_eps.add(int(m.group(1)))
        if not parquet_eps:
            return []
        video_keys = self.video_keys
        if not video_keys or self.video_path_template is None:
            return sorted(parquet_eps)
        ready: list[int] = []
        for ep in sorted(parquet_eps):
            if all(
                self.episode_video_path(ep, vk).is_file() for vk in video_keys
            ):
                ready.append(ep)
        return ready

    # ------------------------------------------------------------------
    # Row access (for non-video fields — actions, states, timestamps, etc.)
    # ------------------------------------------------------------------

    def load_episode_rows(self, raw_ep: int) -> "pa.Table":
        """Load an episode's parquet as a pyarrow Table.

        Cached in an LRU keyed by raw episode id.  Parquets are small
        (~100 rows at PushT scale, few KB on disk), so first-load cost
        is a few ms and subsequent calls are O(1) until eviction.
        """
        ep = int(raw_ep)
        if ep in self._rows_cache:
            self._rows_cache.move_to_end(ep)
            return self._rows_cache[ep]
        import pyarrow.parquet as pq
        path = self.episode_parquet_path(ep)
        if not path.is_file():
            raise FileNotFoundError(
                f"LeRobotShardSource: parquet for episode {ep} not on "
                f"disk: {path}"
            )
        table = pq.read_table(str(path))
        self._rows_cache[ep] = table
        while len(self._rows_cache) > self._rows_cache_maxsize:
            self._rows_cache.popitem(last=False)
        return table

    def load_episode_row_dict(
        self, raw_ep: int, frame_idx: int,
    ) -> dict[str, Any]:
        """Convenience accessor: one row as a column→value dict.

        Useful for consumer ``__getitem__`` paths that need a single
        frame's non-video fields.
        """
        table = self.load_episode_rows(raw_ep)
        slice_dict = table.slice(frame_idx, 1).to_pydict()
        return {k: v[0] for k, v in slice_dict.items()}

    def load_episode_window(
        self, raw_ep: int, frame_indices: list[int],
    ) -> "pa.Table":
        """Pull a subset of rows (for delta_timestamps windowing).

        Equivalent to ``dataset.hf_dataset.select(frame_indices)`` from
        :class:`FastLeRobotDataset` but per-episode scope and no global
        index required.
        """
        table = self.load_episode_rows(raw_ep)
        return table.take(frame_indices)

    # ------------------------------------------------------------------
    # Torch-tensor convenience wrappers
    #
    # Consumers (e.g. LeRobotShuffleBufferDataset) want dict[str, Tensor]
    # like LeRobet's ``hf_transform_to_torch``-applied ``hf_dataset[idx]``
    # yields.  These helpers run the equivalent conversion per row /
    # per window so the consumer doesn't have to know about pyarrow.
    # ------------------------------------------------------------------

    def load_episode_row_torch(
        self, raw_ep: int, frame_idx: int,
    ) -> dict[str, Any]:
        """Single row → dict of torch tensors (same shape as the old
        ``dataset.hf_dataset[sample_idx]`` return).
        """
        row_dict = self.load_episode_row_dict(raw_ep, frame_idx)
        return _row_dict_to_torch(row_dict)

    def load_episode_window_torch(
        self, raw_ep: int, frame_indices: list[int],
    ) -> dict[str, Any]:
        """Multi-row window → dict of stacked torch tensors."""
        table = self.load_episode_window(raw_ep, frame_indices)
        return _window_table_to_torch(table)


# ---------------------------------------------------------------------------
# Tensor conversion helpers (mirrors LeRobet's hf_transform_to_torch but
# applied per-row or per-window so we don't need a live transform on a
# materialized hf_dataset).
# ---------------------------------------------------------------------------


def _row_dict_to_torch(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a single row (column→scalar/list) into torch tensors."""
    import torch

    out: dict[str, Any] = {}
    for key, val in row_dict.items():
        if isinstance(val, list):
            out[key] = torch.tensor(val)
        elif isinstance(val, (int, float, bool)):
            out[key] = torch.tensor(val)
        else:
            # Non-tensor types pass through unchanged (e.g. str, None).
            out[key] = val
    return out


def _window_table_to_torch(table: "pa.Table") -> dict[str, Any]:
    """Convert a multi-row pyarrow Table into dict[str, stacked tensors]."""
    import torch

    out: dict[str, Any] = {}
    for name in table.column_names:
        col = table.column(name).to_pylist()   # list of values, one per row
        if not col:
            out[name] = torch.tensor([])
            continue
        first = col[0]
        if isinstance(first, list):
            out[name] = torch.tensor(col)
        elif isinstance(first, (int, float, bool)):
            out[name] = torch.tensor(col)
        else:
            out[name] = col
    return out
