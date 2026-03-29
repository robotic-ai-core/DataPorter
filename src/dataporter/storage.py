"""Storage backends for PrefetchedSource.

Three implementations:
  - ``ShardStorage``: reads from a directory of Parquet shard files (text).
    Grows as new shards appear, handles deferred eviction.
  - ``MemoryStorage``: in-memory dict with LRU eviction (video frames).
    Fixed capacity, instant access. Single-process only.
  - ``SharedMemoryStorage``: pre-allocated shared memory tensor buffer.
    Works across forked DataLoader workers (num_workers > 0). Fixed shape.

All implement the ``Storage`` protocol so ``PrefetchedSource`` can
use them interchangeably.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Protocol, runtime_checkable

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Storage(Protocol):
    """Protocol for a key-value storage backend."""

    def get(self, idx: int) -> Any | None:
        """Get item by index, or None if not available."""
        ...

    def __len__(self) -> int:
        """Number of items currently available."""
        ...

    def __contains__(self, idx: int) -> bool:
        """Check if index is available."""
        ...

    def evict(self, n: int = 1) -> int:
        """Evict up to n items. Returns count evicted."""
        ...

    @property
    def capacity(self) -> int | None:
        """Max items (None = unlimited)."""
        ...

    def refresh(self) -> None:
        """Rescan for new data (no-op for in-memory)."""
        ...


# ---------------------------------------------------------------------------
# ShardStorage — Parquet shard files on disk
# ---------------------------------------------------------------------------


class ShardStorage:
    """Storage backed by a directory of Parquet shard files.

    Each shard is a Parquet file with a text column. Items are
    addressed by global row index across all shards.

    Supports:
      - Deferred eviction: ``schedule_eviction()`` marks shards for
        deletion, executed on next ``refresh()``.
      - Epoch-aware eviction: ``freeze()`` / ``unfreeze()`` defer eviction
        during an epoch so the shard list stays stable for resumption.
      - State dict: ``state_dict()`` / ``load_state_dict()`` save and
        restore the exact shard list for deterministic resume.

    Args:
        data_dir: Directory containing shard_*.parquet files.
        text_column: Column name to read.
        refresh_interval: Seconds between directory rescans.
        max_shards: Auto-evict oldest shards when count exceeded.
        max_cache_gb: Auto-evict oldest shards when total size exceeded (GB).
            Both limits can be set — eviction triggers when either is exceeded.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        refresh_interval: float = 30.0,
        max_shards: int | None = None,
        max_cache_gb: float | None = None,
    ):
        self._dir = Path(data_dir)
        self._text_column = text_column
        self._refresh_interval = refresh_interval
        self._max_shards = max_shards
        self._max_cache_bytes = int(max_cache_gb * 1_073_741_824) if max_cache_gb else None
        self._last_refresh = 0.0

        self._shards: list[tuple[Path, int]] = []
        self._cumulative_rows: list[int] = []
        self._total_rows = 0
        self._shard_texts: dict[int, list[str]] = {}
        self._pending_evictions: set[Path] = set()

        # Epoch-aware eviction: when frozen, eviction is deferred
        self._frozen = False
        # Pinned shard list from state_dict (for resume)
        self._pinned_shards: list[str] | None = None

        self.refresh()

    def get(self, idx: int) -> dict[str, str] | None:
        self._maybe_refresh()
        self._maybe_evict_excess()
        if self._total_rows == 0:
            return None
        idx = idx % self._total_rows
        shard_idx, local_row = self._locate(idx)

        try:
            if shard_idx not in self._shard_texts:
                pf = pq.ParquetFile(str(self._shards[shard_idx][0]))
                col = pf.read().column(self._text_column)
                self._shard_texts[shard_idx] = col.to_pylist()
            return {"text": self._shard_texts[shard_idx][local_row]}
        except (FileNotFoundError, KeyError, IndexError):
            self._last_refresh = 0.0
            self.refresh()
            if self._total_rows == 0:
                return None
            idx = idx % self._total_rows
            shard_idx, local_row = self._locate(idx)
            if shard_idx not in self._shard_texts:
                pf = pq.ParquetFile(str(self._shards[shard_idx][0]))
                col = pf.read().column(self._text_column)
                self._shard_texts[shard_idx] = col.to_pylist()
            return {"text": self._shard_texts[shard_idx][local_row]}

    def __len__(self) -> int:
        self._maybe_refresh()
        return self._total_rows

    def __contains__(self, idx: int) -> bool:
        return 0 <= idx < self._total_rows

    @property
    def capacity(self) -> int | None:
        return self._max_shards

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def schedule_eviction(self, shard_path: str | Path) -> None:
        self._pending_evictions.add(Path(shard_path))

    @property
    def pending_eviction_count(self) -> int:
        return len(self._pending_evictions)

    def evict(self, n: int = 1) -> int:
        """Schedule n oldest shards for eviction (executed on next refresh)."""
        evicted = 0
        for path, _ in self._shards[:n]:
            self._pending_evictions.add(path)
            evicted += 1
        return evicted

    def freeze(self) -> None:
        """Freeze the shard list — defer all eviction until ``unfreeze()``.

        Use during an epoch to keep the index stable for resumption.
        New shards from the prefetcher are still picked up on refresh.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze — execute any pending evictions on next refresh."""
        self._frozen = False
        self._pinned_shards = None

    def state_dict(self) -> dict:
        """Save shard list and row counts for deterministic resume."""
        return {
            "shard_names": [p.name for p, _ in self._shards],
            "shard_rows": [n for _, n in self._shards],
            "total_rows": self._total_rows,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore pinned shard list from checkpoint.

        On next ``refresh()``, only shards in this list are used
        (others are ignored, not evicted). Call ``unfreeze()`` after
        the epoch completes to allow eviction and new shards.
        """
        self._pinned_shards = state.get("shard_names")
        self._frozen = True
        # Force a refresh to apply the pinned list
        self._last_refresh = 0.0
        self.refresh()

    def refresh(self) -> None:
        now = monotonic()
        self._last_refresh = now

        # Close caches before evicting
        self._shard_texts.clear()

        # Execute pending evictions (only if not frozen)
        if not self._frozen:
            for path in list(self._pending_evictions):
                try:
                    if path.exists():
                        manifest = path.with_suffix(".companions.json")
                        if manifest.exists():
                            manifest.unlink()
                        path.unlink()
                except (FileNotFoundError, OSError):
                    pass
            self._pending_evictions.clear()

        paths = sorted(self._dir.glob("*.parquet"))

        # Auto-evict if over limits (only if not frozen)
        if not self._frozen:
            if self._max_shards is not None and len(paths) > self._max_shards:
                for p in paths[: len(paths) - self._max_shards]:
                    try:
                        p.unlink()
                    except (FileNotFoundError, OSError):
                        pass
                paths = sorted(self._dir.glob("*.parquet"))

            if self._max_cache_bytes is not None:
                total = sum(p.stat().st_size for p in paths)
                while total > self._max_cache_bytes and paths:
                    victim = paths.pop(0)
                    try:
                        total -= victim.stat().st_size
                        victim.unlink()
                    except (FileNotFoundError, OSError):
                        pass

        # If pinned, filter to only the pinned shard names
        if self._pinned_shards is not None:
            pinned_set = set(self._pinned_shards)
            paths = [p for p in paths if p.name in pinned_set]

        shards = []
        cumulative = []
        total = 0
        for p in paths:
            try:
                pf = pq.ParquetFile(str(p))
                n = pf.metadata.num_rows
                shards.append((p, n))
                total += n
                cumulative.append(total)
            except Exception:
                continue

        self._shards = shards
        self._cumulative_rows = cumulative
        self._total_rows = total

    def _maybe_refresh(self) -> None:
        if monotonic() - self._last_refresh >= self._refresh_interval:
            self.refresh()

    def _maybe_evict_excess(self) -> None:
        """Quick eviction check — throttled to every 0.5 seconds.

        Evicts oldest shards when either limit is exceeded:
        - ``max_shards``: shard count limit
        - ``max_cache_gb``: total cache size limit

        Throttled because glob + stat costs ~1ms for 200 files.
        """
        if self._frozen:
            return
        if self._max_shards is None and self._max_cache_bytes is None:
            return
        now = monotonic()
        if now - getattr(self, "_last_evict_check", 0.0) < 0.5:
            return
        self._last_evict_check = now

        paths = sorted(self._dir.glob("*.parquet"))

        # Evict by count
        if self._max_shards is not None and len(paths) > self._max_shards:
            for p in paths[: len(paths) - self._max_shards]:
                try:
                    p.unlink()
                except (FileNotFoundError, OSError):
                    pass
            paths = sorted(self._dir.glob("*.parquet"))

        # Evict by size
        if self._max_cache_bytes is not None:
            total = sum(p.stat().st_size for p in paths)
            while total > self._max_cache_bytes and paths:
                victim = paths.pop(0)
                try:
                    total -= victim.stat().st_size
                    victim.unlink()
                except (FileNotFoundError, OSError):
                    pass

    def _locate(self, idx: int) -> tuple[int, int]:
        shard_idx = bisect_right(self._cumulative_rows, idx)
        local = idx - (self._cumulative_rows[shard_idx - 1] if shard_idx > 0 else 0)
        return shard_idx, local


# ---------------------------------------------------------------------------
# MemoryStorage — in-memory dict with LRU eviction
# ---------------------------------------------------------------------------


class MemoryStorage:
    """Storage backed by an in-memory OrderedDict with LRU eviction.

    Items are stored as arbitrary values (frame tensors, dicts, etc.).
    When capacity is reached, oldest items are evicted.

    Args:
        capacity: Max items in storage. None = unlimited.
    """

    def __init__(self, capacity: int | None = None):
        self._capacity = capacity
        self._data: OrderedDict[int, Any] = OrderedDict()

    def get(self, idx: int) -> Any | None:
        if idx in self._data:
            self._data.move_to_end(idx)
            return self._data[idx]
        return None

    def put(self, idx: int, value: Any) -> None:
        """Store an item. Evicts oldest if at capacity."""
        if idx in self._data:
            self._data.move_to_end(idx)
            self._data[idx] = value
            return
        if self._capacity is not None and len(self._data) >= self._capacity:
            self._data.popitem(last=False)
        self._data[idx] = value

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, idx: int) -> bool:
        return idx in self._data

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def evict(self, n: int = 1) -> int:
        evicted = 0
        for _ in range(min(n, len(self._data))):
            self._data.popitem(last=False)
            evicted += 1
        return evicted

    def refresh(self) -> None:
        pass  # No-op for in-memory storage

    def keys(self) -> list[int]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# SharedMemoryStorage — shared tensor buffer for multi-worker DataLoader
# ---------------------------------------------------------------------------


class SharedMemoryStorage:
    """Storage backed by pre-allocated shared memory tensors.

    Designed for multi-worker DataLoader: the producer fills the buffer
    in the main process, and forked workers read from the same shared
    memory — zero-copy, no IPC overhead.

    All tensors use ``share_memory_()`` so they survive ``fork()``.
    Variable-length episodes are supported via a lengths array.

    Args:
        capacity: Max number of items (episodes) in the buffer.
        max_frames: Max frames per episode (padded with zeros).
        channels: Number of image channels.
        height: Frame height.
        width: Frame width.
        max_keys: Max number of unique keys (episode indices) to track.
            Defaults to capacity * 10 (sparse key space).
    """

    def __init__(
        self,
        capacity: int,
        max_frames: int,
        channels: int,
        height: int,
        width: int,
        max_keys: int | None = None,
    ):
        import torch

        self._capacity = capacity
        self._max_frames = max_frames
        max_keys = max_keys or capacity * 10

        # Shared memory tensors — survive fork()
        self._buffer = torch.zeros(
            capacity, max_frames, channels, height, width, dtype=torch.uint8
        ).share_memory_()
        self._lengths = torch.zeros(capacity, dtype=torch.int32).share_memory_()

        # Slot management: index_map[ep_idx] = slot (-1 = not present)
        self._index_map = torch.full(
            (max_keys,), -1, dtype=torch.int32
        ).share_memory_()

        # Ring buffer for slot allocation (FIFO eviction)
        self._slot_keys = torch.full(
            (capacity,), -1, dtype=torch.int32
        ).share_memory_()  # slot → ep_idx that occupies it
        self._next_slot = torch.zeros(1, dtype=torch.int32).share_memory_()
        self._count = torch.zeros(1, dtype=torch.int32).share_memory_()

    def put(self, idx: int, value: Any) -> None:
        """Store frames for an episode.

        Args:
            idx: Episode index (key).
            value: Tensor of shape [num_frames, C, H, W] (uint8), or a dict
                with a "frames" key containing such a tensor.
        """
        import torch

        if isinstance(value, dict):
            frames = value["frames"]
        else:
            frames = value

        n_frames = min(frames.shape[0], self._max_frames)

        # Check if already stored
        if idx < len(self._index_map) and self._index_map[idx] >= 0:
            slot = self._index_map[idx].item()
            self._buffer[slot, :n_frames] = frames[:n_frames]
            self._lengths[slot] = n_frames
            return

        # Allocate a slot (ring buffer — evicts oldest on wrap)
        slot = self._next_slot.item() % self._capacity
        self._next_slot[0] = slot + 1

        # Evict previous occupant if any
        old_key = self._slot_keys[slot].item()
        if old_key >= 0 and old_key < len(self._index_map):
            self._index_map[old_key] = -1

        # Write
        self._buffer[slot, :n_frames] = frames[:n_frames]
        if n_frames < self._max_frames:
            self._buffer[slot, n_frames:] = 0
        self._lengths[slot] = n_frames
        self._slot_keys[slot] = idx
        if idx < len(self._index_map):
            self._index_map[idx] = slot
        self._count[0] = min(self._count[0] + 1, self._capacity)

    def get(self, idx: int) -> dict[str, Any] | None:
        """Get frames for an episode. Returns dict with 'frames' tensor."""
        if idx >= len(self._index_map):
            return None
        slot_val = self._index_map[idx]
        if slot_val < 0:
            return None
        slot = int(slot_val)
        n_frames = int(self._lengths[slot])
        if n_frames == 0:
            return None
        return {"frames": self._buffer[slot, :n_frames]}

    def __len__(self) -> int:
        return self._count.item()

    def __contains__(self, idx: int) -> bool:
        return (
            idx < len(self._index_map)
            and self._index_map[idx].item() >= 0
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    def evict(self, n: int = 1) -> int:
        """Evict n oldest items (by slot order)."""
        evicted = 0
        current = len(self)
        # Find the oldest occupied slots
        oldest_slot = (self._next_slot.item() - current) % self._capacity
        for i in range(min(n, current)):
            slot = (oldest_slot + i) % self._capacity
            key = self._slot_keys[slot].item()
            if key >= 0 and key < len(self._index_map):
                self._index_map[key] = -1
            self._slot_keys[slot] = -1
            self._lengths[slot] = 0
            self._count[0] = max(0, self._count[0] - 1)
            evicted += 1
        return evicted

    def refresh(self) -> None:
        pass  # No-op

    def keys(self) -> list[int]:
        """Return list of episode indices currently in the buffer."""
        result = []
        for slot in range(self._capacity):
            key = self._slot_keys[slot].item()
            if key >= 0 and self._lengths[slot].item() > 0:
                result.append(key)
        return result

    def clear(self) -> None:
        self._index_map.fill_(-1)
        self._slot_keys.fill_(-1)
        self._lengths.fill_(0)
        self._buffer.fill_(0)
        self._next_slot.fill_(0)
        self._count.fill_(0)

    def state_dict(self) -> dict:
        """Save which episodes are in the buffer.

        Does NOT save the frame data — that must be re-decoded on resume.
        The ``priority_keys`` list tells the producer which episodes to
        decode first so the buffer is warm immediately.
        """
        return {
            "episode_keys": self.keys(),
            "capacity": self._capacity,
            "max_frames": self._max_frames,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the priority decode list.

        The buffer itself is cleared (frames can't be checkpointed
        efficiently). The returned ``priority_keys`` are stored for
        the producer to decode first.
        """
        self._priority_keys = state.get("episode_keys", [])

    @property
    def priority_keys(self) -> list[int]:
        """Episode indices to decode first on resume (set by load_state_dict)."""
        return getattr(self, "_priority_keys", [])
