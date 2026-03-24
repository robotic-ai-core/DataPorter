"""Storage backends for PrefetchedSource.

Two implementations:
  - ``ShardStorage``: reads from a directory of Parquet shard files (text).
    Grows as new shards appear, handles deferred eviction.
  - ``MemoryStorage``: in-memory dict with LRU eviction (video frames).
    Fixed capacity, instant access.

Both implement the ``Storage`` protocol so ``PrefetchedSource`` can
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

    Supports deferred eviction: ``schedule_eviction()`` marks shards
    for deletion, executed on next ``refresh()``.

    Args:
        data_dir: Directory containing shard_*.parquet files.
        text_column: Column name to read.
        refresh_interval: Seconds between directory rescans.
        max_shards: Auto-evict oldest shards when exceeded.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        refresh_interval: float = 30.0,
        max_shards: int | None = None,
    ):
        self._dir = Path(data_dir)
        self._text_column = text_column
        self._refresh_interval = refresh_interval
        self._max_shards = max_shards
        self._last_refresh = 0.0

        self._shards: list[tuple[Path, int]] = []
        self._cumulative_rows: list[int] = []
        self._total_rows = 0
        self._shard_texts: dict[int, list[str]] = {}
        self._pending_evictions: set[Path] = set()

        self.refresh()

    def get(self, idx: int) -> dict[str, str] | None:
        self._maybe_refresh()
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

    def refresh(self) -> None:
        now = monotonic()
        self._last_refresh = now

        # Close caches before evicting
        self._shard_texts.clear()

        # Execute pending evictions
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

        # Auto-evict if over max_shards
        if self._max_shards is not None and len(paths) > self._max_shards:
            for p in paths[: len(paths) - self._max_shards]:
                try:
                    p.unlink()
                except (FileNotFoundError, OSError):
                    pass
            paths = sorted(self._dir.glob("*.parquet"))

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
