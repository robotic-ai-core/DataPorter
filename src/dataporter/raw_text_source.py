"""Simple Parquet source for raw text documents.

Reads string columns from a growing directory of Parquet files.
No fixed-length assumptions — each row is a variable-length string.
Rescans for new shards periodically.

Supports **deferred eviction**: callers schedule shards for deletion
via ``schedule_eviction()``. Actual deletion happens during the next
``refresh()`` after all file handles and caches for those shards are
closed. This eliminates the race condition where a DataLoader worker
tries to read a shard that was just deleted.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from pathlib import Path
from time import monotonic

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class RawTextSource:
    """Random-access source over a directory of Parquet text shards.

    Each ``__getitem__`` returns a dict with the text column value.
    Periodically rescans the directory for new shards written by
    the prefetcher.

    Eviction is deferred: call ``schedule_eviction(path)`` to mark
    a shard for deletion. The file is only deleted during the next
    ``refresh()`` after handles are closed.

    Args:
        data_dir: Directory containing .parquet files.
        text_column: Column name containing text strings.
        refresh_interval_seconds: How often to rescan for new shards.
        max_shards: If set, automatically schedule eviction of oldest
            shards when count exceeds this. None = no auto-eviction.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        refresh_interval_seconds: float = 30.0,
        max_shards: int | None = None,
    ):
        self._dir = Path(data_dir)
        self._text_column = text_column
        self._refresh_interval = refresh_interval_seconds
        self._max_shards = max_shards
        self._last_refresh = 0.0

        # Shard metadata: [(path, num_rows), ...]
        self._shards: list[tuple[Path, int]] = []
        self._cumulative_rows: list[int] = []
        self._total_rows = 0

        # Lazy file handles and text caches
        self._tables: dict[int, pq.ParquetFile] = {}
        self._shard_texts: dict[int, list[str]] = {}

        # Deferred eviction: paths scheduled for deletion
        self._pending_evictions: set[Path] = set()

        self._refresh()

    def schedule_eviction(self, shard_path: str | Path) -> None:
        """Schedule a shard for deletion on next refresh.

        The file is not deleted immediately — it stays readable until
        the next ``refresh()`` closes all handles and rebuilds the index.
        Safe to call from any thread/process.
        """
        self._pending_evictions.add(Path(shard_path))

    @property
    def pending_eviction_count(self) -> int:
        return len(self._pending_evictions)

    def _execute_pending_evictions(self) -> int:
        """Delete all pending eviction files. Returns count deleted."""
        if not self._pending_evictions:
            return 0
        deleted = 0
        for path in list(self._pending_evictions):
            try:
                if path.exists():
                    # Also delete companion manifest if present
                    manifest = path.with_suffix(".companions.json")
                    if manifest.exists():
                        manifest.unlink()
                    path.unlink()
                    deleted += 1
                    logger.debug(f"Evicted shard: {path.name}")
            except FileNotFoundError:
                pass  # Already gone (another process deleted it)
            except OSError as e:
                logger.warning(f"Failed to evict {path.name}: {e}")
        self._pending_evictions.clear()
        return deleted

    def _auto_evict_oldest(self) -> None:
        """Schedule oldest shards for eviction if over max_shards."""
        if self._max_shards is None:
            return
        n_over = len(self._shards) - self._max_shards
        if n_over <= 0:
            return
        # Schedule the oldest n_over shards (list is sorted by name)
        for path, _ in self._shards[:n_over]:
            self._pending_evictions.add(path)

    def _refresh(self) -> None:
        """Rescan directory for Parquet shards.

        Executes pending evictions first (handles are closed during
        index rebuild), then scans for current files.
        """
        now = monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now

        # Close all handles and caches before evicting
        self._tables.clear()
        self._shard_texts.clear()

        # Execute deferred evictions (handles now closed)
        self._execute_pending_evictions()

        paths = sorted(self._dir.glob("*.parquet"))

        # Auto-evict if over max_shards (schedule + execute immediately
        # since handles are already closed)
        if self._max_shards is not None and len(paths) > self._max_shards:
            n_over = len(paths) - self._max_shards
            for p in paths[:n_over]:
                try:
                    manifest = p.with_suffix(".companions.json")
                    if manifest.exists():
                        manifest.unlink()
                    p.unlink()
                    logger.debug(f"Auto-evicted shard: {p.name}")
                except (FileNotFoundError, OSError):
                    pass
            # Re-scan after eviction
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

        if shards:
            logger.debug(f"RawTextSource: {len(shards)} shards, {total:,} rows")

    def __len__(self) -> int:
        self._refresh()
        return self._total_rows

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range ({self._total_rows})")
        shard_idx = bisect_right(self._cumulative_rows, idx)
        local = idx - (self._cumulative_rows[shard_idx - 1] if shard_idx > 0 else 0)
        return shard_idx, local

    def _get_table(self, shard_idx: int) -> pq.ParquetFile:
        if shard_idx not in self._tables:
            self._tables[shard_idx] = pq.ParquetFile(
                str(self._shards[shard_idx][0])
            )
        return self._tables[shard_idx]

    def __getitem__(self, idx: int) -> dict[str, str]:
        self._refresh()
        if self._total_rows == 0:
            raise IndexError("No data available")
        idx = idx % self._total_rows
        shard_idx, local_row = self._locate(idx)

        try:
            if shard_idx not in self._shard_texts:
                pf = self._get_table(shard_idx)
                col = pf.read().column(self._text_column)
                self._shard_texts[shard_idx] = col.to_pylist()
            text = self._shard_texts[shard_idx][local_row]
        except (FileNotFoundError, KeyError, IndexError):
            # Shard gone (external deletion) — force refresh and retry
            self._last_refresh = 0.0
            self._refresh()
            if self._total_rows == 0:
                raise IndexError("No data available after refresh")
            idx = idx % self._total_rows
            shard_idx, local_row = self._locate(idx)
            if shard_idx not in self._shard_texts:
                pf = self._get_table(shard_idx)
                col = pf.read().column(self._text_column)
                self._shard_texts[shard_idx] = col.to_pylist()
            text = self._shard_texts[shard_idx][local_row]

        return {"text": text}
