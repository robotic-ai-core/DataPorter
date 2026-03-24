"""Simple Parquet source for raw text documents.

Reads string columns from a growing directory of Parquet files.
No fixed-length assumptions — each row is a variable-length string.
Rescans for new shards periodically.
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

    Args:
        data_dir: Directory containing .parquet files.
        text_column: Column name containing text strings.
        refresh_interval_seconds: How often to rescan for new shards.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        refresh_interval_seconds: float = 30.0,
    ):
        self._dir = Path(data_dir)
        self._text_column = text_column
        self._refresh_interval = refresh_interval_seconds
        self._last_refresh = 0.0

        # Shard metadata: [(path, num_rows), ...]
        self._shards: list[tuple[Path, int]] = []
        self._cumulative_rows: list[int] = []  # cumsum of rows
        self._total_rows = 0

        # Lazy file handles and text caches
        self._tables: dict[int, pq.ParquetFile] = {}
        self._shard_texts: dict[int, list[str]] = {}

        self._refresh()

    def _refresh(self):
        """Rescan directory for Parquet shards."""
        now = monotonic()
        if now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now

        paths = sorted(self._dir.glob("*.parquet"))
        if len(paths) == len(self._shards):
            return  # No new shards

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
                continue  # Skip corrupt/partial shards

        self._shards = shards
        self._cumulative_rows = cumulative
        self._total_rows = total
        self._tables.clear()
        self._shard_texts.clear()

        if shards:
            logger.debug(f"RawTextSource: {len(shards)} shards, {total:,} rows")

    def __len__(self) -> int:
        self._refresh()
        return self._total_rows

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map global idx to (shard_idx, local_row)."""
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
        idx = idx % self._total_rows
        shard_idx, local_row = self._locate(idx)

        # Cache the text column per shard — avoids re-reading Parquet
        # on every __getitem__. Each shard is ~10k strings, ~5MB in RAM.
        if shard_idx not in self._shard_texts:
            pf = self._get_table(shard_idx)
            col = pf.read().column(self._text_column)
            self._shard_texts[shard_idx] = col.to_pylist()

        text = self._shard_texts[shard_idx][local_row]
        return {"text": text}
