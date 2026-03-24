"""Simple Parquet source for raw text documents.

Thin wrapper around ``ShardStorage`` that implements the ``DataSource``
protocol. Maintains backward compatibility with existing callers.

For new code, prefer using ``PrefetchedSource(ShardStorage(...))`` directly.
"""

from __future__ import annotations

from pathlib import Path

from .storage import ShardStorage


class RawTextSource:
    """Random-access source over a directory of Parquet text shards.

    Delegates to ``ShardStorage`` for all operations.

    Args:
        data_dir: Directory containing .parquet files.
        text_column: Column name containing text strings.
        refresh_interval_seconds: How often to rescan for new shards.
        max_shards: Auto-evict oldest shards when exceeded.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        refresh_interval_seconds: float = 30.0,
        max_shards: int | None = None,
    ):
        self._storage = ShardStorage(
            data_dir=data_dir,
            text_column=text_column,
            refresh_interval=refresh_interval_seconds,
            max_shards=max_shards,
        )

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, idx: int) -> dict[str, str]:
        item = self._storage.get(idx)
        if item is None:
            raise IndexError(f"Index {idx} not available")
        return item

    @property
    def shard_count(self) -> int:
        return self._storage.shard_count

    def schedule_eviction(self, shard_path) -> None:
        self._storage.schedule_eviction(shard_path)

    @property
    def pending_eviction_count(self) -> int:
        return self._storage.pending_eviction_count

    def refresh(self) -> None:
        self._storage._last_refresh = 0.0
        self._storage.refresh()
