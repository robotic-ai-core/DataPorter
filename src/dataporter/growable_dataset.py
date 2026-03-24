"""Growable Parquet dataset that rescans for new shards.

Wraps a directory of Parquet shard files and periodically rescans to pick
up new shards written by the ParquetPrefetcher. Unlike ParquetTokenDataset,
this dataset does NOT require a split subdirectory — it reads directly from
the given directory.

The dataset uses streaming mode (row-group LRU cache) since shards arrive
incrementally and preloading the full dataset at init would miss future shards.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class _RowGroupCache:
    """LRU cache for decompressed Parquet row groups."""

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get(self, key: tuple[int, int]) -> np.ndarray | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple[int, int], value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def invalidate(self, shard_idx: int) -> None:
        """Remove all entries for a given shard index."""
        keys_to_remove = [k for k in self._cache if k[0] == shard_idx]
        for k in keys_to_remove:
            del self._cache[k]


class GrowableParquetDataset(Dataset):
    """Random-access dataset over a growing directory of Parquet shards.

    Periodically rescans the directory for new shard files. Designed to
    work with ParquetPrefetcher which writes new shards in the background.

    Args:
        data_dir: Directory containing shard_*.parquet files.
        refresh_interval_seconds: How often to rescan for new shards.
        cache_size: Number of decompressed row groups to keep in LRU cache.
        column: Name of the column to read from Parquet files.
    """

    def __init__(
        self,
        data_dir: str | Path,
        refresh_interval_seconds: float = 30.0,
        cache_size: int = 32,
        column: str = "input_ids",
    ):
        self._data_dir = Path(data_dir)
        self._refresh_interval = refresh_interval_seconds
        self._cache_size = cache_size
        self._column = column

        # Per-shard metadata
        self._shard_paths: list[Path] = []
        self._shard_names: set[str] = set()
        self._rg_offsets: list[int] = []
        self._rg_map: list[tuple[int, int]] = []  # (shard_idx, rg_idx)
        self._rg_sizes: list[int] = []
        self._total_rows = 0
        self._seq_len_val: int | None = None

        # Lazy handles (spawn-safe)
        self._pf_handles: list[pq.ParquetFile | None] = []
        self._cache: _RowGroupCache | None = None
        self._last_refresh = 0.0

        # Initial scan
        self.refresh()

    def refresh(self) -> None:
        """Rescan directory for new (and removed) shards."""
        now = monotonic()
        current_shards = sorted(self._data_dir.glob("shard_*.parquet"))
        current_names = {p.name for p in current_shards}

        # Detect removed shards (eviction)
        removed = self._shard_names - current_names
        if removed or current_names != self._shard_names:
            # Full rebuild of index if any shards removed or new ones found
            self._rebuild_index(current_shards)

        self._last_refresh = now

    def _rebuild_index(self, shard_paths: list[Path]) -> None:
        """Rebuild the row-group index from scratch."""
        self._shard_paths = shard_paths
        self._shard_names = {p.name for p in shard_paths}
        self._rg_offsets = []
        self._rg_map = []
        self._rg_sizes = []

        cumulative = 0
        for shard_idx, shard_path in enumerate(shard_paths):
            try:
                pf = pq.ParquetFile(shard_path)
            except Exception:
                logger.warning(f"Skipping unreadable shard: {shard_path}")
                continue
            for rg_idx in range(pf.metadata.num_row_groups):
                n_rows = pf.metadata.row_group(rg_idx).num_rows
                self._rg_offsets.append(cumulative)
                self._rg_map.append((shard_idx, rg_idx))
                self._rg_sizes.append(n_rows)
                cumulative += n_rows

        self._total_rows = cumulative

        # Invalidate handles and cache (shards may have been evicted)
        self._pf_handles = [None] * len(shard_paths)
        self._cache = None
        self._seq_len_val = None

    def _maybe_refresh(self) -> None:
        """Refresh if enough time has elapsed since last scan."""
        if monotonic() - self._last_refresh >= self._refresh_interval:
            self.refresh()

    def __len__(self) -> int:
        self._maybe_refresh()
        return self._total_rows

    @property
    def shard_count(self) -> int:
        return len(self._shard_paths)

    @property
    def dataset_name(self) -> str:
        return self._data_dir.name

    @property
    def seq_len(self) -> int:
        if self._seq_len_val is None:
            if self._total_rows == 0:
                raise RuntimeError("No data available yet")
            sample = self[0]
            self._seq_len_val = sample["input_ids"].shape[0]
        return self._seq_len_val

    def _ensure_handle(self, shard_idx: int) -> pq.ParquetFile:
        """Lazily open a ParquetFile handle (spawn-safe)."""
        if self._pf_handles[shard_idx] is None:
            self._pf_handles[shard_idx] = pq.ParquetFile(self._shard_paths[shard_idx])
        return self._pf_handles[shard_idx]

    def _ensure_cache(self) -> _RowGroupCache:
        if self._cache is None:
            self._cache = _RowGroupCache(self._cache_size)
        return self._cache

    def _resolve(self, idx: int) -> tuple[int, int, int]:
        """Map global index -> (shard_idx, rg_idx_in_shard, local_row)."""
        rg_global = bisect_right(self._rg_offsets, idx) - 1
        local_row = idx - self._rg_offsets[rg_global]
        shard_idx, rg_in_shard = self._rg_map[rg_global]
        return shard_idx, rg_in_shard, local_row

    def _read_row_group(self, shard_idx: int, rg_idx: int) -> np.ndarray:
        """Read and cache a row group."""
        cache = self._ensure_cache()
        key = (shard_idx, rg_idx)
        cached = cache.get(key)
        if cached is not None:
            return cached

        pf = self._ensure_handle(shard_idx)
        rg_table = pf.read_row_group(rg_idx, columns=[self._column])
        flat = rg_table.column(self._column).combine_chunks()
        values = flat.values.to_numpy(zero_copy_only=False).astype(np.uint16)
        seq_len = len(flat[0])
        arr = values.reshape(-1, seq_len)
        cache.put(key, arr)
        return arr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._maybe_refresh()

        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        shard_idx, rg_idx, local_row = self._resolve(idx)
        tokens = self._read_row_group(shard_idx, rg_idx)[local_row]

        input_ids = torch.as_tensor(tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
