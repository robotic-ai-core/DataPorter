"""Random-access dataset over zstd-compressed Parquet shards.

Two access modes:
- **preload** (default): Reads all shards into a single numpy array at init.
  O(1) random access with zero per-sample overhead.  Best when dataset fits
  in RAM (~2 bytes/token, e.g. 500K × 512 = 0.5 GB).
- **streaming**: Row-group-level LRU cache with lazy file handles (spawn-safe).
  Trades CPU decompression for lower memory.  Use for datasets that don't fit
  in RAM.
"""

import logging
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .types import TextSample

logger = logging.getLogger(__name__)


def _read_shard_to_numpy(path: Path) -> np.ndarray:
    """Read a Parquet shard into a uint16 numpy array via zero-copy Arrow."""
    table = pq.read_table(path, columns=["input_ids"])
    # list<uint16> → flatten to 1D then reshape
    flat = table.column("input_ids").combine_chunks()
    values = flat.values.to_numpy(zero_copy_only=False).astype(np.uint16)
    seq_len = len(flat[0])
    return values.reshape(-1, seq_len)


class _RowGroupCache:
    """LRU cache for decompressed Parquet row groups.

    Thread-local per DataLoader worker — no locking needed.
    """

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


class ParquetTokenDataset(Dataset):
    """Random-access dataset over zstd-compressed Parquet shards.

    Args:
        data_dir: Root directory containing train/ and val/ subdirectories.
        split: Dataset split ("train" or "val").
        cache_size: Number of decompressed row groups to keep in LRU cache
            (only used when preload=False).
        preload: If True, read all shards into RAM at init for O(1) access.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        cache_size: int = 32,
        preload: bool = True,
    ):
        self._data_dir = Path(data_dir)
        self._split = split
        self._cache_size = cache_size
        self._preload = preload

        split_dir = self._data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._shard_paths = sorted(split_dir.glob("shard_*.parquet"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shard files found in {split_dir}")

        if preload:
            self._init_preloaded()
        else:
            self._init_streaming()

    # ------------------------------------------------------------------
    # Preloaded mode: single contiguous numpy array
    # ------------------------------------------------------------------

    def _init_preloaded(self) -> None:
        t0 = monotonic()
        arrays = [_read_shard_to_numpy(p) for p in self._shard_paths]
        self._data = np.concatenate(arrays, axis=0)
        self._total_rows = self._data.shape[0]
        self._seq_len_val = self._data.shape[1]
        elapsed = monotonic() - t0
        mb = self._data.nbytes / 1e6
        logger.info(
            f"Preloaded {self._split}: {self._total_rows:,} rows, "
            f"{mb:.0f} MB in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Streaming mode: row-group LRU cache (original path)
    # ------------------------------------------------------------------

    def _init_streaming(self) -> None:
        self._data = None
        self._rg_offsets: list[int] = []
        self._rg_map: list[tuple[int, int]] = []
        self._rg_sizes: list[int] = []

        cumulative = 0
        for shard_idx, shard_path in enumerate(self._shard_paths):
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.metadata.num_row_groups):
                n_rows = pf.metadata.row_group(rg_idx).num_rows
                self._rg_offsets.append(cumulative)
                self._rg_map.append((shard_idx, rg_idx))
                self._rg_sizes.append(n_rows)
                cumulative += n_rows

        self._total_rows = cumulative
        self._seq_len_val: int | None = None
        self._pf_handles: list[pq.ParquetFile] | None = None
        self._cache: _RowGroupCache | None = None

    def __len__(self) -> int:
        return self._total_rows

    @property
    def dataset_name(self) -> str:
        return self._data_dir.name

    @property
    def seq_len(self) -> int:
        if self._seq_len_val is None:
            sample = self[0]
            self._seq_len_val = sample["input_ids"].shape[0]
        return self._seq_len_val

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _ensure_handles(self) -> tuple[list[pq.ParquetFile], _RowGroupCache]:
        """Lazily open file handles and cache (spawn-safe)."""
        if self._pf_handles is None:
            self._pf_handles = [pq.ParquetFile(p) for p in self._shard_paths]
            self._cache = _RowGroupCache(self._cache_size)
        return self._pf_handles, self._cache

    def _resolve(self, idx: int) -> tuple[int, int, int]:
        """Map global index → (shard_idx, rg_idx_in_shard, local_row)."""
        rg_global = bisect_right(self._rg_offsets, idx) - 1
        local_row = idx - self._rg_offsets[rg_global]
        shard_idx, rg_in_shard = self._rg_map[rg_global]
        return shard_idx, rg_in_shard, local_row

    def _read_row_group(self, shard_idx: int, rg_idx: int) -> np.ndarray:
        """Read and cache a row group's input_ids column."""
        handles, cache = self._ensure_handles()
        key = (shard_idx, rg_idx)
        cached = cache.get(key)
        if cached is not None:
            return cached

        pf = handles[shard_idx]
        rg_table = pf.read_row_group(rg_idx, columns=["input_ids"])
        flat = rg_table.column("input_ids").combine_chunks()
        values = flat.values.to_numpy(zero_copy_only=False).astype(np.uint16)
        seq_len = len(flat[0])
        arr = values.reshape(-1, seq_len)
        cache.put(key, arr)
        return arr

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> TextSample:
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._total_rows})")

        if self._data is not None:
            # Preloaded: direct numpy indexing
            tokens = self._data[idx]
        else:
            # Streaming: row group lookup
            shard_idx, rg_idx, local_row = self._resolve(idx)
            tokens = self._read_row_group(shard_idx, rg_idx)[local_row]

        input_ids = torch.as_tensor(tokens, dtype=torch.long)
        return TextSample(input_ids=input_ids, labels=input_ids.clone())
