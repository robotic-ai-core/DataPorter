"""Text prefetcher — streams HF docs to Parquet text shards.

Writes raw text strings to Parquet (no tokenization). DataLoader workers
handle tokenization in parallel via TransformableDataset.

Each offset runs in its own thread for parallel I/O and better shuffle
quality (shards contain docs from different regions of the dataset).
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .prefetcher import BasePrefetcher

logger = logging.getLogger(__name__)


class TextPrefetcher(BasePrefetcher):
    """Background threads that stream HF docs to Parquet text shards.

    Each offset gets its own thread — parallel I/O improves throughput
    and ensures shards contain docs from different dataset regions
    for better shuffle quality.

    Args:
        dataset: HuggingFace dataset ID.
        data_dir: Subdirectory within the dataset.
        text_field: Column name for text.
        output_dir: Local directory for Parquet shards.
        min_shards: Block until this many shards are available.
        max_shards: Evict oldest shards when exceeded (None = no limit).
        max_rows_per_shard: Rows per shard file.
        stream_shuffle_buffer: HF stream shuffle buffer size.
        offsets: List of stream offsets for data diversity.
        seed: Random seed for eviction and shuffling.
        _dataset_factory: Override dataset loading (for testing).
            Signature: (offset: int) -> iterable of dicts.
    """

    def __init__(
        self,
        dataset: str,
        data_dir: str | None = None,
        text_field: str = "text",
        output_dir: str | Path = "/tmp/text_prefetch",
        min_shards: int = 5,
        max_shards: int | None = 100,
        max_rows_per_shard: int = 10_000,
        stream_shuffle_buffer: int = 10_000,
        offsets: list[int] | None = None,
        seed: int = 42,
        _dataset_factory: Callable | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            min_shards=min_shards,
            max_shards=max_shards,
            eviction="stochastic_oldest",
            seed=seed,
        )
        self._dataset = dataset
        self._data_dir = data_dir
        self._text_field = text_field
        self._max_rows_per_shard = max_rows_per_shard
        self._shuffle_buffer = stream_shuffle_buffer
        self._offsets = offsets or [0, 2_000_000]
        self._dataset_factory = _dataset_factory

    def _load_dataset(self, offset: int) -> Any:
        """Load an HF streaming dataset at the given offset.

        Uses the shared rate limiter from hf_client for 429 protection.
        """
        if self._dataset_factory is not None:
            return self._dataset_factory(offset)

        from .hf_client import hf_load_dataset

        ds = hf_load_dataset(
            self._dataset,
            data_dir=self._data_dir,
            split="train",
            streaming=True,
        ).shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)

        if offset > 0:
            ds = ds.skip(offset)
        return ds

    def _run_inner(self) -> None:
        """Spawn one thread per offset, wait for all to finish."""
        if len(self._offsets) == 1:
            self._stream_offset(self._offsets[0])
            return

        threads = []
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def _worker(offset: int) -> None:
            try:
                self._stream_offset(offset)
            except BaseException as e:
                with errors_lock:
                    errors.append(e)

        for offset in self._offsets:
            t = threading.Thread(
                target=_worker,
                args=(offset,),
                daemon=True,
                name=f"text-stream-{offset}",
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            raise errors[0]

    def _stream_offset(self, offset: int) -> None:
        """Stream docs from one offset and write shards."""
        rng = random.Random(self._seed + offset)
        schema = pa.schema([("text", pa.string())])
        ds = self._load_dataset(offset)
        buffer: list[str] = []

        for doc in ds:
            if self._stop_event.is_set():
                break

            text = doc.get(self._text_field, "")
            if not text or not text.strip():
                continue

            buffer.append(text)

            if len(buffer) >= self._max_rows_per_shard:
                self._write_shard(buffer, schema)
                buffer.clear()
                self._maybe_evict(rng)

        if buffer and not self._stop_event.is_set():
            self._write_shard(buffer, schema)
            self._maybe_evict(rng)

    def _write_shard(self, texts: list[str], schema: pa.Schema) -> None:
        shard_path = self._next_shard_path()
        table = pa.table({"text": texts}, schema=schema)
        pq.write_table(table, str(shard_path), compression="zstd")

        logger.info(f"Wrote shard: {len(texts)} docs -> {shard_path.name}")
        self._check_min_ready()
