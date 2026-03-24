"""Lightweight text prefetcher — downloads HF docs to Parquet text shards.

Writes raw text strings to Parquet (no tokenization). DataLoader workers
handle tokenization in parallel via TransformableParquetDataset.

Much simpler than DataPorter's ParquetPrefetcher — no companion files,
no eviction, just a background thread writing text shards as fast as
the network allows.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class TextPrefetcher:
    """Background thread that streams HF docs to Parquet text shards.

    Args:
        dataset: HuggingFace dataset ID.
        data_dir: Subdirectory within the dataset.
        text_field: Column name for text.
        output_dir: Local directory for Parquet shards.
        min_shards: Block until this many shards are available.
        max_rows_per_shard: Rows per shard file.
        stream_shuffle_buffer: HF stream shuffle buffer size.
        offsets: List of stream offsets for data diversity.
    """

    def __init__(
        self,
        dataset: str,
        data_dir: str | None = None,
        text_field: str = "text",
        output_dir: str | Path = "/tmp/text_prefetch",
        min_shards: int = 5,
        max_rows_per_shard: int = 10_000,
        stream_shuffle_buffer: int = 10_000,
        offsets: list[int] | None = None,
    ):
        self._dataset = dataset
        self._data_dir = data_dir
        self._text_field = text_field
        self._output_dir = Path(output_dir)
        self._min_shards = min_shards
        self._max_rows_per_shard = max_rows_per_shard
        self._shuffle_buffer = stream_shuffle_buffer
        self._offsets = offsets or [0, 2_000_000]

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._shard_count = 0
        self._min_ready = threading.Event()
        self._error: BaseException | None = None

    @property
    def shard_count(self) -> int:
        return self._shard_count

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait_for_min(self, timeout: float = 300.0):
        """Block until min_shards are written."""
        if not self._min_ready.wait(timeout=timeout):
            if self._error:
                raise RuntimeError(f"Prefetcher failed: {self._error}") from self._error
            raise TimeoutError(
                f"Prefetcher didn't produce {self._min_shards} shards in {timeout}s"
            )
        if self._error:
            raise RuntimeError(f"Prefetcher failed: {self._error}") from self._error

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        try:
            self._produce()
        except BaseException as e:
            self._error = e
            self._min_ready.set()  # Unblock waiter

    def _produce(self):
        from datasets import load_dataset

        schema = pa.schema([("text", pa.string())])

        for offset in self._offsets:
            if self._stop_event.is_set():
                break

            ds = load_dataset(
                self._dataset,
                data_dir=self._data_dir,
                split="train",
                streaming=True,
            ).shuffle(buffer_size=self._shuffle_buffer)

            buffer: list[str] = []
            skipped = 0

            for i, doc in enumerate(ds):
                if self._stop_event.is_set():
                    break

                if i < offset:
                    skipped += 1
                    if skipped >= offset:
                        pass  # Now past the offset
                    else:
                        continue

                text = doc.get(self._text_field, "")
                if not text or not text.strip():
                    continue

                buffer.append(text)

                if len(buffer) >= self._max_rows_per_shard:
                    self._write_shard(buffer, schema)
                    buffer.clear()

            # Flush remaining
            if buffer and not self._stop_event.is_set():
                self._write_shard(buffer, schema)

    def _write_shard(self, texts: list[str], schema: pa.Schema):
        shard_path = self._output_dir / f"shard_{self._shard_count:06d}.parquet"
        table = pa.table({"text": texts}, schema=schema)
        pq.write_table(table, str(shard_path), compression="zstd")

        self._shard_count += 1
        logger.info(f"Wrote shard {self._shard_count}: {len(texts)} docs → {shard_path.name}")

        if self._shard_count >= self._min_shards:
            self._min_ready.set()
