"""Text prefetcher — streams HF docs to Parquet text shards.

Writes raw text strings to Parquet (no tokenization). DataLoader workers
handle tokenization in parallel via TransformableDataset.

Architecture (within the prefetcher process):

    Stream thread 0 ──→ ┐
    Stream thread 1 ──→ ├──→ doc queue ──→ writer thread ──→ .tmp → .parquet
    Stream thread N ──→ ┘

- Stream threads pull docs from HF and push to a shared queue. When a
  stream exhausts its offset, it restarts from a new random offset.
  Network stays saturated until stop() or max_shards is reached.
- Writer thread drains the queue and writes Parquet shards. Disk I/O
  overlaps fully with network I/O.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .prefetcher import BasePrefetcher, atomic_write

logger = logging.getLogger(__name__)

_SENTINEL = None  # Signals a stream thread has exited


class TextPrefetcher(BasePrefetcher):
    """Background process that streams HF docs to Parquet text shards.

    Producer threads stream from HF, pushing docs to a shared queue.
    A writer thread drains the queue into Parquet shards. Network I/O
    and disk I/O overlap fully — the network stays saturated.

    When a stream exhausts its data, it restarts from a new random
    offset. Streaming only stops when stop() is called or all data
    has been written.

    Args:
        dataset: HuggingFace dataset ID.
        data_dir: Subdirectory within the dataset.
        text_field: Column name for text.
        output_dir: Local directory for Parquet shards.
        min_shards: Block until this many shards are available.
        max_shards: Evict oldest shards when exceeded (None = no limit).
        max_rows_per_shard: Rows per shard file.
        stream_shuffle_buffer: HF stream shuffle buffer size.
        offsets: List of initial stream offsets for data diversity.
        seed: Random seed for eviction and shuffling.
        queue_size: Max docs buffered between producers and writer.
        max_restarts: Max times a stream restarts after exhaustion.
            None = unlimited (stream until stop/max_shards).
        _dataset_factory: Override dataset loading (for testing).
            Forces thread mode (lambdas aren't picklable).
    """

    def __init__(
        self,
        output_dir: str | Path,
        dataset: str = "",
        data_dir: str | None = None,
        text_field: str = "text",
        min_shards: int = 5,
        max_shards: int | None = 100,
        max_rows_per_shard: int = 10_000,
        stream_shuffle_buffer: int = 10_000,
        offsets: list[int] | None = None,
        seed: int = 42,
        queue_size: int = 10_000,
        max_restarts: int | None = None,
        _dataset_factory: Callable | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            min_shards=min_shards,
            max_shards=max_shards,
            eviction="stochastic_oldest",
            seed=seed,
        )
        self._use_thread = _dataset_factory is not None
        self._dataset = dataset
        self._data_dir = data_dir
        self._text_field = text_field
        self._max_rows_per_shard = max_rows_per_shard
        self._shuffle_buffer = stream_shuffle_buffer
        self._offsets = offsets or [0, 2_000_000]
        self._dataset_factory = _dataset_factory
        self._queue_size = queue_size
        self._max_restarts = max_restarts

    def _get_init_kwargs(self) -> dict[str, Any]:
        return dict(
            dataset=self._dataset,
            data_dir=self._data_dir,
            text_field=self._text_field,
            output_dir=str(self._output_dir),
            min_shards=self._min_shards,
            max_shards=self._max_shards,
            max_rows_per_shard=self._max_rows_per_shard,
            stream_shuffle_buffer=self._shuffle_buffer,
            offsets=self._offsets,
            seed=self._seed,
            queue_size=self._queue_size,
            max_restarts=self._max_restarts,
        )

    def _load_dataset(self, offset: int) -> Any:
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _run_inner(self) -> None:
        self._shard_counter = self.shard_count

        doc_queue: queue.Queue[str | None] = queue.Queue(maxsize=self._queue_size)
        n_producers = len(self._offsets)

        # Start producer threads (one per offset)
        producers = []
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        for i, offset in enumerate(self._offsets):
            t = threading.Thread(
                target=self._producer,
                args=(offset, i, doc_queue, errors, errors_lock),
                daemon=True,
                name=f"stream-{offset}",
            )
            producers.append(t)
            t.start()

        # Writer runs on this thread (the main thread of the child process)
        try:
            self._writer(doc_queue, n_producers)
        finally:
            # Ensure all producers stop
            self._stop_event.set()
            for t in producers:
                t.join(timeout=5)

        if errors:
            raise errors[0]

    # ------------------------------------------------------------------
    # Producer: stream docs from HF → queue
    # ------------------------------------------------------------------

    def _producer(
        self,
        initial_offset: int,
        producer_idx: int,
        doc_queue: queue.Queue,
        errors: list,
        errors_lock: threading.Lock,
    ) -> None:
        """Stream docs from HF, push text to queue. Restart on exhaustion."""
        rng = random.Random(self._seed + initial_offset)
        offset = initial_offset
        restarts = 0

        try:
            while not self._stop_event.is_set():
                ds = self._load_dataset(offset)

                for doc in ds:
                    if self._stop_event.is_set():
                        return

                    text = doc.get(self._text_field, "")
                    if not text or not text.strip():
                        continue

                    # Block if queue is full — backpressure from writer
                    while not self._stop_event.is_set():
                        try:
                            doc_queue.put(text, timeout=0.5)
                            break
                        except queue.Full:
                            continue

                # Stream exhausted — restart from random offset
                if self._max_restarts is not None and restarts >= self._max_restarts:
                    break
                restarts += 1
                offset = rng.randint(0, 10_000_000)
                logger.debug(
                    f"Producer {producer_idx} restarting at offset {offset} "
                    f"(restart {restarts})"
                )
        except BaseException as e:
            with errors_lock:
                errors.append(e)
        finally:
            # Signal writer that this producer is done
            doc_queue.put(_SENTINEL)

    # ------------------------------------------------------------------
    # Writer: queue → Parquet shards
    # ------------------------------------------------------------------

    def _writer(self, doc_queue: queue.Queue, n_producers: int) -> None:
        """Drain doc queue and write Parquet shards."""
        rng = random.Random(self._seed)
        schema = pa.schema([("text", pa.string())])
        buffer: list[str] = []
        producers_done = 0

        while producers_done < n_producers:
            if self._stop_event.is_set():
                break

            try:
                text = doc_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is _SENTINEL:
                producers_done += 1
                continue

            buffer.append(text)

            if len(buffer) >= self._max_rows_per_shard:
                self._write_shard(buffer, schema)
                buffer.clear()
                self._maybe_evict(rng)

        # Flush remaining
        if buffer and not self._stop_event.is_set():
            self._write_shard(buffer, schema)
            self._maybe_evict(rng)

    def _write_shard(self, texts: list[str], schema: pa.Schema) -> None:
        tmp_path, final_path = self._next_shard_tmp_path()
        table = pa.table({"text": texts}, schema=schema)
        pq.write_table(table, str(tmp_path), compression="zstd")
        atomic_write(tmp_path, final_path)

        logger.info(f"Wrote shard: {len(texts)} docs -> {final_path.name}")
        self._check_min_ready()
