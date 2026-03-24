"""Text prefetcher — streams HF docs to Parquet text shards.

Writes raw text strings to Parquet (no tokenization). DataLoader workers
handle tokenization in parallel via TransformableDataset.

Architecture (within the prefetcher process):

    Stream thread 0 ──→ ┐                          ┌──→ .tmp → .parquet
    Stream thread 1 ──→ ├──→ DualThresholdBuffer ──┤
    Stream thread N ──→ ┘                          └──→ .tmp → .parquet

- Producers push docs into a dual-threshold buffer. They pause when the
  buffer hits high_water and resume when it drains to low_water. The
  hysteresis gap prevents rapid start/stop oscillation.
- Writer drains the buffer into Parquet shards. It always has docs
  to write (buffer never fully drains while producers are active).
- Streams restart from random offsets on exhaustion.
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .prefetcher import BasePrefetcher, atomic_write

logger = logging.getLogger(__name__)

_SENTINEL = object()  # Signals a producer thread has exited


class DualThresholdBuffer:
    """Thread-safe buffer with high/low water mark flow control.

    Producers call ``put()`` which blocks when the buffer exceeds
    ``high_water``. They unblock when the buffer drains to ``low_water``.
    The consumer calls ``get_batch()`` to drain up to N items at once.

    Args:
        high_water: Producers pause when buffer size >= this.
        low_water: Producers resume when buffer size <= this.
    """

    def __init__(self, high_water: int = 8000, low_water: int = 3000):
        if low_water >= high_water:
            raise ValueError("low_water must be < high_water")
        self._high_water = high_water
        self._low_water = low_water
        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._paused = False

    def put(self, item: Any, stop_event: threading.Event | None = None) -> bool:
        """Add an item. Blocks if buffer is above high_water.

        Returns False if stop_event was set while waiting.
        """
        with self._not_full:
            while len(self._buf) >= self._high_water:
                if stop_event and stop_event.is_set():
                    return False
                self._paused = True
                self._not_full.wait(timeout=0.5)
            self._buf.append(item)
            self._not_empty.notify()
        return True

    def get_batch(
        self, max_items: int, stop_event: threading.Event | None = None
    ) -> list:
        """Get up to max_items from the buffer. Blocks if buffer is empty.

        Returns empty list if stop_event was set while waiting.
        """
        with self._not_empty:
            while len(self._buf) == 0:
                if stop_event and stop_event.is_set():
                    return []
                self._not_empty.wait(timeout=0.5)

            items = []
            while self._buf and len(items) < max_items:
                items.append(self._buf.popleft())

            # Resume producers if below low_water
            if len(self._buf) <= self._low_water and self._paused:
                self._paused = False
                self._not_full.notify_all()

        return items

    def put_sentinel(self) -> None:
        """Add a sentinel (never blocked by high_water)."""
        with self._not_full:
            self._buf.append(_SENTINEL)
            self._not_empty.notify()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused


class TextPrefetcher(BasePrefetcher):
    """Background process that streams HF docs to Parquet text shards.

    Producer threads stream from HF, pushing docs into a dual-threshold
    buffer. Producers pause at high_water, resume at low_water — the
    writer always has docs and producers idle smoothly when ahead.

    When a stream exhausts its data, it restarts from a new random
    offset. Streaming only stops when stop() is called.

    Args:
        output_dir: Local directory for Parquet shards.
        dataset: HuggingFace dataset ID.
        data_dir: Subdirectory within the dataset.
        text_field: Column name for text.
        min_shards: Block until this many shards are available.
        max_shards: Evict oldest shards when exceeded (None = no limit).
        max_rows_per_shard: Rows per shard file.
        stream_shuffle_buffer: HF stream shuffle buffer size.
        offsets: List of initial stream offsets for data diversity.
        seed: Random seed for eviction and shuffling.
        high_water: Producers pause when buffer reaches this size.
            Default: 3 × max_rows_per_shard (writer always has full shards).
        low_water: Producers resume when buffer drains to this size.
            Default: 1 × max_rows_per_shard (at least one full shard buffered).
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
        high_water: int | None = None,
        low_water: int | None = None,
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
        self._high_water = high_water if high_water is not None else max_rows_per_shard * 3
        self._low_water = low_water if low_water is not None else max_rows_per_shard
        self._max_restarts = max_restarts

    def _get_init_kwargs(self) -> dict[str, Any]:
        return dict(
            output_dir=str(self._output_dir),
            dataset=self._dataset,
            data_dir=self._data_dir,
            text_field=self._text_field,
            min_shards=self._min_shards,
            max_shards=self._max_shards,
            max_rows_per_shard=self._max_rows_per_shard,
            stream_shuffle_buffer=self._shuffle_buffer,
            offsets=self._offsets,
            seed=self._seed,
            high_water=self._high_water,
            low_water=self._low_water,
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

        buf = DualThresholdBuffer(
            high_water=self._high_water,
            low_water=self._low_water,
        )
        n_producers = len(self._offsets)

        producers = []
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        for i, offset in enumerate(self._offsets):
            t = threading.Thread(
                target=self._producer,
                args=(offset, i, buf, errors, errors_lock),
                daemon=True,
                name=f"stream-{offset}",
            )
            producers.append(t)
            t.start()

        try:
            self._writer(buf, n_producers)
        finally:
            self._stop_event.set()
            for t in producers:
                t.join(timeout=5)

        if errors:
            raise errors[0]

    # ------------------------------------------------------------------
    # Producer: stream docs from HF → buffer
    # ------------------------------------------------------------------

    def _producer(
        self,
        initial_offset: int,
        producer_idx: int,
        buf: DualThresholdBuffer,
        errors: list,
        errors_lock: threading.Lock,
    ) -> None:
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

                    if not buf.put(text, stop_event=self._stop_event):
                        return  # stop_event was set

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
            buf.put_sentinel()

    # ------------------------------------------------------------------
    # Writer: buffer → Parquet shards
    # ------------------------------------------------------------------

    def _writer(self, buf: DualThresholdBuffer, n_producers: int) -> None:
        rng = random.Random(self._seed)
        schema = pa.schema([("text", pa.string())])
        shard_buf: list[str] = []
        producers_done = 0

        while producers_done < n_producers:
            if self._stop_event.is_set():
                break

            # Drain up to one shard's worth of docs at once
            items = buf.get_batch(
                self._max_rows_per_shard - len(shard_buf),
                stop_event=self._stop_event,
            )

            for item in items:
                if item is _SENTINEL:
                    producers_done += 1
                    continue
                shard_buf.append(item)

            if len(shard_buf) >= self._max_rows_per_shard:
                self._write_shard(shard_buf, schema)
                shard_buf.clear()
                self._maybe_evict(rng)

        # Flush remaining
        if shard_buf and not self._stop_event.is_set():
            self._write_shard(shard_buf, schema)
            self._maybe_evict(rng)

    def _write_shard(self, texts: list[str], schema: pa.Schema) -> None:
        tmp_path, final_path = self._next_shard_tmp_path()
        table = pa.table({"text": texts}, schema=schema)
        pq.write_table(table, str(tmp_path), compression="zstd")
        atomic_write(tmp_path, final_path)

        logger.info(f"Wrote shard: {len(texts)} docs -> {final_path.name}")
        self._check_min_ready()
