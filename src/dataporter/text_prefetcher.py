"""Lightweight text prefetcher — downloads HF docs to Parquet text shards.

Writes raw text strings to Parquet (no tokenization). DataLoader workers
handle tokenization in parallel via TransformableDataset.

The prefetcher thread is I/O-bound (HF streaming + Parquet writing).
CPU-heavy tokenization fans out across DataLoader workers.
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _evict_oldest_shard(shard_dir: Path, rng: random.Random) -> Path | None:
    """Remove a random shard from the oldest 50%."""
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if not shards:
        return None
    oldest_half = shards[: max(1, len(shards) // 2)]
    victim = rng.choice(oldest_half)
    victim.unlink()
    logger.debug(f"Evicted shard: {victim.name}")
    return victim


class TextPrefetcher:
    """Background thread that streams HF docs to Parquet text shards.

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
        seed: Random seed for eviction.
        _dataset_factory: Override dataset loading (for testing).
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
        self._dataset = dataset
        self._data_dir = data_dir
        self._text_field = text_field
        self._output_dir = Path(output_dir)
        self._min_shards = min_shards
        self._max_shards = max_shards
        self._max_rows_per_shard = max_rows_per_shard
        self._shuffle_buffer = stream_shuffle_buffer
        self._offsets = offsets or [0, 2_000_000]
        self._seed = seed
        self._dataset_factory = _dataset_factory

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._shard_counter = 0
        self._lock = threading.Lock()
        self._min_ready = threading.Event()
        self._error: BaseException | None = None

    @property
    def shard_count(self) -> int:
        """Number of shards currently on disk."""
        return len(list(self._output_dir.glob("shard_*.parquet")))

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self):
        """Start background download thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("TextPrefetcher already running")

        # Resume from existing shards
        existing = list(self._output_dir.glob("shard_*.parquet"))
        self._shard_counter = len(existing)
        if self._shard_counter >= self._min_shards:
            self._min_ready.set()

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="text-prefetcher"
        )
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
        """Stop background production."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    def _next_shard_path(self) -> Path:
        with self._lock:
            idx = self._shard_counter
            self._shard_counter += 1
        return self._output_dir / f"shard_{idx:06d}.parquet"

    def _run(self):
        try:
            self._produce()
        except BaseException as e:
            self._error = e
        finally:
            self._min_ready.set()

    def _load_dataset(self, offset: int):
        """Load an HF streaming dataset at the given offset."""
        if self._dataset_factory is not None:
            return self._dataset_factory(offset)

        from datasets import load_dataset

        ds = load_dataset(
            self._dataset,
            data_dir=self._data_dir,
            split="train",
            streaming=True,
        ).shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)

        if offset > 0:
            ds = ds.skip(offset)
        return ds

    def _produce(self):
        rng = random.Random(self._seed)
        schema = pa.schema([("text", pa.string())])

        for offset in self._offsets:
            if self._stop_event.is_set():
                break

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

            # Flush remaining
            if buffer and not self._stop_event.is_set():
                self._write_shard(buffer, schema)
                self._maybe_evict(rng)

    def _write_shard(self, texts: list[str], schema: pa.Schema):
        shard_path = self._next_shard_path()
        table = pa.table({"text": texts}, schema=schema)
        pq.write_table(table, str(shard_path), compression="zstd")

        logger.info(f"Wrote shard: {len(texts)} docs -> {shard_path.name}")

        if not self._min_ready.is_set() and self.shard_count >= self._min_shards:
            self._min_ready.set()
            logger.info(f"Min shards reached ({self._min_shards}), training can start")

    def _maybe_evict(self, rng: random.Random):
        """Evict shards if over max_shards."""
        if self._max_shards is None:
            return
        while self.shard_count > self._max_shards:
            _evict_oldest_shard(self._output_dir, rng)
