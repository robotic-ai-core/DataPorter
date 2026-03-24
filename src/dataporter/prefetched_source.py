"""Unified data source with optional background prefetching.

Wraps any ``Storage`` backend and adds optional background producers
that fill the storage ahead of consumption. Implements the ``DataSource``
protocol so it works directly with ``TransformableDataset``.

Usage — text (Parquet shards on disk):

    storage = ShardStorage("/data/shards", max_shards=100)
    source = PrefetchedSource(storage)
    dataset = TransformableDataset(source, tokenize_transform)

Usage — video (in-memory frame buffer):

    storage = MemoryStorage(capacity=200)
    source = PrefetchedSource(storage, producers=[decode_episodes])
    dataset = TransformableDataset(source, augment_transform)

Usage — video with remote download + decode:

    # LeRobotPrefetcher downloads episodes to disk (separate process)
    # Then PrefetchedSource decodes frames into memory
    storage = MemoryStorage(capacity=200)
    source = PrefetchedSource(storage, producers=[decode_local_episodes])
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Iterator

from .storage import Storage
from .text_prefetcher import DualThresholdBuffer

logger = logging.getLogger(__name__)

# A producer yields (index, value) pairs
Producer = Callable[[], Iterator[tuple[int, Any]]]

_SENTINEL = object()


class PrefetchedSource:
    """Data source backed by Storage with optional background fill.

    Implements ``DataSource`` protocol (``__len__``, ``__getitem__``).
    If producers are provided, background threads fill the storage
    via a dual-threshold buffer. If no producers, reads directly
    from storage (for pre-populated or externally-filled stores).

    Args:
        storage: The storage backend (ShardStorage, MemoryStorage, etc.).
        producers: List of callables, each returning an iterator of
            (index, value) pairs. One thread per producer.
        high_water: Producers pause when this many items are buffered.
        low_water: Producers resume when buffer drains to this level.
        fallback: Called on cache miss if storage returns None.
            Signature: (idx: int) -> value. If None, misses return None.
    """

    def __init__(
        self,
        storage: Storage,
        producers: list[Producer] | None = None,
        high_water: int = 500,
        low_water: int = 100,
        fallback: Callable[[int], Any] | None = None,
    ):
        self._storage = storage
        self._producers = producers or []
        self._fallback = fallback
        self._high_water = high_water
        self._low_water = low_water

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._started = False

    @property
    def storage(self) -> Storage:
        return self._storage

    def start(self) -> None:
        """Start background producer threads (if any)."""
        if self._started or not self._producers:
            return
        self._stop_event.clear()
        self._started = True

        for i, producer_fn in enumerate(self._producers):
            t = threading.Thread(
                target=self._run_producer,
                args=(producer_fn, i),
                daemon=True,
                name=f"prefetch-producer-{i}",
            )
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        """Stop all background producers."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        self._started = False

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, idx: int) -> Any:
        item = self._storage.get(idx)
        if item is not None:
            return item
        # Cache miss — use fallback if available
        if self._fallback is not None:
            value = self._fallback(idx)
            # Store for next access
            if hasattr(self._storage, "put"):
                self._storage.put(idx, value)
            return value
        raise IndexError(f"Index {idx} not available in storage")

    def _run_producer(self, producer_fn: Producer, idx: int) -> None:
        """Background thread: produce items and store them."""
        try:
            for key, value in producer_fn():
                if self._stop_event.is_set():
                    break
                # Wait if storage is at capacity
                if self._storage.capacity is not None:
                    while (
                        len(self._storage) >= self._storage.capacity
                        and not self._stop_event.is_set()
                    ):
                        # Evict oldest to make room
                        self._storage.evict(1)
                if hasattr(self._storage, "put"):
                    self._storage.put(key, value)
        except Exception as e:
            logger.error(f"Producer {idx} error: {e}", exc_info=True)

    def __del__(self):
        if self._started:
            self.stop()
