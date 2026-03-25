"""Unified data source with optional background prefetching.

Wraps any ``Storage`` backend and adds optional background producers
that fill the storage ahead of consumption. Implements the ``DataSource``
protocol so it works directly with ``TransformableDataset``.

Two access modes:

- **Direct mode** (``shuffle_available=False``, default for ShardStorage):
  ``__getitem__(idx)`` passes idx directly to storage. For ShardStorage,
  idx is a global row index. Cache misses use the fallback.

- **Shuffle-from-available mode** (``shuffle_available=True``, default for
  MemoryStorage): ``__len__`` returns the number of items currently in
  storage. ``__getitem__(idx)`` maps to an available key — zero cache
  misses by construction. The DataLoader's sampler shuffles within the
  available pool, which grows as producers fill the buffer.

Usage — text (shards on disk, direct mode):

    storage = ShardStorage("/data/shards", max_shards=100)
    source = PrefetchedSource(storage)
    dataset = TransformableDataset(source, tokenize_transform)

Usage — video (in-memory, shuffle-from-available):

    storage = MemoryStorage(capacity=200)
    source = PrefetchedSource(storage, producers=[decode_episodes],
                              shuffle_available=True)
    dataset = TransformableDataset(source, augment_transform)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterator

from .storage import Storage

logger = logging.getLogger(__name__)

# A producer yields (index, value) pairs
Producer = Callable[[], Iterator[tuple[int, Any]]]


def priority_producer(
    base_producer: Producer,
    priority_keys: list[int],
    decode_fn: Callable[[int], Any],
) -> Producer:
    """Wrap a producer to decode priority_keys first, then continue normally.

    Used on resume: ``priority_keys`` are the episodes that were in the
    buffer at checkpoint time. The producer decodes them first to warm
    the buffer, then hands off to ``base_producer`` for the normal cycle.

    Args:
        base_producer: The normal producer (yields (key, value) pairs).
        priority_keys: Keys to decode first (from state_dict).
        decode_fn: Function to decode a single key → value.
    """

    def wrapped() -> Iterator[tuple[int, Any]]:
        # Phase 1: decode priority keys to warm the buffer
        for key in priority_keys:
            try:
                value = decode_fn(key)
                yield key, value
            except Exception:
                continue  # skip unavailable episodes
        # Phase 2: normal production
        yield from base_producer()

    return wrapped


class PrefetchedSource:
    """Data source backed by Storage with optional background fill.

    Implements ``DataSource`` protocol (``__len__``, ``__getitem__``).

    Args:
        storage: The storage backend (ShardStorage, MemoryStorage, etc.).
        producers: List of callables, each returning an iterator of
            (index, value) pairs. One thread per producer.
        shuffle_available: If True, __getitem__ only serves items currently
            in storage (zero cache misses). __len__ returns available count.
            If False, passes idx directly to storage (may cache-miss).
        fallback: Called on cache miss in direct mode (shuffle_available=False).
            Signature: (idx: int) -> value. Ignored in shuffle-available mode.
        min_available: In shuffle-available mode, wait_for_min() blocks
            until this many items are loaded. Default: 1.
    """

    def __init__(
        self,
        storage: Storage,
        producers: list[Producer] | None = None,
        shuffle_available: bool = False,
        fallback: Callable[[int], Any] | None = None,
        min_available: int = 1,
    ):
        self._storage = storage
        self._producers = producers or []
        self._shuffle_available = shuffle_available
        self._fallback = fallback
        self._min_available = min_available

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._started = False

        # Shuffle-available mode: index mapping (DataLoader idx → storage key)
        self._available_keys: list[int] = []
        self._keys_lock = threading.Lock()
        self._min_ready = threading.Event()
        if not shuffle_available or len(self._storage) >= min_available:
            self._min_ready.set()

    @property
    def storage(self) -> Storage:
        return self._storage

    def state_dict(self) -> dict:
        """Save source state (delegates to storage)."""
        state = {}
        if hasattr(self._storage, "state_dict"):
            state["storage"] = self._storage.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore source state (delegates to storage)."""
        storage_state = state.get("storage")
        if storage_state is not None and hasattr(self._storage, "load_state_dict"):
            self._storage.load_state_dict(storage_state)

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

    def wait_for_min(self, timeout: float = 300.0) -> None:
        """Block until min_available items are loaded."""
        if not self._min_ready.wait(timeout=timeout):
            raise TimeoutError(
                f"Didn't load {self._min_available} items in {timeout}s "
                f"(have {len(self._storage)})"
            )

    def stop(self) -> None:
        """Stop all background producers."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        self._started = False

    def _refresh_available_keys(self) -> None:
        """Update the key list from storage (for shuffle-available mode)."""
        if not self._shuffle_available:
            return
        if hasattr(self._storage, "keys"):
            with self._keys_lock:
                self._available_keys = list(self._storage.keys())

    def __len__(self) -> int:
        if self._shuffle_available:
            self._refresh_available_keys()
            return len(self._available_keys)
        return len(self._storage)

    def __getitem__(self, idx: int) -> Any:
        if self._shuffle_available:
            return self._getitem_available(idx)
        return self._getitem_direct(idx)

    def _getitem_available(self, idx: int) -> Any:
        """Shuffle-available mode: map idx to an available storage key."""
        self._refresh_available_keys()
        with self._keys_lock:
            keys = self._available_keys
        if not keys:
            raise IndexError("No data available in buffer")
        key = keys[idx % len(keys)]
        item = self._storage.get(key)
        if item is not None:
            return item
        # Key was evicted between keys() and get() — refresh and retry
        self._refresh_available_keys()
        with self._keys_lock:
            keys = self._available_keys
        if not keys:
            raise IndexError("No data available after refresh")
        key = keys[idx % len(keys)]
        item = self._storage.get(key)
        if item is None:
            raise IndexError(f"Key {key} not available after retry")
        return item

    def _getitem_direct(self, idx: int) -> Any:
        """Direct mode: pass idx to storage, fallback on miss."""
        item = self._storage.get(idx)
        if item is not None:
            return item
        if self._fallback is not None:
            value = self._fallback(idx)
            if hasattr(self._storage, "put"):
                self._storage.put(idx, value)
            return value
        raise IndexError(f"Index {idx} not available in storage")

    def _run_producer(self, producer_fn: Producer, producer_idx: int) -> None:
        """Background thread: produce items and store them."""
        try:
            for key, value in producer_fn():
                if self._stop_event.is_set():
                    break
                # Evict if at capacity
                if self._storage.capacity is not None:
                    while (
                        len(self._storage) >= self._storage.capacity
                        and not self._stop_event.is_set()
                    ):
                        self._storage.evict(1)
                if hasattr(self._storage, "put"):
                    self._storage.put(key, value)

                # Check min_available
                if not self._min_ready.is_set():
                    if len(self._storage) >= self._min_available:
                        self._min_ready.set()
        except Exception as e:
            logger.error(f"Producer {producer_idx} error: {e}", exc_info=True)

    def __del__(self):
        if self._started:
            self.stop()
