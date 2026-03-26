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
import multiprocessing as mp
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

    **GIL warning:** When training with GPU, use ``use_process=True`` for
    producers that do tensor operations (video decode, frame writes).
    Thread-mode producers hold the GIL during tensor writes, blocking
    CUDA kernel scheduling and causing up to 60% GPU throughput loss.

    Rule of thumb:
      - ``SharedMemoryStorage`` → always use ``use_process=True``
      - ``MemoryStorage`` (CPU-only, no GPU training) → thread mode is fine
      - ``ShardStorage`` → no producers needed (prefetcher is separate process)

    Args:
        storage: The storage backend (ShardStorage, MemoryStorage, etc.).
        producers: List of callables, each returning an iterator of
            (index, value) pairs. One worker per producer.
        shuffle_available: If True, __getitem__ only serves items currently
            in storage (zero cache misses). __len__ returns available count.
            If False, passes idx directly to storage (may cache-miss).
        fallback: Called on cache miss in direct mode (shuffle_available=False).
            Signature: (idx: int) -> value. Ignored in shuffle-available mode.
        min_available: In shuffle-available mode, wait_for_min() blocks
            until this many items are loaded. Default: 1.
        keys_refresh_interval: Seconds between key list refreshes in
            shuffle-available mode. Lower = faster visibility of new items,
            higher = less overhead per __getitem__. Default: 1.0.
        use_process: If True, run producers in forked child processes
            instead of threads. Required for GPU training to avoid GIL
            contention. SharedMemoryStorage tensors are accessible
            cross-process via share_memory_(). Default: True when
            producers are provided (safe for GPU training), False
            when no producers (read-only source).
        use_threads: Explicitly request thread mode. Only use for
            CPU-only workloads where fork isn't safe. Overrides the
            default process mode.
    """

    def __init__(
        self,
        storage: Storage,
        producers: list[Producer] | None = None,
        shuffle_available: bool = False,
        fallback: Callable[[int], Any] | None = None,
        min_available: int = 1,
        keys_refresh_interval: float = 1.0,
        use_process: bool | None = None,
        use_threads: bool = False,
    ):
        self._storage = storage
        self._producers = producers or []
        self._shuffle_available = shuffle_available
        self._fallback = fallback
        self._min_available = min_available

        # Default: process mode when producers exist, thread mode when not
        if use_process is not None:
            self._use_process = use_process
        elif use_threads:
            self._use_process = False
        else:
            self._use_process = len(self._producers) > 0

        self._workers: list[threading.Thread | mp.Process] = []
        # Use multiprocessing primitives when running producers in processes
        if self._use_process:
            ctx = mp.get_context("fork")
            self._stop_event = ctx.Event()
            self._min_ready = ctx.Event()
        else:
            self._stop_event = threading.Event()
            self._min_ready = threading.Event()
        self._started = False

        # Shuffle-available mode: index mapping (DataLoader idx → storage key)
        self._available_keys: list[int] = []
        self._keys_lock = threading.Lock()
        self._keys_refresh_interval = keys_refresh_interval
        self._keys_last_refresh = 0.0
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
        """Start background producer threads or processes (if any).

        When ``use_process=True``, producers run in forked child processes
        to avoid GIL contention with the main training loop. The
        SharedMemoryStorage tensors (created with ``share_memory_()``)
        are accessible from the child process without extra IPC.
        """
        if self._started or not self._producers:
            return
        self._stop_event.clear()
        self._started = True

        for i, producer_fn in enumerate(self._producers):
            if self._use_process:
                ctx = mp.get_context("fork")
                w = ctx.Process(
                    target=self._run_producer,
                    args=(producer_fn, i),
                    daemon=True,
                    name=f"prefetch-producer-{i}",
                )
            else:
                w = threading.Thread(
                    target=self._run_producer,
                    args=(producer_fn, i),
                    daemon=True,
                    name=f"prefetch-producer-{i}",
                )
            self._workers.append(w)
            w.start()

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
        for w in self._workers:
            w.join(timeout=10)
            # Terminate process if it didn't exit cleanly
            if isinstance(w, mp.Process) and w.is_alive():
                w.terminate()
        self._workers.clear()
        self._started = False

    def _refresh_available_keys(self, force: bool = False) -> None:
        """Update the key list from storage (interval-based to avoid overhead)."""
        if not self._shuffle_available:
            return
        now = time.monotonic()
        if not force and now - self._keys_last_refresh < self._keys_refresh_interval:
            return  # use cached key list
        if hasattr(self._storage, "keys"):
            with self._keys_lock:
                self._available_keys = list(self._storage.keys())
            self._keys_last_refresh = now

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
        # Key was evicted between keys() and get() — force refresh and retry
        self._refresh_available_keys(force=True)
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
