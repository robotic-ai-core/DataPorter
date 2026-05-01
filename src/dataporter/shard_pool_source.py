"""Pool-based source over Parquet text shards.

Each DataLoader worker holds N shards in memory and samples uniformly
across them.  When a shard is exhausted, it's replaced from a shuffled
queue.  Memory is bounded: N × ~15MB per worker.

Replaces ``RawTextSource`` for pre-training — no LRU cache, no random
access into 4000+ shards, no OOM risk.

    TextPrefetcher → Parquet shards → ShardStorage (metadata only)
                                            │
                                      ShardPoolSource (N=3 per worker)
                                            │
                                      TransformableDataset
                                            │
                                      ResumableDataLoader
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path

import pyarrow.parquet as pq

from .storage import ShardStorage

logger = logging.getLogger(__name__)

# Default pool size — see test_mixing_quality.py for the benchmark
# showing N=3 is the inflection point for run length and autocorrelation.
_DEFAULT_POOL_SIZE = 3


class ShardPoolSource:
    """Pool-based random-access source over Parquet text shards.

    Each DataLoader worker holds ``pool_size`` shards in memory and
    samples uniformly across them.  ``__getitem__`` ignores the index
    and returns the next pool-sampled doc — the sampler index serves
    only as a progress counter for resumption.

    Workers are assigned disjoint shard partitions so every row is
    seen exactly once per epoch (modulo ``drop_last`` on the last batch).

    Args:
        data_dir: Directory containing shard_*.parquet files.
        text_column: Column name to read.
        pool_size: Number of shards each worker holds in memory.
        seed: Random seed for deterministic shard shuffling.
        refresh_interval: Seconds between directory rescans (for new shards).
        max_cache_gb: Auto-evict oldest shards when total disk size exceeded.
    """

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        pool_size: int = _DEFAULT_POOL_SIZE,
        seed: int = 42,
        refresh_interval: float = 30.0,
        max_cache_gb: float | None = None,
    ):
        self._storage = ShardStorage(
            data_dir=data_dir,
            text_column=text_column,
            refresh_interval=refresh_interval,
            max_cache_gb=max_cache_gb,
        )
        self._text_column = text_column
        self._pool_size = pool_size
        self._seed = seed
        self._epoch = 0

        # Per-worker state — initialized lazily after fork
        self._initialized = False
        self._empty_worker = False
        self._pool: list[list[str]] = []        # loaded shard texts
        self._cursors: list[int] = []            # read position per pool slot
        self._row_orders: list[list[int]] = []   # shuffled row indices per slot
        self._shard_queue: deque[int] = deque()  # remaining shard indices
        self._rng: random.Random = random.Random(seed)

    # ------------------------------------------------------------------
    # DataSource protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return next doc from pool. ``idx`` is ignored (progress only).

        Raises RuntimeError if not enough shards for all workers — this
        means ``wait_for_min`` wasn't called with a high enough threshold.
        """
        if not self._initialized:
            self._init_worker()
        try:
            return {"text": self._next_doc()}
        except IndexError:
            # All assigned shards consumed — return empty for remaining indices
            # (only happens near the end of an epoch due to uneven partitions)
            return {"text": ""}

    # ------------------------------------------------------------------
    # Freeze / unfreeze / state_dict — delegates to ShardStorage
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        self._storage.freeze()

    def unfreeze(self) -> None:
        self._storage.unfreeze()
        self._initialized = False  # re-partition on next access

    def state_dict(self) -> dict:
        return {
            "storage": self._storage.state_dict(),
            "pool_size": self._pool_size,
            "seed": self._seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self._storage.load_state_dict(state["storage"])
        self._pool_size = state.get("pool_size", self._pool_size)
        self._seed = state.get("seed", self._seed)
        self._epoch = state.get("epoch", self._epoch)
        self._initialized = False

    @property
    def shard_count(self) -> int:
        return self._storage.shard_count

    # ------------------------------------------------------------------
    # Worker initialization (after fork)
    # ------------------------------------------------------------------

    @property
    def min_shards_required(self) -> int:
        """Minimum shards needed before workers can initialize.

        Equal to ``num_workers × pool_size`` so every worker gets a full
        pool.  Pass this to ``TextPrefetcher.wait_for_min()`` at startup.
        """
        # num_workers isn't known until fork; use pool_size as the
        # per-worker floor so callers can compute the total.
        return self._pool_size

    def _init_worker(self) -> None:
        """Partition shards across workers and fill the initial pool.

        Called lazily on first ``__getitem__`` — after the DataLoader
        fork, so ``get_worker_info()`` returns the correct worker id.

        Fails fast if there aren't enough shards for all workers.
        The caller (TextDataModule) must ensure enough shards exist
        before creating the DataLoader via ``wait_for_min(num_workers *
        pool_size)``.

        Auto-increments the epoch counter so shard order changes on each
        DataLoader creation without requiring an explicit ``set_epoch()``
        call from the training loop.
        """
        import torch.utils.data

        self._epoch += 1

        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1

        # Refresh to pick up latest shards from prefetcher
        self._storage._last_refresh = 0.0
        self._storage.refresh()

        n_shards = self._storage.shard_count
        min_needed = num_workers  # at least 1 shard per worker

        if n_shards < min_needed:
            raise RuntimeError(
                f"ShardPoolSource: only {n_shards} shards available but "
                f"{num_workers} workers need at least {min_needed}. "
                f"Increase wait_for_min to num_workers × pool_size "
                f"({num_workers} × {self._pool_size} = "
                f"{num_workers * self._pool_size}) before creating "
                f"the DataLoader."
            )

        # Deterministic shard order from (seed, epoch)
        shard_indices = list(range(n_shards))
        rng = random.Random(self._seed + self._epoch)
        rng.shuffle(shard_indices)

        # Interleaved partition: worker k gets indices k, k+W, k+2W, ...
        my_shards = [s for i, s in enumerate(shard_indices)
                     if i % num_workers == worker_id]

        self._empty_worker = False
        self._shard_queue = deque(my_shards)
        self._rng = random.Random(self._seed + self._epoch * 1000 + worker_id)
        self._pool = []
        self._cursors = []
        self._row_orders = []
        self._fill_pool()
        self._initialized = True

        logger.debug(
            f"Worker {worker_id}/{num_workers}: "
            f"{len(my_shards)} shards assigned, pool_size={self._pool_size}"
        )

    def _fill_pool(self) -> None:
        """Load shards from queue into pool up to pool_size."""
        while len(self._pool) < self._pool_size and self._shard_queue:
            shard_idx = self._shard_queue.popleft()
            texts = self._load_shard(shard_idx)
            if texts is None:
                continue  # shard missing from disk — skip
            row_order = list(range(len(texts)))
            self._rng.shuffle(row_order)
            self._pool.append(texts)
            self._cursors.append(0)
            self._row_orders.append(row_order)

    def _load_shard(self, shard_idx: int) -> list[str] | None:
        """Read an entire shard's text column into memory.

        Returns None if the shard file is missing (e.g., evicted between
        init and load).
        """
        path, _ = self._storage._shards[shard_idx]
        try:
            pf = pq.ParquetFile(str(path))
            return pf.read().column(self._text_column).to_pylist()
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Shard {path.name} missing, skipping: {e}")
            return None

    def _next_doc(self) -> str:
        """Sample one doc from the pool, replace exhausted shards."""
        if not self._pool:
            raise IndexError("All shards exhausted in this worker's partition")

        # Pick a random pool slot
        slot = self._rng.randrange(len(self._pool))
        texts = self._pool[slot]
        cursor = self._cursors[slot]
        row_order = self._row_orders[slot]

        text = texts[row_order[cursor]]
        self._cursors[slot] = cursor + 1

        # Shard exhausted → replace from queue
        if self._cursors[slot] >= len(texts):
            self._pool.pop(slot)
            self._cursors.pop(slot)
            self._row_orders.pop(slot)
            self._fill_pool()

        return text
