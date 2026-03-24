"""Async Parquet prefetcher for HuggingFace streaming datasets.

Downloads HF datasets to a local directory of Parquet shards in a background
thread. Existing dataset classes (ParquetTokenDataset, FastLeRobotDataset)
read from the growing directory unchanged — the prefetcher is a transparent
caching layer below them.

Key features:
  - Multi-offset streams for data distribution diversity
  - min_shards / max_shards for shuffle quality and disk caps
  - Stochastic eviction of oldest shards when max_shards exceeded
  - Composable transform pipeline (dependency injection)
  - Companion file co-download (video, images) via CompanionPool
"""

from __future__ import annotations

import json
import logging
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Companion file support
# ---------------------------------------------------------------------------


@dataclass
class CompanionRef:
    """Reference to a companion file that must be co-located with a shard.

    Args:
        remote: Source path/URL to download from (HF hub path, URL, or local).
        local: Local path relative to companion_dir where the file will be stored.
    """

    remote: str
    local: str


CompanionResolver = Callable[[dict[str, Any]], list[CompanionRef]]
"""Callable that extracts companion file references from an HF doc.

Takes a raw document dict, returns a list of CompanionRef objects
describing files to download alongside the Parquet shard.
Returns empty list if no companions needed for this doc.
"""


class CompanionPool:
    """Thread pool that downloads companion files alongside Parquet shards.

    Tracks which companions belong to which shard for atomic eviction.
    A shard is "ready" only when all its companions have been downloaded.

    Args:
        companion_dir: Local directory for downloaded companion files.
        max_workers: Number of download threads.
        download_fn: Callable that downloads a single file.
            Signature: (remote: str, local_path: Path) -> None.
            If None, uses a default that copies from local filesystem
            (useful for testing).
    """

    def __init__(
        self,
        companion_dir: Path,
        max_workers: int = 4,
        download_fn: Callable[[str, Path], None] | None = None,
    ):
        self._companion_dir = Path(companion_dir)
        self._companion_dir.mkdir(parents=True, exist_ok=True)
        self._download_fn = download_fn or _default_download
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="companion-dl"
        )
        self._lock = threading.Lock()
        # shard_name -> list of (CompanionRef, Future)
        self._pending: dict[str, list[tuple[CompanionRef, Future]]] = {}

    def submit(self, shard_name: str, refs: list[CompanionRef]) -> None:
        """Submit companion downloads for a shard."""
        if not refs:
            return
        with self._lock:
            entries = []
            for ref in refs:
                local_path = self._companion_dir / ref.local
                local_path.parent.mkdir(parents=True, exist_ok=True)
                fut = self._pool.submit(self._download_fn, ref.remote, local_path)
                entries.append((ref, fut))
            self._pending[shard_name] = entries

    def is_ready(self, shard_name: str) -> bool:
        """True if all companions for this shard have been downloaded."""
        with self._lock:
            entries = self._pending.get(shard_name)
        if entries is None:
            return True  # No companions registered = always ready
        return all(fut.done() and not fut.exception() for _, fut in entries)

    def wait_ready(self, shard_name: str, timeout: float | None = None) -> bool:
        """Block until all companions for this shard are downloaded."""
        with self._lock:
            entries = self._pending.get(shard_name)
        if entries is None:
            return True
        for _, fut in entries:
            try:
                fut.result(timeout=timeout)
            except Exception:
                return False
        return True

    def get_companion_paths(self, shard_name: str) -> list[Path]:
        """Get local paths of all companions for a shard."""
        with self._lock:
            entries = self._pending.get(shard_name, [])
        return [self._companion_dir / ref.local for ref, _ in entries]

    def evict(self, shard_name: str) -> list[Path]:
        """Delete all companion files for a shard. Returns deleted paths."""
        with self._lock:
            entries = self._pending.pop(shard_name, [])
        deleted = []
        for ref, fut in entries:
            if not fut.done():
                fut.cancel()
            path = self._companion_dir / ref.local
            if path.exists():
                path.unlink()
                deleted.append(path)
        return deleted

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)


def _default_download(remote: str, local_path: Path) -> None:
    """Default download: copy from local filesystem (for testing)."""
    import shutil

    shutil.copy2(remote, local_path)


# ---------------------------------------------------------------------------
# Shard manifest (for atomic eviction across restarts)
# ---------------------------------------------------------------------------


def _write_manifest(shard_path: Path, companion_locals: list[str]) -> None:
    """Write a JSON manifest listing companion files for a shard."""
    if not companion_locals:
        return
    manifest_path = shard_path.with_suffix(".companions.json")
    manifest_path.write_text(json.dumps(companion_locals))


def _read_manifest(shard_path: Path) -> list[str]:
    """Read companion file list from a shard's manifest."""
    manifest_path = shard_path.with_suffix(".companions.json")
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class _ShardWriter:
    """Writes rows to a single Parquet shard file.

    Buffers rows and flushes as row groups. The shard is finalized on close().
    """

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        row_group_size: int = 256,
        compression: str = "zstd",
    ):
        self._path = path
        self._schema = schema
        self._row_group_size = row_group_size
        self._writer = pq.ParquetWriter(str(path), schema, compression=compression)
        self._buffer: list[list[int]] = []
        self._rows_written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def write_row(self, input_ids: list[int]) -> None:
        self._buffer.append(input_ids)
        if len(self._buffer) >= self._row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.table(
            {"input_ids": self._buffer},
            schema=self._schema,
        )
        self._writer.write_table(table)
        self._rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._writer.close()


# ---------------------------------------------------------------------------
# Eviction (shard + companions)
# ---------------------------------------------------------------------------


def _evict_shard(
    shard_dir: Path,
    strategy: str,
    rng: random.Random,
    companion_pool: CompanionPool | None = None,
    companion_dir: Path | None = None,
) -> Path | None:
    """Remove one shard + its companions from disk.

    Returns the path of the evicted shard, or None if no shards to evict.
    """
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if not shards:
        return None

    if strategy == "fifo":
        victim = shards[0]
    elif strategy == "random":
        victim = rng.choice(shards)
    elif strategy == "stochastic_oldest":
        oldest_half = shards[: max(1, len(shards) // 2)]
        victim = rng.choice(oldest_half)
    else:
        raise ValueError(f"Unknown eviction strategy: {strategy}")

    shard_name = victim.name

    # Evict companions (from pool if active, or from manifest on disk)
    if companion_pool is not None:
        companion_pool.evict(shard_name)
    if companion_dir is not None:
        for rel_path in _read_manifest(victim):
            comp_path = companion_dir / rel_path
            if comp_path.exists():
                comp_path.unlink()

    # Delete manifest
    manifest = victim.with_suffix(".companions.json")
    if manifest.exists():
        manifest.unlink()

    # Delete shard
    victim.unlink()
    logger.debug(f"Evicted shard + companions: {shard_name}")
    return victim


# ---------------------------------------------------------------------------
# Ready shards (shard exists + all companions downloaded)
# ---------------------------------------------------------------------------


def _count_ready_shards(
    shard_dir: Path,
    companion_pool: CompanionPool | None = None,
) -> int:
    """Count shards that are fully ready (shard + all companions downloaded)."""
    shards = list(shard_dir.glob("shard_*.parquet"))
    if companion_pool is None:
        return len(shards)
    return sum(1 for s in shards if companion_pool.is_ready(s.name))


# ---------------------------------------------------------------------------
# ParquetPrefetcher
# ---------------------------------------------------------------------------


class ParquetPrefetcher:
    """Async prefetcher that downloads HF datasets to local Parquet shards.

    Runs a background thread that streams from one or more HuggingFace
    dataset offsets, applies an optional transform, and writes Parquet
    shards to a local directory. Optionally co-downloads companion files
    (video, images) via a CompanionPool thread pool.

    Args:
        sources: List of HF dataset configs. Each dict should have at minimum
            a "dataset" key. Optional keys: "data_dir", "split", "offset",
            "text_field".
        output_dir: Local directory for Parquet shard files.
        transform: Optional callable applied to each raw document dict.
            Should return a list of rows (each a list[int] of token IDs)
            or None to skip the document. If None, raw text is written as-is.
        companion_resolver: Optional callable that extracts companion file
            references from each HF doc. Returns list[CompanionRef].
            If None, no companion downloads.
        companion_dir: Directory for downloaded companion files.
            Required if companion_resolver is set.
        companion_workers: Number of threads for companion downloads.
        companion_download_fn: Custom download function for companion files.
            Signature: (remote: str, local_path: Path) -> None.
        min_shards: Block training until this many shards are ready.
        max_shards: Evict oldest shards when this count is exceeded.
        stream_shuffle_buffer: HF-level stream shuffle buffer size.
        eviction: Eviction strategy ("stochastic_oldest", "fifo", "random").
        max_rows_per_shard: Maximum rows per shard file.
        row_group_size: Rows per Parquet row group.
        seed: Random seed for eviction and stream shuffling.
    """

    def __init__(
        self,
        sources: list[dict[str, Any]],
        output_dir: str | Path,
        transform: Callable | None = None,
        companion_resolver: CompanionResolver | None = None,
        companion_dir: str | Path | None = None,
        companion_workers: int = 4,
        companion_download_fn: Callable[[str, Path], None] | None = None,
        min_shards: int = 10,
        max_shards: int = 50,
        stream_shuffle_buffer: int = 10_000,
        eviction: str = "stochastic_oldest",
        max_rows_per_shard: int = 200_000,
        row_group_size: int = 256,
        seed: int = 42,
        _dataset_factory: Callable | None = None,
    ):
        if not sources:
            raise ValueError("At least one source is required")
        if min_shards < 1:
            raise ValueError("min_shards must be >= 1")
        if max_shards < min_shards:
            raise ValueError("max_shards must be >= min_shards")
        if eviction not in ("stochastic_oldest", "fifo", "random"):
            raise ValueError(f"Unknown eviction strategy: {eviction}")
        if companion_resolver is not None and companion_dir is None:
            raise ValueError("companion_dir is required when companion_resolver is set")

        self._sources = sources
        self._output_dir = Path(output_dir)
        self._transform = transform
        self._companion_resolver = companion_resolver
        self._companion_dir = Path(companion_dir) if companion_dir else None
        self._companion_workers = companion_workers
        self._companion_download_fn = companion_download_fn
        self._min_shards = min_shards
        self._max_shards = max_shards
        self._stream_shuffle_buffer = stream_shuffle_buffer
        self._eviction = eviction
        self._max_rows_per_shard = max_rows_per_shard
        self._row_group_size = row_group_size
        self._seed = seed

        self._dataset_factory = _dataset_factory
        self._companion_pool: CompanionPool | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._min_ready = threading.Event()
        self._lock = threading.Lock()
        self._shard_counter = 0
        self._error: Exception | None = None

    def start(self) -> None:
        """Start background async download + write thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Prefetcher already running")

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Start companion pool if needed (resolver for generic, companion_dir for subclasses)
        if self._companion_dir is not None:
            self._companion_pool = CompanionPool(
                companion_dir=self._companion_dir,
                max_workers=self._companion_workers,
                download_fn=self._companion_download_fn,
            )

        # Count existing ready shards (support resuming)
        existing = list(self._output_dir.glob("shard_*.parquet"))
        self._shard_counter = len(existing)
        ready = _count_ready_shards(self._output_dir, self._companion_pool)
        if ready >= self._min_shards:
            self._min_ready.set()

        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="parquet-prefetcher"
        )
        self._thread.start()

    def wait_for_min(self, timeout: float | None = None) -> None:
        """Block until min_shards are ready (shard + companions downloaded).

        Raises RuntimeError if the background thread encountered an error.
        """
        self._min_ready.wait(timeout=timeout)
        if self._error is not None:
            raise RuntimeError(f"Prefetcher failed: {self._error}") from self._error

    def stop(self) -> None:
        """Stop background production and companion downloads."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        if self._companion_pool is not None:
            self._companion_pool.shutdown(wait=False)
            self._companion_pool = None

    @property
    def shard_count(self) -> int:
        """Number of shards currently on disk."""
        return len(list(self._output_dir.glob("shard_*.parquet")))

    @property
    def ready_shard_count(self) -> int:
        """Number of shards that are fully ready (shard + companions)."""
        return _count_ready_shards(self._output_dir, self._companion_pool)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Exception | None:
        return self._error

    def _next_shard_path(self) -> Path:
        with self._lock:
            idx = self._shard_counter
            self._shard_counter += 1
        return self._output_dir / f"shard_{idx:06d}.parquet"

    def _check_min_ready(self) -> None:
        """Set min_ready event if enough shards are ready."""
        if not self._min_ready.is_set():
            ready = _count_ready_shards(self._output_dir, self._companion_pool)
            if ready >= self._min_shards:
                self._min_ready.set()
                logger.info(
                    f"Min shards reached ({self._min_shards}), training can start"
                )

    def _maybe_evict(self, rng: random.Random) -> None:
        """Evict shards if over max_shards."""
        while self.shard_count > self._max_shards:
            _evict_shard(
                self._output_dir,
                self._eviction,
                rng,
                companion_pool=self._companion_pool,
                companion_dir=self._companion_dir,
            )

    def _on_shard_finalized(self, shard_path: Path, companion_refs: list[CompanionRef]) -> None:
        """Called after a shard is written. Submits companions and checks readiness."""
        shard_name = shard_path.name

        if companion_refs and self._companion_pool is not None:
            self._companion_pool.submit(shard_name, companion_refs)
            _write_manifest(shard_path, [ref.local for ref in companion_refs])

        self._check_min_ready()

    def _run(self) -> None:
        """Background thread: stream, transform, write shards."""
        try:
            self._run_inner()
        except Exception as e:
            self._error = e
            logger.error(f"Prefetcher error: {e}", exc_info=True)
        finally:
            self._min_ready.set()

    def _load_dataset(self, src: dict[str, Any]) -> Any:
        """Load a single HF streaming dataset (or use injected factory)."""
        if self._dataset_factory is not None:
            return self._dataset_factory(src)
        from datasets import load_dataset

        return load_dataset(
            src["dataset"],
            data_dir=src.get("data_dir"),
            split=src.get("split", "train"),
            streaming=True,
        )

    def _run_inner(self) -> None:
        rng = random.Random(self._seed)
        schema = pa.schema([("input_ids", pa.list_(pa.uint16()))])

        iterators = []
        for src in self._sources:
            ds = self._load_dataset(src)
            if self._stream_shuffle_buffer > 0:
                ds = ds.shuffle(
                    seed=self._seed,
                    buffer_size=self._stream_shuffle_buffer,
                )
            offset = src.get("offset", 0)
            it = iter(ds.skip(offset) if offset > 0 else ds)
            iterators.append((it, src))

        if not iterators:
            return

        current_writer: _ShardWriter | None = None
        rows_in_shard = 0
        shard_companions: list[CompanionRef] = []

        def _finalize_shard() -> None:
            nonlocal current_writer, rows_in_shard, shard_companions
            if current_writer is not None:
                shard_path = current_writer.path
                current_writer.close()
                current_writer = None
                rows_in_shard = 0
                self._on_shard_finalized(shard_path, shard_companions)
                shard_companions = []
                self._maybe_evict(rng)

        def _ensure_writer() -> _ShardWriter:
            nonlocal current_writer
            if current_writer is None:
                current_writer = _ShardWriter(
                    self._next_shard_path(),
                    schema,
                    row_group_size=self._row_group_size,
                )
            return current_writer

        exhausted = set()
        source_idx = 0

        while not self._stop_event.is_set():
            if len(exhausted) == len(iterators):
                logger.info("All sources exhausted")
                break

            source_idx = source_idx % len(iterators)
            while source_idx in exhausted:
                source_idx = (source_idx + 1) % len(iterators)

            it, src = iterators[source_idx]
            text_field = src.get("text_field", "text")

            batch_size = 64
            for _ in range(batch_size):
                if self._stop_event.is_set():
                    break
                try:
                    doc = next(it)
                except StopIteration:
                    exhausted.add(source_idx)
                    break

                # Extract companion refs before transform (transform may strip fields)
                if self._companion_resolver is not None:
                    refs = self._companion_resolver(doc)
                    if refs:
                        shard_companions.extend(refs)

                if self._transform is not None:
                    rows = self._transform(doc)
                    if rows is None:
                        continue
                else:
                    text = doc.get(text_field, "")
                    if not text.strip():
                        continue
                    rows = [list(text.encode("utf-8"))]

                writer = _ensure_writer()
                for row in rows:
                    writer.write_row(row)
                    rows_in_shard += 1

                    if rows_in_shard >= self._max_rows_per_shard:
                        _finalize_shard()
                        writer = _ensure_writer()

            source_idx += 1

        _finalize_shard()


class MockHFDataset:
    """Mock HF streaming dataset for testing without network access.

    Yields dicts with a "text" field. Supports skip(), shuffle(), and iter().
    """

    def __init__(self, docs: list[dict[str, Any]], seed: int = 42):
        self._docs = docs
        self._offset = 0
        self._seed = seed

    def skip(self, n: int) -> "MockHFDataset":
        clone = MockHFDataset(self._docs, self._seed)
        clone._offset = n
        return clone

    def shuffle(self, seed: int = 42, buffer_size: int = 1000) -> "MockHFDataset":
        rng = random.Random(seed)
        docs = self._docs.copy()
        rng.shuffle(docs)
        clone = MockHFDataset(docs, seed)
        clone._offset = self._offset
        return clone

    def __iter__(self):
        for doc in self._docs[self._offset:]:
            yield doc
