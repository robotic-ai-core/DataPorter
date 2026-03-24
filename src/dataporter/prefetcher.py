"""Base prefetcher and shared utilities for streaming data to local storage.

Provides:
  - ``BasePrefetcher``: Shared lifecycle (start/stop/wait_for_min), threading,
    error propagation, shard counting, and eviction.
  - ``CompanionPool``: Thread pool for co-downloading companion files.
  - ``CompanionRef``: Dataclass for companion file references.
  - Eviction and manifest utilities.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Companion file support
# ---------------------------------------------------------------------------


@dataclass
class CompanionRef:
    """Reference to a companion file that must be co-located with a shard.

    Args:
        remote: Source path/URL to download from.
        local: Local path relative to companion_dir.
    """

    remote: str
    local: str


CompanionResolver = Callable[[dict[str, Any]], list[CompanionRef]]


class CompanionPool:
    """Thread pool that downloads companion files alongside shards.

    Tracks which companions belong to which shard for atomic eviction.
    A shard is "ready" only when all its companions have been downloaded.

    Args:
        companion_dir: Local directory for downloaded companion files.
        max_workers: Number of download threads.
        download_fn: Callable (remote, local_path) -> None.
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
        self._pending: dict[str, list[tuple[CompanionRef, Future]]] = {}

    def submit(self, shard_name: str, refs: list[CompanionRef]) -> None:
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
        with self._lock:
            entries = self._pending.get(shard_name)
        if entries is None:
            return True
        return all(fut.done() and not fut.exception() for _, fut in entries)

    def wait_ready(self, shard_name: str, timeout: float | None = None) -> bool:
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
        with self._lock:
            entries = self._pending.get(shard_name, [])
        return [self._companion_dir / ref.local for ref, _ in entries]

    def evict(self, shard_name: str) -> list[Path]:
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
    import shutil

    shutil.copy2(remote, local_path)


# ---------------------------------------------------------------------------
# Manifest (for atomic eviction across restarts)
# ---------------------------------------------------------------------------


def _write_manifest(shard_path: Path, companion_locals: list[str]) -> None:
    if not companion_locals:
        return
    manifest_path = shard_path.with_suffix(".companions.json")
    manifest_path.write_text(json.dumps(companion_locals))


def _read_manifest(shard_path: Path) -> list[str]:
    manifest_path = shard_path.with_suffix(".companions.json")
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def evict_shard(
    shard_dir: Path,
    strategy: str,
    rng: random.Random,
    companion_pool: CompanionPool | None = None,
    companion_dir: Path | None = None,
    glob_pattern: str = "shard_*.parquet",
) -> Path | None:
    """Remove one shard + its companions from disk.

    Returns the evicted shard path, or None if nothing to evict.
    """
    shards = sorted(shard_dir.glob(glob_pattern))
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

    if companion_pool is not None:
        companion_pool.evict(shard_name)
    if companion_dir is not None:
        for rel_path in _read_manifest(victim):
            comp_path = companion_dir / rel_path
            if comp_path.exists():
                comp_path.unlink()

    manifest = victim.with_suffix(".companions.json")
    if manifest.exists():
        manifest.unlink()

    victim.unlink()
    logger.debug(f"Evicted: {shard_name}")
    return victim


# ---------------------------------------------------------------------------
# BasePrefetcher
# ---------------------------------------------------------------------------


class BasePrefetcher:
    """Base class for background prefetchers.

    Owns lifecycle (start/stop/wait_for_min), threading, error propagation,
    shard counting, and eviction. Subclasses implement ``_run_inner()``.

    Args:
        output_dir: Local directory for shard files.
        min_shards: Block training until this many shards are ready.
        max_shards: Evict shards when exceeded. None = no limit.
        eviction: Eviction strategy ("stochastic_oldest", "fifo", "random").
        seed: Random seed.
        shard_glob: Glob pattern to count shards (default "shard_*.parquet").
    """

    def __init__(
        self,
        output_dir: str | Path,
        min_shards: int = 5,
        max_shards: int | None = 100,
        eviction: str = "stochastic_oldest",
        seed: int = 42,
        shard_glob: str = "shard_*.parquet",
    ):
        if min_shards < 1:
            raise ValueError("min_shards must be >= 1")
        if max_shards is not None and max_shards < min_shards:
            raise ValueError("max_shards must be >= min_shards")
        if eviction not in ("stochastic_oldest", "fifo", "random"):
            raise ValueError(f"Unknown eviction strategy: {eviction}")

        self._output_dir = Path(output_dir)
        self._min_shards = min_shards
        self._max_shards = max_shards
        self._eviction = eviction
        self._seed = seed
        self._shard_glob = shard_glob

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._min_ready = threading.Event()
        self._lock = threading.Lock()
        self._shard_counter = 0
        self._error: BaseException | None = None

    @property
    def shard_count(self) -> int:
        return len(list(self._output_dir.rglob(self._shard_glob)))

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def _next_shard_path(self) -> Path:
        with self._lock:
            idx = self._shard_counter
            self._shard_counter += 1
        return self._output_dir / f"shard_{idx:06d}.parquet"

    def start(self) -> None:
        """Start background download thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"{type(self).__name__} already running")

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Resume from existing shards
        self._shard_counter = self.shard_count
        if self._shard_counter >= self._min_shards:
            self._min_ready.set()

        self._stop_event.clear()
        self._error = None
        self._on_start()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=type(self).__name__
        )
        self._thread.start()

    def _on_start(self) -> None:
        """Hook for subclasses to initialize resources before the thread starts."""

    def wait_for_min(self, timeout: float = 300.0) -> None:
        """Block until min_shards are ready."""
        if not self._min_ready.wait(timeout=timeout):
            if self._error:
                raise RuntimeError(f"Prefetcher failed: {self._error}") from self._error
            raise TimeoutError(
                f"Didn't produce {self._min_shards} shards in {timeout}s"
            )
        if self._error:
            raise RuntimeError(f"Prefetcher failed: {self._error}") from self._error

    def stop(self) -> None:
        """Stop background production."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        self._on_stop()

    def _on_stop(self) -> None:
        """Hook for subclasses to clean up resources after the thread stops."""

    def _check_min_ready(self) -> None:
        if not self._min_ready.is_set() and self.shard_count >= self._min_shards:
            self._min_ready.set()
            logger.info(f"Min shards reached ({self._min_shards}), training can start")

    def _maybe_evict(self, rng: random.Random) -> None:
        if self._max_shards is None:
            return
        while self.shard_count > self._max_shards:
            evict_shard(
                self._output_dir,
                self._eviction,
                rng,
                glob_pattern=self._shard_glob,
            )

    def _run(self) -> None:
        try:
            self._run_inner()
        except BaseException as e:
            self._error = e
            logger.error(f"{type(self).__name__} error: {e}", exc_info=True)
        finally:
            self._min_ready.set()

    def _run_inner(self) -> None:
        """Subclasses implement the actual download logic here."""
        raise NotImplementedError
