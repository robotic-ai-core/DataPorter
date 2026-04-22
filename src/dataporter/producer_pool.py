"""Producer pool for filling ShuffleBuffer.

Spawned child process with asyncio event loop and per-source
ThreadPoolExecutor for parallel video decode. Uses spawn context
(not fork) to avoid deadlocks from inherited HF Arrow mmaps.

Architecture::

    Main process (GPU training)
    |
    +-- ProducerPool process (SPAWNED, fresh)
    |   +-- FastLeRobotDataset per source (own mmap, created in child)
    |   +-- ThreadPoolExecutor per source (sized by weight)
    |   +-- asyncio loop serializes buffer.put() (single writer)
    |
    +-- DataLoader workers (forked, read-only)
        +-- buffer.sample() (shared memory read)
        +-- HF dataset reads (inherited mmap, read-only)
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import multiprocessing as mp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch

from ._blending import WeightedRoundRobinDispatcher
from ._producer_pool_base import BaseProducerPool
from .shuffle_buffer import ShuffleBuffer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProducerConfig:
    """Picklable configuration for a data source producer.

    Holds a :class:`LeRobotShardSource` directly — the shard source's
    ``__getstate__`` drops its row LRU for pickling so spawn cost is
    bounded (few KB of info + episodes metadata) while keeping the
    parent-side construction single-sourced.

    Episode semantics:

    - ``episode_indices``: **raw episode ids** the producer iterates over
      (train split).  Pool's work queue yields these values directly;
      the child's decode function looks them up in the shard source
      by raw id.

    - ``episode_offset``: added to each raw id to form the ShuffleBuffer
      key.  Prevents key collisions when multiple sources share the
      same raw-id space (e.g. two datasets both starting at ep 0).
    """
    source_name: str
    repo_id: str
    shard_source: "LeRobotShardSource"
    episode_indices: list[int]
    weight: float = 1.0
    seed: int = 42
    tolerance_s: float = 1e-4
    video_backend: str = "pyav"
    episode_offset: int = 0
    # Optional producer-side frame transform.  Applied to each decoded
    # episode tensor BEFORE it lands in the ShuffleBuffer — so the buffer
    # allocates shm at the transform's output resolution rather than
    # source resolution.  Must be picklable (the spawn child unpickles
    # ``ProducerConfig``); use a class like ``ResizeFrames`` from
    # :mod:`dataporter.frame_transforms`, not a lambda.
    producer_transform: Callable | None = None

    @classmethod
    def from_source(
        cls,
        source: dict,
        shard_source,
        iteration_episodes: list[int],
        episode_offset: int = 0,
        producer_transform: Callable | None = None,
    ) -> "ProducerConfig":
        """Build a ProducerConfig from a shard source + iteration subset.

        Args:
            source: Source dict with ``repo_id``, ``weight``, optional
                ``tolerance_s`` / ``video_backend``.
            shard_source: A :class:`LeRobotShardSource` — passed through
                unchanged; the spawn child unpickles the same instance
                (with row-LRU dropped) rather than re-constructing one.
            iteration_episodes: RAW episode ids the producer cycles over
                (typically the train split of currently-ready eps).
            episode_offset: Shift applied to raw ids when writing to
                the ShuffleBuffer so multi-source namespaces don't
                collide.
            producer_transform: Optional picklable transform applied to
                decoded frames before buffer.put.
        """
        return cls(
            source_name=source["repo_id"],
            repo_id=source["repo_id"],
            shard_source=shard_source,
            episode_indices=list(iteration_episodes),
            weight=source.get("weight", 1.0),
            tolerance_s=source.get("tolerance_s", 1e-4),
            video_backend=source.get("video_backend", "pyav"),
            episode_offset=episode_offset,
            producer_transform=producer_transform,
        )


# Keep AsyncProducer as a convenience wrapper for thread-based usage
class AsyncProducer:
    """Configuration for a single data source producer.

    For thread-based ProducerPool (legacy API). For spawn-based,
    use ProducerConfig instead.

    Args:
        source_name: Human-readable source name (for logging).
        decode_fn: Callable that decodes an episode.
        episode_indices: List of episode indices to cycle through.
        weight: Relative sampling weight (controls blend ratio).
        seed: Random seed for episode shuffle order.
    """

    def __init__(
        self,
        source_name: str,
        decode_fn: Callable[[int], torch.Tensor],
        episode_indices: list[int],
        weight: float = 1.0,
        seed: int = 42,
        episode_offset: int = 0,
    ):
        self.source_name = source_name
        self.decode_fn = decode_fn
        self.episode_indices = episode_indices
        self.weight = weight
        self.seed = seed
        self.episode_offset = episode_offset


# ---------------------------------------------------------------------------
# Spawn-based producer pool (primary path)
# ---------------------------------------------------------------------------

# Per-decode timeout. If a single video decode takes longer than this,
# it's treated as a failure (likely ffmpeg hanging on symlinked blobs).
_DECODE_TIMEOUT_S = 30.0


def _spawn_pool_entry(
    buffer: ShuffleBuffer,
    configs: list[ProducerConfig],
    total_workers: int,
    warmup_target: int,
    warmup_event,
    stop_event,
    error_queue=None,
    update_queue=None,
) -> None:
    """Entry point for the spawned producer process.

    Runs a smoke test (one decode per source) before starting the async
    loop. If the first decode hangs or fails, the error is raised
    immediately — don't wait 300s for warmup timeout.

    Errors are sent to ``error_queue`` (if provided) so the parent
    process can surface them in the training log — child-process
    logger output may not reach the parent's log handler.

    INFO-level logging is force-configured on entry: spawn children start
    with fresh logging state (default root level = WARNING), so without
    ``basicConfig(force=True)`` our milestone logs were silently dropped.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [child-producer] %(message)s",
        force=True,
    )
    import time as _time
    _child_start = _time.monotonic()
    logger.info(
        f"[child] spawn started; {len(configs)} source(s), "
        f"total_workers={total_workers}, warmup_target={warmup_target}"
    )

    def _report_error(msg: str) -> None:
        logger.error(msg)
        if error_queue is not None:
            try:
                error_queue.put_nowait(msg)
            except Exception:
                pass

    try:
        # Smoke test: verify each source can decode at least one episode.
        #
        # Construction is done synchronously (no timeout) in the main
        # thread: FastLeRobotDataset(episodes=[...]) scales linearly
        # with episode count and for realistic dataset sizes (~18k)
        # exceeds the 30s decode budget.  Running it inside the
        # ThreadPoolExecutor deadlocks on shutdown when construction
        # blows past timeout (Python threads can't be cancelled, and
        # `with ex:` calls shutdown(wait=True)).  Moving construction
        # out keeps the 30s timeout useful for its actual purpose
        # (catching hanging ffmpeg decodes on symlinked blobs) without
        # blocking legitimate slow-dataset startups.
        decode_fns = {}
        for cfg in configs:
            fn = _make_child_decode_fn(cfg)
            decode_fns[cfg.source_name] = fn

            first_ep = cfg.episode_indices[0] if cfg.episode_indices else 0
            logger.info(
                f"[child] smoke test starting: {cfg.source_name} "
                f"ep {first_ep} (source_root={cfg.shard_source.root}, "
                f"episodes_count={len(cfg.episode_indices)})"
            )

            # Phase 1 — construction (no timeout, visible via instrumentation).
            prime_t0 = _time.monotonic()
            fn.prime()
            logger.info(
                f"[child] smoke phase-1 (construction) done: "
                f"{cfg.source_name} in {_time.monotonic() - prime_t0:.1f}s"
            )

            # Phase 2 — decode one episode under the 30s timeout.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                decode_t0 = _time.monotonic()
                future = ex.submit(fn, first_ep)
                try:
                    frames = future.result(timeout=_DECODE_TIMEOUT_S)
                    if frames is None:
                        raise RuntimeError("decode returned None")
                    logger.info(
                        f"[child] smoke phase-2 (decode) OK: "
                        f"{cfg.source_name} ep {first_ep} → "
                        f"{frames.shape} in "
                        f"{_time.monotonic() - decode_t0:.1f}s"
                    )
                except concurrent.futures.TimeoutError:
                    raise RuntimeError(
                        f"Smoke test: decode of {cfg.source_name} "
                        f"ep {first_ep} hung for {_DECODE_TIMEOUT_S}s. "
                        f"Video decode may be blocked on symlinked blobs. "
                        f"source_root={cfg.shard_source.root}"
                    )

        logger.info(
            f"[child] all smoke tests passed in "
            f"{_time.monotonic() - _child_start:.1f}s; entering async loop"
        )
        asyncio.run(_run_spawn_pool(
            buffer, configs, total_workers,
            warmup_target, warmup_event, stop_event,
            decode_fns=decode_fns,
            update_queue=update_queue,
        ))
    except Exception as e:
        _report_error(f"ProducerPool child error: {e}")
    finally:
        warmup_event.set()


def _make_child_decode_fn(config: ProducerConfig) -> Callable[[int], torch.Tensor]:
    """Create a decode function inside the child process.

    The shard source is carried through ``ProducerConfig.shard_source`` —
    unpickled fresh in the child (row LRU dropped via ``__getstate__``)
    so there's no re-read of ``info.json`` and no drift risk between
    parent and child views.

    Accepts RAW episode ids (from ``config.episode_indices``) directly;
    no positional/raw translation needed because the source is raw-indexed
    throughout.
    """
    _state = {}

    import time as _time

    def prime() -> None:
        """No-op phase separator (source already unpickled).

        Kept for API symmetry with the older flow — the smoke test uses
        ``prime() + decode()`` so construction can't be swallowed by the
        30s decode timeout.  With the shard source carried through the
        config, there's nothing to construct here; the call just marks
        the parent→child handoff in logs.
        """
        if "source" in _state:
            return
        _state["source"] = config.shard_source
        logger.info(
            f"[child:{config.source_name}] shard source ready "
            f"(root={config.shard_source.root})"
        )

    def decode(ep_idx: int) -> torch.Tensor:
        prime()
        source = _state["source"]

        video_keys = source.video_keys
        if not video_keys:
            raise RuntimeError(
                f"Dataset {config.source_name} has no video keys — "
                "cannot decode frames"
            )
        vid_key = video_keys[0]
        # ep_idx is the RAW episode id — source methods take raw directly.
        raw_ep_id = int(ep_idx)
        num_frames = source.episode_frame_count(raw_ep_id)
        video_path = source.episode_video_path(raw_ep_id, vid_key)
        # Resolve symlinks — HF hub-cache stores video files as
        # symlinks to ../../blobs/<hash>. pyav/ffmpeg may hang on
        # symlinked paths in spawned process contexts.
        t_resolve = _time.monotonic()
        video_path = video_path.resolve()
        resolve_dt = _time.monotonic() - t_resolve

        from .fast_lerobot_dataset import decode_episode_frames
        t_decode = _time.monotonic()
        frames = decode_episode_frames(
            video_path, num_frames,
            source.fps, config.tolerance_s, config.video_backend,
        )
        decode_dt = _time.monotonic() - t_decode
        if config.producer_transform is not None:
            frames = config.producer_transform(frames)
        # Only log the first decode per source to avoid chatty output
        # in the hot path.  `first_decode_logged` is set after the first
        # successful return.
        if not _state.get("first_decode_logged"):
            logger.info(
                f"[child:{config.source_name}] first decode OK: "
                f"ep {raw_ep_id} → {tuple(frames.shape)} "
                f"(resolve={resolve_dt * 1000:.1f}ms, "
                f"decode={decode_dt * 1000:.0f}ms, "
                f"video_path={video_path})"
            )
            _state["first_decode_logged"] = True
        return frames

    decode.prime = prime
    return decode


async def _run_spawn_pool(
    buffer: ShuffleBuffer,
    configs: list[ProducerConfig],
    total_workers: int,
    warmup_target: int,
    warmup_event,
    stop_event,
    decode_fns: dict[str, Callable] | None = None,
    update_queue=None,
) -> None:
    """Async event loop in spawned child: parallel decode via executors.

    Dispatches ``total_workers`` decode tasks concurrently across
    per-source ThreadPoolExecutors. Results are collected as they
    complete and written to the buffer. This gives true parallelism
    (N decodes in flight) instead of sequential await.

    When ``update_queue`` is provided, an async task polls it for
    ``(source_name, new_episode_list)`` messages and swaps the
    per-source iterator atomically.  The new list takes effect on the
    very next ``_next_dispatch()`` call for that source.
    """
    loop = asyncio.get_event_loop()

    # Reuse smoke-tested decode functions if provided (dataset already
    # lazily initialized from the smoke test call). Otherwise create new.
    if decode_fns is None:
        decode_fns = {}
        for cfg in configs:
            decode_fns[cfg.source_name] = _make_child_decode_fn(cfg)

    # Distribute thread workers by weight
    total_weight = sum(cfg.weight for cfg in configs)
    executors = {}
    workers_per_source = {}
    for cfg in configs:
        n = max(1, round(total_workers * cfg.weight / total_weight))
        executors[cfg.source_name] = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix=f"decode-{cfg.source_name}",
        )
        workers_per_source[cfg.source_name] = n

    # Episode iterators (infinite, shuffled)
    iterators = {}
    for cfg in configs:
        rng = random.Random(cfg.seed)
        iterators[cfg.source_name] = _episode_iterator(cfg.episode_indices, rng)

    weight_map = {cfg.source_name: cfg.weight for cfg in configs}
    offset_map = {cfg.source_name: cfg.episode_offset for cfg in configs}

    # Consecutive failure tracking — escalate after _MAX_CONSECUTIVE_FAILURES
    _MAX_CONSECUTIVE_FAILURES = 50
    _consecutive_failures = 0
    _last_failure_msg = ""

    async def _decode_one(source_name: str, ep_idx: int) -> tuple[str, int, torch.Tensor | None]:
        """Dispatch one decode with timeout to catch hanging ffmpeg."""
        nonlocal _consecutive_failures, _last_failure_msg
        try:
            frames = await asyncio.wait_for(
                loop.run_in_executor(
                    executors[source_name], decode_fns[source_name], ep_idx,
                ),
                timeout=_DECODE_TIMEOUT_S,
            )
            _consecutive_failures = 0  # reset on success
            return source_name, ep_idx, frames
        except asyncio.TimeoutError:
            _consecutive_failures += 1
            _last_failure_msg = (
                f"{source_name} ep {ep_idx}: decode hung for "
                f"{_DECODE_TIMEOUT_S}s (ffmpeg blocked on video file?)"
            )
            if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"ProducerPool: {_consecutive_failures} consecutive decode "
                    f"timeouts — video decode is hanging. Last: "
                    f"{_last_failure_msg}"
                )
            logger.warning(
                f"Decode timeout ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}): "
                f"{_last_failure_msg}"
            )
            return source_name, ep_idx, None
        except Exception as e:
            _consecutive_failures += 1
            _last_failure_msg = f"{source_name} ep {ep_idx}: {e}"
            if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"ProducerPool: {_consecutive_failures} consecutive decode "
                    f"failures — all episodes are failing. Last error: "
                    f"{_last_failure_msg}"
                ) from e
            logger.warning(
                f"Decode failed ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}): "
                f"{_last_failure_msg}"
            )
            return source_name, ep_idx, None

    # Weighted dispatch: fill the pipeline with total_workers tasks
    dispatcher = WeightedRoundRobinDispatcher(weight_map)
    pending: set[asyncio.Task] = set()

    def _next_dispatch() -> tuple[str, int]:
        """Pick next (source, episode) using weighted round-robin."""
        name = dispatcher.next()
        ep_idx = next(iterators[name])
        return name, ep_idx

    async def _poll_updates() -> None:
        """Background task: pull update_episodes messages from the parent.

        Swaps ``iterators[source_name]`` to a fresh iterator over the new
        list as soon as a message arrives.  The currently-dispatched
        decode tasks finish on the old list; subsequent dispatches pick up
        the new list.
        """
        if update_queue is None:
            return
        while not stop_event.is_set():
            try:
                msg = update_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.1)
                continue
            try:
                source_name, new_episodes = msg
                if source_name not in iterators:
                    logger.warning(
                        f"update_episodes: unknown source "
                        f"{source_name!r}, ignoring"
                    )
                    continue
                rng = random.Random(
                    next(
                        (c.seed for c in configs if c.source_name == source_name),
                        42,
                    )
                )
                iterators[source_name] = _episode_iterator(
                    list(new_episodes), rng,
                )
                logger.info(
                    f"ProducerPool: {source_name} iterator swapped to "
                    f"{len(new_episodes)} episodes"
                )
            except Exception as e:
                logger.warning(f"update_episodes handler: {e}")

    update_task = asyncio.create_task(_poll_updates())

    # Seed the pipeline with total_workers concurrent decodes
    for _ in range(total_workers):
        name, ep_idx = _next_dispatch()
        task = asyncio.create_task(_decode_one(name, ep_idx))
        pending.add(task)

    while not stop_event.is_set():
        # Wait for any decode to complete
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            source_name, ep_idx, frames = task.result()
            if frames is not None:
                buffer.put(offset_map[source_name] + ep_idx, frames)

            # Check warmup
            if not warmup_event.is_set() and len(buffer) >= warmup_target:
                warmup_event.set()
                logger.info(
                    f"ProducerPool warmup complete: "
                    f"{len(buffer)} items in buffer"
                )

            # Dispatch a replacement task
            if not stop_event.is_set():
                name, ep_idx = _next_dispatch()
                new_task = asyncio.create_task(_decode_one(name, ep_idx))
                pending.add(new_task)

        # Backpressure: pause dispatching when buffer is full
        while len(buffer) >= buffer.capacity and not stop_event.is_set():
            await asyncio.sleep(0.05)

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    for ex in executors.values():
        ex.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Thread-based fallback (for when spawn isn't needed)
# ---------------------------------------------------------------------------

def _run_thread_pool(
    buffer: ShuffleBuffer,
    producers: list[AsyncProducer],
    warmup_target: int,
    warmup_event,
    stop_event,
) -> None:
    """Thread-based producer loop (fallback for simple cases)."""
    import time

    iterators = {}
    for p in producers:
        rng = random.Random(p.seed)
        iterators[p.source_name] = _episode_iterator(p.episode_indices, rng)

    weight_map = {p.source_name: p.weight for p in producers}
    decode_map = {p.source_name: p.decode_fn for p in producers}
    offset_map = {p.source_name: p.episode_offset for p in producers}
    dispatcher = WeightedRoundRobinDispatcher(weight_map)

    consecutive_failures = 0
    max_consecutive = 50

    while not stop_event.is_set():
        name = dispatcher.next()
        ep_idx = next(iterators[name])

        try:
            frames = decode_map[name](ep_idx)
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive:
                raise RuntimeError(
                    f"ProducerPool: {consecutive_failures} consecutive decode "
                    f"failures. Last: {name} ep {ep_idx}: {e}"
                ) from e
            logger.warning(
                f"Decode failed ({consecutive_failures}/{max_consecutive}): "
                f"{name} ep {ep_idx}: {e}"
            )
            continue

        consecutive_failures = 0  # reset on success
        buffer.put(offset_map[name] + ep_idx, frames)

        if not warmup_event.is_set() and len(buffer) >= warmup_target:
            warmup_event.set()
            logger.info(
                f"ProducerPool warmup complete: {len(buffer)} items in buffer"
            )

        while len(buffer) >= buffer.capacity and not stop_event.is_set():
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _episode_iterator(episode_indices: list[int], rng: random.Random):
    """Infinite iterator that cycles through episodes in shuffled order."""
    while True:
        order = list(episode_indices)
        rng.shuffle(order)
        yield from order


# ---------------------------------------------------------------------------
# ProducerPool
# ---------------------------------------------------------------------------

class ProducerPool(BaseProducerPool):
    """Manages background episode decoding into a ShuffleBuffer.

    Two modes:

    1. **Spawn mode** (default when ``configs`` provided): spawns a
       fresh child process with its own shard source and
       ThreadPoolExecutor per source.  No inherited state, no fork
       deadlocks, true parallelism.
    2. **Thread mode** (fallback when ``producers`` provided): daemon
       thread in the main process.  Simpler but shares the GIL.  Used
       when the decode closure isn't picklable (mostly tests).

    Lifecycle (``start`` / ``stop`` / ``wait_for_warmup`` /
    ``update_episodes`` / ``is_alive`` / error-queue drain) is
    inherited from :class:`BaseProducerPool`.  This subclass owns only
    the mode selection and :meth:`_create_worker`.

    Args:
        buffer: The ShuffleBuffer to fill.
        configs: List of ProducerConfig (spawn mode).  Picklable.
        producers: List of AsyncProducer (thread mode).  Not picklable.
        total_workers: Total decode threads distributed across sources.
        warmup_target: Fill buffer to this level before signaling ready.
    """

    def __init__(
        self,
        buffer: ShuffleBuffer,
        configs: list[ProducerConfig] | None = None,
        producers: list[AsyncProducer] | None = None,
        total_workers: int = 4,
        warmup_target: int | None = None,
    ):
        if configs is None and producers is None:
            raise ValueError("Either configs or producers must be provided")
        if configs is not None and producers is not None:
            raise ValueError("Cannot specify both configs and producers")
        # source_name is the routing key for update_episodes(); duplicates
        # would silently broadcast to the wrong per-source iterator.
        if configs is not None:
            names = [c.source_name for c in configs]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"ProducerPool: duplicate source_name in configs: "
                    f"{names!r} — each ProducerConfig must have a unique "
                    f"source_name"
                )

        self._buffer = buffer
        self._configs = configs
        self._producers = producers
        self._total_workers = total_workers
        self._warmup_target = warmup_target or buffer.capacity
        self._use_spawn = configs is not None
        self._worker = None

        if self._use_spawn:
            ctx = mp.get_context("spawn")
            self._warmup_event = ctx.Event()
            self._stop_event = ctx.Event()
            self._error_queue = ctx.Queue()
            # Parent → child control channel for update_episodes().
            self._update_queue = ctx.Queue()
        else:
            import threading
            self._warmup_event = threading.Event()
            self._stop_event = threading.Event()
            self._error_queue = None
            self._update_queue = None

    def _create_worker(self):
        """Spawn a fresh ``Process`` (spawn mode) or ``Thread`` (thread
        mode) bound to this pool's config.  Inherits ``start``/``stop``
        etc. from :class:`BaseProducerPool`.
        """
        if self._use_spawn:
            ctx = mp.get_context("spawn")
            return ctx.Process(
                target=_spawn_pool_entry,
                args=(
                    self._buffer,
                    self._configs,
                    self._total_workers,
                    self._warmup_target,
                    self._warmup_event,
                    self._stop_event,
                    self._error_queue,
                    self._update_queue,
                ),
                daemon=True,
                name="producer-pool",
            )
        import threading
        return threading.Thread(
            target=_run_thread_pool,
            args=(
                self._buffer,
                self._producers,
                self._warmup_target,
                self._warmup_event,
                self._stop_event,
            ),
            daemon=True,
            name="producer-pool",
        )
