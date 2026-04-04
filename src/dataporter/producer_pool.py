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

from .shuffle_buffer import ShuffleBuffer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProducerConfig:
    """Picklable configuration for a data source producer.

    All fields are primitives — no dataset objects, no closures.
    The spawned child process creates its own dataset from these params.

    ``episode_offset`` shifts raw episode indices into a source-unique
    namespace before writing to the ShuffleBuffer.  This prevents key
    collisions when multiple sources share the same raw episode indices
    (e.g. both start at 0).
    """
    source_name: str
    repo_id: str
    root: str
    episode_indices: list[int]
    weight: float = 1.0
    seed: int = 42
    tolerance_s: float | None = None
    episode_offset: int = 0


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

def _spawn_pool_entry(
    buffer: ShuffleBuffer,
    configs: list[ProducerConfig],
    total_workers: int,
    warmup_target: int,
    warmup_event,
    stop_event,
) -> None:
    """Entry point for the spawned producer process."""
    try:
        asyncio.run(_run_spawn_pool(
            buffer, configs, total_workers,
            warmup_target, warmup_event, stop_event,
        ))
    except Exception as e:
        logger.error(f"ProducerPool error: {e}", exc_info=True)
    finally:
        warmup_event.set()


def _make_child_decode_fn(config: ProducerConfig) -> Callable[[int], torch.Tensor]:
    """Create a decode function inside the child process.

    Lazily initializes a fresh FastLeRobotDataset on first call —
    the child gets its own Arrow mmap, independent of the parent.
    """
    from pathlib import Path

    _state = {}

    def decode(ep_idx: int) -> torch.Tensor:
        if "ds" not in _state:
            from .fast_lerobot_dataset import FastLeRobotDataset
            kwargs = {"root": Path(config.root)}
            if config.tolerance_s is not None:
                kwargs["tolerance_s"] = config.tolerance_s
            _state["ds"] = FastLeRobotDataset(
                config.repo_id,
                delta_timestamps={"observation.image": [0.0]},
                **kwargs,
            )
            logger.info(
                f"[child] Loaded {config.source_name} from {config.root}"
            )
        ds = _state["ds"]

        from lerobot.common.datasets.video_utils import decode_video_frames

        if not ds.meta.video_keys:
            raise RuntimeError(
                f"Dataset {config.source_name} has no video keys — "
                "cannot decode frames"
            )
        vid_key = ds.meta.video_keys[0]
        ep_start = ds.episode_data_index["from"][ep_idx].item()
        ep_end = ds.episode_data_index["to"][ep_idx].item()
        num_frames = ep_end - ep_start
        all_ts = [i / ds.fps for i in range(num_frames)]
        video_path = ds.root / ds.meta.get_video_file_path(ep_idx, vid_key)
        all_frames = decode_video_frames(
            video_path, all_ts, ds.tolerance_s, ds.video_backend,
        )
        if all_frames.dim() == 5:
            all_frames = all_frames.squeeze(0)
        return (all_frames * 255).to(torch.uint8)

    return decode


async def _run_spawn_pool(
    buffer: ShuffleBuffer,
    configs: list[ProducerConfig],
    total_workers: int,
    warmup_target: int,
    warmup_event,
    stop_event,
) -> None:
    """Async event loop in spawned child: parallel decode via executors.

    Dispatches ``total_workers`` decode tasks concurrently across
    per-source ThreadPoolExecutors. Results are collected as they
    complete and written to the buffer. This gives true parallelism
    (N decodes in flight) instead of sequential await.
    """
    loop = asyncio.get_event_loop()

    # Build decode functions (fresh datasets in this process)
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

    async def _decode_one(source_name: str, ep_idx: int) -> tuple[str, int, torch.Tensor | None]:
        """Dispatch one decode to the source's executor."""
        try:
            frames = await loop.run_in_executor(
                executors[source_name], decode_fns[source_name], ep_idx,
            )
            return source_name, ep_idx, frames
        except Exception as e:
            logger.warning(f"Decode failed for {source_name} ep {ep_idx}: {e}")
            return source_name, ep_idx, None

    # Weighted dispatch: fill the pipeline with total_workers tasks
    tokens = {name: 0.0 for name in weight_map}
    pending: set[asyncio.Task] = set()

    def _next_dispatch() -> tuple[str, int]:
        """Pick next (source, episode) using weighted round-robin."""
        name = max(tokens, key=lambda n: weight_map[n] - tokens[n])
        ep_idx = next(iterators[name])
        tokens[name] += 1
        if all(t >= weight_map[n] for n, t in tokens.items()):
            for k in tokens:
                tokens[k] = 0.0
        return name, ep_idx

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

    tokens = {p.source_name: 0.0 for p in producers}
    weight_map = {p.source_name: p.weight for p in producers}
    decode_map = {p.source_name: p.decode_fn for p in producers}
    offset_map = {p.source_name: p.episode_offset for p in producers}

    while not stop_event.is_set():
        name = max(tokens, key=lambda n: weight_map[n] - tokens[n])
        ep_idx = next(iterators[name])

        try:
            frames = decode_map[name](ep_idx)
        except Exception as e:
            logger.warning(f"Decode failed for {name} ep {ep_idx}: {e}")
            continue

        buffer.put(offset_map[name] + ep_idx, frames)

        tokens[name] += 1
        if all(t >= weight_map[n] for n, t in tokens.items()):
            tokens = {n: 0.0 for n in tokens}

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

class ProducerPool:
    """Manages background episode decoding into a ShuffleBuffer.

    Two modes:

    1. **Spawn mode** (default when ``configs`` provided): spawns a fresh
       child process with its own datasets and ThreadPoolExecutor per
       source. No inherited state, no fork deadlocks, true parallelism.

    2. **Thread mode** (fallback when ``producers`` provided): daemon
       thread in the main process. Simpler but shares GIL. Used when
       decode_fn closures aren't picklable.

    Args:
        buffer: The ShuffleBuffer to fill.
        configs: List of ProducerConfig (spawn mode). Picklable.
        producers: List of AsyncProducer (thread mode). Not picklable.
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

        self._buffer = buffer
        self._configs = configs
        self._producers = producers
        self._total_workers = total_workers
        self._warmup_target = warmup_target or buffer.capacity
        self._use_spawn = configs is not None

        self._worker = None  # Process or Thread

        if self._use_spawn:
            ctx = mp.get_context("spawn")
            self._warmup_event = ctx.Event()
            self._stop_event = ctx.Event()
        else:
            import threading
            self._warmup_event = threading.Event()
            self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the producer (spawn process or daemon thread)."""
        if self._worker is not None and (
            hasattr(self._worker, 'is_alive') and self._worker.is_alive()
        ):
            raise RuntimeError("ProducerPool already running")

        self._stop_event.clear()
        self._warmup_event.clear()

        if self._use_spawn:
            ctx = mp.get_context("spawn")
            self._worker = ctx.Process(
                target=_spawn_pool_entry,
                args=(
                    self._buffer,
                    self._configs,
                    self._total_workers,
                    self._warmup_target,
                    self._warmup_event,
                    self._stop_event,
                ),
                daemon=True,
                name="producer-pool",
            )
        else:
            import threading
            self._worker = threading.Thread(
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

        self._worker.start()

    def wait_for_warmup(self, timeout: float = 300.0) -> None:
        """Block until warmup_target items are in the buffer."""
        if not self._warmup_event.wait(timeout=timeout):
            raise TimeoutError(
                f"ProducerPool didn't fill {self._warmup_target} items "
                f"in {timeout}s (have {len(self._buffer)})"
            )

    def stop(self) -> None:
        """Stop the producer."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
            if hasattr(self._worker, 'terminate') and self._worker.is_alive():
                self._worker.terminate()
            self._worker = None

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()
