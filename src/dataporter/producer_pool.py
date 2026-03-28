"""Async producer pool for filling ShuffleBuffer.

Single child process running an asyncio event loop. Coordinates multiple
producers (one per data source) with weighted scheduling. Each producer
dispatches video decode to a ProcessPoolExecutor.

Architecture::

    Main process (GPU training)
    |
    +-- ProducerPool process (asyncio event loop)
        |
        +-- Thread pool A (weight=3, 3 threads) -- decode source A episodes
        +-- Thread pool B (weight=1, 1 thread)  -- decode source B episodes
        |
        +-- ShuffleBuffer.put() (single writer, serialized by event loop)

Workers (in main process) call ShuffleBuffer.sample() — read-only, no decode.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch

from .shuffle_buffer import ShuffleBuffer

logger = logging.getLogger(__name__)


class AsyncProducer:
    """Configuration for a single data source producer.

    Args:
        source_name: Human-readable source name (for logging).
        decode_fn: Callable that decodes an episode. Signature:
            ``(episode_idx: int) -> torch.Tensor`` returning
            frames as ``[T, C, H, W]`` uint8.
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
    ):
        self.source_name = source_name
        self.decode_fn = decode_fn
        self.episode_indices = episode_indices
        self.weight = weight
        self.seed = seed


def _pool_entry(
    buffer: ShuffleBuffer,
    producers: list[AsyncProducer],
    total_workers: int,
    warmup_target: int,
    warmup_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """Entry point for the producer pool child process."""
    try:
        asyncio.run(_run_pool(
            buffer, producers, total_workers,
            warmup_target, warmup_event, stop_event,
        ))
    except Exception as e:
        logger.error(f"ProducerPool error: {e}", exc_info=True)
    finally:
        warmup_event.set()  # unblock parent even on error


async def _run_pool(
    buffer: ShuffleBuffer,
    producers: list[AsyncProducer],
    total_workers: int,
    warmup_target: int,
    warmup_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """Async event loop: weighted scheduling + executor dispatch."""
    loop = asyncio.get_event_loop()

    # Distribute workers by weight
    total_weight = sum(p.weight for p in producers)
    executors = {}
    iterators = {}

    for p in producers:
        n_workers = max(1, round(total_workers * p.weight / total_weight))
        executors[p.source_name] = ThreadPoolExecutor(max_workers=n_workers)
        iterators[p.source_name] = _episode_iterator(p)

    # Weighted round-robin tokens
    tokens = {p.source_name: 0.0 for p in producers}
    weight_map = {p.source_name: p.weight for p in producers}

    while not stop_event.is_set():
        # Pick producer with highest deficit
        name = max(tokens, key=lambda n: weight_map[n] - tokens[n])
        producer = next(p for p in producers if p.source_name == name)

        ep_idx = next(iterators[name])

        # Dispatch decode to executor
        try:
            frames = await loop.run_in_executor(
                executors[name], producer.decode_fn, ep_idx
            )
        except Exception as e:
            logger.warning(f"Decode failed for {name} ep {ep_idx}: {e}")
            continue

        buffer.put(ep_idx, frames)

        # Update tokens
        tokens[name] += 1
        if all(t >= weight_map[n] for n, t in tokens.items()):
            tokens = {n: 0.0 for n in tokens}

        # Check warmup
        if not warmup_event.is_set() and len(buffer) >= warmup_target:
            warmup_event.set()
            logger.info(
                f"ProducerPool warmup complete: {len(buffer)} items in buffer"
            )

        # Backpressure: if buffer full, yield to let workers consume
        while len(buffer) >= buffer.capacity and not stop_event.is_set():
            await asyncio.sleep(0.01)

    # Shutdown executors
    for ex in executors.values():
        ex.shutdown(wait=False, cancel_futures=True)


def _episode_iterator(producer: AsyncProducer):
    """Infinite iterator that cycles through episodes in shuffled order."""
    rng = random.Random(producer.seed)
    while True:
        order = list(producer.episode_indices)
        rng.shuffle(order)
        yield from order


class ProducerPool:
    """Manages background episode decoding into a ShuffleBuffer.

    Single child process with asyncio event loop. Coordinates multiple
    data sources with weighted scheduling.

    Args:
        buffer: The ShuffleBuffer to fill.
        producers: List of AsyncProducer configs (one per source).
        total_workers: Total decode workers distributed across sources.
        warmup_target: Fill buffer to this level before signaling ready.
            Default: buffer capacity (fill completely before training).
    """

    def __init__(
        self,
        buffer: ShuffleBuffer,
        producers: list[AsyncProducer],
        total_workers: int = 4,
        warmup_target: int | None = None,
    ):
        self._buffer = buffer
        self._producers = producers
        self._total_workers = total_workers
        self._warmup_target = warmup_target or buffer.capacity
        self._process: mp.Process | None = None

        ctx = mp.get_context("fork")
        self._warmup_event = ctx.Event()
        self._stop_event = ctx.Event()

    def start(self) -> None:
        """Spawn child process running the producer pool."""
        if self._process is not None and self._process.is_alive():
            raise RuntimeError("ProducerPool already running")

        self._stop_event.clear()
        self._warmup_event.clear()

        ctx = mp.get_context("fork")
        self._process = ctx.Process(
            target=_pool_entry,
            args=(
                self._buffer,
                self._producers,
                self._total_workers,
                self._warmup_target,
                self._warmup_event,
                self._stop_event,
            ),
            daemon=True,
            name="producer-pool",
        )
        self._process.start()

    def wait_for_warmup(self, timeout: float = 300.0) -> None:
        """Block until warmup_target items are in the buffer."""
        if not self._warmup_event.wait(timeout=timeout):
            raise TimeoutError(
                f"ProducerPool didn't fill {self._warmup_target} items "
                f"in {timeout}s (have {len(self._buffer)})"
            )

    def stop(self) -> None:
        """Stop the producer pool process."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()
