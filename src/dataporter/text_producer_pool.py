"""Text producer pool — spawned child that fills a TokenShuffleBuffer.

Parallels :mod:`producer_pool` but the kernel is user-supplied
tokenization (+ optional augmentation) instead of video decode.

Architecture::

    Main process (GPU training)
    |
    +-- TextPrefetcher process (writes parquet shards to shard_dir)
    |
    +-- TextProducerPool process (SPAWNED, fresh)
    |   +-- Discovers shards in shard_dir
    |   +-- ThreadPoolExecutor tokenizes rows in parallel (Rust tokenizers
    |       release the GIL, so threads give real parallelism)
    |   +-- Writes tokens to TokenShuffleBuffer (single writer)
    |
    +-- DataLoader workers (forked, read-only)
        +-- TokenShuffleBuffer.sample() → batches

The kernel (``tokenize_fn``) is an arbitrary picklable callable
``str → (tokens, loss_mask)``.  Projects supply one that matches their
tokenizer + augmentation needs (see ``autofpv/data/streaming_tokenize.py``).

The tokenizer itself **must not be loaded in the parent** — the callable's
``__init__`` should store only data (paths, kwargs) and the ``__call__``
should lazy-init on first use in the child.  HF tokenizers pickle slowly
and sometimes hang on vocab-extension state, so we avoid it entirely.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol

import torch

from .token_shuffle_buffer import TokenShuffleBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel protocol
# ---------------------------------------------------------------------------


class TokenizeFn(Protocol):
    """Picklable tokenize+augment callable run inside the producer child.

    Implementations should load the tokenizer lazily (on first ``__call__``)
    so ``__init__`` state is pure data and pickles cleanly across the
    ``spawn`` boundary.  Returning ``None`` signals the row should be
    dropped (e.g. empty text after augmentation).
    """

    def __call__(
        self, raw_text: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TextProducerConfig:
    """Picklable config for a text producer.

    Args:
        source_name: Label for logging.
        shard_dir: Directory where parquet shards arrive (typically
            TextPrefetcher's cache_dir).
        tokenize_fn: Picklable callable that turns raw text into
            ``(tokens, loss_mask)`` or ``None`` to skip the row.
        text_column: Parquet column holding raw text.
        shard_glob: Glob pattern for shard files under ``shard_dir``.
        seed: RNG seed for shard and row shuffling.
        thread_workers: Worker threads for parallel tokenization inside
            the producer process.  Keep at 1 unless the kernel releases
            the GIL (Rust tokenizers do).
        shard_rescan_s: How often to rescan ``shard_dir`` when exhausted.
        drop_if_buffer_full_s: Sleep between polls when the buffer is at
            capacity (back-pressure).
    """
    source_name: str
    shard_dir: str
    tokenize_fn: TokenizeFn
    text_column: str = "text"
    shard_glob: str = "shard_*.parquet"
    seed: int = 42
    thread_workers: int = 1
    shard_rescan_s: float = 15.0
    drop_if_buffer_full_s: float = 0.05


# ---------------------------------------------------------------------------
# Child entry point
# ---------------------------------------------------------------------------


def _scan_shards(shard_dir: Path, pattern: str) -> list[Path]:
    return sorted(shard_dir.glob(pattern))


def _shard_iterator(
    shard_dir: Path, pattern: str, rescan_s: float,
    rng: random.Random, stop_event,
):
    """Yield shard paths forever, rescanning when exhausted.

    Shards are shuffled within each pass so multiple epochs over a fixed
    directory don't produce identical document order.
    """
    seen: set[str] = set()
    while not stop_event.is_set():
        shards = _scan_shards(shard_dir, pattern)
        if not shards:
            if stop_event.wait(timeout=rescan_s):
                return
            continue

        # Shuffle this epoch's shards
        shard_paths = list(shards)
        rng.shuffle(shard_paths)

        for path in shard_paths:
            if stop_event.is_set():
                return
            yield path
            seen.add(path.name)

        # Exhausted pass — rescan.  If no new shards since last pass, sleep
        # so we don't spin.
        if stop_event.wait(timeout=rescan_s):
            return


def _spawn_pool_entry(
    buffer: TokenShuffleBuffer,
    config: TextProducerConfig,
    warmup_target: int,
    warmup_event,
    stop_event,
    error_queue,
) -> None:
    """Entry point for the spawned text-producer process.

    Smoke-tests the tokenize kernel on the first available shard row
    before entering the main loop so configuration errors surface
    instantly instead of hanging the warmup timeout.
    """
    def _report_error(msg: str) -> None:
        logger.error(msg)
        try:
            error_queue.put_nowait(msg)
        except Exception:
            pass

    try:
        import pyarrow.parquet as pq  # local import — heavy

        shard_dir = Path(config.shard_dir)
        rng = random.Random(config.seed)

        # Smoke test: find first shard, tokenize first row — any failure
        # (import error, tokenizer load, vocab mismatch) surfaces now.
        t0 = time.monotonic()
        first_shard = None
        while not stop_event.is_set() and (time.monotonic() - t0) < 120:
            shards = _scan_shards(shard_dir, config.shard_glob)
            if shards:
                first_shard = shards[0]
                break
            stop_event.wait(timeout=1.0)
        if first_shard is None:
            raise RuntimeError(
                f"TextProducerPool: no shards appeared in {shard_dir} "
                "within 120s — is TextPrefetcher running?"
            )

        table = pq.read_table(first_shard, columns=[config.text_column])
        if len(table) == 0:
            raise RuntimeError(
                f"TextProducerPool: first shard {first_shard.name} has 0 rows"
            )
        sample_text = table[config.text_column][0].as_py()
        probe = config.tokenize_fn(sample_text)
        if probe is None:
            raise RuntimeError(
                "TextProducerPool: tokenize_fn returned None on smoke-test "
                "row — kernel may reject every input"
            )
        probe_tokens, probe_mask = probe
        logger.info(
            f"[text-producer] Smoke test OK: {first_shard.name} → "
            f"{probe_tokens.shape[0]} tokens"
        )

        # Main loop: iterate shards, tokenize rows, push to buffer.
        doc_idx = 0
        with ThreadPoolExecutor(
            max_workers=max(1, config.thread_workers),
            thread_name_prefix=f"tokenize-{config.source_name}",
        ) as executor:
            for shard_path in _shard_iterator(
                shard_dir, config.shard_glob,
                config.shard_rescan_s, rng, stop_event,
            ):
                try:
                    table = pq.read_table(
                        shard_path, columns=[config.text_column],
                    )
                except Exception as e:
                    logger.warning(
                        f"[text-producer] Failed to read {shard_path.name}: {e}"
                    )
                    continue

                texts = table[config.text_column].to_pylist()
                row_order = list(range(len(texts)))
                rng.shuffle(row_order)

                # Parallel tokenize in batches so we don't queue millions
                # of futures when shards are huge.
                batch_size = max(32, config.thread_workers * 8)
                for i in range(0, len(row_order), batch_size):
                    if stop_event.is_set():
                        return
                    batch_idx = row_order[i : i + batch_size]
                    batch_texts = [texts[j] for j in batch_idx]

                    for result in executor.map(config.tokenize_fn, batch_texts):
                        if stop_event.is_set():
                            return
                        if result is None:
                            continue
                        tokens, mask = result
                        if tokens.numel() == 0:
                            continue

                        # Back-pressure: wait when the buffer is full so
                        # we don't tokenize-and-discard.  Capped wait —
                        # if workers aren't consuming, producer logs and
                        # keeps pushing (eviction will happen).
                        spin_start = time.monotonic()
                        while len(buffer) >= buffer.capacity:
                            if stop_event.is_set():
                                return
                            if (time.monotonic() - spin_start) > 5.0:
                                break
                            stop_event.wait(
                                timeout=config.drop_if_buffer_full_s
                            )

                        buffer.put(doc_idx, tokens, mask)
                        doc_idx += 1

                        if (
                            not warmup_event.is_set()
                            and len(buffer) >= warmup_target
                        ):
                            warmup_event.set()
                            logger.info(
                                f"[text-producer] Warmup complete: "
                                f"{len(buffer)} items in buffer"
                            )
    except Exception as e:
        _report_error(f"TextProducerPool child error: {e}")
    finally:
        warmup_event.set()


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class TextProducerPool:
    """Background process that tokenizes shards into a TokenShuffleBuffer.

    Args:
        buffer: Shared-memory token buffer to fill.
        config: Producer config (shards, tokenize_fn, parallelism).
        warmup_target: Wait until the buffer holds this many items before
            ``wait_for_warmup`` returns.  Defaults to half the buffer
            capacity — full capacity is aggressive and stalls startup on
            slow tokenizers.
    """

    def __init__(
        self,
        buffer: TokenShuffleBuffer,
        config: TextProducerConfig,
        warmup_target: int | None = None,
    ):
        self._buffer = buffer
        self._config = config
        self._warmup_target = warmup_target or max(1, buffer.capacity // 2)

        ctx = mp.get_context("spawn")
        self._warmup_event = ctx.Event()
        self._stop_event = ctx.Event()
        self._error_queue = ctx.Queue()
        self._worker: mp.Process | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("TextProducerPool already running")
        self._stop_event.clear()
        self._warmup_event.clear()

        ctx = mp.get_context("spawn")
        self._worker = ctx.Process(
            target=_spawn_pool_entry,
            args=(
                self._buffer,
                self._config,
                self._warmup_target,
                self._warmup_event,
                self._stop_event,
                self._error_queue,
            ),
            daemon=True,
            name=f"text-producer-{self._config.source_name}",
        )
        self._worker.start()

    def wait_for_warmup(self, timeout: float = 300.0) -> None:
        """Block until the buffer hits ``warmup_target``.

        Errors reported by the child are re-raised here so they appear
        in the parent's training log — child-process logger output
        often doesn't reach the parent's handler.
        """
        if not self._warmup_event.wait(timeout=timeout):
            child_error = self._drain_error_queue()
            if child_error:
                raise RuntimeError(
                    f"TextProducerPool child failed: {child_error}"
                )
            raise TimeoutError(
                f"TextProducerPool didn't fill {self._warmup_target} "
                f"items in {timeout}s (have {len(self._buffer)})"
            )
        child_error = self._drain_error_queue()
        if child_error:
            raise RuntimeError(
                f"TextProducerPool child failed: {child_error}"
            )

    def _drain_error_queue(self) -> str | None:
        try:
            return self._error_queue.get_nowait()
        except Exception:
            return None

    @property
    def error(self) -> str | None:
        """Return the first queued error without consuming it."""
        return self._drain_error_queue()

    @property
    def is_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=10)
            if self._worker.is_alive():
                self._worker.terminate()
            self._worker = None
