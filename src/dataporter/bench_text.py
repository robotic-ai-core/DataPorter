"""Throughput + capacity-sizing benchmarks for the text streaming path.

Two scenarios, runnable as a CLI::

    python -m dataporter.bench_text throughput
    python -m dataporter.bench_text capacity

``throughput`` measures raw tokenizer ceiling, producer rate across
thread counts, and end-to-end DataLoader rate.

``capacity`` sweeps buffer capacity against a simulated consumer rate
and reports the resample ratio (how often workers re-read an already-
sampled doc) — the metric that actually matters for training data
diversity.
"""

from __future__ import annotations

import argparse
import random
import resource
import shutil
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .text_producer_pool import TextProducerConfig, TextProducerPool
from .token_shuffle_buffer import TokenShuffleBuffer
from .token_shuffle_buffer_dataset import TokenShuffleBufferDataset


class _GPT2Tokenize:
    """Picklable tokenize callable for the benchmarks.

    Spawn children can't pickle closures, so this lives at module scope.
    """
    def __init__(self, seq_len: int = 512):
        self.seq_len = seq_len
        self._tok = None

    def __call__(self, text: str):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained("gpt2")
        ids = self._tok.encode(
            text, truncation=True, max_length=self.seq_len,
        )
        if not ids:
            return None
        return (
            torch.tensor(ids, dtype=torch.int32),
            torch.tensor([1] * len(ids), dtype=torch.uint8),
        )


def _seed_shards(
    shard_dir: Path, n_shards: int = 20, rows_per_shard: int = 500,
    chars_per_row: int = 2400,
) -> None:
    """Write realistic-length parquet shards (≈354 GPT-2 tokens/row)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    words = (
        "the quick brown fox jumps over the lazy dog while considering "
        "distributed systems and compilation strategies for modern "
        "hardware architectures tokens vocabulary"
    ).split()
    for s in range(n_shards):
        texts = []
        for _ in range(rows_per_shard):
            chunks, cur = [], 0
            while cur < chars_per_row:
                w = rng.choice(words)
                chunks.append(w)
                cur += len(w) + 1
            texts.append(" ".join(chunks))
        pq.write_table(
            pa.table({"text": texts}),
            shard_dir / f"shard_{s:06d}.parquet",
        )


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def bench_raw_tokenizer(seq_len: int = 512, n_docs: int = 1000) -> None:
    """In-process GPT-2 tokenizer baseline — the ceiling for pipeline rate."""
    print("=== Raw tokenizer (no pipeline, single thread) ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    rng = random.Random(42)
    words = "quick brown fox jumps over lazy dog tokens".split()
    texts = []
    for _ in range(n_docs):
        chunks, cur = [], 0
        while cur < 2400:
            w = rng.choice(words)
            chunks.append(w)
            cur += len(w) + 1
        texts.append(" ".join(chunks))

    for t in texts[:10]:  # warm
        tok.encode(t, truncation=True, max_length=seq_len)

    t0 = time.monotonic()
    total = 0
    for t in texts:
        total += len(tok.encode(t, truncation=True, max_length=seq_len))
    dt = time.monotonic() - t0
    print(f"  {len(texts) / dt:.0f} docs/s  "
          f"{total / dt:.0f} tokens/s  "
          f"(avg {total / len(texts):.0f} tok/doc)")


def bench_producer(
    producer_threads: int, seq_len: int = 512,
    capacity: int = 4096, measure_seconds: float = 5.0,
) -> None:
    """Producer-only rate.  Buffer is larger than what fills in
    ``measure_seconds`` so back-pressure doesn't throttle."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _seed_shards(td)
        buf = TokenShuffleBuffer(
            capacity=capacity, seq_len=seq_len, vocab_size=50257,
        )
        cfg = TextProducerConfig(
            source_name="bench", shard_dir=str(td),
            tokenize_fn=_GPT2Tokenize(seq_len=seq_len),
            thread_workers=producer_threads,
        )
        pool = TextProducerPool(buf, cfg, warmup_target=128)
        pool.start()
        pool.wait_for_warmup(timeout=60.0)

        start = int(buf._write_head)
        t0 = time.monotonic()
        time.sleep(measure_seconds)
        end = int(buf._write_head)
        dt = time.monotonic() - t0
        rows = end - start
        print(f"  threads={producer_threads}: "
              f"{rows / dt:.0f} docs/s  "
              f"(~{rows * 354 / dt:.0f} tokens/s)  "
              f"resident={_rss_mb():.0f} MB")
        pool.stop()


def bench_end_to_end(
    num_workers: int, producer_threads: int,
    seq_len: int = 512, batch_size: int = 32,
    capacity: int = 2048, n_batches: int = 300,
) -> None:
    """Full pipeline + DataLoader.  Reports both producer (fresh) rate
    and consumer (batch) rate so resample is visible."""
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _seed_shards(td)
        buf = TokenShuffleBuffer(
            capacity=capacity, seq_len=seq_len, vocab_size=50257,
        )
        cfg = TextProducerConfig(
            source_name="bench", shard_dir=str(td),
            tokenize_fn=_GPT2Tokenize(seq_len=seq_len),
            thread_workers=producer_threads,
        )
        pool = TextProducerPool(buf, cfg, warmup_target=capacity // 2)
        pool.start()
        pool.wait_for_warmup(timeout=120.0)

        ds = TokenShuffleBufferDataset(
            buf, epoch_length=n_batches * batch_size, padded=True,
        )
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers,
            worker_init_fn=TokenShuffleBufferDataset.worker_init_fn,
            persistent_workers=num_workers > 0,
        )
        it = iter(loader)
        _ = next(it)  # warm workers

        prod_start = int(buf._write_head)
        t0 = time.monotonic()
        rows, real_tokens = 0, 0
        for i, batch in enumerate(it):
            rows += batch["input_ids"].shape[0]
            real_tokens += int(batch["length"].sum())
            if i + 1 >= n_batches:
                break
        dt = time.monotonic() - t0
        produced = int(buf._write_head) - prod_start

        print(f"  nw={num_workers}, pt={producer_threads}, "
              f"bs={batch_size}, cap={capacity}:")
        print(f"    consumer: {rows / dt:.0f} docs/s  "
              f"{real_tokens / dt:.0f} real tok/s")
        print(f"    producer: {produced / dt:.0f} fresh docs/s")
        resample = 100.0 * max(0, rows - produced) / max(rows, 1)
        print(f"    resample: {resample:.0f}%")
        pool.stop()


# ---------------------------------------------------------------------------
# Capacity sweep — resample ratio as a function of buffer size
# ---------------------------------------------------------------------------


def show_shm_budget() -> None:
    u = shutil.disk_usage("/dev/shm")
    print(f"=== /dev/shm: {u.total / 1e9:.1f} GB total, "
          f"{u.free / 1e9:.1f} GB free ===")
    for seq_len in [512, 1024, 2048, 4096]:
        per_slot = seq_len * 5  # int32 tokens + uint8 mask
        cap_free = int(u.free * 0.5 / per_slot)
        cap_total = int(u.total * 0.5 / per_slot)
        print(f"  seq_len={seq_len:>5}: {per_slot / 1024:.1f} KB/slot; "
              f"max capacity @50% free={cap_free:>7}, "
              f"@50% total={cap_total:>7}")


def sweep_capacity(
    consumer_rate_docs_per_s: float,
    capacities: list[int],
    producer_threads: int = 8,
    seq_len: int = 512,
    measure_s: float = 6.0,
) -> None:
    """For a simulated consumer rate, show how capacity affects resample.

    Larger capacity → producer spends less wall-clock time in the
    back-pressure wait → more fresh docs/s → lower resample ratio.
    """
    print(f"\n=== consumer={consumer_rate_docs_per_s:.0f} docs/s "
          f"(simulated), producer_threads={producer_threads} ===")
    print(f"{'capacity':>10} {'prod d/s':>10} {'fresh/s':>10} "
          f"{'resample%':>10} {'reads/doc':>10}")

    for cap in capacities:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _seed_shards(td)
            buf = TokenShuffleBuffer(
                capacity=cap, seq_len=seq_len, vocab_size=50257,
            )
            cfg = TextProducerConfig(
                source_name="cap", shard_dir=str(td),
                tokenize_fn=_GPT2Tokenize(seq_len=seq_len),
                thread_workers=producer_threads,
            )
            pool = TextProducerPool(
                buf, cfg, warmup_target=min(cap, 128),
            )
            pool.start()
            pool.wait_for_warmup(timeout=60.0)

            period = 1.0 / consumer_rate_docs_per_s
            start_writes = int(buf._write_head)
            reads, fresh = 0, set()
            rng = random.Random(0)
            t0 = time.monotonic()
            next_t = t0
            while (time.monotonic() - t0) < measure_s:
                now = time.monotonic()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                try:
                    key, _, _, _ = buf.sample(rng)
                    fresh.add(key)
                    reads += 1
                except IndexError:
                    pass
                next_t += period
            dt = time.monotonic() - t0
            end_writes = int(buf._write_head)
            produced = end_writes - start_writes

            resample_pct = 100.0 * (reads - len(fresh)) / max(reads, 1)
            avg_reads = reads / max(len(fresh), 1)
            print(f"{cap:>10} {produced / dt:>10.0f} "
                  f"{len(fresh) / dt:>10.1f} {resample_pct:>9.1f}% "
                  f"{avg_reads:>10.2f}")
            pool.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("throughput", help="Producer + end-to-end rates")
    tp.add_argument("--seq-len", type=int, default=512)
    tp.add_argument(
        "--producer-threads", type=int, nargs="+", default=[1, 2, 4, 8],
    )

    cp = sub.add_parser("capacity", help="Capacity vs resample sweep")
    cp.add_argument("--seq-len", type=int, default=512)
    cp.add_argument(
        "--consumer-rates", type=float, nargs="+",
        default=[100, 500, 2000],
    )
    cp.add_argument(
        "--capacities", type=int, nargs="+",
        default=[512, 2048, 8192],
    )

    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    if args.cmd == "throughput":
        bench_raw_tokenizer(seq_len=args.seq_len)
        print("\n=== Producer throughput (5s measurement windows) ===")
        for t in args.producer_threads:
            bench_producer(producer_threads=t, seq_len=args.seq_len)
        print("\n=== End-to-end (DataLoader pulls from the buffer) ===")
        bench_end_to_end(
            num_workers=4, producer_threads=8,
            seq_len=args.seq_len, capacity=8192,
        )

    elif args.cmd == "capacity":
        show_shm_budget()
        for rate in args.consumer_rates:
            sweep_capacity(
                consumer_rate_docs_per_s=rate,
                capacities=args.capacities,
                seq_len=args.seq_len,
            )


if __name__ == "__main__":
    _main()
