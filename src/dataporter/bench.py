"""DataLoader throughput benchmark.

Measures how fast a DataLoader can produce batches (no GPU compute).
This gives the upper bound on training throughput — if the pipeline
can't keep up, the GPU starves.

Usage as library::

    from dataporter.bench import benchmark_dataloader
    results = benchmark_dataloader(my_dataloader, n_batches=200)
    print(f"Avg: {results['avg_ms']:.1f}ms, Tokens/s: {results['tokens_per_sec']:,.0f}")

Usage as CLI::

    python -m dataporter.bench --help
"""

from __future__ import annotations

import time
from typing import Any

from torch.utils.data import DataLoader


def benchmark_dataloader(
    dataloader: DataLoader,
    n_batches: int = 200,
    n_warmup: int = 20,
    tokens_per_batch: int | None = None,
    gpu_compute_ms: float = 770.0,
    accum_steps: int = 4,
) -> dict[str, Any]:
    """Benchmark a DataLoader's batch production throughput.

    Consumes batches as fast as possible (no GPU compute) and reports
    timing statistics.

    Args:
        dataloader: The DataLoader to benchmark.
        n_batches: Number of batches to time (after warmup).
        n_warmup: Warmup batches (excluded from timing).
        tokens_per_batch: If provided, reports tokens/sec.
        gpu_compute_ms: Assumed GPU compute time per batch for
            estimating ``data_pct`` and step time.
        accum_steps: Gradient accumulation steps for step time estimate.

    Returns:
        Dict with timing stats: avg_ms, p50_ms, p95_ms, p99_ms,
        tokens_per_sec, batches_per_sec, data_pct_est, step_time_est.
    """
    it = iter(dataloader)

    # Warmup
    first_batch = None
    for i in range(n_warmup):
        batch = next(it)
        if i == 0:
            first_batch = batch

    # Auto-detect tokens_per_batch from first batch
    if tokens_per_batch is None and first_batch is not None:
        ids = first_batch.get("input_ids")
        if ids is not None:
            tokens_per_batch = ids.numel()

    # Benchmark
    times: list[float] = []
    for _ in range(n_batches):
        t0 = time.time()
        next(it)
        times.append(time.time() - t0)

    times_ms = [t * 1000 for t in times]
    avg = sum(times_ms) / len(times_ms)
    sorted_times = sorted(times_ms)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    total_time = sum(times)

    results: dict[str, Any] = {
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "batches_per_sec": n_batches / total_time,
    }

    if tokens_per_batch:
        results["tokens_per_sec"] = n_batches * tokens_per_batch / total_time

    # Training impact estimates
    data_pct = avg / (avg + gpu_compute_ms) * 100
    step_time = (avg + gpu_compute_ms) * accum_steps / 1000
    results["data_pct_est"] = data_pct
    results["step_time_est"] = step_time

    return results


def print_report(
    results: dict[str, Any],
    n_batches: int = 200,
    tokens_per_batch: int | None = None,
    gpu_compute_ms: float = 770.0,
    accum_steps: int = 4,
    total_steps: int = 15000,
) -> None:
    """Print a formatted benchmark report."""
    print(f"\n{'='*50}")
    print("Pipeline Throughput Benchmark")
    print(f"{'='*50}")
    print(f"  Batches:        {n_batches}")
    if tokens_per_batch:
        print(f"  Tokens/batch:   {tokens_per_batch:,}")
    print()
    print(f"  Avg batch time: {results['avg_ms']:.1f} ms")
    print(f"  P50 batch time: {results['p50_ms']:.1f} ms")
    print(f"  P95 batch time: {results['p95_ms']:.1f} ms")
    print(f"  P99 batch time: {results['p99_ms']:.1f} ms")
    print()
    if "tokens_per_sec" in results:
        print(f"  Tokens/sec:     {results['tokens_per_sec']:,.0f}")
    print(f"  Batches/sec:    {results['batches_per_sec']:.1f}")
    print(f"{'='*50}")
    print(f"\n  Estimated training impact ({gpu_compute_ms:.0f}ms GPU, accum={accum_steps}):")
    print(f"    data_pct:   {results['data_pct_est']:.1f}%")
    print(f"    step time:  {results['step_time_est']:.2f}s")
    print(f"    {total_steps} steps: {results['step_time_est'] * total_steps / 3600:.1f}h")
