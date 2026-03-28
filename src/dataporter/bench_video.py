"""Video pipeline throughput benchmark.

Measures SharedMemoryStorage + PrefetchedSource performance with
synthetic data (no real video decode). Identifies bottlenecks in the
buffer hit path, key refresh, and DataLoader integration.

Usage::

    from dataporter.bench_video import benchmark_video_pipeline
    results = benchmark_video_pipeline()
    print_video_report(results)
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .prefetched_source import PrefetchedSource
from .storage import SharedMemoryStorage
from .transformable_dataset import TransformableDataset


def benchmark_video_pipeline(
    capacity: int = 200,
    max_frames: int = 30,
    channels: int = 3,
    height: int = 96,
    width: int = 96,
    n_batches: int = 200,
    n_warmup: int = 20,
    batch_size: int = 256,
    num_workers: int = 0,
    gpu_compute_ms: float = 150.0,
) -> dict[str, Any]:
    """Benchmark the video data pipeline with synthetic data.

    Pre-fills SharedMemoryStorage with random frames, then measures
    DataLoader throughput through PrefetchedSource.

    Returns dict with timing stats and bottleneck analysis.
    """
    results: dict[str, Any] = {}

    # --- 1. Storage operations micro-benchmark ---
    storage = SharedMemoryStorage(
        capacity=capacity, max_frames=max_frames,
        channels=channels, height=height, width=width,
        max_keys=capacity * 2,
    )

    # Fill storage
    t0 = time.perf_counter()
    for i in range(capacity):
        frames = torch.randint(0, 255, (max_frames, channels, height, width), dtype=torch.uint8)
        storage.put(i, frames)
    fill_ms = (time.perf_counter() - t0) * 1000
    results["fill_ms"] = fill_ms
    results["fill_per_episode_ms"] = fill_ms / capacity

    # Benchmark storage.get (raw, no PrefetchedSource)
    times = []
    for _ in range(1000):
        idx = torch.randint(0, capacity, (1,)).item()
        t0 = time.perf_counter()
        item = storage.get(idx)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    results["storage_get_avg_us"] = sum(times) / len(times)
    results["storage_get_p99_us"] = sorted(times)[int(len(times) * 0.99)]

    # Benchmark storage.keys()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = storage.keys()
    keys_ms = (time.perf_counter() - t0) * 1000 / 100
    results["storage_keys_ms"] = keys_ms

    # Benchmark __len__ (calls .item() on shared tensor)
    t0 = time.perf_counter()
    for _ in range(10000):
        _ = len(storage)
    len_us = (time.perf_counter() - t0) * 1e6 / 10000
    results["storage_len_us"] = len_us

    # --- 2. PrefetchedSource overhead ---
    source = PrefetchedSource(storage, shuffle_available=True)

    # Benchmark __len__ (includes key refresh)
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = len(source)
    source_len_us = (time.perf_counter() - t0) * 1e6 / 1000
    results["source_len_us"] = source_len_us

    # Benchmark __getitem__ (includes key refresh + storage.get)
    times = []
    for _ in range(1000):
        idx = torch.randint(0, len(source), (1,)).item()
        t0 = time.perf_counter()
        item = source[idx]
        times.append((time.perf_counter() - t0) * 1e6)
    results["source_getitem_avg_us"] = sum(times) / len(times)
    results["source_getitem_p99_us"] = sorted(times)[int(len(times) * 0.99)]

    # --- 3. DataLoader throughput ---
    def transform(src, idx):
        item = src[idx]
        return {"frames": item["frames"].float() / 255.0}

    ds = TransformableDataset(source, transform)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    it = iter(loader)

    # Warmup
    for _ in range(min(n_warmup, len(loader) - n_batches - 1)):
        next(it)

    # Benchmark
    batch_times = []
    for _ in range(min(n_batches, len(loader) - n_warmup)):
        t0 = time.perf_counter()
        batch = next(it)
        batch_times.append((time.perf_counter() - t0) * 1000)

    if batch_times:
        sorted_bt = sorted(batch_times)
        results["dl_avg_ms"] = sum(batch_times) / len(batch_times)
        results["dl_p50_ms"] = sorted_bt[len(sorted_bt) // 2]
        results["dl_p95_ms"] = sorted_bt[int(len(sorted_bt) * 0.95)]
        results["dl_p99_ms"] = sorted_bt[int(len(sorted_bt) * 0.99)]
        results["dl_batches_per_sec"] = len(batch_times) / (sum(batch_times) / 1000)

        # Training impact
        avg = results["dl_avg_ms"]
        results["data_pct_est"] = avg / (avg + gpu_compute_ms) * 100
        results["step_time_est_ms"] = avg + gpu_compute_ms

    # --- 4. Bottleneck analysis ---
    bottlenecks = []
    if results["storage_keys_ms"] > 1.0:
        bottlenecks.append(
            f"storage.keys() takes {results['storage_keys_ms']:.1f}ms — "
            f"called on every __getitem__ in shuffle mode. "
            f"Fix: cache the key list, refresh periodically instead of every call."
        )
    if results["storage_get_avg_us"] > 50:
        bottlenecks.append(
            f"storage.get() takes {results['storage_get_avg_us']:.0f}us — "
            f".item() calls on shared tensors are expensive. "
            f"Fix: use raw tensor indexing without .item()."
        )
    if results.get("source_getitem_avg_us", 0) > 200:
        bottlenecks.append(
            f"PrefetchedSource.__getitem__ takes "
            f"{results['source_getitem_avg_us']:.0f}us — "
            f"key refresh dominates. Fix: refresh on interval, not every call."
        )
    if results.get("dl_p99_ms", 0) > gpu_compute_ms * 0.5:
        bottlenecks.append(
            f"DataLoader p99 = {results.get('dl_p99_ms', 0):.0f}ms — "
            f"tail latency may stall GPU. Check worker contention."
        )
    results["bottlenecks"] = bottlenecks

    return results


def print_video_report(results: dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*55}")
    print("Video Pipeline Throughput Benchmark")
    print(f"{'='*55}")

    print("\n  Storage micro-benchmarks:")
    print(f"    fill:         {results['fill_per_episode_ms']:.2f} ms/episode")
    print(f"    get:          {results['storage_get_avg_us']:.1f} us (p99: {results['storage_get_p99_us']:.1f} us)")
    print(f"    keys():       {results['storage_keys_ms']:.2f} ms")
    print(f"    __len__:      {results['storage_len_us']:.1f} us")

    print(f"\n  PrefetchedSource:")
    print(f"    __len__:      {results['source_len_us']:.1f} us")
    print(f"    __getitem__:  {results['source_getitem_avg_us']:.1f} us (p99: {results['source_getitem_p99_us']:.1f} us)")

    if "dl_avg_ms" in results:
        print(f"\n  DataLoader:")
        print(f"    avg:          {results['dl_avg_ms']:.1f} ms")
        print(f"    p50:          {results['dl_p50_ms']:.1f} ms")
        print(f"    p95:          {results['dl_p95_ms']:.1f} ms")
        print(f"    p99:          {results['dl_p99_ms']:.1f} ms")
        print(f"    batches/sec:  {results['dl_batches_per_sec']:.1f}")
        print(f"\n  Training impact estimate:")
        print(f"    data_pct:     {results['data_pct_est']:.1f}%")
        print(f"    step time:    {results['step_time_est_ms']:.0f} ms")

    # ShuffleBuffer results (if present)
    if "sb_put_avg_us" in results:
        print(f"\n  ShuffleBuffer:")
        print(f"    put:          {results['sb_put_avg_us']:.1f} us (p99: {results['sb_put_p99_us']:.1f} us)")
        print(f"    sample:       {results['sb_sample_avg_us']:.1f} us (p99: {results['sb_sample_p99_us']:.1f} us)")
        print(f"    put_pct:      {results['sb_put_pct_of_decode']:.1f}% of decode time")
        print(f"    verdict:      {results['sb_gil_verdict']}")

    if results.get("bottlenecks"):
        print(f"\n  {'!'*50}")
        print("  BOTTLENECKS DETECTED:")
        for b in results["bottlenecks"]:
            print(f"    - {b}")
        print(f"  {'!'*50}")

    print(f"\n{'='*55}")


# ---------------------------------------------------------------------------
# ShuffleBuffer + GIL contention benchmark
# ---------------------------------------------------------------------------


def benchmark_shuffle_buffer(
    capacity: int = 200,
    max_frames: int = 30,
    channels: int = 3,
    height: int = 96,
    width: int = 96,
    simulated_decode_ms: float = 50.0,
) -> dict[str, Any]:
    """Benchmark ShuffleBuffer put/sample latency and GIL contention risk.

    Measures:
    - put() latency: time to copy frames into shared memory
    - sample() latency: time to read random frames from shared memory
    - put as % of simulated decode time: if >5%, GIL contention is a concern

    Args:
        capacity: Buffer capacity.
        max_frames: Max frames per episode.
        channels: Image channels.
        height: Frame height.
        width: Frame width.
        simulated_decode_ms: Assumed decode time per episode for ratio calc.
    """
    from .shuffle_buffer import ShuffleBuffer
    import random

    buf = ShuffleBuffer(
        capacity=capacity, max_frames=max_frames,
        channels=channels, height=height, width=width,
    )

    frame_bytes = max_frames * channels * height * width
    results: dict[str, Any] = {
        "frame_bytes": frame_bytes,
        "frame_mb": frame_bytes / 1e6,
    }

    # Pre-generate frames (exclude alloc from timing)
    frames = torch.randint(0, 255, (max_frames, channels, height, width), dtype=torch.uint8)

    # --- put() benchmark ---
    put_times = []
    for i in range(min(500, capacity * 2)):
        t0 = time.perf_counter()
        buf.put(i % (capacity * 10), frames)
        put_times.append((time.perf_counter() - t0) * 1e6)

    sorted_put = sorted(put_times)
    results["sb_put_avg_us"] = sum(put_times) / len(put_times)
    results["sb_put_p50_us"] = sorted_put[len(sorted_put) // 2]
    results["sb_put_p99_us"] = sorted_put[int(len(sorted_put) * 0.99)]

    # --- sample() benchmark ---
    rng = random.Random(42)
    sample_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        buf.sample(rng)
        sample_times.append((time.perf_counter() - t0) * 1e6)

    sorted_sample = sorted(sample_times)
    results["sb_sample_avg_us"] = sum(sample_times) / len(sample_times)
    results["sb_sample_p50_us"] = sorted_sample[len(sorted_sample) // 2]
    results["sb_sample_p99_us"] = sorted_sample[int(len(sorted_sample) * 0.99)]

    # --- GIL contention analysis ---
    decode_us = simulated_decode_ms * 1000
    put_pct = results["sb_put_avg_us"] / decode_us * 100
    results["sb_put_pct_of_decode"] = put_pct

    if put_pct < 2:
        results["sb_gil_verdict"] = "OK — put() is negligible vs decode"
    elif put_pct < 10:
        results["sb_gil_verdict"] = "WATCH — put() is measurable, monitor at scale"
    else:
        results["sb_gil_verdict"] = (
            "WARNING — put() takes {:.0f}% of decode time. "
            "Consider separating decode and buffer write into separate processes."
        ).format(put_pct)

    # --- Resolution scaling ---
    results["resolutions"] = {}
    for label, h, w in [("96x96", 96, 96), ("224x224", 224, 224), ("256x256", 256, 256)]:
        test_frames = torch.randint(0, 255, (max_frames, channels, h, w), dtype=torch.uint8)
        test_buf = ShuffleBuffer(capacity=10, max_frames=max_frames, channels=channels, height=h, width=w)
        times = []
        for i in range(50):
            t0 = time.perf_counter()
            test_buf.put(i, test_frames)
            times.append((time.perf_counter() - t0) * 1e6)
        avg = sum(times) / len(times)
        mb = max_frames * channels * h * w / 1e6
        pct = avg / decode_us * 100
        results["resolutions"][label] = {
            "put_avg_us": avg,
            "frame_mb": mb,
            "put_pct_of_decode": pct,
        }

    return results


def print_shuffle_buffer_report(results: dict[str, Any]) -> None:
    """Print ShuffleBuffer benchmark report."""
    print(f"\n{'='*60}")
    print("ShuffleBuffer GIL Contention Analysis")
    print(f"{'='*60}")

    print(f"\n  Frame size:     {results['frame_mb']:.2f} MB/episode")
    print(f"\n  put() latency:")
    print(f"    avg:          {results['sb_put_avg_us']:.1f} us")
    print(f"    p50:          {results['sb_put_p50_us']:.1f} us")
    print(f"    p99:          {results['sb_put_p99_us']:.1f} us")
    print(f"\n  sample() latency:")
    print(f"    avg:          {results['sb_sample_avg_us']:.1f} us")
    print(f"    p99:          {results['sb_sample_p99_us']:.1f} us")
    print(f"\n  GIL contention (vs {results['sb_put_pct_of_decode']:.1f}% of 50ms decode):")
    print(f"    Verdict:      {results['sb_gil_verdict']}")

    print(f"\n  Resolution scaling:")
    print(f"    {'Resolution':<12} {'put (us)':<12} {'Frame (MB)':<12} {'% of decode':<12}")
    print(f"    {'-'*48}")
    for label, data in results["resolutions"].items():
        print(f"    {label:<12} {data['put_avg_us']:<12.1f} {data['frame_mb']:<12.2f} {data['put_pct_of_decode']:<12.1f}%")

    print(f"\n{'='*60}")
