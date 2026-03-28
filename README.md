# DataPorter

PyTorch data loading utilities for seamless training resumption, streaming prefetch, and multi-modal data pipelines.

## Overview

DataPorter provides:

- **Resumable DataLoader**: Exact sample-level resume with `state_dict()` / `load_state_dict()`
- **Streaming Prefetch**: Background process downloads HF datasets to local Parquet while training reads
- **Unified Storage**: `ShardStorage` (disk), `MemoryStorage` (RAM), `SharedMemoryStorage` (cross-process tensors)
- **PrefetchedSource**: Composable data source with background producers, shuffle-from-available, and zero cache misses
- **LeRobot Integration**: Episode prefetcher, shared frame buffer, blended multi-source DataModule
- **Instrumentation**: `TimedDataLoader` for accurate fetch latency, `bench.py` and `bench_video.py` for pipeline profiling

## Installation

```bash
git submodule add https://github.com/neil-tan/DataPorter.git external/DataPorter
pip install -e external/DataPorter/
```

## Quick Start

### Resumable DataLoader

```python
from dataporter import ResumableDataLoader

# Drop-in replacement for DataLoader with exact resume
loader = ResumableDataLoader(dataset, batch_size=32, shuffle=True, num_workers=-1)

# Save / restore
state = loader.state_dict()
loader.load_state_dict(state)
```

### Text Streaming Pipeline

```python
from dataporter import TextPrefetcher, RawTextSource, TransformableDataset

# Background process streams HF docs to local Parquet shards
prefetcher = TextPrefetcher(
    output_dir="/data/cache",
    dataset="HuggingFaceTB/smollm-corpus",
    data_dir="fineweb-edu-dedup",
    offsets=[0, 2_000_000],       # parallel streams for diversity
    min_shards=5,
)
prefetcher.start()
prefetcher.wait_for_min()

# Workers tokenize in parallel (no single-threaded bottleneck)
source = RawTextSource("/data/cache", max_shards=200)
dataset = TransformableDataset(source, my_tokenize_transform)
loader = ResumableDataLoader(dataset, batch_size=32, num_workers=-1)
```

### Video Frame Pipeline (LeRobot)

```python
from dataporter import SharedMemoryStorage, PrefetchedSource

# Shared memory buffer — visible to all DataLoader workers (zero-copy)
storage = SharedMemoryStorage(
    capacity=200, max_frames=50, channels=3, height=96, width=96,
)

# Background process decodes video frames into shared buffer
source = PrefetchedSource(storage, producers=[decode_episodes], shuffle_available=True)
source.start()
source.wait_for_min()

# DataLoader workers read from shared memory — no cache misses
dataset = TransformableDataset(source, augment_transform)
loader = ResumableDataLoader(dataset, batch_size=256, num_workers=-1)
```

### LeRobot Dataset Download

```python
from dataporter import LeRobotPrefetcher

# Downloads entire dataset via snapshot_download (one batched HF API call)
prefetcher = LeRobotPrefetcher(
    repo_id="lerobot/pusht",
    output_dir="/data/pusht",
    min_shards=50,
)
prefetcher.start()
prefetcher.wait_for_min()
```

## Architecture

```
Source (HF Hub / local) → Prefetcher (background process) → Storage → PrefetchedSource → TransformableDataset → DataLoader
```

### Storage Backends

| Backend | Use Case | Cross-process | Eviction |
|---|---|---|---|
| `ShardStorage` | Text (Parquet on disk) | Yes (filesystem) | Deferred (safe for readers) |
| `MemoryStorage` | Small data, tests | No | LRU |
| `SharedMemoryStorage` | Video frames (multi-worker) | Yes (`share_memory_()`) | Ring buffer FIFO |

### Prefetchers

| Prefetcher | Data Type | Download Method |
|---|---|---|
| `TextPrefetcher` | Text (HF streaming) | Parallel offset streams → Parquet shards |
| `LeRobotPrefetcher` | Video (LeRobot episodes) | `snapshot_download` (batched) |

Both run in separate processes (forkserver) to avoid GIL contention with CUDA training.

### Key Features

- **Dual-threshold buffer**: Producers pause at high-water, resume at low-water. Smooth flow control, no bursting.
- **Atomic shard writes**: `.tmp` → `os.rename()` → `.parquet`. Readers never see partial files.
- **Deferred eviction**: Shards scheduled for deletion, executed after file handles close. Zero race conditions.
- **Scientific resumption**: `state_dict()` pins shard list + DataLoader position. Deterministic resume.
- **`num_workers=-1`**: Scales with CPU cores (`ceil(cores/8)`). Prevents GPU starvation on cloud machines.
- **HF rate limiter**: Shared token bucket across all HF API calls. Prevents 429 rate limit errors.
- **`return_uint8`**: Video frames stay uint8 through DataLoader (7x faster batches). Convert on GPU.

## Instrumentation

```python
from dataporter import TimedDataLoader

loader = TimedDataLoader(base_loader)
for batch in loader:
    train_step(batch)
    print(f"Fetch: {loader.fetch_ema_ms:.1f}ms")  # actual DataLoader __next__ time
```

```python
from dataporter.bench import benchmark_dataloader
results = benchmark_dataloader(loader, n_batches=200)
# → avg_ms, p50_ms, p95_ms, p99_ms, data_pct_est, tokens_per_sec

from dataporter.bench_video import benchmark_video_pipeline
results = benchmark_video_pipeline(capacity=200)
# → storage_get_us, source_getitem_us, dl_avg_ms, bottlenecks
```

## Notes

- Works as a drop-in replacement for `torch.utils.data.DataLoader`
- Save/restore with `state_dict()` / `load_state_dict()` for exact resume
- Optional: call `set_epoch(n)` for per-epoch shuffle variation
- PrefetchedSource producers always run as processes (no thread mode — avoids 60% GPU throughput loss from GIL contention)
