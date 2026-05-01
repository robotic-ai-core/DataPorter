# Streaming Parquet Prefetcher — Design & Implementation Plan

## Problem

Two projects (Mamba QAT text pre-training and ProtoWorld video world model) both
suffer from HF dataset streaming bottlenecks during training. The GPU sits idle
30-50% of the time waiting for data. Current workarounds (threads, spawn workers)
have fundamental limitations.

## Design

A **transparent Parquet prefetcher** that makes remote HF data look like a local
directory of Parquet shards. Everything above it (datasets, workers, transforms)
stays unchanged.

### Architecture

```
                    ┌─────────────────────────────┐
                    │   ParquetPrefetcher          │
                    │   (async producer, main proc)│
                    │                              │
HF Streams ───────→│  download → transform →      │
(multi-offset)      │  write Parquet shards to     │
                    │  local directory              │
                    └──────────┬──────────────────┘
                               │ local filesystem
                               │ (growing dir of .parquet files)
                               │
              ┌────────────────┴────────────────┐
              │                                  │
     ParquetTokenDataset              FastLeRobotDataset
       (text, unchanged)              (video, unchanged)
              │                                  │
         DataLoader                         DataLoader
       (num_workers=4)                  (num_workers=4)
       (fork is safe —                  (fork is safe —
        no HF state in workers)          no HF state in workers)
```

### Key Components

#### 1. `ParquetPrefetcher`

Runs in the main process. Manages async HF streams and writes Parquet shards
to a local directory.

```python
class ParquetPrefetcher:
    def __init__(
        self,
        sources: list[dict],           # HF dataset configs
        output_dir: str | Path,
        transform: Callable | None,    # e.g., tokenize_and_chunk for text
        min_shards: int = 10,          # Block training until this many ready
        max_shards: int = 50,          # Evict oldest when exceeded
        stream_shuffle_buffer: int = 10_000,  # HF-level stream shuffle
        eviction: str = "stochastic_oldest",
        max_rows_per_shard: int = 200_000,
    ):
        ...

    def start(self) -> None:
        """Start background async download + write thread."""

    def wait_for_min(self) -> None:
        """Block until min_shards are available."""

    def stop(self) -> None:
        """Stop background production."""

    @property
    def shard_count(self) -> int:
        """Number of shards currently on disk."""
```

#### 2. `GrowableParquetDataset`

Extends `ParquetTokenDataset` to periodically rescan the directory for new
shards. `__len__` reflects currently available data.

```python
class GrowableParquetDataset(ParquetTokenDataset):
    def __init__(self, data_dir, refresh_interval_seconds=30, ...):
        ...

    def refresh(self) -> None:
        """Rescan directory for new shards."""

    def __len__(self) -> int:
        return self._current_row_count  # grows as prefetcher writes
```

#### 3. Multi-offset streams for data distribution

```python
sources = [
    {"dataset": "HuggingFaceTB/smollm-corpus", "data_dir": "fineweb-edu-dedup", "offset": 0},
    {"dataset": "HuggingFaceTB/smollm-corpus", "data_dir": "fineweb-edu-dedup", "offset": 1_000_000},
    {"dataset": "HuggingFaceTB/smollm-corpus", "data_dir": "fineweb-edu-dedup", "offset": 5_000_000},
]
```

Each stream contributes to the same output directory. Shards contain docs from
different parts of the dataset, ensuring within-shard diversity.

#### 4. Four layers of shuffle

1. **HF stream shuffle** (`buffer_size=10_000`): Mixes the stream globally
2. **Multi-offset streams**: Mixes distant regions of the dataset
3. **Ring buffer eviction** (stochastic oldest): Keeps temporal diversity in the shard pool
4. **DataLoader shuffle**: Standard batch-level shuffling

### Data Distribution / Eviction

When `max_shards` is reached, evict to make room for new shards:

- **`stochastic_oldest`** (default): Remove a random shard from the oldest 50%.
  Keeps some old shards for diversity while making room for new data.
- **`fifo`**: Simple oldest-first eviction.
- **`random`**: Remove any random shard.

### Consumer Rate vs Streaming Rate

- **Streaming faster than consumption** (normal): Full directory, random access,
  zero latency. Identical to local pre-downloaded mode.
- **Consumption catches up**: Reduced diversity (fewer shards), but no crash.
  Prefetcher catches up during validation or GPU idle time.
- **Consumption exceeds streaming**: Dataset recycles existing shards (multiple
  epochs over available data). Prefetcher continues filling in background.

The dataset's `__len__` reflects currently available shards. The DataLoader's
sampler draws from what's available. No blocking after initial `min_shards` fill.

### Transform Pipeline (Dependency Injection)

Transforms are composable callables applied at the appropriate stage:

```python
# Text: transform runs in prefetcher (before Parquet write)
# Raw HF doc → tokenized chunks → Parquet shard
prefetcher = ParquetPrefetcher(
    sources=[...],
    transform=compose(tokenize_bpe8k, chunk_seq1024),
    ...
)
# Workers read pre-tokenized chunks, only need to convert to tensors

# Video: NO transform in prefetcher (download raw files)
# Workers decode video frames + augment
prefetcher = ParquetPrefetcher(
    sources=[...],
    transform=None,  # just download
    ...
)
# Workers: FastLeRobotDataset handles video decode + augment
```

The rule: **I/O-bound work in the prefetcher, CPU-bound work in workers.**
For text, tokenization is cheap enough to run in the prefetcher. For video,
decoding is heavy and should parallelize across workers.

### Integration with DataPorter

```
external/DataPorter/
├── src/dataporter/
│   ├── resumable.py               # Existing: ResumableDataLoader
│   ├── prefetcher.py              # New: ParquetPrefetcher
│   ├── growable_dataset.py        # New: GrowableParquetDataset
│   └── transforms.py             # New: compose(), common transforms
```

### Usage — Mamba QAT Text Pre-training

```python
from dataporter import ParquetPrefetcher, GrowableParquetDataset

# Start streaming in background
prefetcher = ParquetPrefetcher(
    sources=[
        {"dataset": "HuggingFaceTB/smollm-corpus", "data_dir": "fineweb-edu-dedup", "offset": 0},
        {"dataset": "HuggingFaceTB/smollm-corpus", "data_dir": "fineweb-edu-dedup", "offset": 2_000_000},
    ],
    output_dir="/tmp/cache/smollm",
    transform=compose(tokenize("bpe_8k"), chunk(seq_len=1024)),
    min_shards=10,
    max_shards=50,
    stream_shuffle_buffer=10_000,
)
prefetcher.start()
prefetcher.wait_for_min()  # blocks ~10 seconds

# Standard dataset + dataloader (unchanged)
dataset = GrowableParquetDataset("/tmp/cache/smollm", split="train")
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
```

### Usage — ProtoWorld Video World Model

```python
prefetcher = ParquetPrefetcher(
    sources=[
        {"dataset": "lerobot/pusht"},
        {"dataset": "neiltan/pusht-synthetic"},
    ],
    output_dir="/tmp/cache/pusht",
    transform=None,  # download raw episodes (parquet + mp4)
    min_shards=5,
    max_shards=100,
)
prefetcher.start()
prefetcher.wait_for_min()

# Existing FastLeRobotDataset, unchanged
dataset = FastLeRobotDataset("/tmp/cache/pusht", ...)
dataloader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=True)
```

### Testing Strategy

1. **Unit tests**: Prefetcher writes shards, GrowableDataset picks them up
2. **Eviction tests**: Verify shard count stays within bounds
3. **Distribution tests**: Verify multi-offset streams produce diverse shards
4. **Rate tests**: Consumer faster than producer, producer faster than consumer
5. **Integration**: End-to-end with DataLoader, verify no data corruption
6. **Throughput benchmark**: Compare `perf/data_pct` against local Parquet baseline

### Success Criteria

- `perf/data_pct` < 10% (vs current 51%)
- Zero code changes in existing Dataset classes
- Works on fresh vast.ai instances (no persistent cache)
- Graceful degradation when streaming is slow

### Open Questions

- Should the prefetcher support resuming from a partially filled directory?
- Should evicted shards be tracked to avoid re-downloading?
- How to handle multi-GPU (DDP) — each rank gets its own prefetcher, or shared?
- Should the async loop use `asyncio` or `concurrent.futures.ThreadPoolExecutor`?

### Reference: Current Codebase

- `autofpv/data/tokenization/pretokenize.py` — existing blocking pretokenization
- `autofpv/data/tokenization/parquet_dataset.py` — `ParquetTokenDataset`
- `autofpv/data/tokenization/stream_dataset.py` — current streaming (to be replaced)
- `autofpv/data/tokenization/chunking.py` — `TokenChunker`
- `autofpv/data/tokenization/parquet_writer.py` — `ParquetShardWriter`
- `external/DataPorter/src/dataporter/resumable.py` — `ResumableDataLoader`
- `projects/protoworld/data/datamodule.py` — ProtoWorld data loading
