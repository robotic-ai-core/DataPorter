# DataPorter

PyTorch data loading utilities for seamless training resumption and memory optimization. A drop-in replacement for PyTorch DataLoader with checkpoint/resume capabilities.

## Features

- **Exact Resume** - Resume training from the precise sample where interrupted
- **Memory Optimization** - Reduce memory usage by 50-87% with dtype conversions
- **Fault Tolerance** - Automatic detection and handling of corrupted data samples
- **Automatic Strategy** - Unified resumption strategy with automatic environment detection
- **Drop-in Replacement** - Compatible with existing PyTorch DataLoader code
- **Production Ready** - Battle-tested in large-scale training environments

## Installation

### As a Git Submodule

```bash
# Add as submodule
git submodule add https://github.com/neil-tan/DataPorter.git external/DataPorter

# Install in editable mode
pip install -e external/DataPorter/
```

### Direct Installation

```bash
# From source
git clone https://github.com/neil-tan/DataPorter.git
cd DataPorter
pip install -e .
```

---

## TL;DR (Quickstart)

### Basic Resumable DataLoader

```python
from dataporter import ResumableDataLoader

dataloader = ResumableDataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        train_step(batch)
```

### Save and Resume

```python
# Save state
state = dataloader.state_dict()
torch.save({'dataloader': state, 'model': model.state_dict()}, 'checkpoint.pt')

# Resume
ckpt = torch.load('checkpoint.pt')
dataloader = ResumableDataLoader(dataset, batch_size=32, shuffle=True)
dataloader.load_state_dict(ckpt['dataloader'])
# Continues from exact sample position
```

### Memory Optimization with Dtype Conversion

```python
from dataporter import ResumableDataLoader, KeyBasedDtypeConverter

# Define dtype conversions
converter = KeyBasedDtypeConverter([
    {"path": "observation.image", "dtype": "float16"},
    {"path": "action", "dtype": "float32"},
])

dataloader = ResumableDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    converter=converter,
)
# Images now use 50% less memory!
```

---

## ResumableDataLoader

Drop-in replacement for `torch.utils.data.DataLoader` with checkpoint/resume capabilities.

### Constructor Parameters

```python
ResumableDataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    prefetch_factor=2,           # Only if num_workers > 0
    persistent_workers=False,    # Only if num_workers > 0
    converter=None,              # Optional KeyBasedDtypeConverter
    # ... all other DataLoader parameters
)
```

### State Management

```python
# Save current position
state = dataloader.state_dict()
# {'epoch': 0, 'samples_seen': 1024, 'seed': 42, ...}

# Restore position
dataloader.load_state_dict(state)

# Set epoch for different shuffle per epoch
dataloader.set_epoch(epoch)
```

### Automatic Environment Detection

ResumableDataLoader automatically detects:
- **Single-node training**: Uses `ResumableSampler`
- **Distributed training**: Uses `ResumableDistributedSampler`
- **No action needed** - just use `ResumableDataLoader` everywhere

---

## KeyBasedDtypeConverter

Converts tensor dtypes in batches based on key paths. Reduces memory usage significantly for large tensors.

### Configuration Formats

**List Format (YAML-friendly):**

```python
converter = KeyBasedDtypeConverter([
    {"path": "observation.image", "dtype": "float16"},
    {"path": "observation.depth", "dtype": "float16"},
    {"path": "action", "dtype": "float32"},
])
```

**Dict Format (Python):**

```python
converter = KeyBasedDtypeConverter({
    "observation.image": "float16",
    "observation.depth": "float16",
    "action": "float32",
})
```

### Supported Dtypes

| Category | Types |
|----------|-------|
| Floating Point | `float16`, `bfloat16`, `float32`, `float64` |
| Integer | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16` |
| Boolean | `bool` |

### Memory Savings

| Conversion | Memory Savings |
|------------|----------------|
| `float32` → `float16` | 50% |
| `float32` → `bfloat16` | 50% |
| `float32` → `int8` | 75% |
| `float64` → `float32` | 50% |

### Nested Key Access

The converter supports dot-notation for nested dictionaries:

```python
batch = {
    "observation": {
        "image": tensor_float32,  # Will be converted to float16
        "state": tensor_float32,  # Unchanged
    },
    "action": tensor_float32,     # Will be converted to float32
}

converter = KeyBasedDtypeConverter([
    {"path": "observation.image", "dtype": "float16"},
])
```

---

## Integration with Lightning DataModule

### Basic Integration

```python
import lightning.pytorch as L
from dataporter import ResumableDataLoader, KeyBasedDtypeConverter

class MyDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        dtype_conversions: list | dict | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype_conversions = dtype_conversions

    def train_dataloader(self):
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True,
            'persistent_workers': self.num_workers > 0,
        }

        # Add dtype converter if configured
        if self.dtype_conversions is not None:
            if isinstance(self.dtype_conversions, list):
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)
            else:
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)

        return ResumableDataLoader(self.train_dataset, **loader_kwargs)
```

### YAML Configuration

```yaml
data:
  class_path: myproject.data.MyDataModule
  init_args:
    batch_size: 256
    num_workers: 8
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
      - path: "observation.depth"
        dtype: "float16"
```

---

## FaultTolerantWrapper

Wraps datasets to handle corrupted or problematic samples gracefully.

```python
from dataporter import FaultTolerantWrapper

# Wrap dataset with fault tolerance
wrapped_dataset = FaultTolerantWrapper(
    dataset,
    skip_nan=True,           # Skip samples with NaN values
    skip_inf=True,           # Skip samples with Inf values
    max_consecutive_errors=5, # Max errors before raising
    retry_with_next=True,    # Try next sample on error
)

dataloader = ResumableDataLoader(wrapped_dataset, batch_size=32)
```

### Error Handling Options

| Parameter | Description |
|-----------|-------------|
| `skip_nan` | Skip samples containing NaN values |
| `skip_inf` | Skip samples containing Inf values |
| `max_consecutive_errors` | Maximum consecutive errors before raising |
| `retry_with_next` | On error, try next sample instead of raising |
| `error_callback` | Custom callback for error handling |

---

## Complete Integration Example

Here's a complete example based on the ProtoWorld project:

```python
# myproject/data/datamodule.py
import lightning.pytorch as L
from dataporter import ResumableDataLoader, KeyBasedDtypeConverter

class LeRobotDataModule(L.LightningDataModule):
    def __init__(
        self,
        repo_id: str,
        batch_size: int = 256,
        num_workers: int = 8,
        context_length: int = 4,
        auto_steps: int = 2,
        dtype_conversions: list | dict | None = None,
        augmentation: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_length = context_length
        self.auto_steps = auto_steps
        self.dtype_conversions = dtype_conversions
        self.augmentation = augmentation

    def setup(self, stage=None):
        # Load dataset and create train/val splits
        self.train_dataset = self._create_dataset(split="train")
        self.val_dataset = self._create_dataset(split="val")

    def train_dataloader(self):
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True,
            'persistent_workers': self.num_workers > 0,
            'prefetch_factor': 2 if self.num_workers > 0 else None,
        }

        # Memory optimization via dtype conversion
        if self.dtype_conversions:
            if isinstance(self.dtype_conversions, list):
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)
            else:
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)

        return ResumableDataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': False,
        }

        if self.dtype_conversions:
            if isinstance(self.dtype_conversions, list):
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)
            else:
                loader_kwargs['converter'] = KeyBasedDtypeConverter(self.dtype_conversions)

        return ResumableDataLoader(self.val_dataset, **loader_kwargs)
```

### Configuration

```yaml
# configs/training.yaml
data:
  class_path: myproject.data.LeRobotDataModule
  init_args:
    repo_id: lerobot/pusht
    batch_size: 256
    num_workers: 8
    context_length: 4
    auto_steps: 2
    dtype_conversions:
      - path: "observation.image"
        dtype: "float16"
    augmentation:
      temporal:
        enabled: true
        p: 0.5
      mirror:
        enabled: true
        p: 0.3
```

### Memory Impact

With the above configuration:
- **Image tensor** (batch=256, frames=6, channels=3, height=96, width=96):
  - `float32`: ~452 MB per batch
  - `float16`: ~226 MB per batch
  - **Savings: 226 MB per batch**

- **DataLoader prefetch buffers** (8 workers × 2 prefetch):
  - Total potential savings: **~3.6 GB**

---

## Distributed Training

ResumableDataLoader works seamlessly with distributed training:

```python
# No changes needed - automatic detection
dataloader = ResumableDataLoader(dataset, batch_size=32, shuffle=True)

# Internally uses ResumableDistributedSampler when:
# - torch.distributed.is_initialized() returns True
# - Multiple processes detected
```

### Manual Distributed Configuration

```python
from dataporter import ResumableDataLoader
from dataporter.samplers import ResumableDistributedSampler

sampler = ResumableDistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

dataloader = ResumableDataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
)
```

---

## API Reference

### ResumableDataLoader

| Method | Description |
|--------|-------------|
| `state_dict()` | Returns checkpoint state dict |
| `load_state_dict(state)` | Restores from checkpoint |
| `set_epoch(epoch)` | Set epoch for shuffle seeding |

### KeyBasedDtypeConverter

| Method | Description |
|--------|-------------|
| `__call__(batch)` | Convert batch tensors |
| `get_conversions()` | Return conversion rules |

### FaultTolerantWrapper

| Method | Description |
|--------|-------------|
| `__getitem__(idx)` | Get item with error handling |
| `get_error_summary()` | Return error statistics |
| `reset_error_counts()` | Reset error counters |

---

## Notes

- Works as a drop-in replacement for `torch.utils.data.DataLoader`
- Save/restore with `state_dict()` / `load_state_dict()` for exact resume
- Optional: call `set_epoch(n)` for different shuffle per epoch
- Dtype conversions happen at batch collation time (after batching, before GPU transfer)
- Automatic distributed detection - no configuration needed
