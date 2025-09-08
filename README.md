# DataPorter

PyTorch data loading utilities for seamless training resumption and memory optimization. A drop-in replacement for PyTorch DataLoader with checkpoint/resume capabilities.

## Overview

Resumable PyTorch dataloading with exact position restore, optional memory optimization, and fault-tolerant dataset wrapping.

### Key Features

- **Exact Resume**: Resume training from the precise sample where interrupted
- **Memory Optimization**: Reduce memory usage by 50-87% with dtype conversions
- **Fault Tolerance**: Automatic detection and handling of corrupted data samples
- **Automatic Strategy**: Unified resumption strategy with automatic environment detection
- **Drop-in Replacement**: Compatible with existing PyTorch DataLoader code
- **Production Ready**: Battle-tested in large-scale training environments

## Installation

### As a Git Submodule

```bash
# Add as submodule
git submodule add https://github.com/neil-tan/DataPorter.git lib/DataPorter

# Install in editable mode
pip install -e lib/DataPorter/
```

### Direct Installation

```bash
# From source
git clone https://github.com/neil-tan/DataPorter.git
cd DataPorter
pip install -e .
```

## Minimal Usage

```python
from dataporter import ResumableDataLoader

dataloader = ResumableDataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        train_step(batch)
```

Resume:

```python
ckpt = torch.load('checkpoint.pt')
dataloader = ResumableDataLoader(dataset, batch_size=32, shuffle=True)
dataloader.load_state_dict(ckpt['dataloader'])
```

## Notes

- Works as a drop-in replacement for `torch.utils.data.DataLoader`
- Save/restore with `state_dict()` / `load_state_dict()` for exact resume
- Optional: call `set_epoch(n)` if you want a different shuffle per epoch or per‑epoch counters; not required for scientific resumption