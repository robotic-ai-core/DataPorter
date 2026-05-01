"""Dataset with worker-side transforms over a random-access source.

Reads raw data from any source (Parquet, memory, etc.) and applies
a transform in each DataLoader worker.  The transform handles
tokenization, chunking, augmentation, etc.

This single class supports both pre-training (pack multiple docs)
and SFT (single example with loss mask) — just swap the transform.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch
from torch.utils.data import Dataset


class DataSource(Protocol):
    """Protocol for a random-access data source."""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict[str, Any]: ...


# Transform: (source, idx) -> dict of tensors
WorkerTransform = Callable[["DataSource", int], dict[str, torch.Tensor]]


class TransformableDataset(Dataset):
    """Dataset with worker-side transforms.

    The source provides raw data via random access.  The transform
    runs in each DataLoader worker, parallelizing CPU-heavy work
    (tokenization, decoding, augmentation).

    Args:
        source: Data source with ``__len__`` and ``__getitem__``.
        transform: Callable ``(source, idx) -> {key: Tensor, ...}``.
    """

    def __init__(self, source: DataSource, transform: WorkerTransform):
        self._source = source
        self._transform = transform

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._transform(self._source, idx)
