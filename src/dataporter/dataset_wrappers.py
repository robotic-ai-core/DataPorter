"""Reusable dataset wrappers for multi-source data loading.

Provides lightweight wrappers that compose with any PyTorch Dataset:
- KeyFilterDataset: strips samples to a fixed set of keys (for blended collation)
- AugmentedDataset: applies per-sample transform pipeline
- DomainIdDataset: stamps a fixed integer domain_id onto every sample
"""

from typing import Callable

import torch
from torch.utils.data import Dataset


class KeyFilterDataset(Dataset):
    """Dataset wrapper that strips samples to a fixed set of keys.

    Used when blending datasets with different schemas — ensures all
    samples have identical keys for the collator.

    Args:
        dataset: Inner dataset returning dict samples.
        allowed_keys: Set of keys to keep. All others are dropped.
    """

    def __init__(self, dataset: Dataset, allowed_keys: set[str]):
        self.dataset = dataset
        self.allowed_keys = allowed_keys

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        return {k: v for k, v in item.items() if k in self.allowed_keys}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset}, "
            f"allowed_keys={len(self.allowed_keys)} keys)"
        )


class AugmentedDataset(Dataset):
    """Dataset wrapper that applies per-sample transforms.

    Wraps a dataset and applies a transform pipeline to each sample.
    Multiple transforms can be composed using torchvision.transforms.Compose.

    Args:
        dataset: Inner dataset returning dict samples.
        transform: Callable that takes and returns a sample dict.
            None = passthrough.

    Example:
        >>> from torchvision.transforms import Compose
        >>> transform = Compose([MyTransform1(), MyTransform2()])
        >>> augmented = AugmentedDataset(train_dataset, transform)
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset}, "
            f"transform={self.transform})"
        )


class DomainIdDataset(Dataset):
    """Stamp a fixed ``domain_id`` integer onto every sample.

    Used by ``BlendedLeRobotDataModule`` to tag each per-source val
    dataset with its source index, matching the streaming train path's
    in-line ``item["domain_id"] = src_idx`` injection.  Models that opt
    into conditional-decoder training (Option 1 of the blend-
    distribution-conflict mitigation) read this field from the batch.

    Args:
        dataset: Inner dataset returning dict samples.
        domain_id: Integer to stamp under the ``"domain_id"`` key.

    Notes:
        - Stored as a 0-dim ``torch.long`` tensor so default_collate
          stacks it cleanly into a ``(B,)`` LongTensor.
        - Idempotent on re-wrap — if the inner dataset already provides
          ``domain_id``, this overwrites it (so the outermost wrapper
          wins, matching the user's mental model).
    """

    def __init__(self, dataset: Dataset, domain_id: int):
        self.dataset = dataset
        self.domain_id = int(domain_id)
        self._tensor = torch.tensor(self.domain_id, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["domain_id"] = self._tensor.clone()
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset}, "
            f"domain_id={self.domain_id})"
        )
