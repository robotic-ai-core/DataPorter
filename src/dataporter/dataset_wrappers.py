"""Reusable dataset wrappers for multi-source data loading.

Provides lightweight wrappers that compose with any PyTorch Dataset:
- KeyFilterDataset: strips samples to a fixed set of keys (for blended collation)
- AugmentedDataset: applies per-sample transform pipeline
- SourceTagDataset: stamps a fixed string ``source_tag`` onto every sample,
  matching the convention used by ``autofpv.data.blended_text_datamodule``
  and ``autofpv.data.sample_spec``.
"""

from typing import Callable

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


class SourceTagDataset(Dataset):
    """Stamp a fixed ``source_tag`` string onto every sample.

    Mirrors the convention from ``autofpv.data._adapters.StampSourceTag``
    and the ``source_tag`` field in ``autofpv.data.sample_spec.SampleSpec`` —
    a string identifier per sample that downstream consumers can use for
    per-source loss aggregation, conditional decoder routing, or
    domain-aware augmentation.

    Used by ``BlendedLeRobotDataModule`` to tag each per-source val
    dataset with its source name, matching the streaming train path's
    in-line ``item["source_tag"] = source["name"]`` injection.

    Args:
        dataset: Inner dataset returning dict samples.
        source_tag: String label to stamp under the ``"source_tag"`` key.
            Should match a name in the parent datamodule's source list
            so consumers can map back to indices via
            ``datamodule.source_tag_to_idx``.

    Notes:
        - Stored as a Python str (not a tensor) — default_collate gathers
          per-batch into a ``list[str]`` of length B without stacking.
          Models that need integer indices for embedding lookup can
          convert via ``datamodule.source_tag_to_idx``.
        - Idempotent on re-wrap — if the inner dataset already provides
          ``source_tag``, this overwrites it (outermost wrapper wins).
    """

    def __init__(self, dataset: Dataset, source_tag: str):
        self.dataset = dataset
        self.source_tag = str(source_tag)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        item["source_tag"] = self.source_tag
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset={self.dataset}, "
            f"source_tag={self.source_tag!r})"
        )
