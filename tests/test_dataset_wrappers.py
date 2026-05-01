"""Tests for KeyFilterDataset and AugmentedDataset."""

import pytest
import torch
from torch.utils.data import Dataset

from dataporter.dataset_wrappers import AugmentedDataset, KeyFilterDataset


class DictDataset(Dataset):
    """Simple dict dataset for testing."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@pytest.fixture
def sample_dataset():
    return DictDataset([
        {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)},
        {"a": torch.tensor(4), "b": torch.tensor(5), "c": torch.tensor(6)},
        {"a": torch.tensor(7), "b": torch.tensor(8), "c": torch.tensor(9)},
    ])


class TestKeyFilterDataset:
    def test_strips_extra_keys(self, sample_dataset):
        filtered = KeyFilterDataset(sample_dataset, {"a", "b"})
        item = filtered[0]
        assert set(item.keys()) == {"a", "b"}
        assert "c" not in item

    def test_preserves_values(self, sample_dataset):
        filtered = KeyFilterDataset(sample_dataset, {"a", "c"})
        item = filtered[1]
        assert item["a"].item() == 4
        assert item["c"].item() == 6

    def test_len_delegates(self, sample_dataset):
        filtered = KeyFilterDataset(sample_dataset, {"a"})
        assert len(filtered) == 3

    def test_empty_allowed_keys(self, sample_dataset):
        filtered = KeyFilterDataset(sample_dataset, set())
        assert filtered[0] == {}

    def test_all_keys_allowed(self, sample_dataset):
        filtered = KeyFilterDataset(sample_dataset, {"a", "b", "c"})
        item = filtered[0]
        assert set(item.keys()) == {"a", "b", "c"}


class TestAugmentedDataset:
    def test_applies_transform(self, sample_dataset):
        def double_a(item):
            return {**item, "a": item["a"] * 2}

        aug = AugmentedDataset(sample_dataset, double_a)
        assert aug[0]["a"].item() == 2
        assert aug[1]["a"].item() == 8

    def test_none_transform_passthrough(self, sample_dataset):
        aug = AugmentedDataset(sample_dataset, None)
        assert aug[0]["a"].item() == 1

    def test_len_delegates(self, sample_dataset):
        aug = AugmentedDataset(sample_dataset)
        assert len(aug) == 3

    def test_compose_transforms(self, sample_dataset):
        from torchvision.transforms import Compose

        def add_one(item):
            return {**item, "a": item["a"] + 1}

        def double(item):
            return {**item, "a": item["a"] * 2}

        aug = AugmentedDataset(sample_dataset, Compose([add_one, double]))
        # (1 + 1) * 2 = 4
        assert aug[0]["a"].item() == 4


class TestCombined:
    def test_filter_then_augment(self, sample_dataset):
        """KeyFilterDataset + AugmentedDataset compose correctly."""
        filtered = KeyFilterDataset(sample_dataset, {"a", "b"})
        aug = AugmentedDataset(filtered, lambda item: {
            k: v + 10 for k, v in item.items()
        })
        item = aug[0]
        assert set(item.keys()) == {"a", "b"}
        assert item["a"].item() == 11
        assert item["b"].item() == 12
