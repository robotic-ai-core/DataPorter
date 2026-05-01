"""Tests for CachedDataset."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataporter import CachedDataset, get_cache_root


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 100, shape: tuple = (3, 32, 32)):
        self.size = size
        self.shape = shape
        # Generate deterministic data
        torch.manual_seed(42)
        self.data = torch.randn(size, *shape)
        self.labels = torch.arange(size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": self.data[idx],
            "label": self.labels[idx],
            "metadata": {"index": idx, "name": f"sample_{idx}"},
        }


class NestedDataset(Dataset):
    """Dataset with nested tensor structure."""

    def __init__(self, size: int = 50):
        self.size = size
        torch.manual_seed(42)
        self.rgb = torch.randn(size, 3, 64, 64)
        self.depth = torch.randn(size, 1, 64, 64)
        self.pose = torch.randn(size, 4, 4)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "rgb": self.rgb[idx],
            "depth": self.depth[idx],
            "pose": self.pose[idx],
            "mask": {
                "rgb": torch.ones(1),
                "depth": torch.ones(1),
            },
            "info": {
                "dataset": "test",
                "index": idx,
            },
        }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCachedDatasetBasic:
    """Basic tests for CachedDataset."""

    def test_simple_caching(self, temp_cache_dir):
        """Test basic caching functionality."""
        dataset = SimpleDataset(size=10)
        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        # Check cache was created
        assert cached.cache_dir.exists()
        assert (cached.cache_dir / "cache_spec.json").exists()
        assert (cached.cache_dir / "metadata.json").exists()

        # Check samples match
        for i in range(len(dataset)):
            original = dataset[i]
            cached_sample = cached[i]

            assert torch.allclose(original["image"], cached_sample["image"])
            assert original["label"] == cached_sample["label"]

    def test_cache_reuse(self, temp_cache_dir):
        """Test that cache is reused on second instantiation."""
        dataset = SimpleDataset(size=10)
        cache_spec = {"version": 1, "test": "reuse"}

        # First instantiation - should populate cache
        cached1 = CachedDataset(
            dataset=dataset,
            cache_spec=cache_spec,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )
        cache_key1 = cached1.cache_key

        # Second instantiation - should reuse cache
        cached2 = CachedDataset(
            dataset=dataset,
            cache_spec=cache_spec,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )
        cache_key2 = cached2.cache_key

        assert cache_key1 == cache_key2

        # Verify data matches
        assert torch.allclose(cached1[0]["image"], cached2[0]["image"])

    def test_cache_invalidation_on_version_change(self, temp_cache_dir):
        """Test that cache is invalidated when version changes."""
        dataset = SimpleDataset(size=10)

        # Create cache with version 1
        cached1 = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )
        key1 = cached1.cache_key

        # Create cache with version 2 - should have different key
        cached2 = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 2},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )
        key2 = cached2.cache_key

        assert key1 != key2

    def test_len(self, temp_cache_dir):
        """Test that length is preserved."""
        dataset = SimpleDataset(size=25)
        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        assert len(cached) == len(dataset) == 25


class TestCachedDatasetNested:
    """Tests for nested tensor structures."""

    def test_nested_tensors(self, temp_cache_dir):
        """Test caching of nested tensor structures."""
        dataset = NestedDataset(size=10)
        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        for i in range(len(dataset)):
            original = dataset[i]
            cached_sample = cached[i]

            assert torch.allclose(original["rgb"], cached_sample["rgb"])
            assert torch.allclose(original["depth"], cached_sample["depth"])
            assert torch.allclose(original["pose"], cached_sample["pose"])
            assert torch.allclose(original["mask"]["rgb"], cached_sample["mask"]["rgb"])
            assert torch.allclose(original["mask"]["depth"], cached_sample["mask"]["depth"])


class TestCachedDatasetTransforms:
    """Tests for transforms."""

    def test_transform_applied(self, temp_cache_dir):
        """Test that transforms are applied before caching."""
        dataset = SimpleDataset(size=10, shape=(3, 64, 64))

        def resize_transform(sample):
            # Simulate resize by taking center crop
            img = sample["image"]
            sample["image"] = img[:, 16:48, 16:48]  # 32x32
            return sample

        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            transforms=resize_transform,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        # Check transformed shape
        sample = cached[0]
        assert sample["image"].shape == (3, 32, 32)

    def test_different_transforms_different_cache(self, temp_cache_dir):
        """Test that different transforms produce different cache keys."""
        dataset = SimpleDataset(size=10)

        def transform_a(sample):
            sample["image"] = sample["image"] * 2
            return sample

        def transform_b(sample):
            sample["image"] = sample["image"] * 3
            return sample

        cached_a = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            transforms=transform_a,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        cached_b = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            transforms=transform_b,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        # Different transforms should produce different cache keys
        assert cached_a.cache_key != cached_b.cache_key


class TestCachedDatasetDtype:
    """Tests for dtype handling."""

    def test_float16_caching(self, temp_cache_dir):
        """Test caching with float16 conversion."""
        dataset = SimpleDataset(size=10)

        def to_float16(sample):
            sample["image"] = sample["image"].half()
            return sample

        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1, "dtype": "float16"},
            transforms=to_float16,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        sample = cached[0]
        assert sample["image"].dtype == torch.float16

    def test_mixed_dtypes(self, temp_cache_dir):
        """Test caching with mixed dtypes."""
        dataset = SimpleDataset(size=10)

        def mixed_dtype_transform(sample):
            sample["image"] = sample["image"].half()  # float16
            sample["label"] = sample["label"].long()  # int64
            return sample

        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            transforms=mixed_dtype_transform,
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        sample = cached[0]
        assert sample["image"].dtype == torch.float16
        assert sample["label"].dtype == torch.int64


class TestCachedDatasetCacheInfo:
    """Tests for cache info and management."""

    def test_get_cache_info(self, temp_cache_dir):
        """Test cache info retrieval."""
        dataset = SimpleDataset(size=10)
        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        info = cached.get_cache_info()

        assert "cache_key" in info
        assert "cache_dir" in info
        assert info["cache_exists"] is True
        assert info["n_samples"] == 10
        assert "cache_size_mb" in info
        assert info["cache_size_mb"] > 0

    def test_clear_cache(self, temp_cache_dir):
        """Test cache clearing."""
        dataset = SimpleDataset(size=10)
        cached = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        assert cached.cache_dir.exists()

        cached.clear_cache()

        assert not cached.cache_dir.exists()


class TestCacheKeyHashing:
    """Tests for cache key computation."""

    def test_same_spec_same_key(self, temp_cache_dir):
        """Test that same spec produces same key."""
        dataset = SimpleDataset(size=10)

        cached1 = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1, "resolution": (160, 160)},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        cached2 = CachedDataset(
            dataset=dataset,
            cache_spec={"version": 1, "resolution": (160, 160)},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        assert cached1.cache_key == cached2.cache_key

    def test_different_dataset_size_different_key(self, temp_cache_dir):
        """Test that different dataset sizes produce different keys."""
        dataset1 = SimpleDataset(size=10)
        dataset2 = SimpleDataset(size=20)

        cached1 = CachedDataset(
            dataset=dataset1,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        cached2 = CachedDataset(
            dataset=dataset2,
            cache_spec={"version": 1},
            cache_dir=temp_cache_dir,
            show_progress=False,
        )

        assert cached1.cache_key != cached2.cache_key


class TestCacheRoot:
    """Tests for cache root functionality."""

    def test_get_cache_root_default(self, monkeypatch):
        """Test default cache root."""
        monkeypatch.delenv("HF_HOME", raising=False)
        root = get_cache_root()
        assert "huggingface" in str(root)
        assert "dataporter_cache" in str(root)

    def test_get_cache_root_custom(self, monkeypatch, temp_cache_dir):
        """Test custom cache root via HF_HOME."""
        monkeypatch.setenv("HF_HOME", str(temp_cache_dir))
        root = get_cache_root()
        assert str(temp_cache_dir) in str(root)
