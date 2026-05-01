"""
CachedDataset - Disk-cached dataset wrapper with memory-mapped storage.

Provides transparent caching of dataset samples to disk, eliminating redundant
preprocessing on subsequent runs. Uses memory-mapped numpy arrays for efficient
random access without loading entire dataset into memory.

Key features:
- Automatic cache key hashing from dataset spec and transforms
- Memory-mapped storage for fast random access
- Eager population with progress bar
- Cache validation on load (spec matching)
- Respects HF_HOME environment variable for cache location

Example:
    ```python
    from dataporter import CachedDataset

    # Wrap any dataset with caching
    cached = CachedDataset(
        dataset=MyDataset(...),
        cache_spec={
            "version": 1,
            "resolution": (160, 160),
            "dtype": "float16",
        },
        transforms=my_transform_fn,  # Applied before caching
    )

    # First run: populates cache (shows progress bar)
    # Subsequent runs: loads from cache instantly
    sample = cached[0]
    ```
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _IndexedTransformDataset(Dataset):
    """Helper dataset that applies transforms and returns (idx, sample) pairs.

    Used internally by CachedDataset for parallel cache population.
    """

    def __init__(self, source: Dataset, transforms: Optional[Callable] = None):
        self.source = source
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Tuple[int, Any]:
        sample = self.source[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return idx, sample


def _identity_collate(batch: List) -> List:
    """Identity collate function that preserves batch as-is.

    Used by CachedDataset for parallel cache population.
    """
    return batch


def get_cache_root() -> Path:
    """Get the root cache directory, respecting HF_HOME environment variable.

    Returns:
        Path to cache root directory
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf_home) / "datasets" / "dataporter_cache"


def _tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def _numpy_to_tensor(array: np.ndarray) -> Tensor:
    """Convert a numpy array to PyTorch tensor."""
    # Handle numpy scalars (when indexing mmap with single index)
    if isinstance(array, (np.generic, np.ndarray)) and array.ndim == 0:
        return torch.tensor(array.item())
    # Ensure we have a proper array (not a scalar)
    array = np.asarray(array)
    return torch.from_numpy(array.copy())  # copy() to handle non-contiguous mmap


def _get_numpy_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """Map PyTorch dtype to numpy dtype."""
    dtype_map = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bfloat16: np.float32,  # No bfloat16 in numpy, use float32
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    return dtype_map.get(torch_dtype, np.float32)


def _serialize_for_hash(obj: Any) -> str:
    """Serialize an object to a string for hashing.

    Handles common types including tensors, numpy arrays, and nested structures.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return json.dumps(obj)
    elif isinstance(obj, (list, tuple)):
        return json.dumps([_serialize_for_hash(x) for x in obj])
    elif isinstance(obj, dict):
        return json.dumps({k: _serialize_for_hash(v) for k, v in sorted(obj.items())})
    elif isinstance(obj, (Tensor, np.ndarray)):
        return f"array:shape={tuple(obj.shape)},dtype={obj.dtype}"
    elif hasattr(obj, '__class__'):
        # For objects, use class name and repr
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}:{repr(obj)[:200]}"
    else:
        return str(obj)[:200]


class CachedDataset(Dataset):
    """Dataset wrapper that caches processed samples to disk.

    Wraps any PyTorch Dataset and caches the output of __getitem__ calls
    to memory-mapped numpy arrays on disk. On subsequent runs with the
    same cache_spec, samples are loaded directly from cache.

    The cache key is computed from:
    - Dataset class name and module
    - Dataset length
    - cache_spec dictionary (user-provided, should include version)
    - transforms repr (if provided)

    Storage format:
    - Each tensor field is stored as a separate .npy file (memory-mapped)
    - Metadata stored in cache_spec.json
    - Non-tensor fields stored in metadata.json

    Args:
        dataset: The source dataset to wrap
        cache_spec: Dict of cache parameters for hashing. Should include
            a 'version' key that you bump to invalidate cache.
        transforms: Optional callable applied to samples before caching.
            If None, samples are cached as-is from the source dataset.
        cache_dir: Override cache directory. If None, uses HF_HOME/datasets/dataporter_cache/
        eager: If True, populate entire cache on first access. If False, cache lazily.
        num_workers: Number of workers for eager population (0 = main process)
        show_progress: Show progress bar during eager population

    Example:
        ```python
        cached = CachedDataset(
            dataset=TartanAirDataset(scenes=["abandonedfactory"], resolution=(640, 480)),
            cache_spec={
                "version": 1,
                "resolution": (160, 160),
                "dtype": "float16",
            },
            transforms=lambda x: resize_and_convert(x, (160, 160), torch.float16),
        )
        ```
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_spec: Dict[str, Any],
        transforms: Optional[Callable[[Any], Any]] = None,
        cache_dir: Optional[Path] = None,
        eager: bool = True,
        num_workers: int = 0,
        show_progress: bool = True,
    ):
        self.source_dataset = dataset
        self.cache_spec = cache_spec
        self.transforms = transforms
        self.eager = eager
        self.num_workers = num_workers
        self.show_progress = show_progress

        # Compute cache key
        self._cache_key = self._compute_cache_key()

        # Set up cache directory
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir) / self._cache_key
        else:
            self._cache_dir = get_cache_root() / self._cache_key

        # Cache state
        self._initialized = False
        self._mmap_arrays: Dict[str, np.ndarray] = {}
        self._tensor_fields: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._sample_metadata: List[Dict[str, Any]] = []

        # Initialize cache
        self._initialize_cache()

    def _compute_cache_key(self) -> str:
        """Compute a unique cache key from dataset and spec."""
        key_parts = {
            "dataset_class": f"{self.source_dataset.__class__.__module__}.{self.source_dataset.__class__.__name__}",
            "dataset_len": len(self.source_dataset),
            "cache_spec": self.cache_spec,
        }

        # Add transforms repr if provided
        if self.transforms is not None:
            key_parts["transforms"] = repr(self.transforms)[:500]

        # Try to get additional dataset info if available
        if hasattr(self.source_dataset, 'get_cache_key'):
            key_parts["dataset_cache_key"] = self.source_dataset.get_cache_key()
        elif hasattr(self.source_dataset, '__dict__'):
            # Filter to simple types for hashing
            simple_attrs = {}
            for k, v in self.source_dataset.__dict__.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, (str, int, float, bool, tuple, list)):
                    simple_attrs[k] = v
            if simple_attrs:
                key_parts["dataset_attrs"] = simple_attrs

        # Serialize and hash
        serialized = _serialize_for_hash(key_parts)
        hash_value = hashlib.sha256(serialized.encode()).hexdigest()[:16]

        # Create human-readable prefix
        class_name = self.source_dataset.__class__.__name__.lower()
        version = self.cache_spec.get("version", 0)

        return f"{class_name}_v{version}_{hash_value}"

    def _initialize_cache(self):
        """Initialize cache - either load existing or create new."""
        spec_file = self._cache_dir / "cache_spec.json"

        if spec_file.exists():
            # Try to load existing cache
            if self._validate_and_load_cache():
                logger.info(f"Loaded cache from {self._cache_dir}")
                self._initialized = True
                return
            else:
                logger.warning(f"Cache validation failed, rebuilding: {self._cache_dir}")
                self._clear_cache()

        # Create new cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self.eager:
            self._populate_cache()

        self._initialized = True

    def _validate_and_load_cache(self) -> bool:
        """Validate existing cache and load memory maps if valid.

        Returns:
            True if cache is valid and loaded, False otherwise
        """
        spec_file = self._cache_dir / "cache_spec.json"
        metadata_file = self._cache_dir / "metadata.json"

        try:
            with open(spec_file, 'r') as f:
                saved_spec = json.load(f)

            # Check if spec matches
            if saved_spec.get("cache_spec") != self.cache_spec:
                logger.warning("Cache spec mismatch")
                return False

            if saved_spec.get("dataset_len") != len(self.source_dataset):
                logger.warning("Dataset length mismatch")
                return False

            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                    self._tensor_fields = self._metadata.get("tensor_fields", [])
                    self._sample_metadata = self._metadata.get("sample_metadata", [])

            # Load memory-mapped arrays
            for field in self._tensor_fields:
                npy_file = self._cache_dir / f"{field}.npy"
                if not npy_file.exists():
                    logger.warning(f"Missing cache file: {npy_file}")
                    return False
                self._mmap_arrays[field] = np.load(npy_file, mmap_mode='r')

            return True

        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            return False

    def _clear_cache(self):
        """Clear existing cache directory."""
        import shutil
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._mmap_arrays = {}
        self._tensor_fields = []
        self._metadata = {}
        self._sample_metadata = []

    def _populate_cache(self):
        """Populate the entire cache eagerly using parallel data loading."""
        from torch.utils.data import DataLoader

        n_samples = len(self.source_dataset)

        if n_samples == 0:
            logger.warning("Source dataset is empty, nothing to cache")
            return

        logger.info(f"Populating cache with {n_samples} samples using {self.num_workers} workers...")

        # Create a wrapper dataset that applies transforms and returns (idx, sample)
        indexed_dataset = _IndexedTransformDataset(self.source_dataset, self.transforms)

        # Get first sample to determine structure (single-threaded for safety)
        _, first_sample = indexed_dataset[0]

        # Analyze sample structure
        tensor_info = self._analyze_sample_structure(first_sample)
        self._tensor_fields = list(tensor_info.keys())

        # Create memory-mapped arrays for each tensor field
        arrays = {}
        for field, info in tensor_info.items():
            shape = (n_samples,) + info["shape"]
            dtype = info["numpy_dtype"]
            npy_file = self._cache_dir / f"{field}.npy"

            # Create empty mmap file
            arr = np.lib.format.open_memmap(
                npy_file,
                mode='w+',
                dtype=dtype,
                shape=shape,
            )
            arrays[field] = arr

        # Initialize metadata storage
        sample_metadata = [None] * n_samples

        # Create DataLoader for parallel loading
        # Use spawn to avoid CUDA context issues
        mp_context = 'spawn' if self.num_workers > 0 else None
        loader = DataLoader(
            indexed_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_identity_collate,
            multiprocessing_context=mp_context,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

        # Process samples (DataLoader handles parallelism, writing is sequential)
        iterator = loader
        if self.show_progress:
            iterator = tqdm(iterator, desc="Caching samples", total=n_samples)

        for batch in iterator:
            idx, sample = batch[0]  # batch_size=1, so unpack single item

            # Store tensor data
            for field in self._tensor_fields:
                value = self._get_nested_value(sample, field)
                arrays[field][idx] = _tensor_to_numpy(value)

            # Store non-tensor metadata
            sample_metadata[idx] = self._extract_non_tensor_data(sample)

        # Flush all arrays
        for arr in arrays.values():
            arr.flush()

        # Save metadata
        self._metadata = {
            "tensor_fields": self._tensor_fields,
            "tensor_info": {k: {"shape": v["shape"], "dtype": str(v["torch_dtype"])}
                          for k, v in tensor_info.items()},
            "sample_metadata": sample_metadata,
        }
        self._sample_metadata = sample_metadata

        with open(self._cache_dir / "metadata.json", 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)

        # Save cache spec
        spec_data = {
            "cache_spec": self.cache_spec,
            "dataset_len": n_samples,
            "dataset_class": f"{self.source_dataset.__class__.__module__}.{self.source_dataset.__class__.__name__}",
            "cache_key": self._cache_key,
        }
        with open(self._cache_dir / "cache_spec.json", 'w') as f:
            json.dump(spec_data, f, indent=2)

        # Reopen arrays in read-only mode
        for field in self._tensor_fields:
            npy_file = self._cache_dir / f"{field}.npy"
            self._mmap_arrays[field] = np.load(npy_file, mmap_mode='r')

        logger.info(f"Cache populated: {self._cache_dir}")

    def _get_transformed_sample(self, idx: int) -> Any:
        """Get a sample from source dataset with transforms applied."""
        sample = self.source_dataset[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _analyze_sample_structure(self, sample: Any) -> Dict[str, Dict]:
        """Analyze a sample to determine tensor fields and their properties.

        Returns:
            Dict mapping field paths to {shape, torch_dtype, numpy_dtype}
        """
        tensor_info = {}

        def analyze_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, Tensor):
                tensor_info[prefix] = {
                    "shape": tuple(obj.shape),
                    "torch_dtype": obj.dtype,
                    "numpy_dtype": _get_numpy_dtype(obj.dtype),
                }
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    analyze_recursive(value, new_prefix)
            elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], Tensor):
                # List/tuple of tensors - treat as single stacked tensor
                for i, value in enumerate(obj):
                    new_prefix = f"{prefix}.{i}" if prefix else str(i)
                    analyze_recursive(value, new_prefix)

        analyze_recursive(sample)
        return tensor_info

    def _extract_non_tensor_data(self, sample: Any) -> Dict[str, Any]:
        """Extract non-tensor data from a sample for metadata storage."""
        non_tensor = {}

        def extract_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, Tensor):
                pass  # Skip tensors
            elif isinstance(obj, np.ndarray):
                pass  # Skip numpy arrays
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_recursive(value, new_prefix)
            elif isinstance(obj, (list, tuple)):
                if len(obj) > 0 and isinstance(obj[0], (Tensor, np.ndarray)):
                    pass  # Skip tensor lists
                else:
                    non_tensor[prefix] = obj
            elif isinstance(obj, (str, int, float, bool, type(None))):
                non_tensor[prefix] = obj

        extract_recursive(sample)
        return non_tensor

    def _get_nested_value(self, obj: Any, path: str) -> Any:
        """Get a value from a nested structure using dot-separated path."""
        parts = path.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, (list, tuple)):
                current = current[int(part)]
            else:
                raise KeyError(f"Cannot navigate to {part} in {type(current)}")
        return current

    def _set_nested_value(self, obj: Dict, path: str, value: Any):
        """Set a value in a nested dict structure using dot-separated path."""
        parts = path.split('.')
        current = obj
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _reconstruct_sample(self, idx: int) -> Dict[str, Any]:
        """Reconstruct a sample from cached data."""
        sample = {}

        # Load tensor data from mmap arrays
        for field in self._tensor_fields:
            array_data = self._mmap_arrays[field][idx]
            tensor = _numpy_to_tensor(array_data)
            self._set_nested_value(sample, field, tensor)

        # Add non-tensor metadata
        if idx < len(self._sample_metadata):
            for key, value in self._sample_metadata[idx].items():
                if key not in sample:  # Don't overwrite tensor data
                    self._set_nested_value(sample, key, value)

        return sample

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Any:
        if not self._initialized:
            self._initialize_cache()

        # If cache is populated, load from cache
        if self._mmap_arrays:
            return self._reconstruct_sample(idx)

        # Lazy caching: get from source, cache, return
        sample = self._get_transformed_sample(idx)
        # Note: lazy caching of individual samples not implemented yet
        # Would need to handle dynamic mmap array creation
        return sample

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    @property
    def cache_key(self) -> str:
        """Return the cache key."""
        return self._cache_key

    def get_cache_info(self) -> Dict[str, Any]:
        """Return information about the cache."""
        info = {
            "cache_key": self._cache_key,
            "cache_dir": str(self._cache_dir),
            "cache_exists": self._cache_dir.exists(),
            "tensor_fields": self._tensor_fields,
            "n_samples": len(self.source_dataset),
        }

        if self._cache_dir.exists():
            # Calculate cache size
            total_size = sum(
                f.stat().st_size for f in self._cache_dir.iterdir() if f.is_file()
            )
            info["cache_size_mb"] = total_size / (1024 * 1024)

        return info

    def clear_cache(self):
        """Clear the cache and reset state."""
        self._clear_cache()
        self._initialized = False
        logger.info(f"Cache cleared: {self._cache_dir}")
