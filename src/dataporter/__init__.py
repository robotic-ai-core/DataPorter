"""
DataPorter - PyTorch data loading utilities for seamless training resumption and memory optimization
"""

__version__ = "0.1.0"

from .converters import KeyBasedDtypeConverter
from .base_wrapper import BaseDatasetWrapper
from .generic_wrapper import GenericDatasetWrapper, DataLoaderDtypeWrapper
from .fault_tolerant_wrapper import FaultTolerantDatasetWrapper
from .wrappers import UnifiedHFDatasetWrapper
from .samplers import ResumableSampler, ResumableDistributedSampler
from .resumable import ResumableDataLoader, create_resumable_dataloader
from .strategies import ResumptionStrategy, UnifiedResumptionStrategy
from .cached import CachedDataset, get_cache_root
from .prefetcher import BasePrefetcher, CompanionPool, CompanionRef
from .lerobot_prefetcher import LeRobotPrefetcher
from .transforms import compose, get_tokenizer
from .hf_client import hf_download, hf_snapshot, hf_load_dataset
from .text_prefetcher import TextPrefetcher
from .raw_text_source import RawTextSource
from .transformable_dataset import TransformableDataset

__all__ = [
    'KeyBasedDtypeConverter',
    'BaseDatasetWrapper',
    'GenericDatasetWrapper',
    'DataLoaderDtypeWrapper',
    'FaultTolerantDatasetWrapper',
    'UnifiedHFDatasetWrapper',
    'ResumableSampler',
    'ResumableDistributedSampler',
    'ResumableDataLoader',
    'create_resumable_dataloader',
    # Strategy classes
    'ResumptionStrategy',
    'UnifiedResumptionStrategy',
    # Caching
    'CachedDataset',
    'get_cache_root',
    # Prefetcher base + companions
    'BasePrefetcher',
    'CompanionPool',
    'CompanionRef',
    'LeRobotPrefetcher',
    # Text streaming (raw text prefetch + worker-side transform)
    'TextPrefetcher',
    'RawTextSource',
    'TransformableDataset',
    # HF rate-limited client
    'hf_download',
    'hf_snapshot',
    'hf_load_dataset',
    # Utilities
    'compose',
    'get_tokenizer',
]
