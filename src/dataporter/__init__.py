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
from .shard_pool_source import ShardPoolSource
from .storage import ShardStorage, MemoryStorage, SharedMemoryStorage
from .prefetched_source import PrefetchedSource, priority_producer
from .transformable_dataset import TransformableDataset
from .timed_loader import TimedDataLoader
from .dataset_wrappers import KeyFilterDataset, AugmentedDataset
from .shuffle_buffer import ShuffleBuffer
from ._producer_pool_base import BaseProducerPool
from .producer_pool import ProducerPool, AsyncProducer
from .shuffle_buffer_dataset import ShuffleBufferDataset
from .lerobot_shuffle_buffer_dataset import LeRobotShuffleBufferDataset
from .token_shuffle_buffer import TokenShuffleBuffer
from .text_producer_pool import (
    TextProducerConfig, TextProducerPool, TokenizeFn,
)
from .token_shuffle_buffer_dataset import TokenShuffleBufferDataset
from .growing_dataset_callback import GrowingDatasetCallback
from .frame_transforms import (
    ResizeFrames, FrameCompose, probe_output_shape,
)
from .lerobot_shard_source import LeRobotShardSource
from .interfaces import (
    EpisodicSource,
    TemporalEpisodicSource,
    EpisodicPrefetcher,
    ProducerConfigProtocol,
)

# Lazy imports for optional dependencies (lerobot, lightning)
def __getattr__(name):
    if name == "FastLeRobotDataset":
        from .fast_lerobot_dataset import FastLeRobotDataset
        return FastLeRobotDataset
    if name == "BlendedLeRobotDataModule":
        from .blended_lerobot_datamodule import BlendedLeRobotDataModule
        return BlendedLeRobotDataModule
    if name == "scan_available_episodes":
        from .blended_lerobot_datamodule import scan_available_episodes
        return scan_available_episodes
    if name == "check_hf_cache_populated":
        from .blended_lerobot_datamodule import check_hf_cache_populated
        return check_hf_cache_populated
    if name == "hf_cache_repo_path":
        from .blended_lerobot_datamodule import hf_cache_repo_path
        return hf_cache_repo_path
    if name == "write_cache_sentinel":
        from .blended_lerobot_datamodule import write_cache_sentinel
        return write_cache_sentinel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # Storage + unified source
    'ShardStorage',
    'MemoryStorage',
    'SharedMemoryStorage',
    'PrefetchedSource',
    'priority_producer',
    # Text streaming
    'TextPrefetcher',
    'RawTextSource',
    'ShardPoolSource',
    'TransformableDataset',
    # HF rate-limited client
    'hf_download',
    'hf_snapshot',
    'hf_load_dataset',
    # Dataset wrappers
    'KeyFilterDataset',
    'AugmentedDataset',
    # LeRobot integration (lazy — requires lerobot)
    'FastLeRobotDataset',
    'BlendedLeRobotDataModule',
    'scan_available_episodes',
    'check_hf_cache_populated',
    'hf_cache_repo_path',
    'write_cache_sentinel',
    # Growing dataset integration (Lightning callback)
    'GrowingDatasetCallback',
    # Frame-level transforms for the producer-side pipeline
    'ResizeFrames',
    'FrameCompose',
    'probe_output_shape',
    # Live, lazy LeRobot shard source
    'LeRobotShardSource',
    # Public Protocols (structural-typing contracts for extension)
    'EpisodicSource',
    'TemporalEpisodicSource',
    'EpisodicPrefetcher',
    'ProducerConfigProtocol',
    # Shuffle buffer (video pipeline)
    'ShuffleBuffer',
    'BaseProducerPool',
    'ProducerPool',
    'AsyncProducer',
    'ShuffleBufferDataset',
    'LeRobotShuffleBufferDataset',
    # Shuffle buffer (text pipeline)
    'TokenShuffleBuffer',
    'TextProducerConfig',
    'TextProducerPool',
    'TokenizeFn',
    'TokenShuffleBufferDataset',
    # Instrumentation
    'TimedDataLoader',
    # Utilities
    'compose',
    'get_tokenizer',
]
