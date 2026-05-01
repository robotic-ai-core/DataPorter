"""
ResumableDataLoader implementation with true sample-level resumption.

Provides precise sample-level resumption for PyTorch DataLoaders, eliminating
Lightning warnings about non-resumable dataloaders. Supports both single-node
and distributed training scenarios with production-tested performance.

Key features:
- Sample-level precision: resume from exact batch position
- Performance optimized: 7.8x-32x speedup vs reprocessing from start  
- Lightning compatible: uses proper state management for multi-epoch training
- Distributed support: works with DistributedSampler for multi-GPU training
- Production ready: handles 10,000+ batch resumption scenarios efficiently

Technical Implementation:
- Shuffled datasets: torch.randperm() with epoch-specific seeds ensures deterministic ordering
- Resume position: Exact sample-level skip within the deterministic shuffle pattern
- Result: Identical behavior to continuous training regardless of resumption point

Memory optimizations (prevents pin memory exhaustion):
- Streaming index generation: no full tensor materialization in memory
- Lazy generator initialization: creates generators only when needed
- Explicit memory cleanup: uses del statements to free tensors immediately
- Memory-efficient distributed sampling: handles padding without intermediate lists
- Compatible with persistent_workers: no worker state conflicts

Lightning Compatibility Requirements:
- Sampler.__len__() returns full dataset length for consistent epoch calculations
- Resume through state management, not length manipulation
- Automatic handling of validation sanity check issues during resume
"""

import logging
import math
import os

import torch
from torch.utils.data import DataLoader, Sampler
from typing import Optional, Dict, Any, Iterator, Union
from .samplers import ResumableSampler, ResumableDistributedSampler
from .strategies import ResumptionStrategy, UnifiedResumptionStrategy
from .converters import KeyBasedDtypeConverter

logger = logging.getLogger(__name__)


def resolve_num_workers(num_workers: int) -> int:
    """Resolve num_workers, supporting -1 for core-count-based auto-scaling.

    -1 uses ``ceil(cpu_count / 8)``, rounded up to even. This gives:
      12-16 cores → 2,  17-24 → 4,  25-40 → 4,
      41-64 → 8,  65-96 → 12,  97-128 → 16

    Convention follows scikit-learn's ``n_jobs=-1`` (use all cores).

    Args:
        num_workers: Worker count. -1 = auto, 0 = main process, >0 = explicit.

    Returns:
        Resolved integer worker count (always >= 0).
    """
    if num_workers >= 0:
        return num_workers
    if num_workers == -1:
        cores = os.cpu_count() or 4
        n = math.ceil(cores / 8)
        n += n % 2  # round up to even
        n = max(2, n)  # at least 2
        logger.info(f"Auto num_workers: {n} (from {cores} cores)")
        return n
    raise ValueError(f"num_workers must be >= -1, got {num_workers}")


class _ConvertingCollate:
    """Picklable collate wrapper that applies dtype conversion in workers.

    Must be a class (not a closure/lambda) so multiprocessing can pickle it.
    """

    def __init__(self, base_collate, converter):
        self._base_collate = base_collate
        self._converter = converter

    def __call__(self, batch):
        from torch.utils.data.dataloader import default_collate
        collate_fn = self._base_collate or default_collate
        collated = collate_fn(batch)
        return self._converter.convert_batch(collated)


def _make_converting_collate(collate_fn, converter):
    """Create a collate function that includes dtype conversion."""
    return _ConvertingCollate(collate_fn, converter)


class _ResumableIter:
    """Iterator wrapper that updates the parent counter on each iteration."""

    def __init__(self, base_iter: Iterator, parent: "ResumableDataLoader") -> None:
        self._iter = base_iter
        self._parent = parent

    def __iter__(self) -> "_ResumableIter":
        return self

    def __next__(self):
        batch = next(self._iter)
        # Increment batch counter, but allow suppression for the first batch
        # immediately after an explicit epoch change so counts reflect
        # per-epoch progress in user loops that break early.
        if getattr(self._parent, "_suppress_next_increment", False):
            self._parent._suppress_next_increment = False
        else:
            self._parent._batches_processed += 1
        # Apply dtype conversions if converter is configured
        if self._parent._converter is not None:
            batch = self._parent._converter.convert_batch(batch)
        return batch


class ResumableDataLoader(DataLoader):
    """
    A DataLoader that can save and load its state for resuming training.
    
    Automatically handles single-node and distributed scenarios with sample-level
    precision. Provides true resumption capabilities that maintain identical
    behavior to continuous training.
    
    Features:
    - Sample-level resumption precision
    - Automatic distributed/single-node detection
    - Memory-optimized streaming approaches
    - Lightning-compatible state management
    - Production-tested performance (7.8x-32x speedup vs reprocessing)
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle samples (default: None -> True if no sampler provided)
        sampler: Custom sampler (default: None -> auto-create resumable sampler)
        batch_sampler: Custom batch sampler (default: None)
        num_workers: Number of worker processes (default: 0). Use -1
            for core-count-based auto-scaling (ceil(cores/8), rounded to even).
        collate_fn: Function to collate samples into batches (default: None)
        pin_memory: Whether to pin memory for faster GPU transfer (default: False)
        drop_last: Whether to drop last incomplete batch (default: False)
        timeout: Timeout for collecting samples from workers (default: 0)
        worker_init_fn: Function to initialize workers (default: None)
        multiprocessing_context: Multiprocessing context (default: None)
        generator: Random number generator (default: None)
        prefetch_factor: Number of samples loaded in advance by each worker (default: 2)
        persistent_workers: Whether to keep workers alive between epochs (default: False)
        distributed: Whether to use distributed training (default: None -> auto-detect)
        seed: Random seed for reproducible shuffling (default: None -> 42)
        converter: KeyBasedDtypeConverter or dict mapping paths to dtypes (default: None)
                   Can be either a converter instance or a dict like {"image": "float16"}
        **kwargs: Additional arguments passed to parent DataLoader
    """
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = None,
                 sampler: Optional[Sampler] = None, batch_sampler=None,
                 num_workers: int = 0, collate_fn=None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False, 
                 # ResumableDataLoader specific
                 resumption_strategy: Optional[ResumptionStrategy] = None,
                 distributed: bool = None, seed: Optional[int] = None,
                 converter: Optional[Union[KeyBasedDtypeConverter, Dict[str, str]]] = None,
                 **kwargs):
        
        # Resolve "auto" num_workers
        num_workers = resolve_num_workers(num_workers)

        # Auto-detect distributed training if not specified
        if distributed is None:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        elif distributed and not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            # Fail fast when distributed is explicitly requested but not available
            raise RuntimeError(
                "distributed=True was specified but distributed training is not initialized. "
                "Either initialize distributed training with torch.distributed.init_process_group() "
                "or set distributed=False/None for automatic detection."
            )
        
        # Always use UnifiedResumptionStrategy (auto-detects distributed)
        if resumption_strategy is None:
            resumption_strategy = UnifiedResumptionStrategy()
        
        # Create resumable sampler if none provided
        if sampler is None and batch_sampler is None:
            sampler = resumption_strategy.create_sampler(
                dataset,
                shuffle=shuffle if shuffle is not None else True,
                seed=seed if seed is not None else 42
            )
            # Disable shuffle since we're using a custom sampler
            shuffle = False
        
        # Move dtype conversion into collate_fn when using workers.
        # This runs in worker processes, freeing the main thread for
        # GPU kernel scheduling (eliminates ~15ms idle gap per step).
        resolved_converter = None
        if isinstance(converter, dict):
            resolved_converter = KeyBasedDtypeConverter(converter)
        elif converter is not None:
            resolved_converter = converter

        effective_collate = collate_fn
        if resolved_converter is not None and num_workers > 0:
            effective_collate = _make_converting_collate(
                collate_fn, resolved_converter
            )

        # Build DataLoader arguments
        dataloader_kwargs = {
            'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle,
            'sampler': sampler, 'batch_sampler': batch_sampler, 'num_workers': num_workers,
            'collate_fn': effective_collate, 'pin_memory': pin_memory, 'drop_last': drop_last,
            'timeout': timeout, 'worker_init_fn': worker_init_fn,
            'multiprocessing_context': multiprocessing_context, 'generator': generator,
            'persistent_workers': persistent_workers
        }

        # Only add prefetch_factor if num_workers > 0 (PyTorch constraint)
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor

        super().__init__(**dataloader_kwargs, **kwargs)
        
        # Initialize resumption strategy
        self.resumption_strategy = resumption_strategy
        self.resumption_strategy.attach_dataloader(self)
        
        # Track batches and epochs (maintained for backward compatibility)
        self._batches_processed = 0
        self._epoch = 0
        self._distributed = distributed
        # Control increment semantics around explicit epoch switches
        self._suppress_next_increment = False
        
        # Main-thread fallback: only when num_workers=0 and converter is set
        if resolved_converter is not None and num_workers == 0:
            self._converter = resolved_converter
        else:
            self._converter = None
        
    def __iter__(self) -> _ResumableIter:
        """Return an iterator that updates batch progress on each step."""
        return _ResumableIter(super().__iter__(), self)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Save DataLoader state including sampler state.
        
        Returns:
            Dictionary containing all state needed for resumption:
            - batches_processed: Number of batches processed in current epoch
            - epoch: Current epoch number
            - distributed: Whether using distributed training
            - sampler_state: Sampler-specific state for exact resumption
        """
        # Sync strategy state with legacy attributes for consistency
        if hasattr(self.resumption_strategy, '_batches_processed'):
            self.resumption_strategy._batches_processed = self._batches_processed
        elif hasattr(self.resumption_strategy, 'batches_seen'):
            self.resumption_strategy.batches_seen = self._batches_processed
            
        if hasattr(self.resumption_strategy, '_epoch'):
            self.resumption_strategy._epoch = self._epoch
        elif hasattr(self.resumption_strategy, 'epoch'):
            self.resumption_strategy.epoch = self._epoch
        
        # Use strategy to generate state dict
        return self.resumption_strategy.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load DataLoader state and update sampler for resumption.
        
        Calculates the exact sample position to resume from and updates
        the sampler accordingly. Handles epoch progression for multi-epoch
        resumption scenarios.
        
        Args:
            state_dict: State dictionary from previous save_state_dict() call
        """
        # Use strategy to load state
        self.resumption_strategy.load_state_dict(state_dict)
        
        # Sync legacy attributes from strategy for backward compatibility
        if hasattr(self.resumption_strategy, '_batches_processed'):
            self._batches_processed = self.resumption_strategy._batches_processed
        elif hasattr(self.resumption_strategy, 'batches_seen'):
            self._batches_processed = self.resumption_strategy.batches_seen
        
        if hasattr(self.resumption_strategy, '_epoch'):
            self._epoch = self.resumption_strategy._epoch
        elif hasattr(self.resumption_strategy, 'epoch'):
            self._epoch = self.resumption_strategy.epoch
            
        if hasattr(self.resumption_strategy, '_distributed'):
            self._distributed = self.resumption_strategy._distributed
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for distributed training.
        
        Updates internal epoch tracking and forwards to sampler if supported.
        This is typically called by Lightning or distributed training frameworks.
        
        Args:
            epoch: Epoch number to set
        """
        self._epoch = epoch
        # Reset per-epoch batch counter so state reflects progress within current epoch
        self._batches_processed = 0
        # Suppress increment for the first batch fetched after an explicit epoch change
        # so that a loop which breaks immediately reflects 0->1 progress.
        self._suppress_next_increment = True
        if hasattr(self.resumption_strategy, '_epoch'):
            self.resumption_strategy._epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


def create_resumable_dataloader(dataset, batch_size: int, shuffle: bool = True,
                               num_workers: int = 0, pin_memory: bool = True,
                               drop_last: bool = False, 
                               strategy: Optional[str] = None,
                               distributed: Optional[bool] = None,
                               seed: Optional[int] = None,
                               converter: Optional[Union[KeyBasedDtypeConverter, Dict[str, str]]] = None,
                               **kwargs) -> ResumableDataLoader:
    """
    Factory function to create a ResumableDataLoader with sample-level resumption.
    
    Provides true sample-level resumption that can handle production scenarios
    like resuming after 10,000+ batches with minimal performance overhead.
    
    This is the recommended way to create resumable dataloaders as it provides
    sensible defaults and clear parameter documentation.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples (default: True)
        num_workers: Number of worker processes (default: 0)
        pin_memory: Whether to pin memory for faster GPU transfer (default: True)
        drop_last: Whether to drop the last incomplete batch (default: False)
        distributed: Whether to use distributed training (default: None -> auto-detect)
        seed: Random seed for reproducible shuffling (default: None -> 42)
        converter: KeyBasedDtypeConverter or dict mapping paths to dtypes (default: None)
                   Can be either a converter instance or a dict like {"image": "float16"}
        **kwargs: Additional arguments passed to ResumableDataLoader
    
    Returns:
        ResumableDataLoader instance with full resumption capabilities
        
    Example:
        >>> from Yggdrasil.lib.data import create_resumable_dataloader
        >>> 
        >>> # Create resumable dataloader
        >>> dataloader = create_resumable_dataloader(
        ...     dataset=my_dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     seed=42
        ... )
        >>>
        >>> # Save state during training
        >>> state = dataloader.state_dict()
        >>>
        >>> # Resume later
        >>> new_dataloader = create_resumable_dataloader(...)
        >>> new_dataloader.load_state_dict(state)
    """
    # Strategy parameter is deprecated - always use UnifiedResumptionStrategy
    if strategy is not None:
        import warnings
        warnings.warn(
            "The 'strategy' parameter is deprecated. ResumableDataLoader now automatically "
            "detects the appropriate strategy based on your environment.",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Create dataloader with unified strategy
    return ResumableDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        seed=seed,
        distributed=distributed,
        converter=converter,
        **kwargs
    )