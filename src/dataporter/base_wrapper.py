"""
Abstract base class for dataset wrappers with dtype conversion support.
Provides common functionality for all dataset wrappers in Yggdrasil.
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Union
from .converters import KeyBasedDtypeConverter
import logging

logger = logging.getLogger(__name__)


class BaseDatasetWrapper(Dataset, ABC):
    """
    Abstract base class for dataset wrappers with dtype conversion.
    
    This class provides:
    - Common dtype conversion functionality
    - Standardized initialization
    - Shared utility methods
    - Consistent logging
    
    Subclasses must implement:
    - __getitem__: How to retrieve and process items
    - _process_item: How to process raw items before dtype conversion
    """
    
    def __init__(
        self,
        base_dataset: Union[Dataset, Any],
        dtype_conversions: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize the base wrapper.
        
        Args:
            base_dataset: The underlying dataset to wrap
            dtype_conversions: Dict mapping paths to target dtypes
                             e.g. {"inputs.image": "float16"}
            **kwargs: Additional arguments for subclasses
        """
        self.base_dataset = base_dataset
        self.dtype_conversions = dtype_conversions
        
        # Initialize dtype converter
        self.dtype_converter = KeyBasedDtypeConverter(dtype_conversions)
        
        # Log configuration
        self._log_initialization()
        
        # Allow subclasses to perform additional initialization
        self._post_init(**kwargs)
    
    def _post_init(self, **kwargs):
        """
        Hook for subclasses to perform additional initialization.
        Override this method instead of __init__ for cleaner inheritance.
        """
        pass
    
    def _log_initialization(self):
        """Log initialization details."""
        wrapper_name = self.__class__.__name__
        if self.dtype_conversions:
            logger.info(
                f"{wrapper_name} initialized with {len(self.dtype_conversions)} "
                f"dtype conversion rules"
            )
            # Details logged at converter level if needed
        else:
            logger.info(f"{wrapper_name} initialized without dtype conversions")
    
    def __len__(self) -> int:
        """Return the length of the underlying dataset."""
        if hasattr(self.base_dataset, '__len__'):
            return len(self.base_dataset)
        else:
            # For iterable datasets
            return 2**31 - 1
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the dataset.
        
        Subclasses must implement this method to define how to:
        1. Retrieve the raw item
        2. Process it using _process_item
        3. Apply dtype conversions
        """
        pass
    
    @abstractmethod
    def _process_item(self, item: Any) -> Any:
        """
        Process a raw item before dtype conversion.
        
        This method should handle any dataset-specific processing
        like image transformations, tokenization, etc.
        
        Args:
            item: Raw item from the base dataset
            
        Returns:
            Processed item ready for dtype conversion
        """
        pass
    
    def _apply_dtype_conversions(self, item: Any) -> Any:
        """
        Apply dtype conversions to a processed item.
        
        Args:
            item: Processed item
            
        Returns:
            Item with dtype conversions applied
        """
        if self.dtype_conversions:
            return self.dtype_converter.convert_batch(item)
        return item
    
    def set_epoch(self, epoch: int):
        """
        Set epoch for datasets that support it (e.g., for proper shuffling).
        Recursively searches through dataset chain to find a dataset with set_epoch.
        
        Args:
            epoch: The current epoch number
        """
        def _recursive_set_epoch(dataset, epoch):
            """Recursively search for and call set_epoch on nested datasets."""
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
                # Successfully set epoch
                return True
            elif hasattr(dataset, 'dataset'):
                # Handle PyTorch Subset and similar wrappers
                return _recursive_set_epoch(dataset.dataset, epoch)
            elif hasattr(dataset, 'base_dataset'):
                # Handle other dataset wrappers
                return _recursive_set_epoch(dataset.base_dataset, epoch)
            return False
        
        _recursive_set_epoch(self.base_dataset, epoch)
    
    def get_dtype_summary(self) -> Dict[str, str]:
        """
        Get a summary of configured dtype conversions.
        
        Returns:
            Dictionary mapping paths to target dtypes
        """
        return self.dtype_converter.get_conversion_summary()
    
    @property
    def is_iterable(self) -> bool:
        """Check if the underlying dataset is iterable."""
        from torch.utils.data import IterableDataset
        return (
            isinstance(self.base_dataset, IterableDataset) or
            hasattr(self.base_dataset, '_ex_iterable') or
            not hasattr(self.base_dataset, '__len__')
        )
    
    def __iter__(self):
        """
        Iterate through the dataset (works for both map-style and iterable).
        """
        if self.is_iterable:
            # For iterable datasets
            for item in self.base_dataset:
                processed = self._process_item(item)
                yield self._apply_dtype_conversions(processed)
        else:
            # For map-style datasets
            for idx in range(len(self)):
                yield self[idx]