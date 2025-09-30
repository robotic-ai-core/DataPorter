"""
Generic dataset wrapper with dtype conversion for any PyTorch dataset.
This provides a flexible way to apply dtype conversions to existing datasets.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from .base_wrapper import BaseDatasetWrapper
from .converters import KeyBasedDtypeConverter
import logging
import os

# Import debug utils only if debug mode is enabled
_DEBUG_MODE = os.environ.get('DATAPORTER_ADVANCED_DEBUG', 'false').lower() == 'true'
if _DEBUG_MODE:
    from .debug_utils import get_advanced_debug_info, format_debug_report

logger = logging.getLogger(__name__)


class GenericDatasetWrapper(BaseDatasetWrapper):
    """
    A generic wrapper that applies dtype conversions to any PyTorch dataset.

    This is particularly useful for:
    - Reducing VRAM usage by converting to lower precision dtypes
    - Working with existing datasets without modifying their code
    - Applying conversions based on configurable paths

    Example:
        ```python
        # Wrap any existing dataset
        wrapped_dataset = GenericDatasetWrapper(
            original_dataset,
            dtype_conversions={
                "observation.image": "float16",  # 50% memory savings
                "action": "float16",
                "next.done": "uint8"  # Boolean only needs 1 byte
            }
        )
        ```
    """

    def _post_init(self, custom_path_mapping: Optional[Dict[str, str]] = None,
                   skip_validation: bool = False, **kwargs):
        """
        Additional initialization for GenericDatasetWrapper.

        Args:
            custom_path_mapping: Optional dict to remap paths before conversion
                               e.g. {"images": "observation.image"} to handle different naming
            skip_validation: If True, skip NaN/Inf validation for better performance
        """
        self.custom_path_mapping = custom_path_mapping or {}
        self.skip_validation = skip_validation
    
    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the dataset and apply dtype conversions.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            The item with dtype conversions applied
        """
        
        # Collect debug info for potential error reporting
        debug_context = {'index': idx}
        
        try:
            # Get item from base dataset
            item = self.base_dataset[idx]
            debug_context['item_retrieved'] = True
        except Exception as e:
            # Log comprehensive error info
            logger.error(f"Failed to get item at index {idx} from base dataset: {type(self.base_dataset).__name__}")
            logger.error(f"Exception: {str(e)}")
            logger.error(f"Dataset length: {len(self.base_dataset) if hasattr(self.base_dataset, '__len__') else 'unknown'}")
            raise
        
        try:
            # Process the item (applies path mapping if configured)
            processed_item = self._process_item(item)
            debug_context['item_processed'] = True
            
            # Store item structure for debugging (only if we later fail)
            if isinstance(processed_item, dict):
                debug_context['keys'] = list(processed_item.keys())
                debug_context['dtypes'] = {
                    k: str(v.dtype) if hasattr(v, 'dtype') else type(v).__name__
                    for k, v in processed_item.items()
                }
        except Exception as e:
            # Log error with context
            logger.error(f"Failed to process item at index {idx}")
            logger.error(f"Item type: {type(item)}")
            if isinstance(item, dict):
                logger.error(f"Item keys: {list(item.keys())}")
            logger.error(f"Exception: {str(e)}")
            raise
        
        try:
            # Apply dtype conversions (handled by base class)
            converted_item = self._apply_dtype_conversions(processed_item)

            # Validate converted item for NaN/Inf values (skip if disabled for performance)
            if not self.skip_validation:
                self._validate_converted_item(converted_item, processed_item, idx)

            return converted_item
            
        except ValueError:
            # Re-raise ValueError (NaN/Inf detection) - already logged
            raise
        except Exception as e:
            # Log comprehensive debug info on conversion failure
            logger.error(f"Failed to apply dtype conversions at index {idx}")
            logger.error(f"Debug context: {debug_context}")
            
            if isinstance(processed_item, dict):
                logger.error("Item structure before conversion:")
                for key, val in list(processed_item.items())[:5]:  # Limit to first 5 keys
                    if hasattr(val, 'shape'):
                        logger.error(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                        if hasattr(val, 'min') and hasattr(val, 'max'):
                            try:
                                logger.error(f"    Range: [{val.min():.6f}, {val.max():.6f}]")
                            except:
                                pass
            
            # Collect advanced debug info if enabled
            if _DEBUG_MODE:
                logger.error("Collecting advanced debug information...")
                try:
                    debug_info = get_advanced_debug_info()
                    report = format_debug_report(debug_info, e)
                    logger.error(f"\n{report}")
                    
                    # Optional: Save debug dump
                    if os.environ.get('DATAPORTER_SAVE_DEBUG_DUMPS', 'false').lower() == 'true':
                        from .debug_utils import save_debug_dump
                        dump_file = save_debug_dump(debug_info, e)
                        logger.error(f"Debug dump saved to: {dump_file}")
                except Exception as debug_error:
                    logger.error(f"Failed to collect advanced debug info: {debug_error}")
            
            # Enable debug mode for future conversions (if converter supports it)
            if hasattr(self.dtype_converter, 'enable_debug_mode'):
                self.dtype_converter.enable_debug_mode(True)
                logger.info("Enabled debug mode for dtype converter for future debugging")
            
            logger.error(f"Exception: {str(e)}")
            raise
    
    def _validate_converted_item(self, converted_item: Any, processed_item: Any, idx: int):
        """
        Validate converted item for NaN/Inf values with detailed error reporting.
        
        Args:
            converted_item: Item after conversion
            processed_item: Item before conversion
            idx: Index for error reporting
        """
        if not isinstance(converted_item, dict):
            return
            
        for key, val in converted_item.items():
            if not hasattr(val, 'isnan'):
                continue
                
            # Check for NaN
            if val.isnan().any():
                nan_mask = val.isnan()
                nan_count = nan_mask.sum().item()
                
                logger.error(f"CRITICAL: NaN detected after dtype conversion at index {idx}")
                logger.error(f"  Key: {key}")
                logger.error(f"  Shape: {val.shape}")
                logger.error(f"  Dtype: {val.dtype}")
                logger.error(f"  NaN count: {nan_count}/{val.numel()}")
                
                # Sample NaN locations (first 10)
                nan_locs = nan_mask.nonzero()
                if len(nan_locs[0]) > 0:
                    sample_locs = [tuple(loc[i].item() for loc in nan_locs) 
                                   for i in range(min(10, len(nan_locs[0])))]
                    logger.error(f"  Sample NaN locations: {sample_locs}")
                
                # Check if NaN was present before conversion
                if isinstance(processed_item, dict) and key in processed_item:
                    orig_val = processed_item[key]
                    if hasattr(orig_val, 'isnan'):
                        if orig_val.isnan().any():
                            logger.error(f"  NaN was already present BEFORE conversion")
                        else:
                            logger.error(f"  NaN was INTRODUCED BY conversion")
                            logger.error(f"  Original dtype: {orig_val.dtype}")
                            try:
                                logger.error(f"  Original range: [{orig_val.min():.6f}, {orig_val.max():.6f}]")
                            except:
                                pass
                
                # Get debug info from converter if available
                if hasattr(self.dtype_converter, 'get_debug_info'):
                    debug_info = self.dtype_converter.get_debug_info()
                    if debug_info.get('debug_enabled'):
                        logger.error(f"  Converter debug info: {debug_info}")
                
                raise ValueError(f"NaN detected in {key} after dtype conversion at index {idx}")
            
            # Check for Inf
            if hasattr(val, 'isinf') and val.isinf().any():
                inf_count = val.isinf().sum().item()
                logger.error(f"CRITICAL: Inf detected after dtype conversion at index {idx}")
                logger.error(f"  Key: {key}")
                logger.error(f"  Inf count: {inf_count}")
                raise ValueError(f"Inf detected in {key} after dtype conversion at index {idx}")
    
    def _process_item(self, item: Any) -> Any:
        """
        Process the item, applying path mapping if configured.
        
        Args:
            item: Raw item from base dataset
            
        Returns:
            Processed item
        """
        # Apply custom path mapping if needed
        if self.custom_path_mapping:
            return self._apply_path_mapping(item)
        return item
    
    def _apply_path_mapping(self, item: Any, path: str = "") -> Any:
        """
        Apply custom path mapping to standardize paths before conversion.
        
        This allows handling datasets with different naming conventions.
        For example, mapping "images" -> "observation.image" for consistency.
        """
        if isinstance(item, dict):
            result = {}
            for key, value in item.items():
                current_path = f"{path}.{key}".lstrip('.')
                
                # Check if this path needs remapping
                if current_path in self.custom_path_mapping:
                    new_path = self.custom_path_mapping[current_path]
                    # Create nested structure for the new path
                    self._set_nested_value(result, new_path, value)
                else:
                    result[key] = self._apply_path_mapping(value, current_path)
            return result
        else:
            return item
    
    def _set_nested_value(self, target_dict: dict, path: str, value: Any) -> None:
        """
        Set a value in a nested dictionary using a dot-separated path.
        
        Args:
            target_dict: The dictionary to modify
            path: Dot-separated path (e.g., "observation.image")
            value: The value to set
        """
        keys = path.split('.')
        current = target_dict
        
        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value


class DataLoaderDtypeWrapper:
    """
    A wrapper for DataLoader that ensures dtype conversions are applied.
    
    This is useful when you want to apply conversions at the DataLoader level
    rather than the dataset level, which can be more efficient for some use cases.
    """
    
    def __init__(self, dataloader, dtype_conversions: Optional[Dict[str, str]] = None):
        """
        Initialize the DataLoader wrapper.
        
        Args:
            dataloader: The underlying DataLoader
            dtype_conversions: Dict mapping paths to target dtypes
        """
        self.dataloader = dataloader
        self.dtype_converter = KeyBasedDtypeConverter(dtype_conversions)
    
    def __iter__(self):
        """Iterate through the dataloader, applying conversions to each batch."""
        for batch in self.dataloader:
            if self.dtype_converter.dtype_map:
                batch = self.dtype_converter.convert_batch(batch)
            yield batch
    
    def __len__(self):
        """Return the length of the underlying dataloader."""
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        """Return the batch size of the underlying dataloader."""
        return self.dataloader.batch_size
    
    @property
    def dataset(self):
        """Return the dataset of the underlying dataloader."""
        return self.dataloader.dataset