"""
Key-based dtype conversion system for generic batch processing.
Supports any nested dictionary structure with configurable dtype mappings.
"""

import torch
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class KeyBasedDtypeConverter:
    """Simple dtype converter using exact key paths."""
    
    def __init__(self, dtype_map: Optional[Union[Dict[str, str], list]] = None):
        """
        Initialize the dtype converter.
        
        Args:
            dtype_map: Either:
                      - Dict mapping key paths to target dtypes (legacy format)
                        e.g. {"inputs.image": "float16", "conditions.class_labels": "int64"}
                      - List of dicts with 'path' and 'dtype' keys (new format)
                        e.g. [{"path": "inputs.image", "dtype": "float16"}]
                      If None, no conversions will be performed.
        """
        # Convert list format to dict format internally
        if isinstance(dtype_map, list):
            self.dtype_map = {}
            for item in dtype_map:
                if not isinstance(item, dict) or 'path' not in item or 'dtype' not in item:
                    raise ValueError(
                        f"List format requires dicts with 'path' and 'dtype' keys. Got: {item}"
                    )
                self.dtype_map[item['path']] = item['dtype']
        else:
            self.dtype_map = dtype_map or {}
        self.torch_dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16, 
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "uint16": torch.uint16,  # Added for token IDs optimization
            "bool": torch.bool,
        }
        
        # Store conversion history for debugging (only activated on errors)
        self.conversion_history = []
        self.enable_debug_history = False  # Only enabled when debugging
        
        # Validate dtype mappings
        self._validate_dtype_map()
        
        # Log configuration
        if self.dtype_map:
            logger.info(f"KeyBasedDtypeConverter initialized with {len(self.dtype_map)} conversion rules")
        else:
            logger.info("KeyBasedDtypeConverter initialized with no conversion rules")
    
    def _validate_dtype_map(self):
        """Validate that all specified dtypes are supported."""
        for path, dtype_str in self.dtype_map.items():
            if dtype_str not in self.torch_dtypes:
                supported_dtypes = list(self.torch_dtypes.keys())
                raise ValueError(
                    f"Unsupported dtype '{dtype_str}' for path '{path}'. "
                    f"Supported dtypes: {supported_dtypes}"
                )
    
    def convert_batch(self, batch: Any) -> Any:
        """
        Convert batch based on key mappings.
        
        Args:
            batch: Input batch (typically a dictionary)
            
        Returns:
            Converted batch with same structure but updated dtypes
        """
        if not self.dtype_map:
            return batch
            
        return self._convert_recursive(batch, "")
    
    def _convert_recursive(self, obj: Any, current_path: str) -> Any:
        """
        Recursively convert based on current path.
        
        Args:
            obj: Current object to process
            current_path: Current path in the nested structure
            
        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                new_path = f"{current_path}.{key}".lstrip('.')
                result[key] = self._convert_recursive(value, new_path)
            return result
        
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples by processing each element with indexed path
            result_items = []
            for i, item in enumerate(obj):
                indexed_path = f"{current_path}[{i}]"
                result_items.append(self._convert_recursive(item, indexed_path))
            return type(obj)(result_items)
        
        elif isinstance(obj, torch.Tensor):
            return self._convert_tensor(obj, current_path)
        
        else:
            # Return other types unchanged (strings, numbers, etc.)
            return obj
    
    def _convert_tensor(self, tensor: torch.Tensor, path: str) -> torch.Tensor:
        """
        Convert a single tensor if its path matches a conversion rule.
        
        Args:
            tensor: Input tensor
            path: Current path of the tensor
            
        Returns:
            Converted tensor or original tensor if no rule matches
        """
        # Check if current path matches any conversion rule
        target_dtype_str = self.dtype_map.get(path)
        if target_dtype_str is None:
            return tensor
        
        target_dtype = self.torch_dtypes[target_dtype_str]
        
        # Only convert if dtype actually differs
        if tensor.dtype == target_dtype:
            return tensor
        
        # Store debug info ONLY if debug mode is enabled
        conversion_info = None
        if self.enable_debug_history:
            conversion_info = {
                'path': path,
                'from_dtype': str(tensor.dtype),
                'to_dtype': str(target_dtype),
                'shape': tensor.shape,
                'device': str(tensor.device)
            }
        
        try:
            converted_tensor = tensor.to(target_dtype)
            
            # Only store successful conversions in debug mode
            if self.enable_debug_history and conversion_info:
                conversion_info['status'] = 'success'
                self.conversion_history.append(conversion_info)
                # Limit history size to prevent memory issues
                if len(self.conversion_history) > 1000:
                    self.conversion_history = self.conversion_history[-500:]
            
            return converted_tensor
        except Exception as e:
            # On error, log comprehensive debug info
            logger.error(f"Failed to convert tensor at '{path}':")
            logger.error(f"  From dtype: {tensor.dtype}")
            logger.error(f"  To dtype: {target_dtype}")
            logger.error(f"  Shape: {tensor.shape}")
            logger.error(f"  Device: {tensor.device}")
            logger.error(f"  Tensor stats: min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}")
            logger.error(f"  Error: {e}")
            
            # Store failure in debug history
            if self.enable_debug_history and conversion_info:
                conversion_info['status'] = 'failed'
                conversion_info['error'] = str(e)
                self.conversion_history.append(conversion_info)
            
            return tensor
    
    def get_conversion_summary(self) -> Dict[str, str]:
        """
        Get a summary of all configured conversions.
        
        Returns:
            Dictionary mapping paths to target dtypes
        """
        return self.dtype_map.copy()
    
    def add_conversion_rule(self, path: str, target_dtype: str):
        """
        Add a new conversion rule.
        
        Args:
            path: Key path to match
            target_dtype: Target dtype string
        """
        if target_dtype not in self.torch_dtypes:
            supported_dtypes = list(self.torch_dtypes.keys())
            raise ValueError(
                f"Unsupported dtype '{target_dtype}'. "
                f"Supported dtypes: {supported_dtypes}"
            )
        
        self.dtype_map[path] = target_dtype
        logger.info(f"Added conversion rule: {path} -> {target_dtype}")
    
    def remove_conversion_rule(self, path: str) -> bool:
        """
        Remove a conversion rule.
        
        Args:
            path: Key path to remove
            
        Returns:
            True if rule was removed, False if it didn't exist
        """
        if path in self.dtype_map:
            del self.dtype_map[path]
            logger.info(f"Removed conversion rule for path: {path}")
            return True
        return False
    
    def enable_debug_mode(self, enabled: bool = True):
        """
        Enable or disable debug mode for conversion tracking.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.enable_debug_history = enabled
        if enabled:
            logger.info("Debug mode ENABLED for dtype converter - will track conversion history")
        else:
            logger.info("Debug mode DISABLED for dtype converter")
            self.conversion_history.clear()  # Clear history when disabling
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about recent conversions.
        
        Returns:
            Dictionary with debug information
        """
        if not self.enable_debug_history:
            return {
                'debug_enabled': False,
                'message': 'Debug mode is disabled. Enable with enable_debug_mode(True)'
            }
        
        # Analyze conversion history
        total_conversions = len(self.conversion_history)
        failed_conversions = [c for c in self.conversion_history if c.get('status') == 'failed']
        successful_conversions = [c for c in self.conversion_history if c.get('status') == 'success']
        
        # Group by path to find patterns
        conversions_by_path = {}
        for conv in self.conversion_history:
            path = conv['path']
            if path not in conversions_by_path:
                conversions_by_path[path] = {'success': 0, 'failed': 0}
            if conv.get('status') == 'success':
                conversions_by_path[path]['success'] += 1
            else:
                conversions_by_path[path]['failed'] += 1
        
        return {
            'debug_enabled': True,
            'total_conversions': total_conversions,
            'successful': len(successful_conversions),
            'failed': len(failed_conversions),
            'conversions_by_path': conversions_by_path,
            'recent_failures': failed_conversions[-10:],  # Last 10 failures
            'configured_rules': self.dtype_map
        }