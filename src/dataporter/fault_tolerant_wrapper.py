"""
Fault-tolerant dataset wrapper that can skip problematic samples.

This wrapper adds error tolerance to any dataset, allowing it to skip
corrupted or problematic samples up to a configurable limit per epoch.
"""

import logging
from typing import Any, Dict, Optional, Set, List
import torch
from collections import defaultdict
import warnings

from .base_wrapper import BaseDatasetWrapper

logger = logging.getLogger(__name__)


class FaultTolerantDatasetWrapper(BaseDatasetWrapper):
    """
    Dataset wrapper that can skip problematic samples with configurable error tolerance.
    
    Features:
    - Skip corrupted/problematic samples automatically
    - Configurable error limits (per epoch, per reset, or total)
    - Detailed error tracking and reporting
    - Automatic fallback to next valid sample
    - Option to raise after threshold or continue with warnings
    
    Example:
        ```python
        from dataporter import FaultTolerantDatasetWrapper
        
        # Wrap any dataset with fault tolerance
        tolerant_dataset = FaultTolerantDatasetWrapper(
            base_dataset=my_dataset,
            max_errors_per_epoch=10,
            max_consecutive_errors=3,
            error_action='skip',  # 'skip', 'warn', or 'raise'
            verbose=True
        )
        
        # Use with DataLoader - bad samples will be automatically skipped
        loader = DataLoader(tolerant_dataset, batch_size=32)
        for batch in loader:
            # Batch will only contain valid samples
            process(batch)
        ```
    """
    
    def __init__(
        self,
        base_dataset: Any,
        max_errors_per_epoch: Optional[int] = 10,
        max_consecutive_errors: Optional[int] = 3,
        max_total_errors: Optional[int] = None,
        error_action: str = 'skip',
        retry_with_next: bool = True,
        track_error_indices: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize fault-tolerant wrapper.
        
        Args:
            base_dataset: The dataset to wrap
            max_errors_per_epoch: Maximum errors allowed per epoch (None = unlimited)
            max_consecutive_errors: Maximum consecutive errors before raising (None = unlimited)
            max_total_errors: Maximum total errors across all epochs (None = unlimited)
            error_action: Action on error - 'skip', 'warn', or 'raise'
            retry_with_next: If True, try next index when current fails
            track_error_indices: If True, track which indices caused errors
            verbose: If True, log detailed error information
            **kwargs: Additional arguments passed to base wrapper
        """
        super().__init__(base_dataset=base_dataset, **kwargs)
        
        # Error limits
        self.max_errors_per_epoch = max_errors_per_epoch
        self.max_consecutive_errors = max_consecutive_errors
        self.max_total_errors = max_total_errors
        
        # Error handling
        self.error_action = error_action
        self.retry_with_next = retry_with_next
        self.track_error_indices = track_error_indices
        self.verbose = verbose
        
        # Error tracking
        self.epoch_errors = 0
        self.total_errors = 0
        self.consecutive_errors = 0
        self.error_indices: Set[int] = set()
        self.error_history: List[Dict[str, Any]] = []
        
        # Epoch tracking
        self.current_epoch = 0
        self.samples_this_epoch = 0
        
    def _process_item(self, item: Any) -> Any:
        """
        Process an item (no-op for fault tolerant wrapper).
        
        This wrapper focuses on error handling, not transformation.
        Override this if you need custom processing.
        
        Args:
            item: The item to process
            
        Returns:
            The item unchanged
        """
        return item
    
    def reset_epoch(self):
        """Reset error counts for new epoch."""
        if self.epoch_errors > 0:
            logger.info(f"Epoch {self.current_epoch} completed with {self.epoch_errors} errors "
                       f"({self.samples_this_epoch} successful samples)")
        
        self.current_epoch += 1
        self.epoch_errors = 0
        self.consecutive_errors = 0
        self.samples_this_epoch = 0
        
        if self.verbose:
            logger.info(f"Starting epoch {self.current_epoch} with error tolerance: "
                       f"max {self.max_errors_per_epoch} errors")
    
    def __getitem__(self, idx: int) -> Any:
        """
        Get item with fault tolerance.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            The item if successful, or next valid item if current fails
            
        Raises:
            Exception: If error limits are exceeded
        """
        original_idx = idx
        attempts = 0
        max_attempts = min(100, len(self.base_dataset))  # Prevent infinite loops
        
        while attempts < max_attempts:
            try:
                # Try to get the item
                item = self.base_dataset[idx]
                
                # Validate item (check for NaN, Inf, etc.)
                self._validate_item(item, idx)
                
                # Success - reset consecutive error counter
                if self.consecutive_errors > 0:
                    self.consecutive_errors = 0
                    if self.verbose:
                        logger.info(f"Recovered after {self.consecutive_errors} consecutive errors")
                
                self.samples_this_epoch += 1
                return item
                
            except Exception as e:
                # Track the error
                self._handle_error(idx, e, original_idx)
                
                # Check if we should raise based on limits
                if self._should_raise_error():
                    self._raise_limit_exceeded(e, idx)
                
                # Determine next action
                if not self.retry_with_next:
                    # Return None or a placeholder if configured not to retry
                    return self._get_placeholder_item()
                
                # Try next index
                idx = (idx + 1) % len(self.base_dataset)
                attempts += 1
                
                # Warn if we've wrapped around
                if idx == original_idx:
                    logger.warning(f"Wrapped around to original index {original_idx} "
                                  f"while searching for valid sample")
                    break
        
        # If we get here, we couldn't find a valid sample
        raise RuntimeError(f"Could not find valid sample after {attempts} attempts "
                          f"starting from index {original_idx}")
    
    def _validate_item(self, item: Any, idx: int):
        """
        Validate an item for common issues.
        
        Args:
            item: The item to validate
            idx: Index of the item
            
        Raises:
            ValueError: If validation fails
        """
        if isinstance(item, dict):
            for key, value in item.items():
                if torch.is_tensor(value):
                    # Check for NaN
                    if torch.isnan(value).any():
                        raise ValueError(f"NaN detected in '{key}' at index {idx}")
                    
                    # Check for Inf
                    if torch.isinf(value).any():
                        raise ValueError(f"Inf detected in '{key}' at index {idx}")
                    
                    # Check for extreme values (optional)
                    if value.dtype in [torch.float32, torch.float64]:
                        max_val = value.abs().max().item()
                        if max_val > 1e10:
                            warnings.warn(f"Extreme value {max_val} in '{key}' at index {idx}")
        
        elif torch.is_tensor(item):
            if torch.isnan(item).any():
                raise ValueError(f"NaN detected at index {idx}")
            if torch.isinf(item).any():
                raise ValueError(f"Inf detected at index {idx}")
    
    def _handle_error(self, idx: int, error: Exception, original_idx: int):
        """
        Handle an error when getting an item.
        
        Args:
            idx: Current index that failed
            error: The exception that occurred
            original_idx: Original requested index
        """
        # Update error counts
        self.epoch_errors += 1
        self.total_errors += 1
        self.consecutive_errors += 1
        
        # Track error index
        if self.track_error_indices:
            self.error_indices.add(idx)
            
            # Store detailed error history
            self.error_history.append({
                'epoch': self.current_epoch,
                'index': idx,
                'original_index': original_idx,
                'error_type': type(error).__name__,
                'error_msg': str(error),
                'epoch_errors': self.epoch_errors,
                'total_errors': self.total_errors
            })
        
        # Log the error
        if self.verbose or self.error_action == 'warn':
            logger.warning(f"Error at index {idx} (originally {original_idx}): {error}")
            
            # Format counts with None handling
            epoch_str = f"{self.epoch_errors}/{self.max_errors_per_epoch}" if self.max_errors_per_epoch is not None else str(self.epoch_errors)
            consec_str = f"{self.consecutive_errors}/{self.max_consecutive_errors}" if self.max_consecutive_errors is not None else str(self.consecutive_errors)
            total_str = f"{self.total_errors}/{self.max_total_errors}" if self.max_total_errors is not None else str(self.total_errors)
            
            logger.warning(f"Error counts - Epoch: {epoch_str}, "
                          f"Consecutive: {consec_str}, "
                          f"Total: {total_str}")
    
    def _should_raise_error(self) -> bool:
        """Check if we should raise an error based on limits."""
        if self.error_action == 'raise':
            return True
        
        # Check consecutive errors (only if limit is set)
        if self.max_consecutive_errors is not None and self.consecutive_errors >= self.max_consecutive_errors:
            return True
        
        # Check epoch errors (only if limit is set)
        if self.max_errors_per_epoch is not None and self.epoch_errors > self.max_errors_per_epoch:
            return True
        
        # Check total errors (only if limit is set)
        if self.max_total_errors is not None and self.total_errors > self.max_total_errors:
            return True
        
        return False
    
    def _raise_limit_exceeded(self, original_error: Exception, idx: int):
        """Raise an error when limits are exceeded."""
        # Try to get advanced debug info when we're about to crash
        debug_info = None
        try:
            # Temporarily enable debug mode for this final error
            import os
            os.environ['DATAPORTER_ADVANCED_DEBUG'] = 'true'
            
            # Try to get detailed debug info
            from .debug_utils import get_advanced_debug_info, format_debug_report
            debug_info = get_advanced_debug_info()
            debug_report = format_debug_report(debug_info)
            
            # Save to file
            from pathlib import Path
            debug_dir = Path("./debug_dumps")
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"fault_tolerance_limit_exceeded_{idx}.txt"
            with open(debug_file, 'w') as f:
                f.write(debug_report)
            
            logger.error(f"Advanced debug info saved to: {debug_file}")
        except:
            pass  # Don't fail if debug capture fails
        
        error_msg = f"Error tolerance exceeded at index {idx}.\n"
        
        # Format limit messages based on what was exceeded
        if self.max_errors_per_epoch is not None:
            error_msg += f"Epoch errors: {self.epoch_errors}/{self.max_errors_per_epoch}\n"
        else:
            error_msg += f"Epoch errors: {self.epoch_errors} (no limit)\n"
            
        if self.max_consecutive_errors is not None:
            error_msg += f"Consecutive errors: {self.consecutive_errors}/{self.max_consecutive_errors}\n"
        else:
            error_msg += f"Consecutive errors: {self.consecutive_errors} (no limit)\n"
            
        if self.max_total_errors is not None:
            error_msg += f"Total errors: {self.total_errors}/{self.max_total_errors}\n"
        else:
            error_msg += f"Total errors: {self.total_errors} (no limit)\n"
        
        if self.track_error_indices:
            error_msg += f"\nError indices this epoch: {sorted(self.error_indices)[:10]}"
            if len(self.error_indices) > 10:
                error_msg += f"... ({len(self.error_indices)} total)"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg) from original_error
    
    def _get_placeholder_item(self) -> Any:
        """Get a placeholder item when retry_with_next is False."""
        # This could be customized based on the dataset structure
        # For now, return None and let the DataLoader handle it
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        summary = {
            'current_epoch': self.current_epoch,
            'epoch_errors': self.epoch_errors,
            'total_errors': self.total_errors,
            'error_indices': sorted(self.error_indices) if self.track_error_indices else [],
            'samples_this_epoch': self.samples_this_epoch,
            'error_rate': self.epoch_errors / max(1, self.samples_this_epoch + self.epoch_errors)
        }
        
        if self.error_history:
            # Group errors by type
            error_types = defaultdict(int)
            for error in self.error_history:
                error_types[error['error_type']] += 1
            summary['error_types'] = dict(error_types)
            
            # Most recent errors
            summary['recent_errors'] = self.error_history[-5:]
        
        return summary
    
    def __len__(self) -> int:
        """Return length of dataset (unchanged by fault tolerance)."""
        return len(self.base_dataset)