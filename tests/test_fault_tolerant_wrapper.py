"""
Test suite for FaultTolerantDatasetWrapper.
"""

import pytest
import torch
from unittest.mock import MagicMock, call
from dataporter import FaultTolerantDatasetWrapper


class TestFaultTolerantWrapper:
    """Test fault tolerance features."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset with configurable failures."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        return dataset
    
    @pytest.fixture
    def good_sample(self):
        """A valid sample."""
        return {
            'data': torch.randn(3, 224, 224),
            'label': torch.tensor(1)
        }
    
    @pytest.fixture
    def nan_sample(self):
        """A sample with NaN values."""
        return {
            'data': torch.tensor([float('nan'), 1.0, 2.0]),
            'label': torch.tensor(1)
        }
    
    def test_skip_nan_samples(self, mock_dataset, good_sample, nan_sample):
        """Test that NaN samples are skipped."""
        # Configure mock to return NaN at index 0, good at index 1
        mock_dataset.__getitem__ = MagicMock(
            side_effect=[nan_sample, good_sample]
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_errors_per_epoch=10,
            error_action='skip'
        )
        
        # Should skip index 0 and return index 1
        result = wrapper[0]
        assert result is not None
        assert not torch.isnan(result['data']).any()
        assert wrapper.total_errors == 1
        assert 0 in wrapper.error_indices
    
    def test_consecutive_error_limit(self, mock_dataset):
        """Test that consecutive error limit stops iteration."""
        # Configure mock to always fail
        mock_dataset.__getitem__ = MagicMock(
            side_effect=RuntimeError("Always fails")
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_consecutive_errors=3,
            error_action='skip'
        )
        
        # Should raise after 3 consecutive errors
        with pytest.raises(RuntimeError, match="Error tolerance exceeded"):
            wrapper[0]
        
        assert wrapper.consecutive_errors == 3
    
    def test_epoch_error_limit(self, mock_dataset, good_sample):
        """Test that epoch error limit works."""
        error_count = 0
        def side_effect(idx):
            nonlocal error_count
            if error_count < 5:
                error_count += 1
                raise RuntimeError(f"Error {error_count}")
            return good_sample
        
        mock_dataset.__getitem__ = MagicMock(side_effect=side_effect)
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_errors_per_epoch=4,
            max_consecutive_errors=10,  # High to not trigger
            error_action='skip'
        )
        
        # Should raise after exceeding epoch limit
        with pytest.raises(RuntimeError, match="Error tolerance exceeded"):
            wrapper[0]
        
        assert wrapper.epoch_errors > 4
    
    def test_none_limits_unlimited(self, mock_dataset, good_sample):
        """Test that None limits allow unlimited errors."""
        error_count = 0
        def side_effect(idx):
            nonlocal error_count
            if error_count < 20:  # Many errors
                error_count += 1
                if error_count == 20:
                    return good_sample
                raise RuntimeError(f"Error {error_count}")
            return good_sample
        
        mock_dataset.__getitem__ = MagicMock(side_effect=side_effect)
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_errors_per_epoch=None,  # No limit
            max_consecutive_errors=None,  # No limit
            max_total_errors=None,  # No limit
            error_action='skip',
            verbose=False
        )
        
        # Should eventually return good sample after many errors
        result = wrapper[0]
        assert result is not None
        assert wrapper.total_errors == 19  # 19 errors before success
    
    def test_error_action_raise(self, mock_dataset):
        """Test that error_action='raise' immediately raises."""
        mock_dataset.__getitem__ = MagicMock(
            side_effect=RuntimeError("Test error")
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            error_action='raise'  # Raise immediately
        )
        
        # Should raise immediately
        with pytest.raises(RuntimeError, match="Error tolerance exceeded"):
            wrapper[0]
        
        assert wrapper.total_errors == 1
    
    def test_reset_epoch(self, mock_dataset):
        """Test that reset_epoch resets counters."""
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset
        )
        
        # Simulate some errors
        wrapper.epoch_errors = 5
        wrapper.consecutive_errors = 2
        wrapper.samples_this_epoch = 100
        
        # Reset epoch
        wrapper.reset_epoch()
        
        # Epoch counters should reset
        assert wrapper.epoch_errors == 0
        assert wrapper.consecutive_errors == 0
        assert wrapper.samples_this_epoch == 0
        assert wrapper.current_epoch == 1
        # Total errors should NOT reset
        assert wrapper.total_errors == 0  # Unchanged
    
    def test_error_summary(self, mock_dataset, good_sample):
        """Test that error summary is accurate."""
        # Configure some errors then success
        mock_dataset.__getitem__ = MagicMock(
            side_effect=[
                RuntimeError("Error 1"),
                ValueError("Error 2"),
                good_sample,
                RuntimeError("Error 3"),
                good_sample
            ]
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_consecutive_errors=5,
            track_error_indices=True,
            verbose=False
        )
        
        # Trigger some errors
        wrapper[0]  # 2 errors, then success
        wrapper[0]  # 1 error, then success
        
        summary = wrapper.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert summary['epoch_errors'] == 3
        assert summary['samples_this_epoch'] == 2
        assert 0 in summary['error_indices']
        assert 'RuntimeError' in summary['error_types']
        assert summary['error_types']['RuntimeError'] == 2
        assert summary['error_types']['ValueError'] == 1
    
    def test_inf_detection(self, mock_dataset):
        """Test that Inf values are detected."""
        inf_sample = {
            'data': torch.tensor([float('inf'), 1.0, 2.0]),
            'label': torch.tensor(1)
        }
        
        mock_dataset.__getitem__ = MagicMock(return_value=inf_sample)
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_consecutive_errors=1
        )
        
        # Should detect Inf and raise
        with pytest.raises(RuntimeError, match="Error tolerance exceeded"):
            wrapper[0]
    
    def test_retry_with_next_false(self, mock_dataset):
        """Test retry_with_next=False returns None on error."""
        mock_dataset.__getitem__ = MagicMock(
            side_effect=RuntimeError("Always fails")
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            retry_with_next=False,  # Don't retry
            max_errors_per_epoch=10
        )
        
        # Should return None instead of retrying
        result = wrapper[0]
        assert result is None
        assert wrapper.total_errors == 1
    
    def test_wrap_around_detection(self, mock_dataset):
        """Test that wrap-around is detected."""
        # All samples fail
        mock_dataset.__getitem__ = MagicMock(
            side_effect=RuntimeError("Always fails")
        )
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_consecutive_errors=200,  # High limit
            max_errors_per_epoch=200,   # High limit to avoid hitting it first
            verbose=False
        )
        
        # Should eventually give up after trying many indices
        with pytest.raises(RuntimeError, match="Could not find valid sample"):
            wrapper[0]
    
    def test_mixed_none_and_set_limits(self, mock_dataset, good_sample):
        """Test mixing None and set limits."""
        error_count = 0
        def side_effect(idx):
            nonlocal error_count
            error_count += 1
            if error_count > 6:  # Success after 6 errors
                return good_sample
            raise RuntimeError(f"Error {error_count}")
        
        mock_dataset.__getitem__ = MagicMock(side_effect=side_effect)
        
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=mock_dataset,
            max_errors_per_epoch=None,  # No epoch limit
            max_consecutive_errors=5,   # But limit consecutive
            verbose=False
        )
        
        # Should fail due to consecutive limit even though epoch is unlimited
        with pytest.raises(RuntimeError, match="Error tolerance exceeded"):
            wrapper[0]
        
        assert wrapper.consecutive_errors == 5
        assert wrapper.epoch_errors == 5  # Same since all consecutive


class TestFaultTolerantIntegration:
    """Integration tests with real tensors and transformations."""
    
    def test_with_real_dataset(self):
        """Test with a real PyTorch dataset."""
        from torch.utils.data import TensorDataset
        
        # Create dataset with some bad samples
        data = []
        labels = []
        for i in range(10):
            if i in [2, 5, 7]:  # Bad samples
                data.append(torch.full((3, 32, 32), float('nan')))
            else:
                data.append(torch.randn(3, 32, 32))
            labels.append(torch.tensor(i))
        
        dataset = TensorDataset(torch.stack(data), torch.stack(labels))
        
        # Wrap with fault tolerance
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=dataset,
            max_errors_per_epoch=5,
            verbose=False
        )
        
        # Should skip bad samples
        valid_samples = []
        for i in range(10):
            sample = wrapper[i]
            if sample is not None:
                data_tensor, label = sample
                if not torch.isnan(data_tensor).any():
                    valid_samples.append(label.item())
        
        # Should have skipped indices 2, 5, 7
        assert 2 not in valid_samples
        assert 5 not in valid_samples
        assert 7 not in valid_samples
        assert len(wrapper.error_indices) >= 3
    
    def test_with_dataloader(self):
        """Test that it works with DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dataset
        data = torch.randn(100, 3, 32, 32)
        # Insert some NaN
        data[10] = float('nan')
        data[50] = float('nan')
        labels = torch.arange(100)
        
        dataset = TensorDataset(data, labels)
        
        # Wrap with fault tolerance
        wrapper = FaultTolerantDatasetWrapper(
            base_dataset=dataset,
            max_errors_per_epoch=10,
            verbose=False
        )
        
        # Create DataLoader
        loader = DataLoader(wrapper, batch_size=8, shuffle=False)
        
        # Should be able to iterate without crashing
        batch_count = 0
        for batch_data, batch_labels in loader:
            assert not torch.isnan(batch_data).any()
            batch_count += 1
        
        # Should have processed most batches despite errors
        assert batch_count > 10