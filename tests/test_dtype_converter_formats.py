"""
Tests for KeyBasedDtypeConverter with both dict and list formats.
"""

import pytest
import torch
from dataporter.converters import KeyBasedDtypeConverter


class TestDtypeConverterFormats:
    """Test suite for different dtype converter configuration formats."""
    
    def test_dict_format_basic(self):
        """Test basic dict format (legacy)."""
        converter = KeyBasedDtypeConverter({
            "image": "float16",
            "label": "int32",
            "mask": "uint8"
        })
        
        assert converter.dtype_map == {
            "image": "float16",
            "label": "int32",
            "mask": "uint8"
        }
        
    def test_list_format_basic(self):
        """Test basic list format (new)."""
        converter = KeyBasedDtypeConverter([
            {"path": "image", "dtype": "float16"},
            {"path": "label", "dtype": "int32"},
            {"path": "mask", "dtype": "uint8"}
        ])
        
        assert converter.dtype_map == {
            "image": "float16",
            "label": "int32",
            "mask": "uint8"
        }
    
    def test_list_format_with_nested_paths(self):
        """Test list format with nested paths containing dots."""
        converter = KeyBasedDtypeConverter([
            {"path": "observation.image", "dtype": "float16"},
            {"path": "observation.state", "dtype": "float32"},
            {"path": "metadata.timestamp", "dtype": "float64"}
        ])
        
        assert converter.dtype_map == {
            "observation.image": "float16",
            "observation.state": "float32",
            "metadata.timestamp": "float64"
        }
    
    def test_empty_configurations(self):
        """Test empty configurations."""
        # Empty dict
        converter1 = KeyBasedDtypeConverter({})
        assert converter1.dtype_map == {}
        
        # Empty list
        converter2 = KeyBasedDtypeConverter([])
        assert converter2.dtype_map == {}
        
        # None
        converter3 = KeyBasedDtypeConverter(None)
        assert converter3.dtype_map == {}
    
    def test_invalid_list_format_missing_path(self):
        """Test error when 'path' key is missing."""
        with pytest.raises(ValueError, match="List format requires dicts with 'path' and 'dtype' keys"):
            KeyBasedDtypeConverter([
                {"dtype": "float16"}  # Missing 'path'
            ])
    
    def test_invalid_list_format_missing_dtype(self):
        """Test error when 'dtype' key is missing."""
        with pytest.raises(ValueError, match="List format requires dicts with 'path' and 'dtype' keys"):
            KeyBasedDtypeConverter([
                {"path": "image"}  # Missing 'dtype'
            ])
    
    def test_invalid_list_format_wrong_key_names(self):
        """Test error when using wrong key names."""
        with pytest.raises(ValueError, match="List format requires dicts with 'path' and 'dtype' keys"):
            KeyBasedDtypeConverter([
                {"key": "image", "dtype": "float16"}  # Wrong key name
            ])
        
        with pytest.raises(ValueError, match="List format requires dicts with 'path' and 'dtype' keys"):
            KeyBasedDtypeConverter([
                {"path": "image", "target_dtype": "float16"}  # Wrong dtype key
            ])
    
    def test_invalid_list_format_not_dict(self):
        """Test error when list contains non-dict items."""
        with pytest.raises(ValueError, match="List format requires dicts with 'path' and 'dtype' keys"):
            KeyBasedDtypeConverter([
                "image:float16"  # String instead of dict
            ])
    
    def test_invalid_dtype_in_dict_format(self):
        """Test invalid dtype raises error in dict format."""
        with pytest.raises(ValueError, match="Unsupported dtype 'invalid_type'"):
            KeyBasedDtypeConverter({
                "image": "invalid_type"
            })
    
    def test_invalid_dtype_in_list_format(self):
        """Test invalid dtype raises error in list format."""
        with pytest.raises(ValueError, match="Unsupported dtype 'invalid_type'"):
            KeyBasedDtypeConverter([
                {"path": "image", "dtype": "invalid_type"}
            ])
    
    def test_conversion_with_dict_format(self):
        """Test actual conversion with dict format."""
        converter = KeyBasedDtypeConverter({
            "image": "float16",
            "label": "int32"
        })
        
        batch = {
            "image": torch.randn(2, 3, 224, 224, dtype=torch.float32),
            "label": torch.tensor([1, 0], dtype=torch.int64),
            "other": torch.tensor([1.0, 2.0], dtype=torch.float32)  # Not converted
        }
        
        converted = converter.convert_batch(batch)
        
        assert converted["image"].dtype == torch.float16
        assert converted["label"].dtype == torch.int32
        assert converted["other"].dtype == torch.float32  # Unchanged
    
    def test_conversion_with_list_format(self):
        """Test actual conversion with list format."""
        converter = KeyBasedDtypeConverter([
            {"path": "image", "dtype": "float16"},
            {"path": "label", "dtype": "int32"}
        ])
        
        batch = {
            "image": torch.randn(2, 3, 224, 224, dtype=torch.float32),
            "label": torch.tensor([1, 0], dtype=torch.int64),
            "other": torch.tensor([1.0, 2.0], dtype=torch.float32)  # Not converted
        }
        
        converted = converter.convert_batch(batch)
        
        assert converted["image"].dtype == torch.float16
        assert converted["label"].dtype == torch.int32
        assert converted["other"].dtype == torch.float32  # Unchanged
    
    def test_equivalence_of_formats(self):
        """Test that both formats produce identical results."""
        dict_converter = KeyBasedDtypeConverter({
            "observation.image": "float16",
            "action": "float32",
            "reward": "float16",
            "done": "uint8"
        })
        
        list_converter = KeyBasedDtypeConverter([
            {"path": "observation.image", "dtype": "float16"},
            {"path": "action", "dtype": "float32"},
            {"path": "reward", "dtype": "float16"},
            {"path": "done", "dtype": "uint8"}
        ])
        
        # Should have identical dtype maps
        assert dict_converter.dtype_map == list_converter.dtype_map
        
        # Create test batch
        batch = {
            "observation.image": torch.randn(2, 3, 64, 64, dtype=torch.float32),
            "action": torch.randn(2, 4, dtype=torch.float64),
            "reward": torch.tensor([1.0, 0.5], dtype=torch.float32),
            "done": torch.tensor([False, True], dtype=torch.bool),
            "info": {"step": 10}  # Non-tensor, should be unchanged
        }
        
        # Convert with both
        dict_result = dict_converter.convert_batch(batch.copy())
        list_result = list_converter.convert_batch(batch.copy())
        
        # Results should be identical
        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                assert dict_result[key].dtype == list_result[key].dtype
                assert torch.equal(dict_result[key], list_result[key])
            else:
                assert dict_result[key] == list_result[key]
    
    def test_mixed_valid_and_extra_keys_in_list(self):
        """Test that extra keys in list items are ignored with warning."""
        # This should work but might log warnings about unknown keys
        converter = KeyBasedDtypeConverter([
            {"path": "image", "dtype": "float16", "device": "cuda"},  # Extra 'device' key
            {"path": "label", "dtype": "int32", "comment": "class labels"}  # Extra 'comment' key
        ])
        
        # Should still work, ignoring extra keys
        assert converter.dtype_map == {
            "image": "float16",
            "label": "int32"
        }
    
    def test_supported_dtypes(self):
        """Test all supported dtype strings."""
        supported = [
            "float16", "bfloat16", "float32", "float64",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "bool"
        ]
        
        # Test with list format
        config = [{"path": f"tensor_{i}", "dtype": dtype} 
                  for i, dtype in enumerate(supported)]
        converter = KeyBasedDtypeConverter(config)
        
        # All should be in dtype_map
        for i, dtype in enumerate(supported):
            assert converter.dtype_map[f"tensor_{i}"] == dtype
    
    def test_case_sensitivity(self):
        """Test that dtype names are case-sensitive."""
        # These should fail (uppercase not supported)
        with pytest.raises(ValueError, match="Unsupported dtype 'Float16'"):
            KeyBasedDtypeConverter([
                {"path": "image", "dtype": "Float16"}
            ])
        
        with pytest.raises(ValueError, match="Unsupported dtype 'FLOAT16'"):
            KeyBasedDtypeConverter({"image": "FLOAT16"})


class TestDtypeConverterIntegration:
    """Integration tests with actual data processing."""
    
    def test_nested_batch_structure_with_list_format(self):
        """Test conversion with nested batch structures using list format."""
        converter = KeyBasedDtypeConverter([
            {"path": "observation.image", "dtype": "float16"},
            {"path": "observation.state", "dtype": "float32"},
            {"path": "action", "dtype": "float16"}
        ])
        
        batch = {
            "observation.image": torch.randn(4, 3, 224, 224, dtype=torch.float64),
            "observation.state": torch.randn(4, 10, dtype=torch.float64),
            "action": torch.randn(4, 5, dtype=torch.float32),
            "metadata": {"episode": 1, "step": 100}
        }
        
        converted = converter.convert_batch(batch)
        
        assert converted["observation.image"].dtype == torch.float16
        assert converted["observation.state"].dtype == torch.float32
        assert converted["action"].dtype == torch.float16
        assert converted["metadata"] == {"episode": 1, "step": 100}
    
    def test_memory_reduction_with_list_format(self):
        """Test that memory is actually reduced with conversions."""
        converter = KeyBasedDtypeConverter([
            {"path": "embeddings", "dtype": "float16"},  # float32 -> float16 (50% reduction)
            {"path": "ids", "dtype": "int32"},           # int64 -> int32 (50% reduction)
            {"path": "mask", "dtype": "uint8"}           # int64 -> uint8 (87.5% reduction)
        ])
        
        # Create large tensors
        batch = {
            "embeddings": torch.randn(100, 512, 768, dtype=torch.float32),
            "ids": torch.randint(0, 30000, (100, 512), dtype=torch.int64),
            "mask": torch.randint(0, 2, (100, 512), dtype=torch.int64)
        }
        
        # Calculate original size
        original_size = sum(t.element_size() * t.nelement() for t in batch.values())
        
        # Convert
        converted = converter.convert_batch(batch)
        
        # Calculate new size
        new_size = sum(t.element_size() * t.nelement() for t in converted.values())
        
        # Should have significant reduction
        reduction = 1 - (new_size / original_size)
        assert reduction > 0.5  # At least 50% reduction overall
        
        # Check individual reductions
        assert converted["embeddings"].element_size() == batch["embeddings"].element_size() / 2
        assert converted["ids"].element_size() == batch["ids"].element_size() / 2
        assert converted["mask"].element_size() == batch["mask"].element_size() / 8