"""
Examples of dtype conversion configuration formats in DataPorter.

This module demonstrates both the list format (recommended for YAML configs)
and the dict format (legacy, still supported) for configuring dtype conversions.
"""

import torch
from dataporter import ResumableDataLoader, GenericDatasetWrapper
from dataporter.converters import KeyBasedDtypeConverter
from torch.utils.data import Dataset


# Example dataset for demonstration
class ExampleDataset(Dataset):
    """Simple dataset for demonstrating dtype conversions."""
    
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "observation.image": torch.randn(3, 224, 224, dtype=torch.float32),
            "observation.state": torch.randn(10, dtype=torch.float64),
            "action": torch.randn(5, dtype=torch.float32),
            "reward": torch.tensor([1.0], dtype=torch.float32),
            "done": torch.tensor([False], dtype=torch.bool),
            "metadata": {"episode": idx // 10, "step": idx % 10}
        }


def example_list_format():
    """
    Demonstrates the list format (recommended for YAML configurations).
    
    This format is preferred because:
    - No quotes needed for keys with dots in YAML
    - Consistent with YAML best practices
    - More readable and maintainable
    - Easily extensible for future features
    """
    print("=" * 60)
    print("LIST FORMAT EXAMPLE (Recommended)")
    print("=" * 60)
    
    # Create converter with list format
    converter = KeyBasedDtypeConverter([
        {"path": "observation.image", "dtype": "float16"},  # 50% memory reduction
        {"path": "observation.state", "dtype": "float32"},  # 50% reduction from float64
        {"path": "action", "dtype": "float16"},             # 50% reduction
        {"path": "reward", "dtype": "float16"},             # 50% reduction
        # "done" stays as bool (no conversion needed)
    ])
    
    # Create dataset and dataloader
    dataset = ExampleDataset(size=10)
    dataloader = ResumableDataLoader(
        dataset,
        batch_size=2,
        converter=converter  # Apply conversions
    )
    
    # Process one batch
    batch = next(iter(dataloader))
    
    print("\nDtype conversions applied:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key:20s}: {value.dtype}")
        else:
            print(f"  {key:20s}: {type(value).__name__}")
    
    # Calculate memory savings
    original_batch = dataset[0]
    original_size = sum(
        t.element_size() * t.nelement() 
        for t in original_batch.values() 
        if torch.is_tensor(t)
    )
    
    converted_size = sum(
        t[0].element_size() * t[0].nelement()  # Use first item in batch
        for t in batch.values() 
        if torch.is_tensor(t)
    )
    
    print(f"\nMemory reduction: {(1 - converted_size/original_size)*100:.1f}%")
    return converter


def example_dict_format():
    """
    Demonstrates the dict format (legacy, still supported).
    
    This format requires quoted keys in YAML when keys contain dots.
    It's still supported for backward compatibility.
    """
    print("\n" + "=" * 60)
    print("DICT FORMAT EXAMPLE (Legacy)")
    print("=" * 60)
    
    # Create converter with dict format
    converter = KeyBasedDtypeConverter({
        "observation.image": "float16",  # Requires quotes in YAML
        "observation.state": "float32",
        "action": "float16",
        "reward": "float16",
    })
    
    # Create dataset and dataloader
    dataset = ExampleDataset(size=10)
    dataloader = ResumableDataLoader(
        dataset,
        batch_size=2,
        converter=converter
    )
    
    # Process one batch
    batch = next(iter(dataloader))
    
    print("\nDtype conversions applied:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key:20s}: {value.dtype}")
        else:
            print(f"  {key:20s}: {type(value).__name__}")
    
    return converter


def example_yaml_configuration():
    """
    Shows how these formats look in YAML configuration files.
    """
    print("\n" + "=" * 60)
    print("YAML CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    list_format_yaml = """
# List format (recommended) - clean, no quotes needed
data:
  init_args:
    dtype_conversions:
      - path: observation.image
        dtype: float16
      - path: observation.state  
        dtype: float32
      - path: action
        dtype: float16
      - path: metadata.timestamp
        dtype: float32
    """
    
    dict_format_yaml = """
# Dict format (legacy) - requires quotes for keys with dots
data:
  init_args:
    dtype_conversions:
      "observation.image": "float16"  # Quotes required!
      "observation.state": "float32"  # Quotes required!
      "action": "float16"
      "metadata.timestamp": "float32"  # Quotes required!
    """
    
    print("\nList Format in YAML (Recommended):")
    print(list_format_yaml)
    
    print("\nDict Format in YAML (Legacy):")
    print(dict_format_yaml)


def example_error_handling():
    """
    Demonstrates error handling for invalid configurations.
    """
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLES")
    print("=" * 60)
    
    # Invalid list format - missing 'dtype' key
    try:
        converter = KeyBasedDtypeConverter([
            {"path": "observation.image"}  # Missing 'dtype'!
        ])
    except ValueError as e:
        print(f"\n✓ Correctly caught error for missing 'dtype': {e}")
    
    # Invalid list format - wrong key name
    try:
        converter = KeyBasedDtypeConverter([
            {"key": "observation.image", "dtype": "float16"}  # Wrong key name!
        ])
    except ValueError as e:
        print(f"\n✓ Correctly caught error for wrong key name: {e}")
    
    # Invalid dtype
    try:
        converter = KeyBasedDtypeConverter([
            {"path": "observation.image", "dtype": "invalid"}  # Invalid dtype!
        ])
    except ValueError as e:
        print(f"\n✓ Correctly caught error for invalid dtype: {e}")


def example_memory_optimization():
    """
    Demonstrates memory optimization strategies with dtype conversions.
    """
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION STRATEGIES")
    print("=" * 60)
    
    # Aggressive memory optimization
    aggressive_converter = KeyBasedDtypeConverter([
        {"path": "embeddings", "dtype": "float16"},        # 50% reduction
        {"path": "attention_mask", "dtype": "uint8"},      # 87.5% reduction from int64
        {"path": "token_ids", "dtype": "int16"},          # 75% reduction if vocab < 32k
        {"path": "position_ids", "dtype": "int16"},       # 75% reduction
        {"path": "token_type_ids", "dtype": "uint8"},     # 87.5% reduction (usually 0 or 1)
        {"path": "labels", "dtype": "int16"},             # 75% reduction for classification
    ])
    
    print("\nAggressive optimization for NLP:")
    for path, dtype in aggressive_converter.dtype_map.items():
        print(f"  {path:20s} -> {dtype:8s}")
    
    # Conservative optimization (preserve precision where needed)
    conservative_converter = KeyBasedDtypeConverter([
        {"path": "images", "dtype": "float16"},           # OK for images
        {"path": "logits", "dtype": "float32"},          # Keep precision for logits
        {"path": "losses", "dtype": "float32"},          # Keep precision for losses
        {"path": "masks", "dtype": "uint8"},             # Binary masks
        {"path": "indices", "dtype": "int32"},           # Index arrays
    ])
    
    print("\nConservative optimization (preserving precision):")
    for path, dtype in conservative_converter.dtype_map.items():
        print(f"  {path:20s} -> {dtype:8s}")


def main():
    """Run all examples."""
    
    # Show both formats produce identical results
    print("\n" + "=" * 60)
    print("COMPARING FORMATS")
    print("=" * 60)
    
    list_converter = example_list_format()
    dict_converter = example_dict_format()
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"List format dtype_map: {list_converter.dtype_map}")
    print(f"Dict format dtype_map: {dict_converter.dtype_map}")
    print(f"Formats are equivalent: {list_converter.dtype_map == dict_converter.dtype_map}")
    
    # Show YAML examples
    example_yaml_configuration()
    
    # Show error handling
    example_error_handling()
    
    # Show memory optimization strategies
    example_memory_optimization()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Recommendations:
1. Use list format for new YAML configurations
2. Dict format still works for backward compatibility
3. Choose dtypes based on precision requirements
4. Monitor memory usage to verify optimizations
5. Test model accuracy after dtype conversions
""")


if __name__ == "__main__":
    main()