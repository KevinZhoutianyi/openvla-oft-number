"""
Test script to verify the PlaceShoeDataset loads correctly.

Usage:
    python scripts/test_dataset.py [path_to_hdf5_directory]

If no path is provided, defaults to:
    /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train
"""
import sys
from pathlib import Path

# Add OpenVLA to path (script is in scripts/, so go up one level)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets import PlaceShoeDataset

def test_dataset(hdf5_path=None):
    """Test loading the place shoe dataset."""
    print("="*80)
    print("Testing PlaceShoeDataset")
    print("="*80)
    
    # Dataset path
    if hdf5_path is None:
        hdf5_path = Path("/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train")
    else:
        hdf5_path = Path(hdf5_path)
    
    # Load processor from pretrained model
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # Load dataset
    print(f"\nLoading dataset from: {hdf5_path}")
    dataset = PlaceShoeDataset(
        hdf5_path=hdf5_path,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        action_dim=14,
        use_relative_actions=False,
        camera_view="head",
        use_wrist_image=True,
        wrist_camera="left",
        use_proprio=False,
        predict_stop_token=True,
        compute_stats=True,
        default_instruction="place the shoe",
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Action dimension: 14")
    
    # Test loading a sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"\nSample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
        else:
            print(f"  {key}: {type(value)}, value: {value}")
    
    # Check dataset statistics
    print(f"\nDataset statistics:")
    for ds_name, stats in dataset.dataset_statistics.items():
        print(f"  Dataset: {ds_name}")
        for key, values in stats.items():
            if isinstance(values, dict):
                print(f"    {key}:")
                for stat_name, stat_val in values.items():
                    if hasattr(stat_val, 'shape'):
                        print(f"      {stat_name}: shape={stat_val.shape}, first_5={stat_val[:5]}")
                    else:
                        print(f"      {stat_name}: {stat_val}")
    
    print("\n" + "="*80)
    print("✓ Dataset test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    # Get path from command line or use default
    hdf5_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_dataset(hdf5_path)

