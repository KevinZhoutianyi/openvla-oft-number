"""
Script to inspect HDF5 file structure.
"""
import h5py
import numpy as np

def inspect_hdf5(file_path):
    """Inspect the structure of an HDF5 file."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}, Dtype: {obj.dtype}")
                if obj.size > 0 and obj.size < 10:
                    print(f"{indent}   Value: {obj[()]}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name}")
        
        print("File structure:")
        f.visititems(print_structure)
        
        # Print root level keys
        print(f"\n\nRoot level keys: {list(f.keys())}")
        
        # Try to print some sample data
        print("\n" + "-"*80)
        print("Sample data inspection:")
        print("-"*80)
        
        # Check common keys
        for key in ['observation', 'obs', 'action', 'actions']:
            if key in f:
                print(f"\n'{key}' found at root level")
                if isinstance(f[key], h5py.Dataset):
                    print(f"  Shape: {f[key].shape}, Dtype: {f[key].dtype}")
                elif isinstance(f[key], h5py.Group):
                    print(f"  Subkeys: {list(f[key].keys())}")

if __name__ == "__main__":
    import sys
    
    """
    Usage:
        python scripts/inspect_hdf5.py [path_to_hdf5_file_or_directory]
    
    If no path is provided, defaults to:
        /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train
    """
    import os
    from pathlib import Path
    
    # Get path from command line or use default
    if len(sys.argv) > 1:
        train_dir = Path(sys.argv[1])
    else:
        train_dir = Path("/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train")
    hdf5_files = sorted(train_dir.glob("*.hdf5"))
    
    if hdf5_files:
        print(f"Found {len(hdf5_files)} HDF5 files in train directory")
        print(f"Inspecting first file: {hdf5_files[0]}")
        inspect_hdf5(str(hdf5_files[0]))
    else:
        print("No HDF5 files found!")

