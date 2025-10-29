"""
Generate presentation figures using only standard library + h5py + numpy
Creates simple visualizations without matplotlib
"""
import json
import numpy as np
from pathlib import Path
import h5py

# Results directory
results_dir = Path("/work/hdd/bfdj/tzhou4/checkpoints/loss_comparison_4gpu_20251029_140834")

print("="*80)
print("Extracting Results for Presentation")
print("="*80)

# Load results
print("\n1. Loading results...")
base_file = results_dir / "base_model_results.json"
with open(base_file) as f:
    base_results = json.load(f)

loss_types = ["l1", "l2", "huber", "smooth_l1"]
all_results = {}
for loss_type in loss_types:
    result_file = results_dir / f"{loss_type}_results.json"
    if result_file.exists():
        with open(result_file) as f:
            all_results[loss_type] = json.load(f)

print("   ✓ Loaded results")

# Print summary table
print("\n" + "="*80)
print("RESULTS SUMMARY TABLE (for presentation)")
print("="*80)

print("\nBase Model (Before Training):")
print(f"  Loss:             {base_results['avg_loss']:.6f}")
print(f"  Action Accuracy:  {base_results['avg_action_token_accuracy']*100:.2f}%")
print(f"  L1 Loss:          {base_results['avg_l1_loss']:.6f}")

print("\nTrained Models (100 steps, 4 GPUs):")
print("-"*80)
print(f"{'Loss Type':<12} | {'Val Loss':>10} | {'Accuracy':>10} | {'L1 Loss':>10}")
print("-"*80)
for loss_type in loss_types:
    if loss_type in all_results:
        r = all_results[loss_type]
        print(f"{loss_type.upper():<12} | {r['avg_loss']:>10.6f} | {r['avg_action_token_accuracy']*100:>9.2f}% | {r['avg_l1_loss']:>10.6f}")

# Extract dataset sample images
print("\n" + "="*80)
print("2. Extracting dataset sample images...")
print("="*80)

hdf5_dir = Path("/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train")
hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))

if hdf5_files:
    print(f"\n   Found {len(hdf5_files)} HDF5 files")
    
    # Create output directory for raw images
    output_dir = Path("dataset_images")
    output_dir.mkdir(exist_ok=True)
    
    # Open first file
    with h5py.File(hdf5_files[0], 'r') as f:
        if 'observations' in f and 'images' in f['observations']:
            images_group = f['observations']['images']
            if 'head' in images_group:
                head_images = images_group['head'][:]
                
                print(f"   Total images in first episode: {len(head_images)}")
                print(f"   Image shape: {head_images[0].shape}")
                
                # Save 4 sample images at different points in trajectory
                indices = [0, len(head_images)//4, len(head_images)//2, 3*len(head_images)//4]
                
                for i, idx in enumerate(indices):
                    if idx < len(head_images):
                        img = head_images[idx]
                        # Save as numpy array (you can convert to PNG using PIL/cv2 later or in Overleaf)
                        np.save(output_dir / f"sample_{i}_step{idx}.npy", img)
                        print(f"   ✓ Saved sample_{i}_step{idx}.npy (shape: {img.shape})")
                
                # Also save the first one as main sample
                np.save(output_dir / "dataset_sample.npy", head_images[0])
                print(f"   ✓ Saved dataset_sample.npy")
                
                print(f"\n   → Images saved as numpy arrays in: {output_dir.absolute()}")
                print(f"   → You can convert them to PNG using:")
                print(f"      from PIL import Image; import numpy as np")
                print(f"      img = np.load('sample_0.npy'); Image.fromarray(img).save('sample_0.png')")
else:
    print("   ⚠ No HDF5 files found")

# Create data files for plotting
print("\n" + "="*80)
print("3. Generating data files for plotting...")
print("="*80)

data_dir = Path("plot_data")
data_dir.mkdir(exist_ok=True)

# Comparison data
comparison_data = {
    'loss_types': ['Base Model', 'L1', 'L2', 'Huber', 'Smooth L1'],
    'accuracies': [base_results['avg_action_token_accuracy']*100] + 
                  [all_results[lt]['avg_action_token_accuracy']*100 for lt in loss_types if lt in all_results],
    'l1_losses': [base_results['avg_l1_loss']] + 
                 [all_results[lt]['avg_l1_loss'] for lt in loss_types if lt in all_results],
    'val_losses': [base_results['avg_loss']] + 
                  [all_results[lt]['avg_loss'] for lt in loss_types if lt in all_results],
}

with open(data_dir / "comparison_data.json", 'w') as f:
    json.dump(comparison_data, f, indent=2)
print("   ✓ Saved comparison_data.json")

# Per-dimension errors (from L1 model)
if 'l1' in all_results:
    per_dim_data = {
        'dimensions': list(range(14)),
        'mean_errors': [all_results['l1']['per_dimension_stats'][f'dim_{i}']['mean_abs_error'] for i in range(14)],
        'std_errors': [all_results['l1']['per_dimension_stats'][f'dim_{i}']['std_abs_error'] for i in range(14)],
    }
    with open(data_dir / "per_dim_errors.json", 'w') as f:
        json.dump(per_dim_data, f, indent=2)
    print("   ✓ Saved per_dim_errors.json")

# Improvement data
improvement_data = {
    'loss_types': [lt.upper() for lt in loss_types if lt in all_results],
    'accuracy_improvement': [(all_results[lt]['avg_action_token_accuracy'] - base_results['avg_action_token_accuracy'])*100 
                             for lt in loss_types if lt in all_results],
    'l1_improvement': [base_results['avg_l1_loss'] - all_results[lt]['avg_l1_loss'] 
                       for lt in loss_types if lt in all_results],
}

with open(data_dir / "improvement_data.json", 'w') as f:
    json.dump(improvement_data, f, indent=2)
print("   ✓ Saved improvement_data.json")

print("\n" + "="*80)
print("DATA EXTRACTION COMPLETE")
print("="*80)
print(f"\nData files saved to: {data_dir.absolute()}")
print(f"Image files saved to: {output_dir.absolute() if hdf5_files else 'N/A'}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("\n1. Convert numpy images to PNG (requires PIL/Pillow):")
print("   cd dataset_images")
print("   python3 -c \"")
print("from PIL import Image")
print("import numpy as np")
print("from pathlib import Path")
print("for npy_file in Path('.').glob('*.npy'):")
print("    img = np.load(npy_file)")
print("    Image.fromarray(img).save(npy_file.stem + '.png')")
print('   "')

print("\n2. Use plot_data/*.json to create charts in your preferred tool")
print("   - comparison_data.json -> bar charts")
print("   - per_dim_errors.json -> bar chart with error bars")
print("   - improvement_data.json -> improvement bar charts")

print("\n3. Copy presentation.tex and figures to Overleaf")

print("\n" + "="*80)

