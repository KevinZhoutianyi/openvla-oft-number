# HDF5 Dataset Fine-tuning for OpenVLA

This directory contains everything needed to fine-tune OpenVLA on HDF5 datasets, specifically configured for the "Place Shoe" task.

## 📁 Project Structure

```
/u/tzhou4/fone/openvla-oft-number/
├── docs/                           # Documentation
│   ├── QUICK_START.md             # Quick start guide (START HERE!)
│   ├── FINETUNE_INSTRUCTIONS.md   # Detailed instructions
│   ├── README_FINETUNING.md       # Dataset-specific info
│   └── SETUP_COMPLETE.md          # Setup summary
│
├── scripts/                        # Utility scripts
│   ├── test_dataset.py            # Test dataset loading
│   └── inspect_hdf5.py            # Inspect HDF5 file structure
│
├── vla-scripts/                    # Training scripts
│   ├── finetune_place_shoe.py     # Training script (Place Shoe specific)
│   ├── finetune_hdf5.py           # Training script (Generic HDF5)
│   └── finetune.py                # Original RLDS training script
│
└── prismatic/vla/datasets/         # Dataset loaders
    ├── place_shoe_dataset.py      # PlaceShoeDataset loader
    ├── hdf5_dataset.py            # Generic HDF5Dataset loader
    └── datasets.py                # Original RLDS datasets
```

## 🗂️ Dataset Location

**Data is stored separately in:**
```
/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/
├── train/    # 48 HDF5 training episodes
└── val/      # Validation episodes
```

## 🚀 Quick Start

### 1. Test Dataset Loading
```bash
cd /u/tzhou4/fone/openvla-oft-number
python scripts/test_dataset.py
```

### 2. Inspect HDF5 Files
```bash
python scripts/inspect_hdf5.py
```

### 3. Start Training (4 GPUs)
```bash
cd /u/tzhou4/fone/openvla-oft-number

torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 \
  --max_steps 50000 \
  --num_images_in_input 2
```

## 📖 Documentation

**New to OpenVLA? Start here:**
1. **`docs/QUICK_REFERENCE.md`** - 1-page quick reference (TL;DR)
2. **`docs/DATA_FORMAT_EXPLAINED.md`** - Understand input/output format
3. **`docs/ARCHITECTURE_DIAGRAM.md`** - Visual diagrams & architecture
4. **`docs/QUICK_START.md`** - Quick commands and examples

**Detailed guides:**
- **FINETUNE_INSTRUCTIONS.md** - Step-by-step training instructions
- **README_FINETUNING.md** - Dataset-specific configuration
- **SETUP_COMPLETE.md** - Setup summary

## 🛠️ What's Included

### Custom Dataset Loaders

1. **PlaceShoeDataset** (`prismatic/vla/datasets/place_shoe_dataset.py`)
   - Specialized for your HDF5 dataset structure
   - Handles 14-dimensional actions
   - Supports multiple camera views (head, low, left_wrist, right_wrist)
   - Automatic action normalization
   - Language instruction extraction

2. **HDF5Dataset** (`prismatic/vla/datasets/hdf5_dataset.py`)
   - Generic HDF5 loader for other datasets
   - Customizable for different HDF5 structures

### Training Scripts

1. **finetune_place_shoe.py** - Optimized for Place Shoe dataset
2. **finetune_hdf5.py** - Generic HDF5 training script

### Utility Scripts

1. **test_dataset.py** - Verify dataset loads correctly
2. **inspect_hdf5.py** - Examine HDF5 file structure

## 📊 Dataset Structure

Your HDF5 files contain:
```
├── action: (T, 14) float64          - 14D action vectors
├── head_camera_image: (T, 256, 256, 3) uint8
├── left_wrist_image: (T, 256, 256, 3) uint8
├── right_wrist_image: (T, 256, 256, 3) uint8
├── low_cam_image: (T, 256, 256, 3) uint8
├── relative_action: (T, 14) float64
└── seen/unseen: language instructions
```

## ⚙️ Common Commands

All commands should be run from: `/u/tzhou4/fone/openvla-oft-number/`

```bash
# Navigate to project directory
cd /u/tzhou4/fone/openvla-oft-number

# Test dataset
python scripts/test_dataset.py

# Inspect HDF5 structure
python scripts/inspect_hdf5.py

# Train (single GPU)
python vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 4 --max_steps 50000

# Train (multi-GPU, 4 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 --max_steps 50000

# Check checkpoints
ls -lht /work/hdd/bfdj/tzhou4/checkpoints/
```

## 🔧 Configuration Options

### Camera Setup
- `--camera_view [head|low|left_wrist|right_wrist]` - Primary camera
- `--wrist_camera [left|right|both]` - Secondary wrist camera
- `--num_images_in_input [1|2|3]` - Number of camera views to use

### Training Parameters
- `--batch_size 8` - Batch size per GPU
- `--max_steps 50000` - Total training steps
- `--learning_rate 5e-4` - Learning rate
- `--save_freq 5000` - Checkpoint frequency
- `--lora_rank 32` - LoRA rank

### Actions
- `--action_dim 14` - Action dimension (14 for this dataset)
- `--use_relative_actions` - Use relative actions instead of absolute

## 🔍 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 2

# Use fewer images
--num_images_in_input 1
```

### Dataset Issues
```bash
# Test loading
python scripts/test_dataset.py

# Inspect structure
python scripts/inspect_hdf5.py
```

### Import Errors
Make sure you're in the project directory:
```bash
cd /u/tzhou4/fone/openvla-oft-number
```

## 📦 Checkpoints

Checkpoints are saved to:
```
/work/hdd/bfdj/tzhou4/checkpoints/<run_id>--<step>_chkpt/
```

Each checkpoint contains:
- Merged VLA model
- LoRA adapter
- Action head
- Dataset statistics
- Processor config

## ✅ Quick Checklist

Before training:
- [ ] Dataset is in `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train`
- [ ] Test dataset: `python scripts/test_dataset.py`
- [ ] Inspect files: `python scripts/inspect_hdf5.py`

During training:
- [ ] Monitor GPU memory: `nvidia-smi`
- [ ] Check checkpoints: `ls -lht /work/hdd/bfdj/tzhou4/checkpoints/`
- [ ] Watch L1 loss (should decrease to ~0.01)

## 📚 Additional Resources

- [OpenVLA Paper](https://arxiv.org/abs/2502.19645)
- [Project Website](https://openvla-oft.github.io/)
- Original LIBERO instructions: `LIBERO.md`
- Original ALOHA instructions: `ALOHA.md`

## 💡 Tips

1. **Start small**: Test with 1 GPU and few steps first
2. **Monitor training**: Watch L1 loss in logs
3. **Save frequently**: Use `--save_freq 5000`
4. **Multiple cameras**: `--num_images_in_input 2` often works best
5. **Train longer**: 50k-150k steps depending on dataset size

---

**All code and scripts are in this directory.**  
**Dataset is stored separately in `/work/hdd/bfdj/tzhou4/`**

For questions or issues, see `docs/FINETUNE_INSTRUCTIONS.md` for troubleshooting.

