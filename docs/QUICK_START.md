# OpenVLA Place Shoe Dataset - Quick Start Guide

## 📁 File Organization

All code and scripts are in: `/u/tzhou4/fone/openvla-oft-number/`
Dataset is in: `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/`

## 🧪 Step 1: Test the Dataset

```bash
cd /u/tzhou4/fone/openvla-oft-number
python scripts/test_dataset.py
```

Or specify a custom path:
```bash
python scripts/test_dataset.py /path/to/your/hdf5/files
```

## 🔍 Step 2: Inspect HDF5 Files

```bash
cd /u/tzhou4/fone/openvla-oft-number
python scripts/inspect_hdf5.py
```

Or inspect a specific file:
```bash
python scripts/inspect_hdf5.py /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train/episode_0.hdf5
```

## 🚀 Step 3: Start Training

### Single GPU
```bash
cd /u/tzhou4/fone/openvla-oft-number

python vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 4 \
  --max_steps 50000 \
  --num_images_in_input 2
```

### Multi-GPU (4 GPUs - Recommended)
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

## ⚙️ Key Configuration Options

### Camera Setup
- `--camera_view head` - Use head camera (default)
- `--camera_view low` - Use low camera
- `--wrist_camera left` - Use left wrist camera (default)
- `--wrist_camera right` - Use right wrist camera
- `--num_images_in_input 2` - Use primary + wrist camera

### Training Parameters
- `--batch_size 8` - Batch size per GPU (adjust based on memory)
- `--max_steps 50000` - Total training steps
- `--learning_rate 5e-4` - Learning rate
- `--save_freq 5000` - Save checkpoint every N steps

### Actions
- `--action_dim 14` - Action dimension (14 for this dataset)
- `--use_relative_actions` - Use relative actions instead of absolute

## 📊 Monitor Training

Checkpoints are saved to:
```
/work/hdd/bfdj/tzhou4/checkpoints/<run_id>--<step>_chkpt/
```

Check latest checkpoint:
```bash
ls -lht /work/hdd/bfdj/tzhou4/checkpoints/
```

## 📁 Project Structure

```
/u/tzhou4/fone/openvla-oft-number/
├── docs/
│   ├── QUICK_START.md              # This file
│   ├── FINETUNE_INSTRUCTIONS.md    # Detailed instructions
│   ├── README_FINETUNING.md        # Dataset-specific info
│   └── SETUP_COMPLETE.md           # Setup summary
├── scripts/
│   ├── test_dataset.py             # Test dataset loading
│   └── inspect_hdf5.py             # Inspect HDF5 structure
├── vla-scripts/
│   ├── finetune_place_shoe.py      # Training script (specialized)
│   └── finetune_hdf5.py            # Training script (generic)
└── prismatic/vla/datasets/
    ├── place_shoe_dataset.py       # PlaceShoe dataset loader
    └── hdf5_dataset.py             # Generic HDF5 loader
```

## 🔧 Troubleshooting

### Out of Memory
- Reduce `--batch_size` to 2 or 1
- Use single image: `--num_images_in_input 1`

### Dataset Issues
Run the test script to debug:
```bash
python scripts/test_dataset.py
```

### Check HDF5 Structure
```bash
python scripts/inspect_hdf5.py
```

## 📖 More Documentation

- **Detailed guide**: `docs/FINETUNE_INSTRUCTIONS.md`
- **Setup info**: `docs/SETUP_COMPLETE.md`
- **Dataset info**: `docs/README_FINETUNING.md`

## ✅ Quick Checklist

- [ ] Test dataset loads: `python scripts/test_dataset.py`
- [ ] Inspect HDF5 files: `python scripts/inspect_hdf5.py`
- [ ] Start training with command above
- [ ] Monitor checkpoints in `/work/hdd/bfdj/tzhou4/checkpoints/`

Good luck with your fine-tuning! 🚀

