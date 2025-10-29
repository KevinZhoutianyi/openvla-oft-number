# Fine-tuning OpenVLA on Your HDF5 Dataset

## Dataset Structure

Your HDF5 dataset has been downloaded to `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/`

Structure:
- **train/**: Training episodes (48 HDF5 files)
- **val/**: Validation episodes

Each HDF5 file contains:
- `action`: (T, 14) - 14-dimensional action vectors
- `head_camera_image`: (T, 256, 256, 3) - Third-person view
- `left_wrist_image`: (T, 256, 256, 3) - Left wrist camera
- `right_wrist_image`: (T, 256, 256, 3) - Right wrist camera  
- `low_cam_image`: (T, 256, 256, 3) - Low camera view
- `relative_action`: (T, 14) - Relative actions
- `seen`: Language instructions for seen tasks
- `unseen`: Language instructions for unseen tasks

## Quick Start

### 1. Inspect Your Dataset

```bash
cd /work/hdd/bfdj/tzhou4
python3 inspect_hdf5.py
```

### 2. Customize the HDF5 Dataset Loader

A custom dataset loader has been created at:
`/u/tzhou4/fone/openvla-oft-number/prismatic/vla/datasets/place_shoe_dataset.py`

This loader is specifically configured for your data structure.

### 3. Run Fine-tuning

**Single GPU:**
```bash
cd /u/tzhou4/fone/openvla-oft-number

python vla-scripts/finetune_hdf5.py \
  --vla_path openvla/openvla-7b \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --max_steps 50000 \
  --save_freq 5000 \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio False \
  --wandb_entity "your-entity" \
  --wandb_project "place-shoe-finetune"
```

**Multi-GPU (4 GPUs):**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_hdf5.py \
  --vla_path openvla/openvla-7b \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 50000 \
  --save_freq 5000 \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --use_proprio False \
  --wandb_entity "your-entity" \
  --wandb_project "place-shoe-finetune"
```

## Key Configuration Options

- `--num_images_in_input 2`: Uses head camera + one wrist camera (you can use 1, 2, or 3)
- `--use_l1_regression True`: Use continuous action prediction (recommended)
- `--use_proprio False`: Your dataset doesn't have proprioceptive state
- `--batch_size`: Adjust based on GPU memory (4-8 per GPU recommended)
- `--max_steps`: Total training steps (50k-150k recommended)

## Dataset Statistics

The dataset will automatically compute action normalization statistics on first load.
These will be saved with your checkpoints for proper action denormalization during inference.

## Next Steps

1. **Test the data loader:**
   ```bash
   cd /u/tzhou4/fone/openvla-oft-number
   python -c "from prismatic.vla.datasets.place_shoe_dataset import PlaceShoeDataset; print('Dataset loaded successfully!')"
   ```

2. **Start training** with the command above

3. **Monitor training** via WandB or local logs

## Files Created

1. `/u/tzhou4/fone/openvla-oft-number/prismatic/vla/datasets/hdf5_dataset.py` - Generic HDF5 loader
2. `/u/tzhou4/fone/openvla-oft-number/prismatic/vla/datasets/place_shoe_dataset.py` - Custom loader for your data
3. `/u/tzhou4/fone/openvla-oft-number/vla-scripts/finetune_hdf5.py` - Training script for HDF5 datasets
4. `/work/hdd/bfdj/tzhou4/inspect_hdf5.py` - Script to inspect HDF5 structure

## Troubleshooting

- **OOM errors**: Reduce `--batch_size` or use fewer images (`--num_images_in_input 1`)
- **Action dimension mismatch**: Check that your actions are 14-dimensional (currently configured)
- **Language instruction issues**: The loader will try to extract from 'seen' and 'unseen' keys

Based on the [Google Drive folder](https://drive.google.com/drive/folders/1tVDHgSlxmRlAT-B9dZm0lb_lCfRieZH6) you provided.

