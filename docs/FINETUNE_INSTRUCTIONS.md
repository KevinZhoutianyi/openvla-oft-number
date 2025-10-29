# OpenVLA Fine-tuning Instructions for Place Shoe Dataset

## ✅ Setup Complete

Your HDF5 dataset has been successfully downloaded and set up for OpenVLA fine-tuning!

### Data Location
- **Raw data**: `/work/hdd/bfdj/tzhou4/place_shoe_1/`
- **Training data**: `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train/` (48 episodes)
- **Validation data**: `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val/`

### Dataset Structure
Each HDF5 file contains:
- `action`: (T, 14) - 14-dimensional action vectors
- `head_camera_image`: (T, 256, 256, 3) - Third-person view
- `left_wrist_image`: (T, 256, 256, 3) - Left wrist camera
- `right_wrist_image`: (T, 256, 256, 3) - Right wrist camera
- `low_cam_image`: (T, 256, 256, 3) - Low camera view
- `relative_action`: (T, 14) - Relative actions
- `seen`/`unseen`: Language instructions

---

## 🧪 Step 1: Test the Dataset Loader

First, verify that the dataset loads correctly:

```bash
cd /work/hdd/bfdj/tzhou4
python3 test_dataset.py
```

This will:
- Load the dataset
- Display dataset statistics
- Show a sample data point
- Verify all shapes and types are correct

---

## 🚀 Step 2: Start Fine-tuning

### Option A: Single GPU Training

```bash
cd /u/tzhou4/fone/openvla-oft-number

python vla-scripts/finetune_place_shoe.py \
  --vla_path openvla/openvla-7b \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --max_steps 50000 \
  --save_freq 5000 \
  --num_images_in_input 2 \
  --camera_view head \
  --wrist_camera left \
  --wandb_entity "your-wandb-entity" \
  --wandb_project "place-shoe-finetune"
```

### Option B: Multi-GPU Training (Recommended)

For 4 GPUs:

```bash
cd /u/tzhou4/fone/openvla-oft-number

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_place_shoe.py \
  --vla_path openvla/openvla-7b \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 50000 \
  --save_freq 5000 \
  --num_images_in_input 2 \
  --camera_view head \
  --wrist_camera left \
  --wandb_entity "your-wandb-entity" \
  --wandb_project "place-shoe-finetune"
```

---

## ⚙️ Configuration Options

### Camera Setup
- `--camera_view`: Primary camera view
  - Options: `head` (default), `low`, `left_wrist`, `right_wrist`
- `--wrist_camera`: Secondary wrist camera
  - Options: `left` (default), `right`, `both`
- `--num_images_in_input`: Total number of camera views
  - `1`: Primary camera only
  - `2`: Primary + wrist camera (recommended)
  - `3`: Primary + both wrist cameras

### Actions
- `--action_dim 14`: Action dimension (14 for this dataset)
- `--use_relative_actions`: Use `relative_action` instead of `action` (default: False)

### Training
- `--batch_size`: Batch size per GPU
  - Single GPU: 4-8 (depending on GPU memory)
  - Multi-GPU: 8-16 per GPU
- `--max_steps`: Total training steps
  - Recommended: 50,000-150,000
- `--learning_rate`: Learning rate (default: 5e-4)
- `--num_steps_before_decay`: LR decay step (default: 40,000)

### LoRA
- `--lora_rank`: LoRA rank (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.0)

---

## 📊 Monitoring Training

### Local Logs
Training logs will be saved to:
```
/work/hdd/bfdj/tzhou4/checkpoints/<run_id>/
```

### WandB (Recommended)
Update the WandB settings in the training command:
```bash
--wandb_entity "your-wandb-username" \
--wandb_project "place-shoe-finetune"
```

---

## 💾 Checkpoints

Checkpoints are saved every `--save_freq` steps to:
```
/work/hdd/bfdj/tzhou4/checkpoints/<run_id>--<step>_chkpt/
```

Each checkpoint contains:
- Merged VLA model (base + LoRA)
- LoRA adapter weights
- Action head weights
- Dataset statistics (for action denormalization)
- Processor config

---

## 🔍 Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 2 or 1)
- Use fewer images: `--num_images_in_input 1`
- Enable gradient checkpointing (requires code modification)

### Dataset Loading Issues
- Run the test script: `python3 /work/hdd/bfdj/tzhou4/test_dataset.py`
- Check HDF5 file integrity
- Verify file paths are correct

### Action Dimension Mismatch
- Verify your dataset has 14-dimensional actions
- Check the `action` key in your HDF5 files
- Run inspection: `python3 /work/hdd/bfdj/tzhou4/inspect_hdf5.py`

### Language Instructions
- Default instruction: "place the shoe"
- Modify with: `--default_instruction "your custom instruction"`
- The loader tries to extract from `seen`/`unseen` keys automatically

---

## 📁 Files Created

### OpenVLA Directory (`/u/tzhou4/fone/openvla-oft-number/`)
1. `prismatic/vla/datasets/hdf5_dataset.py` - Generic HDF5 loader
2. `prismatic/vla/datasets/place_shoe_dataset.py` - Custom loader for your data
3. `vla-scripts/finetune_hdf5.py` - Generic HDF5 training script
4. `vla-scripts/finetune_place_shoe.py` - Specialized training script

### Data Directory (`/work/hdd/bfdj/tzhou4/`)
1. `place_shoe_1/` - Your dataset
2. `inspect_hdf5.py` - HDF5 inspection script
3. `test_dataset.py` - Dataset testing script
4. `FINETUNE_INSTRUCTIONS.md` - This file
5. `checkpoints/` - Will be created during training

---

## 🎯 Quick Start Commands

**1. Test dataset:**
```bash
python3 /work/hdd/bfdj/tzhou4/test_dataset.py
```

**2. Start training (4 GPUs):**
```bash
cd /u/tzhou4/fone/openvla-oft-number
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 \
  --max_steps 50000
```

**3. Monitor logs:**
```bash
# Check latest checkpoint directory
ls -lht /work/hdd/bfdj/tzhou4/checkpoints/
```

---

## 📚 Additional Resources

- [OpenVLA Paper](https://arxiv.org/abs/2502.19645)
- [OpenVLA Project Website](https://openvla-oft.github.io/)
- [Google Drive Dataset](https://drive.google.com/drive/folders/1tVDHgSlxmRlAT-B9dZm0lb_lCfRieZH6)

---

## 💡 Tips for Best Results

1. **Start small**: Test with 1 GPU and few steps first
2. **Monitor L1 loss**: Should decrease to ~0.01 or lower
3. **Save checkpoints frequently**: Every 5k-10k steps
4. **Use multiple cameras**: `--num_images_in_input 2` often works better
5. **Train longer if needed**: 50k-150k steps depending on dataset size

---

## ✉️ Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test script to verify dataset loading
3. Check GPU memory usage with `nvidia-smi`
4. Review training logs in the checkpoint directory

Good luck with your fine-tuning! 🚀

