# OpenVLA Fine-tuning Quick Reference

## 📝 TL;DR

**What is OpenVLA?**
- Vision-Language-Action model
- Takes: Images + Text → Outputs: Robot actions

**What you're training:**
- Your dataset: 48 episodes of "place the shoe" task
- Action space: 14 dimensions
- Using LoRA (only ~64M params trainable, not 7B)

**Input/Output:**
```python
Input:  images (224×224) + "place the shoe"
Output: 8 future actions × 14 dimensions = (8, 14) array
```

## 🚀 Essential Commands

```bash
# Navigate to project
cd /u/tzhou4/fone/openvla-oft-number

# Test dataset (DO THIS FIRST!)
python scripts/test_dataset.py

# Train (4 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune_place_shoe.py \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
  --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
  --batch_size 8 --max_steps 50000
```

## 📊 Data Format

### Input
```python
{
    'pixel_values': (3, 224, 224),        # Head camera
    'pixel_values_wrist': (3, 224, 224),  # Wrist camera
    'input_ids': (seq_len,),              # "What action...?"
}
```

### Output
```python
{
    'actions': (8, 14),  # 8 timesteps, 14 action dimensions
    # actions[0] = current action
    # actions[1:8] = next 7 actions
}
```

## 🎯 Training Metrics

**Target Loss:** < 0.01 (excellent: < 0.008)

```
Step 1k:   loss=0.120
Step 5k:   loss=0.035
Step 10k:  loss=0.015  ✓ Good
Step 20k:  loss=0.010  ✓ Great
Step 50k:  loss=0.007  ✓ Excellent
```

## 🔧 Common Adjustments

```bash
# Out of memory? Reduce batch size
--batch_size 4

# Use fewer images
--num_images_in_input 1

# Train longer
--max_steps 100000

# Change cameras
--camera_view low --wrist_camera right
```

## 📁 Key Files

**Read these for details:**
1. `docs/DATA_FORMAT_EXPLAINED.md` - Full data format explanation
2. `docs/ARCHITECTURE_DIAGRAM.md` - Visual diagrams
3. `docs/QUICK_START.md` - All commands

**Code:**
- Dataset: `prismatic/vla/datasets/place_shoe_dataset.py`
- Training: `vla-scripts/finetune_place_shoe.py`
- Action head: `prismatic/models/action_heads.py`

## 🎓 What's Being Trained

```
OpenVLA-7B Base (7B params) - FROZEN
├── + LoRA adapters (~14M params) - TRAINABLE ✓
└── + Action head (~50M params) - TRAINABLE ✓
```

**Total trainable: ~64M params (not 7B!)**

## 📈 Monitor Training

```bash
# Check latest checkpoint
ls -lht /work/hdd/bfdj/tzhou4/checkpoints/

# GPU memory
watch -n 1 nvidia-smi
```

## ❓ Quick Answers

**Q: What are the 14 action dimensions?**
A: Depends on your robot (positions, velocities, gripper, etc.)

**Q: Why 8 timesteps?**
A: Action chunking for temporal consistency and smoother motion

**Q: How long to train?**
A: 50k steps = ~6-8 hours on 4 A100 GPUs (depends on batch size)

**Q: What if I run out of memory?**
A: Reduce `--batch_size` or use `--num_images_in_input 1`

**Q: Where are checkpoints saved?**
A: `/work/hdd/bfdj/tzhou4/checkpoints/<run_id>--<step>_chkpt/`

## 🔍 Debug Checklist

- [ ] `python scripts/test_dataset.py` runs successfully
- [ ] See "Loaded X transitions" message
- [ ] Action statistics show reasonable ranges
- [ ] Sample shapes are correct: (3, 224, 224) for images
- [ ] Loss starts high (~1.5) and decreases
- [ ] Checkpoints appear in the checkpoints directory

## 📞 Need Help?

1. Check `docs/DATA_FORMAT_EXPLAINED.md` for detailed explanation
2. Look at `docs/ARCHITECTURE_DIAGRAM.md` for visual overview
3. Run `python scripts/test_dataset.py` to debug data issues
4. Check `docs/FINETUNE_INSTRUCTIONS.md` for troubleshooting

---

**Dataset:** `/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train`  
**Project:** `/u/tzhou4/fone/openvla-oft-number/`  
**Docs:** All in `docs/` folder

