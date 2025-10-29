# Evaluation Guide for Place Shoe Dataset

This guide explains how to evaluate your trained OpenVLA model on the validation set.

## Quick Start

After training completes, you'll have a checkpoint directory like:
```
/work/hdd/bfdj/tzhou4/checkpoints/openvla-7b+place_shoe+act14+b2+lr-0.0005+lora-r32/
```

To evaluate this checkpoint:

```bash
cd /u/tzhou4/fone/openvla-oft-number

export WANDB_MODE=disabled

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir /work/hdd/bfdj/tzhou4/checkpoints/openvla-7b+place_shoe+act14+b2+lr-0.0005+lora-r32 \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val \
  --batch_size 4
```

**Note**: You can use `batch_size=4` or higher for evaluation since there's no gradient computation (uses less memory).

## Metrics Explained

The evaluation script computes several metrics:

### 1. **Average Loss**
- The cross-entropy loss on the validation set
- Lower is better
- Typical range: 0.5-2.0 for well-trained models

### 2. **Action Token Accuracy**
- Percentage of action tokens predicted correctly (discrete)
- Range: 0.0 to 1.0 (0% to 100%)
- Higher is better
- **Interpretation**: 
  - > 0.8 (80%): Good
  - > 0.9 (90%): Very good
  - > 0.95 (95%): Excellent

### 3. **L1 Loss (Continuous Actions)**
- Mean absolute error between predicted and ground truth actions
- Lower is better
- **Interpretation**:
  - < 0.1: Excellent (actions are very close to ground truth)
  - 0.1 - 0.3: Good
  - 0.3 - 0.5: Acceptable
  - > 0.5: Poor (may need more training)

### 4. **Per-Dimension Errors**
- Shows error for each of the 14 action dimensions
- Helps identify which actions the model struggles with
- **Use this to**:
  - Identify problematic action dimensions
  - Debug whether certain joints/actions are harder to predict
  - Compare relative difficulty of different actions

## Output

The evaluation script produces:

1. **Console output**: Real-time metrics during evaluation
2. **JSON file**: Detailed results saved to `{checkpoint_dir}/eval_results.json`

Example JSON output:
```json
{
  "checkpoint_dir": "/work/hdd/bfdj/tzhou4/checkpoints/openvla-7b+...",
  "hdf5_path": "/work/hdd/bfdj/tzhou4/place_shoe_1/.../val",
  "num_samples": 1642,
  "avg_loss": 0.4523,
  "avg_action_token_accuracy": 0.8734,
  "avg_l1_loss": 0.1245,
  "per_dimension_stats": {
    "dim_0": {
      "mean_abs_error": 0.0823,
      "std_abs_error": 0.0645,
      "median_abs_error": 0.0712,
      "max_abs_error": 0.3421
    },
    ...
  }
}
```

## Evaluation During Training

You can also monitor training metrics in real-time:

1. **If using WandB** (online mode):
   - Go to https://wandb.ai/{your_entity}/openvla
   - View plots of loss, accuracy, and L1 loss over time

2. **If using WandB offline mode**:
   ```bash
   # After training
   cd {checkpoint_dir}
   wandb sync wandb/latest-run
   ```

3. **Check checkpoint directory**:
   ```bash
   ls /work/hdd/bfdj/tzhou4/checkpoints/openvla-7b+place_shoe+act14+b2+lr-0.0005+lora-r32/
   ```
   
   You'll see:
   - `adapter_config.json`: LoRA configuration
   - `adapter_model.safetensors`: Trained weights
   - `dataset_statistics.json`: Dataset normalization stats
   - `config.json`: Model configuration

## Comparing Multiple Checkpoints

To compare different training runs:

```bash
# Evaluate checkpoint 1
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir /work/hdd/bfdj/tzhou4/checkpoints/run1 \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val \
  --batch_size 4

# Evaluate checkpoint 2
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir /work/hdd/bfdj/tzhou4/checkpoints/run2 \
  --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val \
  --batch_size 4

# Compare eval_results.json from both runs
```

## What is "Success Rate"?

**Success rate** refers to the percentage of tasks completed successfully in:
- **Simulation**: Running the trained policy in a simulated environment (e.g., MuJoCo, PyBullet)
- **Real robot**: Deploying the policy on actual hardware

**Why it's not included here**:
- Requires a simulation environment or real robot setup
- The current evaluation only measures **prediction accuracy** on the validation set
- Success rate requires actually executing the actions and checking if the task goal was achieved

**To get success rate**, you would need to:
1. Set up a simulation or real robot environment
2. Load the trained model
3. Run episodes where the policy controls the robot
4. Check if the shoe placement task is completed successfully
5. Compute: `success_rate = successful_episodes / total_episodes`

## Tips

1. **Monitor training loss**: If training loss is still decreasing, train longer
2. **Compare train vs. val**: If train loss is much lower than val loss, you might be overfitting
3. **Check per-dimension errors**: High error in specific dimensions may indicate:
   - Those actions are harder to predict
   - Data for those actions may be noisier
   - You might need more data or different augmentations

## Troubleshooting

### "CUDA out of memory" during eval
- Reduce `--batch_size` (try 2 or 1)

### "Checkpoint not found"
- Check that the path is correct
- Make sure training completed and saved checkpoints

### Very high L1 loss (> 1.0)
- Model may need more training
- Check if normalization statistics are correct
- Verify dataset is loaded properly

### Action accuracy near 0%
- Something is wrong with the checkpoint or data
- Verify the checkpoint path points to a trained model
- Check dataset statistics

