#!/bin/bash
# Quick test script with 4 GPUs: Train for 100 steps, save final checkpoint, and evaluate
# Usage: bash scripts/quick_test_4gpu.sh
# Note: Must be run on a node with 4 GPUs

set -e  # Exit on error

echo "======================================================================"
echo "Quick Test with 4 GPUs: Train 100 steps + Evaluate"
echo "======================================================================"

# Configuration
HDF5_TRAIN="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train"
HDF5_VAL="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val"
CHECKPOINT_DIR="/work/hdd/bfdj/tzhou4/checkpoints"
MAX_STEPS=100
BATCH_SIZE=2          # Per GPU batch size (total = 2 * 4 = 8)
EVAL_BATCH_SIZE=8     # Larger batch for eval since we only use 1 GPU
NUM_GPUS=4

# Disable WandB
export WANDB_MODE=disabled

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH=/u/tzhou4/fone/openvla-oft-number:$PYTHONPATH

cd /u/tzhou4/fone/openvla-oft-number

echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Training steps: ${MAX_STEPS}"
echo ""

echo ""
echo "Step 1: Evaluating BASE model (before training)..."
echo "----------------------------------------------------------------------"

# Create a temporary directory for base model results
BASE_RESULTS_DIR="${CHECKPOINT_DIR}/base_model_eval_4gpu"
mkdir -p ${BASE_RESULTS_DIR}

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir openvla/openvla-7b \
  --hdf5_path ${HDF5_VAL} \
  --batch_size ${EVAL_BATCH_SIZE} \
  --num_images_in_input 1 \
  --use_lora False \
  --output_file ${BASE_RESULTS_DIR}/eval_results_before_training.json

echo ""
echo "Step 2: Training for ${MAX_STEPS} steps on ${NUM_GPUS} GPUs..."
echo "----------------------------------------------------------------------"

torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
  vla-scripts/finetune_place_shoe.py \
  --hdf5_path ${HDF5_TRAIN} \
  --run_root_dir ${CHECKPOINT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --max_steps ${MAX_STEPS} \
  --save_freq ${MAX_STEPS} \
  --num_images_in_input 1 \
  --merge_lora_during_training False

echo ""
echo "Step 3: Finding the checkpoint directory..."
echo "----------------------------------------------------------------------"

# Find the most recent checkpoint directory (with --{MAX_STEPS}_chkpt suffix)
LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR}/openvla-7b+place_shoe*--${MAX_STEPS}_chkpt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    # Try finding any recent checkpoint
    LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR}/openvla-7b+place_shoe* 2>/dev/null | head -1)
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

echo "Found checkpoint: ${LATEST_CHECKPOINT}"

# Check if lora_adapter exists
if [ ! -d "${LATEST_CHECKPOINT}/lora_adapter" ]; then
    echo "ERROR: LoRA adapter directory not found"
    echo "Contents of ${LATEST_CHECKPOINT}:"
    ls -lh "${LATEST_CHECKPOINT}"
    exit 1
fi

echo ""
echo "Step 4: Evaluating TRAINED model (after training)..."
echo "----------------------------------------------------------------------"

# Use lora_adapter directory for evaluation (single GPU is enough)
EVAL_CHECKPOINT="${LATEST_CHECKPOINT}/lora_adapter"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir ${EVAL_CHECKPOINT} \
  --hdf5_path ${HDF5_VAL} \
  --batch_size ${EVAL_BATCH_SIZE} \
  --num_images_in_input 1 \
  --output_file ${EVAL_CHECKPOINT}/eval_results_after_training.json

echo ""
echo "======================================================================"
echo "Test Complete!"
echo "======================================================================"
echo "Checkpoint: ${LATEST_CHECKPOINT}"
echo "LoRA Adapter: ${EVAL_CHECKPOINT}"
echo ""

# Compare results
echo "COMPARISON: Before vs After Training (4 GPUs)"
echo "======================================================================"

if [ -f "${BASE_RESULTS_DIR}/eval_results_before_training.json" ] && [ -f "${EVAL_CHECKPOINT}/eval_results_after_training.json" ]; then
    echo ""
    echo "BEFORE Training (Base Model):"
    echo "----------------------------------------------------------------------"
    python3 -c "
import json
with open('${BASE_RESULTS_DIR}/eval_results_before_training.json') as f:
    r = json.load(f)
    print(f'  Loss: {r[\"avg_loss\"]:.6f}')
    print(f'  Action Accuracy: {r[\"avg_action_token_accuracy\"]*100:.2f}%')
    print(f'  L1 Loss: {r[\"avg_l1_loss\"]:.6f}')
"
    
    echo ""
    echo "AFTER Training (${MAX_STEPS} steps on ${NUM_GPUS} GPUs):"
    echo "----------------------------------------------------------------------"
    python3 -c "
import json
with open('${EVAL_CHECKPOINT}/eval_results_after_training.json') as f:
    r = json.load(f)
    print(f'  Loss: {r[\"avg_loss\"]:.6f}')
    print(f'  Action Accuracy: {r[\"avg_action_token_accuracy\"]*100:.2f}%')
    print(f'  L1 Loss: {r[\"avg_l1_loss\"]:.6f}')
"
    
    echo ""
    echo "IMPROVEMENT:"
    echo "----------------------------------------------------------------------"
    python3 -c "
import json
with open('${BASE_RESULTS_DIR}/eval_results_before_training.json') as f:
    before = json.load(f)
with open('${EVAL_CHECKPOINT}/eval_results_after_training.json') as f:
    after = json.load(f)

loss_change = after['avg_loss'] - before['avg_loss']
acc_change = (after['avg_action_token_accuracy'] - before['avg_action_token_accuracy']) * 100
l1_change = after['avg_l1_loss'] - before['avg_l1_loss']

print(f'  Loss: {loss_change:+.6f} ({\"better\" if loss_change < 0 else \"worse\"})')
print(f'  Action Accuracy: {acc_change:+.2f}% ({\"better\" if acc_change > 0 else \"worse\"})')
print(f'  L1 Loss: {l1_change:+.6f} ({\"better\" if l1_change < 0 else \"worse\"})')
"
    echo ""
else
    echo "Could not find both evaluation results for comparison"
fi

echo "======================================================================"
echo ""
echo "Training Configuration:"
echo "  GPUs used: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Total steps: ${MAX_STEPS}"
echo ""
echo "Results saved to:"
echo "  Before: ${BASE_RESULTS_DIR}/eval_results_before_training.json"
echo "  After:  ${EVAL_CHECKPOINT}/eval_results_after_training.json"
echo ""

