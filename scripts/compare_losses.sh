#!/bin/bash
# Script to compare different regression loss types
# Trains for 100 steps with each loss type and evaluates them
# Usage: bash scripts/compare_losses.sh

set -e  # Exit on error

echo "======================================================================"
echo "Comparing Different Regression Loss Types"
echo "======================================================================"

# Configuration
HDF5_TRAIN="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train"
HDF5_VAL="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val"
CHECKPOINT_DIR="/work/hdd/bfdj/tzhou4/checkpoints"
MAX_STEPS=100
BATCH_SIZE=2
EVAL_BATCH_SIZE=4

# Loss types to test
LOSS_TYPES=("l1" "l2" "huber" "smooth_l1")

# Disable WandB
export WANDB_MODE=disabled

# Set PYTHONPATH to ensure imports work correctly
export PYTHONPATH=/u/tzhou4/fone/openvla-oft-number:$PYTHONPATH

cd /u/tzhou4/fone/openvla-oft-number

# Create results directory
RESULTS_DIR="${CHECKPOINT_DIR}/loss_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo ""
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Evaluate base model once (before any training)
echo "======================================================================"
echo "Step 0: Evaluating BASE model (for reference)..."
echo "======================================================================"

BASE_RESULTS="${RESULTS_DIR}/base_model_results.json"

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/eval_place_shoe.py \
  --checkpoint_dir openvla/openvla-7b \
  --hdf5_path ${HDF5_VAL} \
  --batch_size ${EVAL_BATCH_SIZE} \
  --num_images_in_input 1 \
  --use_lora False \
  --output_file ${BASE_RESULTS}

echo ""

# Loop through each loss type
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    echo "======================================================================"
    echo "Testing Loss Type: ${LOSS_TYPE^^}"
    echo "======================================================================"
    
    # Set huber_delta based on loss type
    if [ "$LOSS_TYPE" == "huber" ]; then
        HUBER_DELTA=1.0
    else
        HUBER_DELTA=1.0  # default, not used for other loss types
    fi
    
    echo ""
    echo "Step 1: Training with ${LOSS_TYPE} loss for ${MAX_STEPS} steps..."
    echo "----------------------------------------------------------------------"
    
    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
      vla-scripts/finetune_place_shoe.py \
      --hdf5_path ${HDF5_TRAIN} \
      --run_root_dir ${CHECKPOINT_DIR} \
      --batch_size ${BATCH_SIZE} \
      --max_steps ${MAX_STEPS} \
      --save_freq ${MAX_STEPS} \
      --num_images_in_input 1 \
      --merge_lora_during_training False \
      --regression_loss_type ${LOSS_TYPE} \
      --huber_delta ${HUBER_DELTA} \
      --run_id_note "loss_${LOSS_TYPE}"
    
    echo ""
    echo "Step 2: Finding checkpoint for ${LOSS_TYPE}..."
    echo "----------------------------------------------------------------------"
    
    # Find the checkpoint with this loss type
    CHECKPOINT=$(ls -td ${CHECKPOINT_DIR}/openvla-7b+place_shoe*loss_${LOSS_TYPE}*--${MAX_STEPS}_chkpt 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found for ${LOSS_TYPE}"
        continue
    fi
    
    echo "Found: ${CHECKPOINT}"
    
    if [ ! -d "${CHECKPOINT}/lora_adapter" ]; then
        echo "ERROR: LoRA adapter not found for ${LOSS_TYPE}"
        continue
    fi
    
    echo ""
    echo "Step 3: Evaluating model trained with ${LOSS_TYPE} loss..."
    echo "----------------------------------------------------------------------"
    
    EVAL_CHECKPOINT="${CHECKPOINT}/lora_adapter"
    EVAL_RESULTS="${RESULTS_DIR}/${LOSS_TYPE}_results.json"
    
    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
      vla-scripts/eval_place_shoe.py \
      --checkpoint_dir ${EVAL_CHECKPOINT} \
      --hdf5_path ${HDF5_VAL} \
      --batch_size ${EVAL_BATCH_SIZE} \
      --num_images_in_input 1 \
      --output_file ${EVAL_RESULTS}
    
    echo ""
    echo "Completed testing ${LOSS_TYPE} loss!"
    echo ""
done

# Generate comparison report
echo "======================================================================"
echo "FINAL COMPARISON REPORT"
echo "======================================================================"

python3 << 'EOF'
import json
import sys
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")

# Load base model results
base_file = results_dir / "base_model_results.json"
if base_file.exists():
    with open(base_file) as f:
        base_results = json.load(f)
    print("\n" + "="*80)
    print("BASE MODEL (Before Training)")
    print("="*80)
    print(f"  Loss:             {base_results['avg_loss']:.6f}")
    print(f"  Action Accuracy:  {base_results['avg_action_token_accuracy']*100:.2f}%")
    print(f"  L1 Loss:          {base_results['avg_l1_loss']:.6f}")
else:
    base_results = None
    print("\n⚠️  Base model results not found")

print("\n" + "="*80)
print("TRAINED MODELS (100 steps)")
print("="*80)

loss_types = ["l1", "l2", "huber", "smooth_l1"]
all_results = {}

for loss_type in loss_types:
    result_file = results_dir / f"{loss_type}_results.json"
    if result_file.exists():
        with open(result_file) as f:
            all_results[loss_type] = json.load(f)

if not all_results:
    print("\n❌ No trained model results found!")
    sys.exit(1)

# Print results for each loss type
for loss_type in loss_types:
    if loss_type not in all_results:
        continue
    
    r = all_results[loss_type]
    print(f"\n{loss_type.upper()} Loss:")
    print("-" * 80)
    print(f"  Loss:             {r['avg_loss']:.6f}")
    print(f"  Action Accuracy:  {r['avg_action_token_accuracy']*100:.2f}%")
    print(f"  L1 Loss:          {r['avg_l1_loss']:.6f}")
    
    if base_results:
        loss_improve = base_results['avg_loss'] - r['avg_loss']
        acc_improve = (r['avg_action_token_accuracy'] - base_results['avg_action_token_accuracy']) * 100
        l1_improve = base_results['avg_l1_loss'] - r['avg_l1_loss']
        
        print(f"  Improvement:")
        print(f"    Loss:           {loss_improve:+.6f} ({'+better' if loss_improve > 0 else 'worse'})")
        print(f"    Accuracy:       {acc_improve:+.2f}% ({'+better' if acc_improve > 0 else 'worse'})")
        print(f"    L1 Loss:        {l1_improve:+.6f} ({'+better' if l1_improve > 0 else 'worse'})")

# Find best performing loss
print("\n" + "="*80)
print("RANKING (by L1 Loss - lower is better)")
print("="*80)

sorted_by_l1 = sorted(all_results.items(), key=lambda x: x[1]['avg_l1_loss'])
for rank, (loss_type, r) in enumerate(sorted_by_l1, 1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank}."))
    print(f"{medal} {loss_type.upper():<12} L1 Loss: {r['avg_l1_loss']:.6f}  |  Accuracy: {r['avg_action_token_accuracy']*100:.2f}%")

print("\n" + "="*80)
print("RANKING (by Action Accuracy - higher is better)")
print("="*80)

sorted_by_acc = sorted(all_results.items(), key=lambda x: x[1]['avg_action_token_accuracy'], reverse=True)
for rank, (loss_type, r) in enumerate(sorted_by_acc, 1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank}."))
    print(f"{medal} {loss_type.upper():<12} Accuracy: {r['avg_action_token_accuracy']*100:.2f}%  |  L1 Loss: {r['avg_l1_loss']:.6f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

best_l1_loss, best_l1_results = sorted_by_l1[0]
best_acc_loss, best_acc_results = sorted_by_acc[0]

if best_l1_loss == best_acc_loss:
    print(f"\n✅ Clear winner: {best_l1_loss.upper()} loss")
    print(f"   - Best L1 Loss: {best_l1_results['avg_l1_loss']:.6f}")
    print(f"   - Best Accuracy: {best_acc_results['avg_action_token_accuracy']*100:.2f}%")
else:
    print(f"\n📊 Trade-off between metrics:")
    print(f"   - {best_l1_loss.upper()} has best L1 Loss ({best_l1_results['avg_l1_loss']:.6f})")
    print(f"   - {best_acc_loss.upper()} has best Accuracy ({best_acc_results['avg_action_token_accuracy']*100:.2f}%)")
    print(f"\n💡 Recommendation: Try {best_l1_loss.upper()} for better continuous action prediction")

print("\n" + "="*80)

EOF

echo ""
echo "======================================================================"
echo "Comparison Complete!"
echo "======================================================================"
echo ""
echo "All results saved to: ${RESULTS_DIR}"
echo ""
echo "Individual result files:"
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${LOSS_TYPE}_results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "  - ${LOSS_TYPE}: ${RESULT_FILE}"
    fi
done
echo ""

