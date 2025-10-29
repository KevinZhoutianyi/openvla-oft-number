#!/bin/bash
# Compare different regression losses with 4 GPUs
# This version correctly evaluates the REGRESSION HEAD, not just token prediction

set -e

export PYTHONPATH=/u/tzhou4/fone/openvla-oft-number:$PYTHONPATH
export WANDB_MODE=disabled

# Paths
HDF5_TRAIN="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train"
HDF5_VAL="/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val"
CHECKPOINT_ROOT="/work/hdd/bfdj/tzhou4/checkpoints"

# Training config
BATCH_SIZE=2  # Per GPU
MAX_STEPS=100
HUBER_DELTA=1.0

# Create unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${CHECKPOINT_ROOT}/loss_comparison_regression_4gpu_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "========================================================================"
echo "     REGRESSION HEAD EVALUATION - Loss Function Comparison"
echo "========================================================================"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Loss types to compare
LOSS_TYPES=("l1" "l2" "huber" "smooth_l1")

echo "========================================================================"
echo "Step 1: Evaluate BASE model (no fine-tuning) - SKIPPED"
echo "========================================================================"
echo "Note: Base model doesn't have regression head, so we skip this"
echo ""

# Train and evaluate each loss type
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Step 2.${LOSS_TYPE}: Training with ${LOSS_TYPE} Loss (4 GPUs)"
    echo "======================================================================"
    
    CURRENT_RUN_DIR="${RESULTS_DIR}/${LOSS_TYPE}_checkpoint"
    
    echo "Training..."
    torchrun --standalone --nnodes 1 --nproc-per-node 4 \
        vla-scripts/finetune_place_shoe.py \
        --hdf5_path ${HDF5_TRAIN} \
        --run_root_dir ${CURRENT_RUN_DIR} \
        --batch_size ${BATCH_SIZE} \
        --max_steps ${MAX_STEPS} \
        --save_freq ${MAX_STEPS} \
        --num_images_in_input 1 \
        --merge_lora_during_training False \
        --regression_loss_type ${LOSS_TYPE} \
        --huber_delta ${HUBER_DELTA}
    
    echo ""
    echo "======================================================================"
    echo "Step 3.${LOSS_TYPE}: Evaluating ${LOSS_TYPE} model with REGRESSION HEAD"
    echo "======================================================================"
    
    # Find the checkpoint directory
    CHECKPOINT_DIR=$(find ${CURRENT_RUN_DIR} -maxdepth 1 -type d -name "*--${MAX_STEPS}_chkpt" | head -1)
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Could not find checkpoint directory for ${LOSS_TYPE}"
        continue
    fi
    
    echo "Evaluating checkpoint: ${CHECKPOINT_DIR}"
    
    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
        vla-scripts/eval_place_shoe_regression.py \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --hdf5_path ${HDF5_VAL} \
        --batch_size 4 \
        --num_images_in_input 1 \
        --use_lora True \
        --regression_loss_type ${LOSS_TYPE} \
        --huber_delta ${HUBER_DELTA} \
        --output_file ${RESULTS_DIR}/${LOSS_TYPE}_results.json
    
    echo "✓ ${LOSS_TYPE} evaluation complete"
done

echo ""
echo "========================================================================"
echo "Step 4: Generating Comparison Report"
echo "========================================================================"

# Generate comparison report
python3 << PYTHON_EOF
import json
import sys
from pathlib import Path
import os

results_dir = Path("${RESULTS_DIR}")

print("\n" + "="*80)
print("REGRESSION HEAD EVALUATION RESULTS (100 steps, 4 GPUs)")
print("="*80)

loss_types = ["l1", "l2", "huber", "smooth_l1"]
all_results = {}

for loss_type in loss_types:
    result_file = results_dir / f"{loss_type}_results.json"
    if result_file.exists():
        with open(result_file) as f:
            all_results[loss_type] = json.load(f)

if not all_results:
    print("\n❌ No results found!")
    sys.exit(1)

# Print results table
print("\nRESULTS TABLE:")
print("-"*80)
print(f"{'Loss Type':<12} | {'L1 Loss':>10}")
print("-"*80)
for loss_type in loss_types:
    if loss_type in all_results:
        r = all_results[loss_type]
        print(f"{loss_type.upper():<12} | {r['avg_l1_loss']:>10.6f}")
print("-"*80)

# Rankings
print("\n" + "="*80)
print("RANKING (by L1 Loss - lower is better)")
print("="*80)

sorted_by_l1 = sorted(all_results.items(), key=lambda x: x[1]['avg_l1_loss'])
for rank, (loss_type, r) in enumerate(sorted_by_l1, 1):
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank}."))
    print(f"{medal} {loss_type.upper():<12} L1 Loss: {r['avg_l1_loss']:.6f}")

# Per-dimension analysis for best model
best_loss_type, best_results = sorted_by_l1[0]
print("\n" + "="*80)
print(f"PER-DIMENSION ERRORS (Best Model: {best_loss_type.upper()})")
print("="*80)
print(f"{'Dimension':<12} | {'Mean Error':>12} | {'Std Error':>12}")
print("-"*80)
for dim in range(14):
    stats = best_results['per_dimension_stats'][f'dim_{dim}']
    print(f"Dim {dim:<8} | {stats['mean_abs_error']:>12.6f} | {stats['std_abs_error']:>12.6f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\n✅ Best loss function: {best_loss_type.upper()}")
print(f"   L1 Loss: {best_results['avg_l1_loss']:.6f}")

# Save summary
summary = {
    'timestamp': "${TIMESTAMP}",
    'results': all_results,
    'ranking': [(lt, r['avg_l1_loss']) for lt, r in sorted_by_l1],
    'best_loss': best_loss_type,
}

with open(results_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📄 Summary saved to: {results_dir / 'summary.json'}")
print("="*80)
PYTHON_EOF

echo ""
echo "======================================================================"
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo "Results directory: ${RESULTS_DIR}"
echo ""

