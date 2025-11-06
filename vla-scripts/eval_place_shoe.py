"""
eval_place_shoe.py

Evaluates a trained OpenVLA model on the Place Shoe validation dataset.

Usage:
    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
      vla-scripts/eval_place_shoe.py \
      --checkpoint_dir /work/hdd/bfdj/tzhou4/checkpoints/openvla-7b+place_shoe+... \
      --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/val \
      --batch_size 4
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Add parent directory to path for prismatic imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable cuDNN to avoid library compatibility issues
torch.backends.cudnn.enabled = False


@dataclass
class EvalConfig:
    # Checkpoint and Data Paths
    checkpoint_dir: Path                                      # Path to checkpoint directory
    hdf5_path: Path                                          # Path to HDF5 validation files
    
    # Model parameters
    use_lora: bool = True                                    # Whether the checkpoint uses LoRA
    batch_size: int = 4                                      # Batch size for evaluation
    num_images_in_input: int = 1                             # Number of images (1: primary only, 2: primary + wrist)
    use_lwe_decoder: bool = False                            # Enable logit-weighted expectation decoder
    lwe_temperature: float = 1.0                             # Softmax temperature for LWE decoder
    
    # Output
    output_file: Optional[str] = None                        # Where to save results (default: checkpoint_dir/eval_results.json)


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    """Compute accuracy of predicted action tokens."""
    # Only compare where mask is True
    masked_preds = predicted_token_ids[mask]
    masked_gt = ground_truth_token_ids[mask]
    accuracy = (masked_preds == masked_gt).float().mean()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    """Compute L1 loss between predicted and ground truth continuous actions."""
    # Extract only the masked (action) tokens
    masked_preds = predicted_token_ids[mask].cpu().numpy()
    masked_gt = ground_truth_token_ids[mask].cpu().numpy()
    
    # Reshape to (batch_size, action_dim)
    num_actions = len(masked_preds)
    if num_actions == 0:
        return torch.tensor(0.0, device=mask.device)
    
    # Convert token IDs to continuous actions
    predicted_actions = action_tokenizer.decode_token_ids_to_actions(masked_preds)
    ground_truth_actions = action_tokenizer.decode_token_ids_to_actions(masked_gt)
    
    # Compute L1 loss
    l1_loss = np.abs(predicted_actions - ground_truth_actions).mean()
    
    return torch.tensor(l1_loss, device=mask.device)


def get_current_action_mask(labels, action_dim=14):
    """Get mask for the current (first) action in the sequence."""
    current_action_mask = torch.zeros_like(labels, dtype=torch.bool)
    count = 0
    for i in range(labels.shape[1]):
        if labels[0, i] != -100:
            current_action_mask[:, i] = True
            count += 1
            if count >= action_dim:
                break
    return current_action_mask


@draccus.wrap()
def eval_place_shoe(cfg: EvalConfig) -> None:
    print(f"\n{'='*80}\nEvaluating Place Shoe Dataset\n{'='*80}\n")
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank := int(os.environ["LOCAL_RANK"]))
    
    # Import and initialize constants for place_shoe
    from prismatic.vla import constants
    constants.detect_robot_platform()
    
    # Load processor from base model
    print(f"Loading processor from openvla/openvla-7b...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    # Validate LWE settings
    if cfg.use_lwe_decoder and cfg.lwe_temperature <= 0:
        raise ValueError("`lwe_temperature` must be positive when LWE decoder is enabled.")

    # Load model from checkpoint
    print(f"Loading model from checkpoint: {cfg.checkpoint_dir}")
    if cfg.use_lora:
        # Load base model first
        from peft import PeftModel
        print("Loading base model...")
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Load LoRA weights
        print(f"Loading LoRA weights from {cfg.checkpoint_dir}...")
        model = PeftModel.from_pretrained(model, cfg.checkpoint_dir)
        # Note: We don't merge for evaluation - it's faster to use LoRA directly
        # model = model.merge_and_unload()  # Only needed for deployment
    else:
        # Load base model directly (no LoRA)
        print("Loading base model (no LoRA)...")
        model = AutoModelForVision2Seq.from_pretrained(
            str(cfg.checkpoint_dir),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    target_models = [model]
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        target_models.append(model.base_model.model)
    for target in target_models:
        if hasattr(target, "config"):
            target.config.use_lwe_decoder = cfg.use_lwe_decoder
            target.config.lwe_temperature = cfg.lwe_temperature
            if not hasattr(target.config, "lwe_loss_weight"):
                target.config.lwe_loss_weight = 1.0

    model = model.to(f"cuda:{rank}")
    model.eval()

    base_model = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        base_model = model.base_model.model

    num_patches = base_model.vision_backbone.get_num_patches() * base_model.vision_backbone.get_num_images_in_input()
    
    # Load action tokenizer
    from prismatic.vla.action_tokenizer import ActionTokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # Load dataset
    print(f"\nLoading validation dataset from: {cfg.hdf5_path}")
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.datasets import PlaceShoeDataset
    
    prompt_builder_fn = PurePromptBuilder
    dataset = PlaceShoeDataset(
        hdf5_path=cfg.hdf5_path,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        compute_stats=False,  # Don't recompute stats for validation
        use_wrist_image=(cfg.num_images_in_input > 1),  # Only use wrist if num_images > 1
    )
    
    # Create dataloader
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )
    
    print(f"Loaded {len(dataset)} validation samples\n")
    
    # Evaluation loop
    print("Starting evaluation...")
    total_loss = 0.0
    total_action_accuracy = 0.0
    total_l1_loss = 0.0
    total_lwe_l1_loss = 0.0
    total_steps = 0
    total_successes = 0.0
    total_success_count = 0
    
    # Track per-dimension errors
    from prismatic.vla.constants import ACTION_DIM
    per_dim_errors = [[] for _ in range(ACTION_DIM)]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch["input_ids"].to(f"cuda:{rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{rank}")
            
            # Handle pixel_values (can be dict or tensor)
            if isinstance(batch["pixel_values"], dict):
                pixel_values = {k: v.to(f"cuda:{rank}") for k, v in batch["pixel_values"].items()}
            else:
                pixel_values = batch["pixel_values"].to(f"cuda:{rank}")
            
            labels = batch["labels"].to(f"cuda:{rank}")
            
            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
            
            loss = outputs.loss
            
            # Get predicted token IDs
            logits = outputs.logits
            predicted_token_ids = logits.argmax(dim=-1)
            
            # Shift to align predictions with labels
            predicted_token_ids = predicted_token_ids[:, :-1]
            ground_truth_token_ids = labels[:, 1:]
            
            # Ensure same sequence length (truncate to shorter length)
            min_len = min(predicted_token_ids.shape[1], ground_truth_token_ids.shape[1])
            predicted_token_ids = predicted_token_ids[:, :min_len]
            ground_truth_token_ids = ground_truth_token_ids[:, :min_len]
            
            # Get mask for current action (first ACTION_DIM tokens)
            current_action_mask = get_current_action_mask(ground_truth_token_ids, action_dim=ACTION_DIM)
            
            # Compute token accuracy
            action_accuracy = compute_token_accuracy(
                predicted_token_ids, 
                ground_truth_token_ids, 
                mask=current_action_mask
            )
            
            # Compute L1 loss / per-dimension errors
            if cfg.use_lwe_decoder and ground_truth_actions is not None:
                action_logits = logits[:, num_patches:-1, :]
                lwe_expectation = base_model.compute_lwe_expectation_from_action_logits(action_logits)
                lwe_expectation = lwe_expectation.view_as(ground_truth_actions).to(ground_truth_actions.dtype)
                lwe_l1_value = torch.nn.functional.l1_loss(lwe_expectation, ground_truth_actions, reduction="mean").item()
                total_lwe_l1_loss += lwe_l1_value
                batch_l1_value = lwe_l1_value

                flat_pred = lwe_expectation.view(-1, ACTION_DIM).cpu().numpy()
                flat_gt = ground_truth_actions.view(-1, ACTION_DIM).cpu().numpy()
                for row_pred, row_gt in zip(flat_pred, flat_gt):
                    for dim in range(ACTION_DIM):
                        per_dim_errors[dim].append(abs(row_pred[dim] - row_gt[dim]))
            else:
                pred_action_tokens = predicted_token_ids[current_action_mask].reshape(-1, ACTION_DIM)
                gt_action_tokens = ground_truth_token_ids[current_action_mask].reshape(-1, ACTION_DIM)

                for i in range(pred_action_tokens.shape[0]):
                    pred_actions = action_tokenizer.decode_token_ids_to_actions(pred_action_tokens[i].cpu().numpy())
                    gt_actions = action_tokenizer.decode_token_ids_to_actions(gt_action_tokens[i].cpu().numpy())

                    for dim in range(ACTION_DIM):
                        per_dim_errors[dim].append(abs(pred_actions[dim] - gt_actions[dim]))

                l1_loss = compute_actions_l1_loss(
                    action_tokenizer,
                    predicted_token_ids[current_action_mask],
                    ground_truth_token_ids[current_action_mask],
                    current_action_mask[current_action_mask],
                )
                batch_l1_value = l1_loss.item()

            success_batch = batch.get("success")
            if success_batch is not None:
                success_batch = success_batch.to(torch.float32)
                valid_mask = ~torch.isnan(success_batch)
                if valid_mask.any():
                    valid_values = success_batch[valid_mask]
                    total_successes += valid_values.sum().item()
                    total_success_count += int(valid_mask.sum().item())
            
            # Accumulate metrics
            total_loss += loss.item()
            total_action_accuracy += action_accuracy.item()
            total_l1_loss += batch_l1_value
            total_steps += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"[{batch_idx + 1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}, "
                      f"Action Acc: {action_accuracy.item():.4f}, "
                      f"L1 Loss: {batch_l1_value:.4f}")
    
    # Compute final metrics
    avg_loss = total_loss / total_steps
    avg_action_accuracy = total_action_accuracy / total_steps
    avg_l1_loss = total_l1_loss / total_steps
    avg_lwe_l1_loss = total_lwe_l1_loss / total_steps if cfg.use_lwe_decoder else None
    success_rate = (total_successes / total_success_count) if total_success_count > 0 else None
    
    # Compute per-dimension statistics
    per_dim_stats = {}
    for dim in range(ACTION_DIM):
        errors = np.asarray(per_dim_errors[dim], dtype=np.float32)
        if errors.size == 0:
            per_dim_stats[f"dim_{dim}"] = {
                "mean_abs_error": 0.0,
                "std_abs_error": 0.0,
                "median_abs_error": 0.0,
                "max_abs_error": 0.0,
            }
        else:
            per_dim_stats[f"dim_{dim}"] = {
                "mean_abs_error": float(np.mean(errors)),
                "std_abs_error": float(np.std(errors)),
                "median_abs_error": float(np.median(errors)),
                "max_abs_error": float(np.max(errors)),
            }
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {len(dataset)}")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average Action Token Accuracy: {avg_action_accuracy:.6f} ({avg_action_accuracy*100:.2f}%)")
    print(f"Average L1 Loss (Continuous Actions): {avg_l1_loss:.6f}")
    if cfg.use_lwe_decoder and avg_lwe_l1_loss is not None:
        print(f"Average LWE L1 Loss: {avg_lwe_l1_loss:.6f}")
    if success_rate is not None:
        print(f"Success Rate: {success_rate:.6f} ({success_rate * 100:.2f}%)")
    print(f"\nPer-Dimension Mean Absolute Errors:")
    for dim in range(ACTION_DIM):
        print(f"  Dim {dim:2d}: {per_dim_stats[f'dim_{dim}']['mean_abs_error']:.6f} "
              f"(std: {per_dim_stats[f'dim_{dim}']['std_abs_error']:.6f}, "
              f"median: {per_dim_stats[f'dim_{dim}']['median_abs_error']:.6f})")
    print(f"{'='*80}\n")
    
    # Save results to file
    output_file = cfg.output_file or str(cfg.checkpoint_dir / "eval_results.json")
    results = {
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "hdf5_path": str(cfg.hdf5_path),
        "num_samples": len(dataset),
        "avg_loss": avg_loss,
        "avg_action_token_accuracy": avg_action_accuracy,
        "avg_l1_loss": avg_l1_loss,
        "per_dimension_stats": per_dim_stats,
        "success_rate": success_rate,
    }
    if cfg.use_lwe_decoder and avg_lwe_l1_loss is not None:
        results["avg_lwe_l1_loss"] = avg_lwe_l1_loss
    
    if rank == 0:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    eval_place_shoe()
