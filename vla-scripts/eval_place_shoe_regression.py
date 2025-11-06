"""
Evaluation script for Place Shoe dataset using the REGRESSION HEAD
This correctly evaluates the trained action head, not just the base VLA tokens
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.vla import constants
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets import PlaceShoeDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from transformers import AutoModelForVision2Seq, AutoProcessor
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.enabled = False

@dataclass
class EvalConfig:
    checkpoint_dir: Path
    hdf5_path: Path
    use_lora: bool = True
    batch_size: int = 4
    num_images_in_input: int = 1
    output_file: Optional[str] = None
    action_dim: int = 14
    regression_loss_type: str = "l1"  # Must match training: 'l1', 'l2'/'mse', 'huber', 'smooth_l1'
    huber_delta: float = 1.0  # Delta parameter for Huber loss

def compute_l1_loss(predicted_actions, ground_truth_actions):
    """Compute L1 loss between predicted and ground truth actions"""
    return torch.abs(predicted_actions - ground_truth_actions).mean().item()

def compute_per_dimension_errors(predicted_actions, ground_truth_actions, action_dim):
    """Compute per-dimension mean absolute errors"""
    errors = torch.abs(predicted_actions - ground_truth_actions)  # (B, act_chunk*act_dim)
    errors = errors.reshape(-1, action_dim)  # Reshape to (B*chunk, act_dim)
    
    per_dim_errors = {}
    for dim in range(action_dim):
        dim_errors = errors[:, dim].cpu().numpy()
        per_dim_errors[f'dim_{dim}'] = {
            'mean_abs_error': float(dim_errors.mean()),
            'std_abs_error': float(dim_errors.std()),
            'max_abs_error': float(dim_errors.max()),
        }
    return per_dim_errors

@draccus.wrap()
def eval_place_shoe_regression(cfg: EvalConfig) -> None:
    """Evaluate the trained regression head on validation set"""
    
    # Setup distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Starting evaluation with regression head...")
    
    # Ensure constants are loaded
    constants.detect_robot_platform()
    
    # Load processor
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    # Load base model
    print(f"[Rank {rank}] Loading model from {cfg.checkpoint_dir}...")
    if cfg.use_lora:
        from peft import PeftModel
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, str(cfg.checkpoint_dir / "lora_adapter"))
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            str(cfg.checkpoint_dir), 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
    
    model = model.to(f"cuda:{rank}")
    model.eval()
    
    # Load action head
    print(f"[Rank {rank}] Loading action head...")
    from prismatic.models.action_heads import L1RegressionActionHead
    
    # Get hidden size from model config
    if hasattr(model, 'config'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model, 'base_model'):
        hidden_size = model.base_model.config.text_config.hidden_size
    else:
        hidden_size = 4096  # Default for Llama-2 7B
    
    print(f"[Rank {rank}] Using hidden size: {hidden_size}")
    print(f"[Rank {rank}] Using regression loss type: {cfg.regression_loss_type}")
    if cfg.regression_loss_type == 'huber':
        print(f"[Rank {rank}] Using huber_delta: {cfg.huber_delta}")
    
    action_head = L1RegressionActionHead(
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        action_dim=cfg.action_dim,  # Just action_dim, not multiplied by chunks!
        loss_type=cfg.regression_loss_type,
        huber_delta=cfg.huber_delta,
    ).to(f"cuda:{rank}")
    
    # Load action head weights
    # Try to find any action_head checkpoint
    action_head_files = list(cfg.checkpoint_dir.glob("action_head--*_checkpoint.pt"))
    
    if action_head_files:
        # Sort by modification time and take the most recent
        action_head_path = max(action_head_files, key=lambda p: p.stat().st_mtime)
        print(f"[Rank {rank}] Loading action head from {action_head_path}")
        state_dict = torch.load(action_head_path, map_location=f"cuda:{rank}")
        # Remove 'module.' prefix if present (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        action_head.load_state_dict(state_dict)
    else:
        print(f"[Rank {rank}] WARNING: No action head checkpoint found!")
        print(f"[Rank {rank}] Available files: {list(cfg.checkpoint_dir.glob('*'))}")
    
    # Convert action head to bfloat16 to match model dtype
    action_head = action_head.to(torch.bfloat16)
    action_head.eval()
    
    # Setup dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder_fn = PurePromptBuilder
    
    dataset = PlaceShoeDataset(
        hdf5_path=cfg.hdf5_path,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        compute_stats=False,
        use_wrist_image=(cfg.num_images_in_input > 1),
    )
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        collate_fn=collator, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Evaluation loop
    all_l1_losses = []
    all_predicted_actions = []
    all_ground_truth_actions = []
    total_successes = 0.0
    total_success_count = 0
    
    print(f"[Rank {rank}] Starting evaluation on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(f"cuda:{rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{rank}")
            
            if isinstance(batch["pixel_values"], dict):
                pixel_values = {k: v.to(f"cuda:{rank}") for k, v in batch["pixel_values"].items()}
            else:
                pixel_values = batch["pixel_values"].to(f"cuda:{rank}")
            
            ground_truth_actions = batch["actions"].to(f"cuda:{rank}")  # (B, chunk_size, action_dim)
            batch_size = ground_truth_actions.shape[0]

            # Flatten actions for comparison
            ground_truth_actions_flat = ground_truth_actions.reshape(batch_size, -1)  # (B, chunk*action_dim)

            success_batch = batch.get("success")
            if success_batch is not None:
                success_batch = success_batch.to(torch.float32)
                valid_mask = ~torch.isnan(success_batch)
                if valid_mask.any():
                    total_successes += success_batch[valid_mask].sum().item()
                    total_success_count += int(valid_mask.sum().item())
            
            # Forward pass to get hidden states
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Get the base model (unwrap PEFT if needed)
                if hasattr(model, 'base_model'):
                    base_model = model.base_model.model  # Unwrap PEFT
                elif hasattr(model, 'model'):
                    base_model = model.model
                else:
                    base_model = model
                
                # Process vision
                if isinstance(pixel_values, dict):
                    pixel_values_list = [pixel_values[k] for k in sorted(pixel_values.keys())]
                    vision_outputs = base_model.vision_backbone(*pixel_values_list)
                else:
                    vision_outputs = base_model.vision_backbone(pixel_values)
                
                # Project vision embeddings to LLM dimension
                projected_patch_embeddings = base_model.projector(vision_outputs)
                
                # Get text embeddings
                inputs_embeds = base_model.language_model.get_input_embeddings()(input_ids)
                
                # Concatenate projected vision and text embeddings
                batch_size = inputs_embeds.shape[0]
                combined_embeds = torch.cat([projected_patch_embeddings, inputs_embeds], dim=1)
                
                # Forward through LLM to get hidden states
                llm_outputs = base_model.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=None,  # Will be computed automatically
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                last_hidden_states = llm_outputs.hidden_states[-1]  # (B, seq_len, D)
            
            # Get number of image patches
            # Try different attribute paths for vision backbone
            num_patches = 729  # Default for DinoV2
            if hasattr(model, 'vision_backbone'):
                if hasattr(model.vision_backbone, 'featurizer'):
                    num_patches = model.vision_backbone.featurizer.patch_embed.num_patches
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'vision_backbone'):
                if hasattr(model.base_model.vision_backbone, 'featurizer'):
                    num_patches = model.base_model.vision_backbone.featurizer.patch_embed.num_patches
            
            # Get hidden states for text portion (after vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]  # (B, text_len, D)
            
            # Get action hidden states (last action_chunk_len tokens)
            action_chunk_len = cfg.action_dim * 8  # 14 * 8 = 112 tokens
            actions_hidden_states = text_hidden_states[:, -action_chunk_len:, :]  # (B, 112, D)
            
            # Predict actions using action head
            # Action head processes each timestep separately and outputs (B, chunk*action_dim, action_dim)
            # Then we need to reshape properly
            predicted_actions = action_head.predict_action(actions_hidden_states)  # (B, chunk*action_dim, action_dim)
            
            # Reshape: (B, chunk*action_dim, action_dim) -> (B, chunk, action_dim) -> (B, chunk*action_dim)
            predicted_actions_flat = predicted_actions.reshape(batch_size, -1)  # (B, chunk*action_dim)
            
            # Compute L1 loss
            l1_loss = compute_l1_loss(predicted_actions_flat, ground_truth_actions_flat)
            all_l1_losses.append(l1_loss)
            
            # Store for per-dimension analysis
            all_predicted_actions.append(predicted_actions_flat.cpu())
            all_ground_truth_actions.append(ground_truth_actions_flat.cpu())
    
    # Aggregate results
    avg_l1_loss = np.mean(all_l1_losses)
    success_rate = (total_successes / total_success_count) if total_success_count > 0 else None
    
    # Compute per-dimension errors
    all_predicted_actions = torch.cat(all_predicted_actions, dim=0)
    all_ground_truth_actions = torch.cat(all_ground_truth_actions, dim=0)
    per_dim_stats = compute_per_dimension_errors(
        all_predicted_actions, 
        all_ground_truth_actions, 
        cfg.action_dim
    )
    
    results = {
        'avg_l1_loss': float(avg_l1_loss),
        'num_batches': len(dataloader),
        'batch_size': cfg.batch_size,
        'per_dimension_stats': per_dim_stats,
        'success_rate': float(success_rate) if success_rate is not None else None,
    }
    
    if rank == 0:
        print("\n" + "="*80)
        print("EVALUATION RESULTS (Regression Head)")
        print("="*80)
        print(f"Average L1 Loss: {avg_l1_loss:.6f}")
        print(f"Number of batches: {len(dataloader)}")
        if success_rate is not None:
            print(f"Success Rate: {success_rate:.6f} ({success_rate * 100:.2f}%)")
        
        print("\nPer-Dimension Errors:")
        for dim in range(cfg.action_dim):
            stats = per_dim_stats[f'dim_{dim}']
            print(f"  Dim {dim}: {stats['mean_abs_error']:.6f} ± {stats['std_abs_error']:.6f}")
        
        # Save results
        if cfg.output_file:
            output_path = Path(cfg.output_file)
        else:
            output_path = cfg.checkpoint_dir / "eval_results_regression.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print("="*80)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    eval_place_shoe_regression()
