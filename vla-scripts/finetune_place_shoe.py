"""
finetune_place_shoe.py

Fine-tune OpenVLA on the "place shoe" HDF5 dataset.

This script is specifically configured for the place shoe dataset structure.

Usage:
    # Single GPU
    python vla-scripts/finetune_place_shoe.py \
        --vla_path openvla/openvla-7b \
        --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
        --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
        --batch_size 4 \
        --max_steps 50000
    
    # Multi-GPU (4 GPUs)
    torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_place_shoe.py \
        --vla_path openvla/openvla-7b \
        --hdf5_path /work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train \
        --run_root_dir /work/hdd/bfdj/tzhou4/checkpoints \
        --batch_size 8 \
        --max_steps 50000
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import wandb

# Add parent directory to path (use insert(0) for priority)
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import PlaceShoeDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Import training utilities from original finetune.py
sys.path.append(str(Path(__file__).parent))
from finetune import (
    count_parameters,
    init_module,
    wrap_ddp,
    run_forward_pass,
    compute_smoothened_metrics,
    log_metrics_to_wandb,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable cuDNN to avoid library compatibility issues
torch.backends.cudnn.enabled = False


@dataclass
class FinetunePlaceShoeConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                                 # Path to OpenVLA model

    # Dataset
    hdf5_path: Path = Path("/work/hdd/bfdj/tzhou4/place_shoe_1/demo_clean_processed/train")
    run_root_dir: Path = Path("/work/hdd/bfdj/tzhou4/checkpoints")       # Checkpoint directory
    action_dim: int = 14                                                  # Action dimension (14 for place shoe dataset)
    use_relative_actions: bool = False                                    # Use relative_action instead of action
    camera_view: str = "head"                                             # Primary camera: "head", "low", "left_wrist", "right_wrist"
    wrist_camera: str = "left"                                            # Wrist camera: "left", "right", "both"
    default_instruction: str = "place the shoe"                           # Default language instruction
    
    # Algorithm and architecture
    use_l1_regression: bool = True                                        # Use L1 regression objective (recommended)
    use_diffusion: bool = False                                           # Use diffusion modeling objective
    num_diffusion_steps_train: int = 50                                   # Number of diffusion steps for training
    regression_loss_type: str = "l1"                                      # Regression loss: 'l1', 'l2'/'mse', 'huber', 'smooth_l1'
    huber_delta: float = 1.0                                              # Delta parameter for Huber loss
    use_film: bool = False                                                # Use FiLM for language conditioning
    num_images_in_input: int = 2                                          # Number of images (1: primary only, 2: primary + wrist)
    use_proprio: bool = False                                             # Use proprioceptive state (not available in this dataset)

    # Training configuration
    batch_size: int = 4                                                   # Batch size per device
    learning_rate: float = 5e-4                                           # Learning rate
    lr_warmup_steps: int = 0                                              # Number of warmup steps
    num_steps_before_decay: int = 40_000                                  # Steps before LR decays by 10x
    grad_accumulation_steps: int = 1                                      # Gradient accumulation steps
    max_steps: int = 50_000                                               # Max training steps
    save_freq: int = 5_000                                                # Checkpoint saving frequency
    save_latest_checkpoint_only: bool = False                             # Save only latest checkpoint
    resume: bool = False                                                  # Resume from checkpoint
    resume_step: Optional[int] = None                                     # Step to resume from
    diffusion_sample_freq: int = 50                                       # Diffusion sampling frequency

    # LoRA
    use_lora: bool = True                                                 # Use LoRA fine-tuning
    lora_rank: int = 32                                                   # LoRA rank
    lora_dropout: float = 0.0                                             # LoRA dropout
    merge_lora_during_training: bool = True                               # Merge LoRA weights during training

    # Logging
    wandb_entity: str = "your-wandb-entity"                               # WandB entity
    wandb_project: str = "place-shoe-finetune"                            # WandB project
    run_id_note: Optional[str] = None                                     # Extra note for run ID
    run_id_override: Optional[str] = None                                 # Override run ID
    wandb_log_freq: int = 10                                              # WandB logging frequency

    # fmt: on


def get_run_id(cfg: FinetunePlaceShoeConfig) -> str:
    """Generate run ID."""
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    
    dataset_name = "place_shoe"
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{dataset_name}"
        f"+act{cfg.action_dim}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}"
    if cfg.num_images_in_input > 1:
        run_id += f"+{cfg.num_images_in_input}imgs"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def save_training_checkpoint(
    cfg, run_dir, log_step, vla, processor, proprio_projector,
    noisy_action_projector, action_head, train_dataset, distributed_state,
) -> None:
    """Save training checkpoint."""
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}")

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        dist.barrier()


@draccus.wrap()
def finetune_place_shoe(cfg: FinetunePlaceShoeConfig) -> None:
    """Fine-tune OpenVLA on place shoe dataset."""
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot do both L1 regression and diffusion!"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA on place shoe dataset")
    print(f"  Model: {cfg.vla_path}")
    print(f"  Dataset: {cfg.hdf5_path}")
    print(f"  Action dim: {cfg.action_dim}")

    # Get run ID
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print constants
    print(f"Action dimension: {cfg.action_dim}")
    print(f"Using camera: {cfg.camera_view}")
    if cfg.num_images_in_input > 1:
        print(f"Using wrist camera: {cfg.wrist_camera}")

    # Handle model loading
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # Initialize action head
    action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim,
            "hidden_dim": vla.module.llm_dim,
            "action_dim": cfg.action_dim,
            "loss_type": cfg.regression_loss_type,
            "huber_delta": cfg.huber_delta,
        },
        to_bf16=True,
    )
    print(f"Using regression loss type: {cfg.regression_loss_type}")

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Place Shoe Dataset
    print(f"Loading place shoe dataset from {cfg.hdf5_path}...")
    train_dataset = PlaceShoeDataset(
        hdf5_path=cfg.hdf5_path,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        action_dim=cfg.action_dim,
        use_relative_actions=cfg.use_relative_actions,
        camera_view=cfg.camera_view,
        use_wrist_image=cfg.num_images_in_input > 1,
        wrist_camera=cfg.wrist_camera,
        use_proprio=False,
        predict_stop_token=True,
        compute_stats=True,
        default_instruction=cfg.default_instruction,
    )

    # Save dataset statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collator, num_workers=4, pin_memory=True,
    )

    # Instantiate optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # Metrics deque
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    print(f"Starting training for {cfg.max_steps} steps...")
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        epoch = 0
        global_step = 0
        
        while global_step < cfg.max_steps:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=False,
                    use_proprio=False,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=False,
                    num_diffusion_steps_train=None,
                )

                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                    log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                    if cfg.lr_warmup_steps > 0:
                        lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                        current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr

                    smoothened_metrics = compute_smoothened_metrics(recent_metrics)

                    if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                        log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                        wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

                    global_step = log_step

                    if global_step > 0 and global_step % cfg.save_freq == 0:
                        save_training_checkpoint(
                            cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla,
                            processor=processor, proprio_projector=None,
                            noisy_action_projector=None, action_head=action_head,
                            train_dataset=train_dataset, distributed_state=distributed_state,
                        )

                    if global_step >= cfg.max_steps:
                        print(f"Max step {cfg.max_steps} reached! Stopping training...")
                        break
                
                if global_step >= cfg.max_steps:
                    break
            
            if global_step >= cfg.max_steps:
                break
            
            print(f"Completed epoch {epoch}, continuing to next epoch...")

    print("Training completed!")


if __name__ == "__main__":
    finetune_place_shoe()

