"""
finetune_hdf5.py

Fine-tune OpenVLA on custom HDF5 dataset.

This is a modified version of finetune.py that works with HDF5 datasets instead of RLDS.

Usage:
    # Single GPU
    python vla-scripts/finetune_hdf5.py \
        --vla_path openvla/openvla-7b \
        --hdf5_path /path/to/your/dataset.hdf5 \
        --run_root_dir /path/to/save/checkpoints
    
    # Multi-GPU (e.g., 4 GPUs)
    torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_hdf5.py \
        --vla_path openvla/openvla-7b \
        --hdf5_path /path/to/your/dataset.hdf5 \
        --run_root_dir /path/to/save/checkpoints
"""

import os
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

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import HDF5Dataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Import training utilities from original finetune.py
import sys
sys.path.append(str(Path(__file__).parent))
from finetune import (
    count_parameters,
    init_module,
    wrap_ddp,
    run_forward_pass,
    compute_smoothened_metrics,
    log_metrics_to_wandb,
    get_run_id as original_get_run_id,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneHDF5Config:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    hdf5_path: Path = Path("data/demos.hdf5")        # Path to HDF5 file or directory containing HDF5 files
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    
    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only latest checkpoint
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # Step number to resume from
    image_aug: bool = False                          # If True, trains with image augmentations (not implemented for HDF5 yet)
    diffusion_sample_freq: int = 50                  # Frequency for sampling in steps (when using diffusion)

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights during training

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID
    run_id_override: Optional[str] = None            # Optional string to override the run ID
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def get_run_id(cfg: FinetuneHDF5Config) -> str:
    """Generate run ID for HDF5 training."""
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    
    dataset_name = cfg.hdf5_path.stem if cfg.hdf5_path.is_file() else cfg.hdf5_path.name
    
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    """Save training checkpoint."""
    # Determine checkpoint paths
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save
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
def finetune_hdf5(cfg: FinetuneHDF5Config) -> None:
    """Fine-tune OpenVLA on HDF5 dataset."""
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one!"
    )

    # Trim trailing forward slash in VLA path
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on HDF5 dataset: `{cfg.hdf5_path}`")

    # Get run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
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

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}"
    )

    # Handle model loading (from HF Hub or local)
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Set number of images in input
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

    # FiLM setup (if applicable)
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # Initialize proprio projector if needed
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # Initialize action head
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load HDF5 Dataset
    print(f"Loading HDF5 dataset from {cfg.hdf5_path}...")
    train_dataset = HDF5Dataset(
        hdf5_path=cfg.hdf5_path,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        action_dim=ACTION_DIM,
        use_wrist_image=cfg.num_images_in_input > 1,
        use_proprio=cfg.use_proprio,
        predict_stop_token=True,
        compute_stats=True,
    )

    # Save dataset statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,  # Can increase for faster data loading
        pin_memory=True,
    )

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )

    # Deque to store recent train metrics
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    print(f"Starting training for {cfg.max_steps} steps...")
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        # Training loop
        epoch = 0
        global_step = 0
        
        while global_step < cfg.max_steps:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                # Compute training metrics and loss
                compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                )

                # Normalize loss
                normalized_loss = loss / cfg.grad_accumulation_steps

                # Backward pass
                normalized_loss.backward()

                # Store recent metrics
                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                # Update weights if gradient accumulation is complete
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                    log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                    # Learning rate warmup
                    if cfg.lr_warmup_steps > 0:
                        lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                        current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr

                    # Compute smoothened metrics
                    smoothened_metrics = compute_smoothened_metrics(recent_metrics)

                    # Log to W&B
                    if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                        log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                        wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

                    global_step = log_step

                    # Save checkpoint
                    if global_step > 0 and global_step % cfg.save_freq == 0:
                        save_training_checkpoint(
                            cfg=cfg,
                            run_dir=run_dir,
                            log_step=log_step,
                            vla=vla,
                            processor=processor,
                            proprio_projector=proprio_projector if cfg.use_proprio else None,
                            noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                            action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                            train_dataset=train_dataset,
                            distributed_state=distributed_state,
                        )

                    # Stop if max steps reached
                    if global_step >= cfg.max_steps:
                        print(f"Max step {cfg.max_steps} reached! Stopping training...")
                        break
                
                if global_step >= cfg.max_steps:
                    break
            
            if global_step >= cfg.max_steps:
                break
            
            print(f"Completed epoch {epoch}, continuing to next epoch...")


if __name__ == "__main__":
    finetune_hdf5()

