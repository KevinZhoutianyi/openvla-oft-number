# OpenVLA Fine-tuning: Data Format Explained

## 🎯 Overview

OpenVLA is a **Vision-Language-Action (VLA)** model that learns to predict robot actions from:
- **Vision**: Camera images
- **Language**: Task instructions
- **Action**: Robot control commands

## 📥 Input Format

### 1. Visual Input (Images)

Your dataset has 4 camera views, and you can choose which ones to use:

```python
# Option 1: Single image (num_images_in_input=1)
visual_input = {
    'pixel_values': [3, 224, 224]  # Primary camera (e.g., head)
}

# Option 2: Two images (num_images_in_input=2) - RECOMMENDED
visual_input = {
    'pixel_values': [3, 224, 224],        # Primary camera (head)
    'pixel_values_wrist': [3, 224, 224]   # Wrist camera (left or right)
}
```

**Your dataset cameras:**
- `head_camera_image` (256×256×3) → resized to 224×224
- `low_cam_image` (256×256×3) → resized to 224×224
- `left_wrist_image` (256×256×3) → resized to 224×224
- `right_wrist_image` (256×256×3) → resized to 224×224

### 2. Language Input (Task Instruction)

Your dataset has language instructions in the `seen`/`unseen` keys:

```python
language_instruction = "place the shoe"  # From your HDF5 files

# Formatted as a question:
prompt = "What action should the robot take to place the shoe?"
```

### 3. Action Output (What the Model Predicts)

**Action Chunking** - The model predicts multiple future actions at once:

```python
# Your dataset:
action_chunk = {
    'shape': (8, 14),  # 8 timesteps × 14 action dimensions
    'current_action': action_chunk[0],   # Shape: (14,)
    'future_actions': action_chunk[1:8]  # Shape: (7, 14)
}
```

**Why action chunking?**
- Predicting future actions improves temporal consistency
- Reduces accumulation of errors
- Default chunk size: 8 timesteps (configurable via `NUM_ACTIONS_CHUNK`)

### 4. Your Dataset's Action Space

```python
# 14-dimensional action vector (example, adjust to your robot):
action = [
    pos_x,      # Position/velocity dimension 1
    pos_y,      # Position/velocity dimension 2
    pos_z,      # Position/velocity dimension 3
    rot_roll,   # Rotation dimension 1
    rot_pitch,  # Rotation dimension 2
    rot_yaw,    # Rotation dimension 3
    gripper,    # Gripper open/close
    # ... 7 more dimensions (depends on your robot)
]
```

## 🔄 Complete Data Flow

### Step 1: Load from HDF5

```python
# From episode_0.hdf5 at timestep t=10
raw_data = {
    'head_camera_image': [256, 256, 3],  # uint8
    'left_wrist_image': [256, 256, 3],   # uint8
    'action': [185, 14],                 # Full episode actions
    'seen': "place the shoe in the box"  # Language instruction
}
```

### Step 2: Process for Model Input

```python
# 1. Extract action chunk (current + next 7 timesteps)
action_chunk = raw_data['action'][t:t+8]  # Shape: (8, 14)

# 2. Resize and normalize images
image_primary = resize_and_normalize(raw_data['head_camera_image'][t])
image_wrist = resize_and_normalize(raw_data['left_wrist_image'][t])

# 3. Create text prompt
prompt = "What action should the robot take to place the shoe in the box?"

# 4. Tokenize everything
model_input = {
    'pixel_values': image_primary,           # (3, 224, 224)
    'pixel_values_wrist': image_wrist,       # (3, 224, 224)
    'input_ids': tokenize(prompt),           # (seq_len,)
    'labels': tokenize(action_chunk),        # (seq_len,) - action tokens
    'actions': action_chunk,                 # (8, 14) - ground truth
}
```

### Step 3: Model Forward Pass

```python
# OpenVLA processes the input:
vision_features = vision_encoder(images)       # Extract visual features
text_features = language_model(input_ids)      # Process text
combined = fuse(vision_features, text_features) # Multimodal fusion

# Action prediction
action_embedding = llm_hidden_states           # From last layer
predicted_actions = action_head(action_embedding) # Shape: (8, 14)
```

### Step 4: Compute Loss

```python
# L1 Regression Loss (recommended for your dataset)
loss = L1Loss(predicted_actions, ground_truth_actions)

# Break it down:
current_action_loss = L1Loss(predicted_actions[0], ground_truth_actions[0])
future_actions_loss = L1Loss(predicted_actions[1:8], ground_truth_actions[1:8])

total_loss = current_action_loss + future_actions_loss
```

## 🎓 What Is Being Trained?

### Trainable Components (with LoRA)

**1. Vision-Language Model (Frozen base, LoRA adapters trainable):**
```
OpenVLA-7B Base Model (7 billion parameters)
├── Vision Encoder (SigLIP) - Frozen + LoRA
├── Projector - Frozen + LoRA
└── Language Model (Llama/Vicuna) - Frozen + LoRA
```

**LoRA (Low-Rank Adaptation):**
- Only train small adapter matrices (rank=32)
- Drastically reduces memory and training time
- ~14M trainable parameters instead of 7B

**2. Action Head (Fully trainable):**
```python
L1RegressionActionHead(
    input_dim=4096,    # From LLM hidden states
    hidden_dim=4096,   # MLP hidden dimension
    action_dim=14      # Your action space
)
# Predicts continuous action values
```

**3. Optional Components:**
- **Proprio Projector**: Maps robot state to embeddings (not used for your dataset)
- **FiLM**: Better language conditioning (optional)
- **Diffusion Head**: Alternative to L1 regression (not recommended for now)

### Training Process

```python
# For each batch:
for batch in dataloader:
    # Forward pass
    predicted_actions = model(
        images=batch['pixel_values'],
        text=batch['input_ids'],
    )
    
    # Compute loss (L1 regression)
    loss = L1Loss(predicted_actions, batch['actions'])
    
    # Backward pass (only updates LoRA + action head)
    loss.backward()
    optimizer.step()
    
    # Loss should decrease to ~0.01 or lower
```

## 📊 Training Statistics

**Action Normalization:**
```python
# Computed once at the start:
action_statistics = {
    'q01': [min_values for each of 14 dims],  # 1st percentile
    'q99': [max_values for each of 14 dims],  # 99th percentile
}

# Actions are normalized to roughly [-1, 1] range
normalized_action = (action - q01) / (q99 - q01) * 2 - 1

# During inference, actions are denormalized:
real_action = (normalized_action + 1) / 2 * (q99 - q01) + q01
```

## 🎯 Example Training Sample

Here's what one complete training sample looks like:

```python
training_sample = {
    # Visual input
    'pixel_values': torch.tensor([3, 224, 224]),      # Head camera
    'pixel_values_wrist': torch.tensor([3, 224, 224]), # Wrist camera
    
    # Text input (tokenized)
    'input_ids': torch.tensor([
        1, 518, 25580, 29962, 1724, 2944, 881, ...  # "What action should..."
    ]),
    
    # Labels (which tokens to predict)
    'labels': torch.tensor([
        -100, -100, -100, ..., 12453, 8765, ...  # -100 = ignore, rest = action tokens
    ]),
    
    # Ground truth actions (for computing L1 loss)
    'actions': torch.tensor([
        [0.123, -0.456, 0.789, ..., 0.321],  # Timestep 0 (current)
        [0.124, -0.455, 0.788, ..., 0.322],  # Timestep 1
        [0.125, -0.454, 0.787, ..., 0.323],  # Timestep 2
        ...                                   # Timesteps 3-7
    ]),  # Shape: (8, 14)
    
    # Metadata
    'dataset_name': 'place_shoe_dataset'
}
```

## 🔍 Checking Your Data

Run this to see your actual data format:

```bash
cd /u/tzhou4/fone/openvla-oft-number
python scripts/test_dataset.py
```

This will show you:
- Exact shapes of all tensors
- Action statistics (min/max values)
- Sample data point
- Whether everything loads correctly

## 📈 Monitoring Training

**Key metrics to watch:**

1. **Total Loss**: Should decrease to ~0.01 or lower
2. **Current Action L1 Loss**: Loss for the immediate next action
3. **Future Actions L1 Loss**: Loss for actions 1-7 steps ahead
4. **Learning Rate**: Starts at 5e-4, decays by 10x after 40k steps

**Example training log:**
```
Step 100:  loss=0.250, curr_l1=0.180, future_l1=0.270
Step 1000: loss=0.080, curr_l1=0.060, future_l1=0.090
Step 5000: loss=0.025, curr_l1=0.018, future_l1=0.028
Step 10000: loss=0.012, curr_l1=0.009, future_l1=0.014  ← Good!
```

## 🎯 Summary

**Input:**
- Images (1-3 camera views)
- Language instruction
- Optional: Proprioceptive state

**Output:**
- Action chunk (8 timesteps × 14 dimensions)

**Training:**
- LoRA adapters on vision-language model (~14M params)
- Action head (fully trained, ~50M params)
- L1 regression loss
- Target loss: ~0.01 or lower

**Your specific setup:**
- 2 images (head + left wrist)
- 14-dimensional actions
- "place the shoe" task
- 48 training episodes
- ~8k training samples (48 episodes × ~175 timesteps each)

---

For more details, see the code:
- Dataset loader: `prismatic/vla/datasets/place_shoe_dataset.py`
- Training script: `vla-scripts/finetune_place_shoe.py`
- Action head: `prismatic/models/action_heads.py`

