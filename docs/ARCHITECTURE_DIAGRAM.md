# OpenVLA Architecture & Training Flow

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenVLA Architecture                      │
└─────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐
│ Head Camera  │  │ Wrist Camera │  │ Language Instruction│
│  (256×256)   │  │  (256×256)   │  │ "place the shoe"    │
└──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘
       │                 │                      │
       │ Resize          │ Resize               │ Tokenize
       ↓                 ↓                      ↓
┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐
│  (3×224×224) │  │  (3×224×224) │  │   input_ids         │
└──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘
       │                 │                      │
       └────────┬────────┘                      │
                ↓                               │
        ┌───────────────┐                       │
        │ Vision Encoder│ (SigLIP)              │
        │   + LoRA      │ ← Trainable adapters  │
        └───────┬───────┘                       │
                │                               │
                ↓                               │
        ┌───────────────┐                       │
        │  Projector    │ ← Trainable (LoRA)    │
        └───────┬───────┘                       │
                │                               │
                ↓                               ↓
        ┌──────────────────────────────────────────┐
        │       Language Model (Llama/Vicuna)      │
        │            7B parameters                 │
        │          + LoRA adapters                 │
        │         ← Trainable adapters             │
        └──────────────────┬───────────────────────┘
                           │
                           ↓
                ┌──────────────────┐
                │  Hidden States   │
                │   (seq_len, 4096)│
                └──────────┬───────┘
                           │
                           ↓
                ┌──────────────────┐
                │   Action Head    │ ← Fully Trainable
                │  L1 Regression   │    (~50M params)
                │  (4096 → 14)     │
                └──────────┬───────┘
                           │
                           ↓
OUTPUT:
                ┌──────────────────┐
                │  Action Chunk    │
                │   (8 × 14)       │
                │                  │
                │ [timestep 0]: 14 │ ← Current action
                │ [timestep 1]: 14 │
                │ [timestep 2]: 14 │
                │     ...          │
                │ [timestep 7]: 14 │ ← 7 steps ahead
                └──────────────────┘
```

## 🔄 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single Training Step                          │
└─────────────────────────────────────────────────────────────────┘

1. LOAD FROM HDF5:
   ┌─────────────────────────────────┐
   │ episode_0.hdf5 [timestep t=10]  │
   ├─────────────────────────────────┤
   │ head_camera_image[10]: (256,256,3) │
   │ left_wrist_image[10]:  (256,256,3) │
   │ action[10:18]:         (8, 14)     │
   │ seen: "place the shoe"             │
   └─────────────────┬───────────────────┘
                     ↓

2. PREPROCESS:
   ┌─────────────────────────────────┐
   │ • Resize images to 224×224      │
   │ • Normalize pixels to [-1, 1]   │
   │ • Tokenize language instruction │
   │ • Normalize actions             │
   └─────────────────┬───────────────┘
                     ↓

3. BATCH (size=8):
   ┌─────────────────────────────────────┐
   │ pixel_values:       (8, 3, 224, 224)│
   │ pixel_values_wrist: (8, 3, 224, 224)│
   │ input_ids:          (8, seq_len)    │
   │ labels:             (8, seq_len)    │
   │ actions:            (8, 8, 14)      │ ← Ground truth
   └─────────────────┬───────────────────┘
                     ↓

4. FORWARD PASS:
   ┌────────────────────────────────────┐
   │ OpenVLA(images, text)              │
   │   → vision_features                │
   │   → text_features                  │
   │   → fused_features                 │
   │   → hidden_states (8, seq_len, 4096) │
   └─────────────────┬──────────────────┘
                     ↓
   ┌────────────────────────────────────┐
   │ ActionHead(hidden_states)          │
   │   → predicted_actions (8, 8, 14)   │
   └─────────────────┬──────────────────┘
                     ↓

5. COMPUTE LOSS:
   ┌────────────────────────────────────────┐
   │ L1Loss(predicted_actions, gt_actions)  │
   │                                        │
   │ loss = |predicted - ground_truth|     │
   │                                        │
   │ Example:                               │
   │   predicted[0,0] = [0.12, -0.45, ...]  │
   │   ground_truth[0,0] = [0.13, -0.44, ...]│
   │   loss = mean(|0.12-0.13|, |-0.45-(-0.44)|, ...) │
   └─────────────────┬──────────────────────┘
                     ↓

6. BACKWARD PASS:
   ┌────────────────────────────────────┐
   │ loss.backward()                    │
   │   → Compute gradients              │
   │   → Update only:                   │
   │     • LoRA adapters (~14M params)  │
   │     • Action head (~50M params)    │
   │                                    │
   │   → Base model stays frozen!       │
   └─────────────────┬──────────────────┘
                     ↓

7. UPDATE WEIGHTS:
   ┌────────────────────────────────────┐
   │ optimizer.step()                   │
   │   → Apply gradients                │
   │   → Learning rate: 5e-4            │
   │   → AdamW optimizer                │
   └────────────────────────────────────┘
```

## 📊 Action Chunking Visualization

```
Timeline:  t=10  t=11  t=12  t=13  t=14  t=15  t=16  t=17  t=18
           ├────┼────┼────┼────┼────┼────┼────┼────┼────>
Episode:   │ ▓▓ │ ▓▓ │ ▓▓ │ ▓▓ │ ▓▓ │ ▓▓ │ ▓▓ │ ▓▓ │
           └────┴────┴────┴────┴────┴────┴────┴────┴────

At timestep t=10, the model sees:
┌──────────────────────────────────────────────────────┐
│ INPUT:                                               │
│   • Image at t=10                                    │
│   • Language: "place the shoe"                       │
│                                                      │
│ OUTPUT (predict 8 future actions):                  │
│   action[0] = action to take NOW at t=10   ◄─ Execute this
│   action[1] = action for t=11              │
│   action[2] = action for t=12              │
│   action[3] = action for t=13              ├─ Predict but
│   action[4] = action for t=14              │  don't use yet
│   action[5] = action for t=15              │
│   action[6] = action for t=16              │
│   action[7] = action for t=17              ◄
└──────────────────────────────────────────────────────┘

Why predict 8 steps?
  ✓ Temporal consistency (smooth motions)
  ✓ Reduces error accumulation
  ✓ Better long-term planning
```

## 🎯 Your Dataset Specifics

```
┌───────────────────────────────────────────────────────────┐
│              Place Shoe Dataset Structure                 │
└───────────────────────────────────────────────────────────┘

EPISODES: 48 HDF5 files
├── episode_0.hdf5
│   ├── head_camera_image:   (185, 256, 256, 3) uint8
│   ├── left_wrist_image:    (185, 256, 256, 3) uint8
│   ├── right_wrist_image:   (185, 256, 256, 3) uint8
│   ├── low_cam_image:       (185, 256, 256, 3) uint8
│   ├── action:              (185, 14) float64
│   ├── relative_action:     (185, 14) float64
│   ├── seen:                "place the shoe in..."
│   └── unseen:              "put the shoe on..."
├── episode_1.hdf5
│   └── ...
└── episode_47.hdf5

TOTAL TRAINING SAMPLES:
  48 episodes × ~178 timesteps per episode = ~8,544 samples
  (accounting for action chunking: 185 - 8 + 1 = 178)

ACTION SPACE (14 dimensions):
  [dim_0, dim_1, dim_2, ..., dim_13]
  ↑                              ↑
  Position/velocity          Gripper?
  (exact meaning depends on your robot)
```

## 🔢 Training Parameters

```
┌─────────────────────────────────────────────────────────┐
│                   Training Configuration                 │
└─────────────────────────────────────────────────────────┘

MODEL:
  Base: OpenVLA-7B (7 billion parameters)
  Trainable: ~64M parameters (LoRA + action head)
  
TRAINING:
  Batch size: 8 per GPU (×4 GPUs = 32 total)
  Learning rate: 5e-4 → 5e-5 (decay at 40k steps)
  Max steps: 50,000
  Optimizer: AdamW
  
DATA:
  Images: 2 (head + left wrist)
  Resolution: 224×224
  Action dim: 14
  Chunk size: 8 timesteps
  
LOSS:
  Type: L1 (Mean Absolute Error)
  Target: ~0.01 or lower
  
CHECKPOINTS:
  Frequency: Every 5,000 steps
  Location: /work/hdd/bfdj/tzhou4/checkpoints/
```

## 📈 Expected Training Progress

```
Step      Loss    Curr L1   Future L1   Status
────────────────────────────────────────────────
0         1.500   1.200     1.600       Random init
100       0.450   0.350     0.500       Learning basics
1,000     0.120   0.090     0.140       Improving
5,000     0.035   0.025     0.042       Getting good
10,000    0.015   0.011     0.018       Almost there ✓
20,000    0.010   0.007     0.012       Excellent ✓✓
40,000    0.008   0.006     0.009       Great ✓✓✓
50,000    0.007   0.005     0.008       Best ✓✓✓✓

Target: Loss < 0.01 is good, < 0.008 is excellent
```

## 🎮 Inference (After Training)

```
┌─────────────────────────────────────────────────────────┐
│                  Using Trained Model                     │
└─────────────────────────────────────────────────────────┘

REAL-TIME DEPLOYMENT:
┌────────────────┐
│ Robot at t=0   │
└────────┬───────┘
         │ Capture image
         ↓
┌────────────────────────┐
│ OpenVLA-OFT            │
│  Input: image, "place  │
│         the shoe"      │
│  Output: [a0...a7]     │ ← Predicted action chunk
└────────┬───────────────┘
         │
         ↓ Execute action[0]
┌────────────────┐
│ Robot moves    │
└────────┬───────┘
         │ t=1
         ↓ Capture new image
┌────────────────────────┐
│ OpenVLA-OFT            │
│  Input: new image      │
│  Output: [a0'...a7']   │ ← New predictions
└────────┬───────────────┘
         │ Execute action[0']
         ↓
       Continue...

Note: Can reuse actions from previous chunk
      or recompute every step (more robust)
```

---

## 📚 Key Takeaways

1. **Input**: Images + Language → **Output**: 8 future actions (14D each)

2. **Training**: Only ~64M params trainable (LoRA + action head), not 7B

3. **Loss**: L1 distance between predicted and ground truth actions

4. **Action Chunking**: Predicts 8 timesteps for temporal consistency

5. **Your Data**: 48 episodes, ~8.5k samples, 14D actions

For implementation details, see:
- `prismatic/vla/datasets/place_shoe_dataset.py`
- `vla-scripts/finetune_place_shoe.py`
- `prismatic/models/action_heads.py`

