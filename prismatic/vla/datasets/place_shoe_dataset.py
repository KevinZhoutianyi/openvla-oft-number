"""
place_shoe_dataset.py

Custom HDF5 Dataset loader for the "place shoe" task dataset.

This dataset has a specific structure:
- action: (T, 14) float64
- head_camera_image: (T, 256, 256, 3) uint8
- left_wrist_image: (T, 256, 256, 3) uint8
- right_wrist_image: (T, 256, 256, 3) uint8
- low_cam_image: (T, 256, 256, 3) uint8
- relative_action: (T, 14) float64
- seen: language instructions for seen tasks
- unseen: language instructions for unseen tasks
"""

import h5py
import numbers
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Type

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK
from transformers import PreTrainedTokenizerBase


class PlaceShoeDataset(Dataset):
    """
    PyTorch Dataset for the "place shoe" robot demonstration data from HDF5 files.
    
    Args:
        hdf5_path: Path to directory containing HDF5 files
        action_tokenizer: ActionTokenizer for converting actions to tokens
        base_tokenizer: Base tokenizer for text
        image_transform: Image transformation function
        prompt_builder_fn: Prompt builder class
        action_dim: Dimension of action space (default: 14 for this dataset)
        use_relative_actions: Whether to use relative_action instead of action (default: False)
        camera_view: Which camera to use as primary ("head", "low", "left_wrist", "right_wrist")
        use_wrist_image: Whether to load wrist camera images (default: True)
        wrist_camera: Which wrist camera to use ("left", "right", or "both")
        use_proprio: Whether to load proprioceptive state (not available in this dataset)
        predict_stop_token: Whether to predict stop token (default: True)
        compute_stats: Whether to compute dataset statistics (default: True)
        default_instruction: Default language instruction if not found (default: "place the shoe")
    """
    
    def __init__(
        self,
        hdf5_path: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        action_dim: int = 14,
        use_relative_actions: bool = False,
        camera_view: str = "head",  # "head", "low", "left_wrist", "right_wrist"
        use_wrist_image: bool = True,
        wrist_camera: str = "left",  # "left", "right", "both"
        use_proprio: bool = False,
        predict_stop_token: bool = True,
        compute_stats: bool = True,
        default_instruction: str = "place the shoe",
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.action_dim = action_dim
        self.use_relative_actions = use_relative_actions
        self.camera_view = camera_view
        self.use_wrist_image = use_wrist_image
        self.wrist_camera = wrist_camera
        self.use_proprio = use_proprio
        self.predict_stop_token = predict_stop_token
        self.default_instruction = default_instruction
        
        # Map camera_view to dataset key
        self.camera_key_map = {
            "head": "head_camera_image",
            "low": "low_cam_image",
            "left_wrist": "left_wrist_image",
            "right_wrist": "right_wrist_image",
        }
        self._success_keys = ("success", "successes", "episode_success", "rollout_success", "task_success")
        self.has_success = False
        
        # Load all episodes and build index
        self.episodes = []
        self._hdf5_files = []
        
        if self.hdf5_path.is_dir():
            # Load from multiple HDF5 files
            hdf5_files = sorted(self.hdf5_path.glob("*.hdf5")) + sorted(self.hdf5_path.glob("*.h5"))
            for file_path in hdf5_files:
                self._load_episodes_from_file(file_path)
            self._hdf5_files = hdf5_files
        else:
            # Load from single HDF5 file
            self._load_episodes_from_file(self.hdf5_path)
            self._hdf5_files = [self.hdf5_path]
        
        # Compute dataset statistics for action normalization
        if compute_stats:
            self.dataset_statistics = self._compute_statistics()
        else:
            # Use identity normalization (no normalization)
            self.dataset_statistics = {
                "place_shoe_dataset": {
                    "action": {
                        "q01": np.zeros((self.action_dim,), dtype=np.float32),
                        "q99": np.ones((self.action_dim,), dtype=np.float32)
                    }
                }
            }
        
        num_files = len(self._hdf5_files) if self._hdf5_files else (1 if self.hdf5_path.exists() else 0)
        print(f"Loaded {len(self.episodes)} transitions from {num_files} HDF5 file(s)")
    
    def _normalize_success_value(self, value: Any) -> Optional[float]:
        """Convert various success annotations to float in [0, 1]."""
        if value is None:
            return None

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                return None

        if isinstance(value, str):
            value_str = value.strip().lower()
            if value_str in {"1", "true", "success", "yes", "y"}:
                return 1.0
            if value_str in {"0", "false", "failure", "fail", "no", "n"}:
                return 0.0
            try:
                return float(value_str)
            except ValueError:
                return None

        if isinstance(value, numbers.Number):
            return float(value)

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            if value.size == 1:
                return self._normalize_success_value(value.reshape(-1)[0])

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            if len(value) == 1:
                return self._normalize_success_value(value[0])

        return None

    def _extract_success_series(self, f: h5py.File) -> Optional[Any]:
        """Try to fetch success annotations from datasets or attributes."""
        for key in self._success_keys:
            if key in f:
                try:
                    return f[key][()]
                except Exception:
                    continue
        for key in self._success_keys:
            if key in f.attrs:
                return f.attrs[key]
        return None

    def _get_success_value_from_series(self, series: Any, timestep: int) -> Optional[float]:
        if series is None:
            return None

        if isinstance(series, np.ndarray):
            if series.ndim == 0:
                return self._normalize_success_value(series.item())
            idx = min(timestep, series.shape[0] - 1)
            return self._normalize_success_value(series[idx])

        if isinstance(series, (list, tuple)):
            if len(series) == 0:
                return None
            idx = min(timestep, len(series) - 1)
            return self._normalize_success_value(series[idx])

        return self._normalize_success_value(series)

    def _load_episodes_from_file(self, file_path: Path) -> None:
        """Load episode metadata from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Get episode length from actions
            if 'action' in f:
                ep_len = len(f['action'])
            elif 'actions' in f:
                ep_len = len(f['actions'])
            else:
                print(f"Warning: Could not find actions in {file_path}")
                return

            success_series = self._extract_success_series(f)
            
            # Store episode metadata for each timestep
            for t in range(ep_len - NUM_ACTIONS_CHUNK + 1):  # Account for action chunking
                success_value = self._get_success_value_from_series(success_series, t)
                if success_value is not None:
                    self.has_success = True
                self.episodes.append({
                    'file_path': str(file_path),
                    'timestep': t,
                    'episode_length': ep_len,
                    'success': success_value,
                })
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics for action normalization."""
        print("Computing dataset statistics...")
        all_actions = []
        
        # Get unique file paths
        unique_files = set(ep['file_path'] for ep in self.episodes)
        
        # Sample actions from dataset
        for file_path in unique_files:
            with h5py.File(file_path, 'r') as f:
                # Load actions
                action_key = 'relative_action' if self.use_relative_actions and 'relative_action' in f else 'action'
                if action_key in f:
                    actions = f[action_key][:]
                elif 'actions' in f:
                    actions = f['actions'][:]
                else:
                    raise ValueError(f"Could not find actions in {file_path}")
                
                all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        # Compute 1st and 99th percentile (for robust normalization)
        q01 = np.percentile(all_actions, 1, axis=0).astype(np.float32)
        q99 = np.percentile(all_actions, 99, axis=0).astype(np.float32)
        
        # Avoid division by zero
        q99 = np.where(np.abs(q99 - q01) < 1e-6, q01 + 1.0, q99)
        
        print(f"Action statistics computed:")
        print(f"  Shape: {all_actions.shape}")
        print(f"  q01 (min): {q01[:5]}... (showing first 5 dims)")
        print(f"  q99 (max): {q99[:5]}... (showing first 5 dims)")
        
        return {
            "place_shoe_dataset": {
                "action": {"q01": q01, "q99": q99}
            }
        }
    
    def _get_language_instruction(self, f: h5py.File) -> str:
        """Extract language instruction from HDF5 file."""
        # Try different keys for language instruction
        for key in ['seen', 'unseen', 'language_instruction', 'language', 'task']:
            if key in f:
                instruction = f[key]
                # Handle different data types
                if isinstance(instruction, h5py.Dataset):
                    if instruction.dtype == 'object':
                        # Object dtype - try to decode
                        try:
                            inst_value = instruction[0]
                            if isinstance(inst_value, bytes):
                                return inst_value.decode('utf-8')
                            return str(inst_value)
                        except:
                            pass
                    else:
                        inst_value = instruction[()]
                        if isinstance(inst_value, bytes):
                            return inst_value.decode('utf-8')
                        elif isinstance(inst_value, np.ndarray):
                            if inst_value.dtype == 'object' and len(inst_value) > 0:
                                inst_str = inst_value[0]
                                if isinstance(inst_str, bytes):
                                    return inst_str.decode('utf-8')
                                return str(inst_str)
                        return str(inst_value)
        
        # Return default instruction if not found
        return self.default_instruction
    
    def load_episode(
        self,
        episode_info: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Load data from HDF5 file for a specific episode and timestep.
        
        Returns:
            image: (H, W, 3) uint8 array
            actions: (action_chunk_size, action_dim) float32 array
            instruction: string
            proprio: None (not available in this dataset)
            wrist_image: (H, W, 3) uint8 array or None
            success: Optional float indicating rollout success
        """
        with h5py.File(episode_info['file_path'], 'r') as f:
            t = episode_info['timestep']
            success_value = episode_info.get('success')
            if success_value is None and self.has_success:
                success_series = self._extract_success_series(f)
                success_value = self._get_success_value_from_series(success_series, t)
            
            # Load primary camera image
            camera_key = self.camera_key_map[self.camera_view]
            if camera_key in f:
                image = f[camera_key][t]
            else:
                raise KeyError(f"Camera view '{self.camera_view}' (key: '{camera_key}') not found in {episode_info['file_path']}")
            
            # Load actions with chunking
            action_key = 'relative_action' if self.use_relative_actions and 'relative_action' in f else 'action'
            if action_key in f:
                actions_full = f[action_key][:]
            elif 'actions' in f:
                actions_full = f['actions'][:]
            else:
                raise KeyError(f"Could not find actions in {episode_info['file_path']}")
            
            # Get action chunk (current + future actions)
            end_t = min(t + NUM_ACTIONS_CHUNK, len(actions_full))
            actions = actions_full[t:end_t]
            
            # Pad if necessary (if we're at the end of the episode)
            if len(actions) < NUM_ACTIONS_CHUNK:
                padding = np.tile(actions[-1:], (NUM_ACTIONS_CHUNK - len(actions), 1))
                actions = np.concatenate([actions, padding], axis=0)
            
            # Load language instruction
            instruction = self._get_language_instruction(f)
            
            # Load wrist image if requested
            wrist_image = None
            if self.use_wrist_image:
                if self.wrist_camera == "left" and "left_wrist_image" in f:
                    wrist_image = f["left_wrist_image"][t]
                elif self.wrist_camera == "right" and "right_wrist_image" in f:
                    wrist_image = f["right_wrist_image"][t]
                elif self.wrist_camera == "both":
                    # For now, just use left wrist if both requested
                    # You can modify this to concatenate both or return a list
                    if "left_wrist_image" in f:
                        wrist_image = f["left_wrist_image"][t]
            
            # Proprio is not available in this dataset
            proprio = None
        
        return image, actions, instruction, proprio, wrist_image, success_value
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        episode_info = self.episodes[idx]
        
        # Load data from HDF5
        image, actions, instruction, proprio, wrist_image, success_value = self.load_episode(episode_info)
        
        # Convert image to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        # Create prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        
        # Get action chunk string
        action_chunk_string = ''.join([self.action_tokenizer(action) for action in actions])
        action_chunk_len = len(action_chunk_string)
        
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction.lower()}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        # Build return dict
        return_dict = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'dataset_name': 'place_shoe_dataset'
        }
        
        # Add wrist image if available
        if self.use_wrist_image and wrist_image is not None:
            if wrist_image.dtype != np.uint8:
                wrist_image = (wrist_image * 255).astype(np.uint8)
            wrist_image_pil = Image.fromarray(wrist_image)
            return_dict['pixel_values_wrist'] = self.image_transform(wrist_image_pil)

        if self.has_success:
            success_tensor = torch.tensor(
                float(success_value) if success_value is not None else float("nan"),
                dtype=torch.float32,
            )
            return_dict['success'] = success_tensor
        
        return return_dict
