"""
hdf5_dataset.py

Custom HDF5 Dataset loader for OpenVLA fine-tuning.

This dataset assumes your HDF5 files have the following structure:
- episodes/episode_<N>/observations/images (H x W x 3 uint8 images)
- episodes/episode_<N>/observations/qpos (proprioceptive state, optional)
- episodes/episode_<N>/actions (action vectors)
- episodes/episode_<N>/language_instruction (text instruction)

You can modify the paths and keys to match your specific HDF5 structure.
"""

import h5py
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


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for loading robot demonstration data from HDF5 files.
    
    Expected HDF5 structure (you can modify load_episode() to match your format):
    - episodes/episode_<N>/observations/images: (T, H, W, 3) uint8 array
    - episodes/episode_<N>/observations/qpos: (T, proprio_dim) float32 array (optional)
    - episodes/episode_<N>/observations/wrist_image: (T, H, W, 3) uint8 array (optional)
    - episodes/episode_<N>/actions: (T, action_dim) float32 array
    - episodes/episode_<N>/language_instruction: string or bytes
    
    Args:
        hdf5_path: Path to HDF5 file or directory containing multiple HDF5 files
        action_tokenizer: ActionTokenizer for converting actions to tokens
        base_tokenizer: Base tokenizer for text
        image_transform: Image transformation function
        prompt_builder_fn: Prompt builder class
        action_dim: Dimension of action space (default: 7)
        use_wrist_image: Whether to load wrist camera images (default: False)
        use_proprio: Whether to load proprioceptive state (default: False)
        predict_stop_token: Whether to predict stop token (default: True)
        compute_stats: Whether to compute dataset statistics (default: True)
    """
    
    def __init__(
        self,
        hdf5_path: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        action_dim: int = 7,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
        predict_stop_token: bool = True,
        compute_stats: bool = True,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.action_dim = action_dim
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio
        self.predict_stop_token = predict_stop_token
        
        # Load all episodes and build index
        self.episodes = []
        self.episode_starts = []  # Start index for each episode
        
        if self.hdf5_path.is_dir():
            # Load from multiple HDF5 files
            hdf5_files = sorted(self.hdf5_path.glob("*.hdf5")) + sorted(self.hdf5_path.glob("*.h5"))
            for file_path in hdf5_files:
                self._load_episodes_from_file(file_path)
        else:
            # Load from single HDF5 file
            self._load_episodes_from_file(self.hdf5_path)
        
        # Compute dataset statistics for action normalization
        if compute_stats:
            self.dataset_statistics = self._compute_statistics()
        else:
            # Use identity normalization (no normalization)
            self.dataset_statistics = {
                "hdf5_dataset": {
                    "action": {
                        "q01": np.zeros((self.action_dim,), dtype=np.float32),
                        "q99": np.ones((self.action_dim,), dtype=np.float32)
                    }
                }
            }
        
        print(f"Loaded {len(self.episodes)} episodes with {len(self)} total transitions")
    
    def _load_episodes_from_file(self, file_path: Path) -> None:
        """Load episode metadata from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Check if file has 'data' or 'episodes' group
            if 'data' in f:
                root = f['data']
            elif 'episodes' in f:
                root = f['episodes']
            else:
                # Try to find episode keys directly
                episode_keys = [k for k in f.keys() if k.startswith('episode')]
                if episode_keys:
                    root = f
                else:
                    raise ValueError(f"Could not find episodes in {file_path}. "
                                   "Expected 'data', 'episodes', or 'episode_*' keys.")
            
            # Get all episode keys
            episode_keys = sorted([k for k in root.keys() if k.startswith('episode')])
            
            for ep_key in episode_keys:
                ep_group = root[ep_key]
                
                # Get episode length
                if 'actions' in ep_group:
                    ep_len = len(ep_group['actions'])
                elif 'obs' in ep_group:
                    # Sometimes actions are stored in obs group
                    ep_len = len(ep_group['obs'])
                else:
                    print(f"Warning: Could not determine episode length for {ep_key} in {file_path}")
                    continue
                
                # Store episode metadata
                self.episode_starts.append(len(self.episodes))
                for t in range(ep_len - NUM_ACTIONS_CHUNK + 1):  # Account for action chunking
                    self.episodes.append({
                        'file_path': str(file_path),
                        'episode_key': ep_key,
                        'timestep': t,
                        'episode_length': ep_len
                    })
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics for action normalization."""
        print("Computing dataset statistics...")
        all_actions = []
        
        # Sample actions from dataset
        for episode_info in self.episodes:
            with h5py.File(episode_info['file_path'], 'r') as f:
                if 'data' in f:
                    ep_group = f['data'][episode_info['episode_key']]
                elif 'episodes' in f:
                    ep_group = f['episodes'][episode_info['episode_key']]
                else:
                    ep_group = f[episode_info['episode_key']]
                
                # Load actions
                if 'actions' in ep_group:
                    actions = ep_group['actions'][:]
                elif 'action' in ep_group:
                    actions = ep_group['action'][:]
                else:
                    raise ValueError(f"Could not find actions in episode {episode_info['episode_key']}")
                
                all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        # Compute 1st and 99th percentile (for robust normalization)
        q01 = np.percentile(all_actions, 1, axis=0).astype(np.float32)
        q99 = np.percentile(all_actions, 99, axis=0).astype(np.float32)
        
        # Avoid division by zero
        q99 = np.where(np.abs(q99 - q01) < 1e-6, q01 + 1.0, q99)
        
        print(f"Action statistics - q01: {q01}, q99: {q99}")
        
        return {
            "hdf5_dataset": {
                "action": {"q01": q01, "q99": q99}
            }
        }
    
    def load_episode(self, episode_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load data from HDF5 file for a specific episode and timestep.
        
        MODIFY THIS METHOD to match your HDF5 structure!
        
        Returns:
            image: (H, W, 3) uint8 array
            actions: (action_chunk_size, action_dim) float32 array
            instruction: string
            proprio: (proprio_dim,) float32 array or None
            wrist_image: (H, W, 3) uint8 array or None
        """
        with h5py.File(episode_info['file_path'], 'r') as f:
            # Navigate to episode group
            if 'data' in f:
                ep_group = f['data'][episode_info['episode_key']]
            elif 'episodes' in f:
                ep_group = f['episodes'][episode_info['episode_key']]
            else:
                ep_group = f[episode_info['episode_key']]
            
            t = episode_info['timestep']
            
            # Load image (MODIFY THESE KEYS to match your HDF5 structure)
            try:
                if 'observations/images' in ep_group:
                    image = ep_group['observations/images'][t]
                elif 'obs/images' in ep_group:
                    image = ep_group['obs/images'][t]
                elif 'observations/image' in ep_group:
                    image = ep_group['observations/image'][t]
                elif 'obs/image' in ep_group:
                    image = ep_group['obs/image'][t]
                else:
                    raise KeyError("Could not find image observations")
            except KeyError as e:
                raise KeyError(f"Error loading image from {episode_info['file_path']}, "
                             f"episode {episode_info['episode_key']}, timestep {t}. "
                             f"Available keys: {list(ep_group.keys())}. Error: {e}")
            
            # Load actions with chunking
            try:
                if 'actions' in ep_group:
                    actions_full = ep_group['actions'][:]
                elif 'action' in ep_group:
                    actions_full = ep_group['action'][:]
                else:
                    raise KeyError("Could not find actions")
                
                # Get action chunk (current + future actions)
                end_t = min(t + NUM_ACTIONS_CHUNK, len(actions_full))
                actions = actions_full[t:end_t]
                
                # Pad if necessary (if we're at the end of the episode)
                if len(actions) < NUM_ACTIONS_CHUNK:
                    padding = np.tile(actions[-1:], (NUM_ACTIONS_CHUNK - len(actions), 1))
                    actions = np.concatenate([actions, padding], axis=0)
                
            except KeyError as e:
                raise KeyError(f"Error loading actions from {episode_info['file_path']}, "
                             f"episode {episode_info['episode_key']}. Error: {e}")
            
            # Load language instruction
            try:
                if 'language_instruction' in ep_group:
                    instruction = ep_group['language_instruction'][()]
                elif 'language' in ep_group:
                    instruction = ep_group['language'][()]
                elif 'task' in ep_group:
                    instruction = ep_group['task'][()]
                else:
                    # Default instruction if not available
                    instruction = "perform the task"
                
                # Decode if bytes
                if isinstance(instruction, bytes):
                    instruction = instruction.decode('utf-8')
                elif isinstance(instruction, np.ndarray):
                    instruction = str(instruction)
                
            except KeyError:
                instruction = "perform the task"
            
            # Load proprio if requested
            proprio = None
            if self.use_proprio:
                try:
                    if 'observations/qpos' in ep_group:
                        proprio = ep_group['observations/qpos'][t]
                    elif 'obs/qpos' in ep_group:
                        proprio = ep_group['obs/qpos'][t]
                    elif 'observations/state' in ep_group:
                        proprio = ep_group['observations/state'][t]
                    elif 'obs/state' in ep_group:
                        proprio = ep_group['obs/state'][t]
                except KeyError:
                    print(f"Warning: proprio requested but not found in {episode_info['episode_key']}")
            
            # Load wrist image if requested
            wrist_image = None
            if self.use_wrist_image:
                try:
                    if 'observations/wrist_image' in ep_group:
                        wrist_image = ep_group['observations/wrist_image'][t]
                    elif 'obs/wrist_image' in ep_group:
                        wrist_image = ep_group['obs/wrist_image'][t]
                    elif 'observations/hand_image' in ep_group:
                        wrist_image = ep_group['observations/hand_image'][t]
                except KeyError:
                    print(f"Warning: wrist_image requested but not found in {episode_info['episode_key']}")
        
        return image, actions, instruction, proprio, wrist_image
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        episode_info = self.episodes[idx]
        
        # Load data from HDF5
        image, actions, instruction, proprio, wrist_image = self.load_episode(episode_info)
        
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
            'dataset_name': 'hdf5_dataset'
        }
        
        # Add wrist image if available
        if self.use_wrist_image and wrist_image is not None:
            if wrist_image.dtype != np.uint8:
                wrist_image = (wrist_image * 255).astype(np.uint8)
            wrist_image_pil = Image.fromarray(wrist_image)
            return_dict['pixel_values_wrist'] = self.image_transform(wrist_image_pil)
        
        # Add proprio if available
        if self.use_proprio and proprio is not None:
            return_dict['proprio'] = torch.from_numpy(proprio.astype(np.float32))
        
        return return_dict

