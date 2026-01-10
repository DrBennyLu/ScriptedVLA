"""
LeRobotDataset适配器
将LeRobot格式的数据集转换为VLA训练格式
兼容HuggingFace上的开源机器人学习数据集
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import json
import h5py


class LeRobotDatasetAdapter(Dataset):
    """
    LeRobot数据集适配器
    将LeRobot格式的数据集转换为VLA训练格式
    
    LeRobot数据集格式：
    - 使用Parquet文件存储元数据（图像路径、动作、状态等）
    - 使用MP4文件存储图像序列（或单独的图像文件）
    - 包含时间窗口信息，支持action chunking
    
    这个适配器将LeRobot格式转换为VLADataset兼容的HDF5格式
    """
    
    def __init__(
        self,
        dataset_path: str,
        image_size: int = 224,
        camera_names: Optional[List[str]] = None,
        use_state: bool = True,
        state_dim: Optional[int] = None,
        action_horizon: int = 4,
        pad_action_chunk: bool = True,
        transform: Optional[callable] = None
    ):
        """
        初始化LeRobot数据集适配器
        
        Args:
            dataset_path: LeRobot数据集路径（可以是HF数据集名称或本地路径）
            image_size: 图像尺寸
            camera_names: 相机名称列表，例如 ["wrist", "base"]
            use_state: 是否使用机器人状态信息
            state_dim: 状态维度（如果为None，从数据中推断）
            action_horizon: 动作序列长度（action chunk大小）
            pad_action_chunk: 如果episode末尾不够action_horizon，是否使用最后一个动作填充
            transform: 图像变换
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.transform = transform
        self.camera_names = camera_names or ["wrist"]
        self.use_state = use_state
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.pad_action_chunk = pad_action_chunk
        
        # 检查是否是HuggingFace数据集名称
        if not self.dataset_path.exists() and "/" in str(dataset_path):
            # 可能是HF数据集名称，需要下载
            self._try_load_from_hf(dataset_path)
        else:
            # 本地路径
            self._load_from_local()
        
        print(f"Loaded {len(self.samples)} samples from LeRobot dataset: {dataset_path}")
        print(f"Cameras: {self.camera_names}, Use state: {self.use_state}, State dim: {self.state_dim}")
        print(f"Action horizon (chunk size): {self.action_horizon}")
    
    def _try_load_from_hf(self, dataset_name: str):
        """
        尝试从HuggingFace加载数据集
        
        Args:
            dataset_name: HF数据集名称，例如 "lerobot/pusht"
        """
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as LeRobotDatasetHF
            from datasets import load_dataset
            
            print(f"Attempting to load LeRobot dataset from HuggingFace: {dataset_name}")
            
            # 尝试使用lerobot库加载
            try:
                lerobot_ds = LeRobotDatasetHF(dataset_name, root="~/.cache/lerobot")
                self.samples = self._convert_from_lerobot_hf(lerobot_ds)
            except Exception as e1:
                print(f"Warning: Failed to load with lerobot library: {e1}")
                # 尝试使用datasets库直接加载
                try:
                    hf_ds = load_dataset(dataset_name, split="train")
                    self.samples = self._convert_from_hf_datasets(hf_ds)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load dataset from HuggingFace: {e1}, {e2}")
        
        except ImportError:
            raise ImportError(
                "LeRobot or datasets library not installed. "
                "Install with: pip install lerobot datasets"
            )
    
    def _load_from_local(self):
        """从本地路径加载LeRobot格式数据集"""
        # 检查是否是LeRobot格式（通常包含info.json和episode_*.hdf5文件）
        info_file = self.dataset_path / "info.json"
        
        if info_file.exists():
            # LeRobot格式：从info.json读取元数据，从HDF5文件读取数据
            self.samples = self._load_lerobot_format()
        else:
            raise ValueError(f"Not a valid LeRobot dataset path: {self.dataset_path}")
    
    def _load_lerobot_format(self) -> List[Dict]:
        """
        加载LeRobot格式的数据集
        
        LeRobot格式通常包含：
        - info.json: 数据集元数据
        - episode_*.hdf5: 每个episode的HDF5文件
        
        转换为VLADataset兼容格式
        """
        samples = []
        info_file = self.dataset_path / "info.json"
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # 获取数据集信息
        num_episodes = info.get("total", {}).get("episodes", 0)
        
        # 如果state_dim未指定，从info中推断
        if self.state_dim is None and self.use_state:
            state_keys = info.get("fps", {}).get("state", [])
            self.state_dim = len(state_keys) if state_keys else 7
        
        # 遍历所有episode文件
        for episode_idx in range(num_episodes):
            episode_file = self.dataset_path / f"episode_{episode_idx:06d}.hdf5"
            
            if not episode_file.exists():
                continue
            
            try:
                with h5py.File(episode_file, 'r') as f:
                    # LeRobot格式通常使用不同的键名
                    # 检查常见的键名格式
                    images_key = None
                    actions_key = None
                    states_key = None
                    
                    for key in f.keys():
                        if 'image' in key.lower() or 'observation' in key.lower():
                            images_key = key
                        elif 'action' in key.lower():
                            actions_key = key
                        elif 'state' in key.lower() and self.use_state:
                            states_key = key
                    
                    if images_key is None or actions_key is None:
                        print(f"Warning: Skipping episode {episode_idx}, missing required keys")
                        continue
                    
                    # 读取数据
                    images = f[images_key]  # 可能是 [T, H, W, C] 或 [T, num_cameras, H, W, C]
                    actions = f[actions_key]  # [T, action_dim]
                    
                    states = None
                    if states_key and self.use_state:
                        states = f[states_key]  # [T, state_dim]
                    
                    # 获取时间步数
                    num_steps = images.shape[0]
                    
                    # 处理图像维度
                    if images.ndim == 4:
                        # [T, H, W, C] - 单相机
                        images = np.expand_dims(images, axis=1)  # [T, 1, H, W, C]
                    elif images.ndim == 5:
                        # [T, num_cameras, H, W, C] - 多相机
                        pass
                    else:
                        print(f"Warning: Unexpected image shape {images.shape}, skipping episode {episode_idx}")
                        continue
                    
                    # 为每个step创建样本（考虑action chunk）
                    for step_idx in range(num_steps):
                        # 检查是否可以形成完整的action chunk
                        if step_idx + self.action_horizon > num_steps:
                            if not self.pad_action_chunk:
                                continue
                        
                        # 提取当前step的图像 [num_cameras, H, W, C]
                        step_images = images[step_idx]
                        
                        # 创建样本
                        sample = {
                            "h5_data": {
                                "h5_file": str(episode_file),
                                "step_idx": step_idx,
                                "num_steps": num_steps,
                                "images": step_images,  # [num_cameras, H, W, C]
                                "episode_actions": np.array(actions),  # [T, action_dim]
                                "episode_images": images,  # [T, num_cameras, H, W, C]
                                "instruction": f"Episode {episode_idx}, Step {step_idx}"
                            },
                            "task_name": info.get("task", "unknown"),
                            "episode_id": episode_idx,
                            "step_id": step_idx
                        }
                        
                        if states is not None and self.use_state:
                            sample["h5_data"]["episode_states"] = np.array(states)  # [T, state_dim]
                        
                        # 创建image_paths标识（用于兼容VLADataset）
                        image_paths = {}
                        for cam_idx, cam_name in enumerate(self.camera_names[:step_images.shape[0]]):
                            image_paths[cam_name] = {
                                'h5_file': str(episode_file),
                                'step_idx': step_idx,
                                'camera_idx': cam_idx,
                                'type': 'hdf5'
                            }
                        sample["image_paths"] = image_paths
                        
                        samples.append(sample)
            
            except Exception as e:
                print(f"Warning: Failed to load episode {episode_idx}: {e}")
                continue
        
        return samples
    
    def _convert_from_lerobot_hf(self, lerobot_ds) -> List[Dict]:
        """
        从LeRobot HF数据集对象转换
        
        Args:
            lerobot_ds: LeRobotDataset对象
            
        Returns:
            转换后的样本列表
        """
        samples = []
        
        # 遍历数据集
        for idx in range(len(lerobot_ds)):
            try:
                sample = lerobot_ds[idx]
                
                # 提取数据
                # LeRobot格式通常包含：observation, action, state等字段
                images = sample.get("observation", {}).get("image", sample.get("image"))
                actions = sample.get("action", [])
                states = sample.get("state", None)
                instruction = sample.get("instruction", "")
                
                # 转换为VLADataset格式
                converted_sample = {
                    "images": images,
                    "action": np.array(actions, dtype=np.float32),
                    "text": instruction,
                    "task_name": sample.get("task", "unknown"),
                    "episode_id": sample.get("episode_id", idx),
                    "step_id": sample.get("index", 0)
                }
                
                if states is not None and self.use_state:
                    converted_sample["state"] = np.array(states, dtype=np.float32)
                
                samples.append(converted_sample)
            
            except Exception as e:
                print(f"Warning: Failed to convert sample {idx}: {e}")
                continue
        
        return samples
    
    def _convert_from_hf_datasets(self, hf_ds) -> List[Dict]:
        """
        从HuggingFace datasets库加载的数据集转换
        
        Args:
            hf_ds: HuggingFace Dataset对象
            
        Returns:
            转换后的样本列表
        """
        samples = []
        
        for idx, sample in enumerate(hf_ds):
            try:
                # HF datasets格式可能不同，需要适配
                # 这里提供一个通用的转换逻辑
                converted_sample = {
                    "action": np.array(sample.get("action", []), dtype=np.float32),
                    "text": sample.get("instruction", sample.get("text", "")),
                    "task_name": sample.get("task", "unknown"),
                    "episode_id": sample.get("episode_id", idx),
                    "step_id": sample.get("index", sample.get("step_id", 0))
                }
                
                # 处理图像（可能是路径或数组）
                images = sample.get("image", sample.get("observation", {}).get("image", None))
                if images is not None:
                    converted_sample["images"] = images
                
                # 处理状态
                if self.use_state:
                    states = sample.get("state", None)
                    if states is not None:
                        converted_sample["state"] = np.array(states, dtype=np.float32)
                
                samples.append(converted_sample)
            
            except Exception as e:
                print(f"Warning: Failed to convert sample {idx}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本，包含action chunk
        
        Returns:
            与VLADataset兼容的格式
        """
        sample = self.samples[idx]
        
        # 如果已经是VLADataset兼容格式（包含h5_data），直接使用
        if "h5_data" in sample:
            # 使用VLADataset的逻辑提取action chunk
            # 这里简化处理，实际应该与VLADataset保持一致
            h5_data = sample["h5_data"]
            step_idx = h5_data["step_idx"]
            num_steps = h5_data["num_steps"]
            episode_actions = h5_data["episode_actions"]  # [T, action_dim]
            
            # 提取action chunk
            end_idx = step_idx + self.action_horizon
            
            if end_idx <= num_steps:
                action_chunk = episode_actions[step_idx:end_idx]  # [action_horizon, action_dim]
            else:
                if self.pad_action_chunk:
                    action_chunk = np.zeros((self.action_horizon, episode_actions.shape[1]), dtype=episode_actions.dtype)
                    available_steps = num_steps - step_idx
                    action_chunk[:available_steps] = episode_actions[step_idx:]
                    if available_steps > 0:
                        action_chunk[available_steps:] = episode_actions[-1]
                    else:
                        action_chunk[:] = episode_actions[-1]
                else:
                    raise ValueError(f"Cannot form action chunk: step_idx={step_idx}, num_steps={num_steps}")
            
            # 提取图像
            step_images = h5_data["images"]  # [num_cameras, H, W, C]
            
            # 转换为tensor
            images_dict = {}
            for cam_idx, cam_name in enumerate(self.camera_names[:step_images.shape[0]]):
                image_array = step_images[cam_idx]  # [H, W, C]
                image = Image.fromarray(image_array)
                image = image.resize((self.image_size, self.image_size))
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                
                if self.transform:
                    image_tensor = self.transform(image_tensor)
                
                images_dict[cam_name] = image_tensor
            
            result = {
                "images": images_dict,
                "text": h5_data["instruction"],
                "action": torch.from_numpy(action_chunk),  # [action_horizon, action_dim]
                "task_name": sample.get("task_name", "unknown"),
                "episode_id": sample.get("episode_id", 0),
                "step_id": sample.get("step_id", 0)
            }
            
            if self.use_state:
                if "episode_states" in h5_data:
                    episode_states = h5_data["episode_states"]  # [T, state_dim]
                    result["state"] = torch.from_numpy(episode_states[step_idx])  # [state_dim]
                else:
                    result["state"] = torch.zeros(self.state_dim, dtype=torch.float32)
            
            return result
        
        # 否则，使用原始格式（需要进一步转换）
        # 这里提供一个基本的实现
        raise NotImplementedError("Direct sample format not yet implemented. Please use HDF5 format.")


def create_lerobot_dataset_from_config(config: Dict) -> LeRobotDatasetAdapter:
    """
    从配置创建LeRobot数据集适配器
    
    Args:
        config: 配置字典，包含lerobot相关配置
        
    Returns:
        LeRobotDatasetAdapter实例
    """
    return LeRobotDatasetAdapter(
        dataset_path=config.get("dataset_path", "./data/lerobot"),
        image_size=config.get("image_size", 224),
        camera_names=config.get("camera_names", ["wrist"]),
        use_state=config.get("use_state", True),
        state_dim=config.get("state_dim", None),
        action_horizon=config.get("action_horizon", 4),
        pad_action_chunk=config.get("pad_action_chunk", True)
    )

