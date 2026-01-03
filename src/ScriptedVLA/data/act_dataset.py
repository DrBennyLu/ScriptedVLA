"""
ACT数据集适配器
将ACT数据集转换为VLA训练格式
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import h5py
import json


class ACTDataset(Dataset):
    """
    ACT (Action Chunking Transformer) 数据集适配器
    将ACT演示数据转换为VLA训练格式
    """
    
    def __init__(
        self,
        dataset_path: str,
        image_size: int = 224,
        chunk_size: int = 1,
        transform: Optional[callable] = None
    ):
        """
        初始化ACT数据集
        
        Args:
            dataset_path: ACT数据集路径
            image_size: 图像尺寸
            chunk_size: 动作块大小
            transform: 图像变换
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.transform = transform
        
        # 加载数据样本
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from ACT dataset")
    
    def _load_samples(self) -> List[Dict]:
        """加载ACT数据样本"""
        samples = []
        
        # ACT数据集通常使用HDF5格式存储
        # 查找所有.h5文件
        h5_files = list(self.dataset_path.rglob("*.h5"))
        
        if not h5_files:
            # 如果没有找到.h5文件，尝试查找其他格式
            # ACT数据集可能也使用其他格式
            json_files = list(self.dataset_path.rglob("*.json"))
            if json_files:
                return self._load_from_json(json_files)
            else:
                print(f"Warning: No data files found in {self.dataset_path}")
                return []
        
        # 从HDF5文件加载
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # ACT数据格式通常包含：
                    # - observations/images: 图像数据
                    # - actions: 动作数据
                    # - qpos: 关节位置（可选）
                    
                    if 'observations' in f:
                        obs_group = f['observations']
                        images = obs_group.get('images', None)
                    elif 'images' in f:
                        images = f['images']
                    else:
                        print(f"Warning: No images found in {h5_file}")
                        continue
                    
                    if 'actions' in f:
                        actions = f['actions']
                    elif 'action' in f:
                        actions = f['action']
                    else:
                        print(f"Warning: No actions found in {h5_file}")
                        continue
                    
                    # 转换为numpy数组
                    images_data = np.array(images)
                    actions_data = np.array(actions)
                    
                    # 确保长度匹配
                    min_len = min(len(images_data), len(actions_data))
                    
                    # 创建样本
                    for i in range(min_len - self.chunk_size):
                        # 使用当前图像和未来动作
                        image = images_data[i]
                        action = actions_data[i:i+self.chunk_size]
                        
                        # 如果chunk_size > 1，展平动作
                        if len(action.shape) > 1:
                            action = action.flatten()
                        
                        samples.append({
                            "image": image,
                            "text": "",  # ACT数据集可能没有文本指令
                            "action": action,
                            "episode_id": h5_file.stem
                        })
            
            except Exception as e:
                print(f"Warning: Failed to load {h5_file}: {e}")
                continue
        
        return samples
    
    def _load_from_json(self, json_files: List[Path]) -> List[Dict]:
        """从JSON文件加载数据"""
        samples = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 根据JSON结构提取数据
                # 这里提供一个通用的处理方式
                if isinstance(data, list):
                    for item in data:
                        if 'image' in item and 'action' in item:
                            samples.append({
                                "image": item['image'],
                                "text": item.get('text', ''),
                                "action": np.array(item['action'], dtype=np.float32),
                                "episode_id": json_file.stem
                            })
                elif isinstance(data, dict):
                    # 如果是字典格式
                    images = data.get('images', [])
                    actions = data.get('actions', [])
                    
                    for i in range(min(len(images), len(actions))):
                        samples.append({
                            "image": images[i],
                            "text": data.get('text', ''),
                            "action": np.array(actions[i], dtype=np.float32),
                            "episode_id": json_file.stem
                        })
            
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 处理图像
        image_data = sample["image"]
        
        # 如果是字符串路径，加载图像
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, np.ndarray):
            # 如果是numpy数组
            if image_data.dtype == np.uint8:
                image = Image.fromarray(image_data)
            else:
                # 如果是归一化的图像，需要转换
                image = Image.fromarray((image_data * 255).astype(np.uint8))
        else:
            image = image_data
        
        # 调整大小
        image = image.resize((self.image_size, self.image_size))
        
        # 转换为张量
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        # 应用变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # 处理动作
        action = torch.from_numpy(np.array(sample["action"], dtype=np.float32))
        
        return {
            "image": image_tensor,
            "text": sample.get("text", ""),
            "action": action,
            "episode_id": sample.get("episode_id", "")
        }


def create_act_dataset_from_config(config: Dict) -> ACTDataset:
    """
    从配置创建ACT数据集
    
    Args:
        config: 配置字典，包含dataset_path等参数
        
    Returns:
        ACTDataset实例
    """
    return ACTDataset(
        dataset_path=config.get("dataset_path", "./data/act"),
        image_size=config.get("image_size", 224),
        chunk_size=config.get("chunk_size", 1)
    )

