# MIT License
#
# Copyright (c) 2024 ScriptedVLA Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Benny Lu
"""
数据归一化和反归一化工具
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Union, Tuple


class Normalizer:
    """
    数据归一化器
    
    用于对action和state进行归一化和反归一化
    """
    
    def __init__(
        self,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        state_min: Optional[np.ndarray] = None,
        state_max: Optional[np.ndarray] = None
    ):
        """
        初始化归一化器
        
        Args:
            action_min: action的最小值 [action_dim]
            action_max: action的最大值 [action_dim]
            state_min: state的最小值 [state_dim]
            state_max: state的最大值 [state_dim]
        """
        self.action_min = action_min
        self.action_max = action_max
        self.state_min = state_min
        self.state_max = state_max
        
        # 计算action的范围
        if action_min is not None and action_max is not None:
            self.action_range = action_max - action_min
            # 避免除零
            self.action_range = np.where(self.action_range == 0, 1.0, self.action_range)
        else:
            self.action_range = None
        
        # 计算state的范围
        if state_min is not None and state_max is not None:
            self.state_range = state_max - state_min
            # 避免除零
            self.state_range = np.where(self.state_range == 0, 1.0, self.state_range)
        else:
            self.state_range = None
    
    def normalize_action(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        归一化action
        
        Args:
            action: 原始action [..., action_dim]
            
        Returns:
            归一化后的action，范围在[-1, 1]
        """
        if self.action_min is None or self.action_max is None:
            return action
        
        is_tensor = isinstance(action, torch.Tensor)
        if is_tensor:
            device = action.device
            dtype = action.dtype
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 归一化到[-1, 1]
        normalized = 2.0 * (action_np - self.action_min) / self.action_range - 1.0
        
        if is_tensor:
            return torch.from_numpy(normalized).to(device=device, dtype=dtype)
        else:
            return normalized
    
    def denormalize_action(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        反归一化action
        
        Args:
            action: 归一化后的action [..., action_dim]，范围在[-1, 1]
            
        Returns:
            反归一化后的action，恢复到原始范围
        """
        if self.action_min is None or self.action_max is None:
            return action
        
        is_tensor = isinstance(action, torch.Tensor)
        if is_tensor:
            device = action.device
            dtype = action.dtype
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 从[-1, 1]反归一化
        denormalized = (action_np + 1.0) / 2.0 * self.action_range + self.action_min
        
        if is_tensor:
            return torch.from_numpy(denormalized).to(device=device, dtype=dtype)
        else:
            return denormalized
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        归一化state
        
        Args:
            state: 原始state [..., state_dim]
            
        Returns:
            归一化后的state，范围在[-1, 1]
        """
        if self.state_min is None or self.state_max is None:
            return state
        
        is_tensor = isinstance(state, torch.Tensor)
        if is_tensor:
            device = state.device
            dtype = state.dtype
            state_np = state.cpu().numpy()
        else:
            state_np = np.array(state)
        
        # 归一化到[-1, 1]
        normalized = 2.0 * (state_np - self.state_min) / self.state_range - 1.0
        
        if is_tensor:
            return torch.from_numpy(normalized).to(device=device, dtype=dtype)
        else:
            return normalized
    
    def denormalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        反归一化state
        
        Args:
            state: 归一化后的state [..., state_dim]，范围在[-1, 1]
            
        Returns:
            反归一化后的state，恢复到原始范围
        """
        if self.state_min is None or self.state_max is None:
            return state
        
        is_tensor = isinstance(state, torch.Tensor)
        if is_tensor:
            device = state.device
            dtype = state.dtype
            state_np = state.cpu().numpy()
        else:
            state_np = np.array(state)
        
        # 从[-1, 1]反归一化
        denormalized = (state_np + 1.0) / 2.0 * self.state_range + self.state_min
        
        if is_tensor:
            return torch.from_numpy(denormalized).to(device=device, dtype=dtype)
        else:
            return denormalized
    
    def to_dict(self) -> Dict:
        """
        将归一化参数转换为字典
        
        Returns:
            包含归一化参数的字典
        """
        result = {}
        if self.action_min is not None:
            result["action_min"] = self.action_min.tolist()
        if self.action_max is not None:
            result["action_max"] = self.action_max.tolist()
        if self.state_min is not None:
            result["state_min"] = self.state_min.tolist()
        if self.state_max is not None:
            result["state_max"] = self.state_max.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Normalizer":
        """
        从字典创建归一化器
        
        Args:
            data: 包含归一化参数的字典
            
        Returns:
            Normalizer实例
        """
        action_min = np.array(data["action_min"]) if "action_min" in data else None
        action_max = np.array(data["action_max"]) if "action_max" in data else None
        state_min = np.array(data["state_min"]) if "state_min" in data else None
        state_max = np.array(data["state_max"]) if "state_max" in data else None
        
        return cls(
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max
        )
    
    def save(self, path: Union[str, Path]):
        """
        保存归一化参数到文件
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Normalizer":
        """
        从文件加载归一化参数
        
        Args:
            path: 文件路径
            
        Returns:
            Normalizer实例
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_normalization_stats_from_episodes_stats(
    episodes_stats_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从episodes_stats.jsonl文件计算全局归一化统计信息
    
    Args:
        episodes_stats_path: episodes_stats.jsonl文件路径
        
    Returns:
        (action_min, action_max, state_min, state_max): 
            action和state的全局最小值和最大值
    """
    episodes_stats_path = Path(episodes_stats_path)
    
    if not episodes_stats_path.exists():
        raise FileNotFoundError(f"无法找到episodes_stats.jsonl文件: {episodes_stats_path}")
    
    action_mins = []
    action_maxs = []
    state_mins = []
    state_maxs = []
    
    print(f"正在从 {episodes_stats_path} 读取归一化统计信息...")
    
    with open(episodes_stats_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                episode_stats = json.loads(line.strip())
                stats = episode_stats.get("stats", {})
                
                # 读取action的min和max
                if "action" in stats:
                    action_stats = stats["action"]
                    if "min" in action_stats and "max" in action_stats:
                        action_mins.append(np.array(action_stats["min"]))
                        action_maxs.append(np.array(action_stats["max"]))
                
                # 读取observation.state的min和max
                if "observation.state" in stats:
                    state_stats = stats["observation.state"]
                    if "min" in state_stats and "max" in state_stats:
                        state_mins.append(np.array(state_stats["min"]))
                        state_maxs.append(np.array(state_stats["max"]))
            
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行处理失败: {e}")
                continue
    
    if not action_mins:
        raise ValueError("未找到action统计信息")
    
    # 计算全局最小值和最大值
    action_min = np.minimum.reduce(action_mins)  # 每个维度取所有episode的最小值
    action_max = np.maximum.reduce(action_maxs)  # 每个维度取所有episode的最大值
    
    state_min = None
    state_max = None
    if state_mins:
        state_min = np.minimum.reduce(state_mins)
        state_max = np.maximum.reduce(state_maxs)
    
    print(f"  找到 {len(action_mins)} 个episode的action统计信息")
    if state_mins:
        print(f"  找到 {len(state_mins)} 个episode的state统计信息")
    print(f"  Action维度: {len(action_min)}")
    if state_min is not None:
        print(f"  State维度: {len(state_min)}")
    
    return action_min, action_max, state_min, state_max


def create_normalizer_from_dataset(
    dataset_path: Union[str, Path],
    episodes_stats_filename: str = "episodes_stats.jsonl"
) -> Normalizer:
    """
    从数据集创建归一化器
    
    Args:
        dataset_path: 数据集路径
        episodes_stats_filename: episodes_stats文件名（默认为episodes_stats.jsonl）
        
    Returns:
        Normalizer实例
    """
    dataset_path = Path(dataset_path)
    episodes_stats_path = dataset_path / "meta" / episodes_stats_filename
    
    action_min, action_max, state_min, state_max = compute_normalization_stats_from_episodes_stats(
        episodes_stats_path
    )
    
    return Normalizer(
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max
    )
