# MIT License
#
# Copyright (c) 2026 ScriptedVLA Contributors
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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class Normalizer:
    """
    数据归一化器：按 action/state 的**每个维度**分别使用独立的 min/max，
    不做全局单一标量，公式对每个元素单独计算。
    """

    def __init__(
        self,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None,
        state_min: Optional[np.ndarray] = None,
        state_max: Optional[np.ndarray] = None
    ):
        """
        初始化归一化器（每维独立 min/max）。

        Args:
            action_min: 每个 action 维度的最小值，形状 [action_dim]
            action_max: 每个 action 维度的最大值，形状 [action_dim]
            state_min: 每个 state 维度的最小值，形状 [state_dim]
            state_max: 每个 state 维度的最大值，形状 [state_dim]
        """
        self.action_min = np.atleast_1d(action_min) if action_min is not None else None
        self.action_max = np.atleast_1d(action_max) if action_max is not None else None
        self.state_min = np.atleast_1d(state_min) if state_min is not None else None
        self.state_max = np.atleast_1d(state_max) if state_max is not None else None

        # 每维独立范围，避免除零
        if self.action_min is not None and self.action_max is not None:
            self.action_range = self.action_max - self.action_min
            self.action_range = np.where(self.action_range == 0, 1.0, self.action_range)
        else:
            self.action_range = None

        if self.state_min is not None and self.state_max is not None:
            self.state_range = self.state_max - self.state_min
            self.state_range = np.where(self.state_range == 0, 1.0, self.state_range)
        else:
            self.state_range = None
    
    def normalize_action(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        按维度归一化 action：每个元素用该维的 min/max 映射到 [-1, 1]。

        Args:
            action: 原始 action [..., action_dim]

        Returns:
            归一化后的 action，每维在 [-1, 1]
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

        # 每维独立: (x - min_i) / range_i -> [-1, 1]
        normalized = 2.0 * (action_np - self.action_min) / self.action_range - 1.0
        
        if is_tensor:
            return torch.from_numpy(normalized).to(device=device, dtype=dtype)
        else:
            return normalized
    
    def denormalize_action(self, action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        按维度反归一化 action：每个元素从 [-1, 1] 映射回该维的 [min_i, max_i]。

        Args:
            action: 归一化后的 action [..., action_dim]，每维在 [-1, 1]

        Returns:
            反归一化后的 action，每维在 [action_min, action_max]
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

        # 每维独立: [-1,1] -> [min_i, max_i]，再按维裁剪
        denormalized = (action_np + 1.0) / 2.0 * self.action_range + self.action_min
        denormalized = np.clip(denormalized, self.action_min, self.action_max)
        
        if is_tensor:
            return torch.from_numpy(denormalized).to(device=device, dtype=dtype)
        else:
            return denormalized
    
    def normalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        按维度归一化 state：每个元素用该维的 min/max 映射到 [-1, 1]。

        Args:
            state: 原始 state [..., state_dim]

        Returns:
            归一化后的 state，每维在 [-1, 1]
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

        # 每维独立: (x - min_i) / range_i -> [-1, 1]
        normalized = 2.0 * (state_np - self.state_min) / self.state_range - 1.0
        
        if is_tensor:
            return torch.from_numpy(normalized).to(device=device, dtype=dtype)
        else:
            return normalized
    
    def denormalize_state(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        按维度反归一化 state：每个元素从 [-1, 1] 映射回该维的 [min_i, max_i]。

        Args:
            state: 归一化后的 state [..., state_dim]，每维在 [-1, 1]

        Returns:
            反归一化后的 state，每维在 [state_min, state_max]
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

        # 每维独立: [-1,1] -> [min_i, max_i]
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
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 episodes_stats.jsonl 按**每个维度**聚合 min/max（不做全局单一标量）。

    Args:
        episodes_stats_path: episodes_stats.jsonl 文件路径

    Returns:
        (action_min, action_max, state_min, state_max):
            形状 [action_dim] / [state_dim]，每个元素对应该维度的 min/max。
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

    # 按维度聚合：每个维度取所有 episode 的 min/max，得到形状 [action_dim] / [state_dim]
    action_min = np.minimum.reduce([np.asarray(m).flatten() for m in action_mins])
    action_max = np.maximum.reduce([np.asarray(m).flatten() for m in action_maxs])
    state_min = np.minimum.reduce([np.asarray(m).flatten() for m in state_mins]) if state_mins else None
    state_max = np.maximum.reduce([np.asarray(m).flatten() for m in state_maxs]) if state_maxs else None

    print(f"  找到 {len(action_mins)} 个 episode 的 action 统计（按维度独立 min/max）")
    if state_mins:
        print(f"  找到 {len(state_mins)} 个 episode 的 state 统计（按维度独立 min/max）")
    print(f"  Action 维度: {len(action_min)}")
    if state_min is not None:
        print(f"  State 维度: {len(state_min)}")

    return action_min, action_max, state_min, state_max


def compute_normalization_stats_from_episodes_stats_items(
    episodes_stats_items: Iterable[Tuple[int, Dict]],
    state_key: str = "observation.state",
    action_key: str = "action",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 LeRobot meta.episodes_stats.items() 按**每个维度**聚合 min/max（不做全局单一标量）。

    用于 state 归一化（传入动作头前）和 action 反归一化（模型输出后），
    每个 action/state 维度使用独立的 min_i / max_i。

    Args:
        episodes_stats_items: lerobot_dataset.meta.episodes_stats.items()，
            即 (episode_index, stats_dict) 的迭代器。
        state_key: state 在 stats 中的键名。
        action_key: action 在 stats 中的键名。

    Returns:
        (action_min, action_max, state_min, state_max):
            形状 [action_dim] / [state_dim]，每个元素为该维度的 min/max。
    """
    action_mins: List[np.ndarray] = []
    action_maxs: List[np.ndarray] = []
    state_mins: List[np.ndarray] = []
    state_maxs: List[np.ndarray] = []

    for _ep_idx, stats in episodes_stats_items:
        if action_key in stats:
            a = stats[action_key]
            if "min" in a and "max" in a:
                action_mins.append(np.asarray(a["min"]).flatten())
                action_maxs.append(np.asarray(a["max"]).flatten())
        if state_key in stats:
            s = stats[state_key]
            if "min" in s and "max" in s:
                state_mins.append(np.asarray(s["min"]).flatten())
                state_maxs.append(np.asarray(s["max"]).flatten())

    if not action_mins:
        raise ValueError("episodes_stats 中未找到 action 统计信息")

    # 按维度聚合，得到每维一个 min/max
    action_min = np.minimum.reduce(action_mins)
    action_max = np.maximum.reduce(action_maxs)
    state_min = np.minimum.reduce(state_mins) if state_mins else None
    state_max = np.maximum.reduce(state_maxs) if state_maxs else None

    return action_min, action_max, state_min, state_max


def create_normalizer_from_lerobot_meta(
    lerobot_dataset: Any,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> Normalizer:
    """
    从已加载的 LeRobot 数据集的 meta.episodes_stats 创建归一化器。

    用于：仅对传入动作头的 state 做归一化；对模型输出的 action 做反归一化。
    LeRobot 中读取的 action 通常已是 [-1, 1]，训练时无需再归一化。

    Args:
        lerobot_dataset: 已加载的 LeRobotDataset 实例（需有 meta.episodes_stats）。
        state_key: state 在 episodes_stats 中的键名。
        action_key: action 在 episodes_stats 中的键名。

    Returns:
        Normalizer 实例（含 state 与 action 的 min/max，用于 state 归一化与 action 反归一化）。
    """
    try:
        items = lerobot_dataset.meta.episodes_stats.items()
    except AttributeError:
        raise AttributeError(
            "lerobot_dataset.meta 没有 episodes_stats；请使用支持 meta.episodes_stats 的 LeRobot 数据集"
        ) from None
    action_min, action_max, state_min, state_max = compute_normalization_stats_from_episodes_stats_items(
        items, state_key=state_key, action_key=action_key
    )
    return Normalizer(
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
    )


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
