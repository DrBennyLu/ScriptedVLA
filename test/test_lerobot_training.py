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
LeRobot数据集训练测试脚本
基于lerobot开源数据集（如libero_object）进行训练测试
使用lerobot 0.3.3版本，支持v2.1格式数据集
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

from ScriptedVLA.model.vla_qwen_groot import QwenGR00TVLAModel
from ScriptedVLA.utils import load_config, get_model_config, get_training_config, get_data_config
from train import create_optimizer, create_scheduler, save_checkpoint

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: lerobot library not installed. Install with: pip install lerobot==0.3.3")
    LeRobotDataset = None


def load_dataset_info(dataset_path: Path) -> dict:
    """
    从数据集meta/info.json中加载信息
    
    Args:
        dataset_path: 数据集根目录路径
        
    Returns:
        info字典
    """
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"无法找到info.json文件: {info_file}")
    
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    return info


def create_delta_timestamps(action_horizon: int, fps: int) -> dict:
    """
    根据action_horizon和fps创建delta_timestamps
    
    Args:
        action_horizon: 动作序列长度（chunk大小）
        fps: 帧率（每秒帧数）
        
    Returns:
        delta_timestamps字典，格式: {"action": [0.1, 0.2, ..., time_window]}
    """
    return {"action": [t / fps for t in range(action_horizon)]}


def load_tasks_from_jsonl(dataset_path: Path) -> dict:
    """
    从meta/tasks.jsonl中加载任务描述（备选方案）
    
    Args:
        dataset_path: 数据集根目录路径
        
    Returns:
        tasks字典，key为task_index，value为任务描述
    """
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        return {}
    
    tasks = {}
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            task_data = json.loads(line.strip())
            # tasks.jsonl每行包含task_index和task描述
            # 使用task_index作为key（如果存在），否则使用行号
            if "task_index" in task_data:
                task_idx = task_data["task_index"]
            else:
                # 如果没有task_index，使用当前字典大小作为索引
                task_idx = len(tasks)
            
            # 获取任务描述
            if "task" in task_data:
                task_desc = task_data["task"]
            elif "description" in task_data:
                task_desc = task_data["description"]
            elif "instruction" in task_data:
                task_desc = task_data["instruction"]
            else:
                # 如果没有找到标准键，使用第一个非元数据键
                non_meta_keys = [k for k in task_data.keys() if k != "task_index"]
                task_desc = str(task_data[non_meta_keys[0]]) if non_meta_keys else ""
            
            tasks[task_idx] = task_desc
    
    return tasks


def get_image_keys_from_info(info: dict) -> list:
    """
    从info.json的features.observation.images下获取图像键名
    
    Args:
        info: info.json字典
        
    Returns:
        图像键名列表，例如 ["observation.images.image", "observation.images.wrist_image"]
    """
    image_keys = []
    if "features" in info:
        features = info["features"]
        for key in features.keys():
            if key.startswith("observation.images."):
                image_keys.append(key)
    
    return sorted(image_keys)  # 排序以确保一致性


def get_state_key_from_info(info: dict) -> str:
    """
    从info.json的features.observation.state下获取状态键名
    
    Args:
        info: info.json字典
        
    Returns:
        状态键名，例如 "observation.state"
    """
    if "features" in info and "observation" in info["features"]:
        obs_features = info["features"]["observation"]
        if "state" in obs_features:
            return "observation.state"
    
    return "observation.state"  # 默认值


def get_state_dim_from_info(info: dict, default_state_dim: int = 7) -> int:
    """
    从info.json的features.observation.state下获取状态维度
    
    Args:
        info: info.json字典
        default_state_dim: 默认状态维度（从配置文件读取），如果无法从info.json获取则返回此值
        
    Returns:
        状态维度，例如 8
    """
    if "features" in info:
        features = info["features"]
        # 查找observation.state
        if "observation.state" in features:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                return int(state_shape[0])
        # 或者从observation.state查找
        elif "observation" in features and "state" in features["observation"]:
            obs_features = features["observation"]
            state_shape = obs_features["state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                return int(state_shape[0])
    
    return default_state_dim  # 返回配置中的默认值


def create_collate_fn(image_keys=None, state_key=None, tasks_dict=None, image_size=None, use_batch_task=True):
    """
    创建collate函数，处理lerobot返回的batch格式
    
    Args:
        image_keys: 图像键名列表（从info.json获取）
        state_key: 状态键名（从info.json获取）
        tasks_dict: 从tasks.jsonl加载的任务描述字典（备选方案）
        image_size: 图像尺寸（可选），如果提供，会调整图像大小
        use_batch_task: 是否优先使用batch["task"]获取任务描述
    
    Returns:
        collate函数
    """
    def collate_fn(batch_list):
        """
        自定义collate函数，处理lerobot返回的batch格式
        
        Args:
            batch_list: 样本列表，每个样本是字典
        
        Returns:
            转换后的batch字典，包含模型输入格式
        """
        # 将样本列表转换为batch字典
        from torch.utils.data._utils.collate import default_collate
        batch_dict = default_collate(batch_list)
        batch_size = len(batch_list)
        
        # 获取图像键名（优先使用传入的image_keys，否则从batch中查找）
        if image_keys:
            available_image_keys = [k for k in image_keys if k in batch_dict]
            if not available_image_keys:
                available_image_keys = [k for k in batch_dict.keys() if 'observation.images' in k.lower()]
        else:
            available_image_keys = [k for k in batch_dict.keys() if 'observation.images' in k.lower()]
        
        if not available_image_keys:
            raise ValueError(f"无法找到图像数据，可用键: {list(batch_dict.keys())}")
        
        # 处理图像：将tensor转换为PIL Image列表
        images_list = []
        if len(available_image_keys) == 1:
            # 单相机：List[PIL.Image]
            images_tensor = batch_dict[available_image_keys[0]]  # [B, C, H, W]
            for i in range(batch_size):
                img_tensor = images_tensor[i]  # [C, H, W]
                img_tensor = img_tensor.permute(1, 2, 0)  # [H, W, C]
                img_array = img_tensor.cpu().numpy()
                
                # 转换为uint8
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                # 如果是单通道，转换为RGB
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                
                img_pil = Image.fromarray(img_array)
                # 调整图像大小（如果需要）
                if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
                    img_pil = img_pil.resize((image_size, image_size))
                images_list.append(img_pil)
        else:
            # 多相机：List[List[PIL.Image]]，按键名排序
            for i in range(batch_size):
                camera_images = []
                for key in sorted(available_image_keys):
                    img_tensor = batch_dict[key][i]  # [C, H, W]
                    img_tensor = img_tensor.permute(1, 2, 0)  # [H, W, C]
                    img_array = img_tensor.cpu().numpy()
                    
                    if img_array.dtype != np.uint8:
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape[2] == 1:
                        img_array = np.repeat(img_array, 3, axis=2)
                    
                    img_pil = Image.fromarray(img_array)
                    # 调整图像大小（如果需要）
                    if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
                        img_pil = img_pil.resize((image_size, image_size))
                    camera_images.append(img_pil)
                images_list.append(camera_images)
        
        # 处理动作：batch["action"] 已经是 [B, action_horizon, action_dim] 格式（lerobot自动创建chunk）
        actions = batch_dict["action"]  # [B, action_horizon, action_dim]
        
        # 处理状态（如果存在）
        states = None
        state_key_to_use = state_key if state_key and state_key in batch_dict else "observation.state"
        if state_key_to_use in batch_dict:
            states = batch_dict[state_key_to_use]  # [B, state_dim]
        
        # 处理文本指令（任务描述）
        texts = []
        if use_batch_task and "task" in batch_dict:
            task_data = batch_dict["task"]
            if isinstance(task_data, torch.Tensor):
                if task_data.dtype == torch.int64:
                    # 是task_index，从tasks_dict中获取描述
                    if tasks_dict:
                        texts = [tasks_dict.get(int(task_data[i].item()), "") for i in range(batch_size)]
                    else:
                        texts = [""] * batch_size
                else:
                    texts = [str(task_data[i].item()) for i in range(batch_size)]
            elif isinstance(task_data, list):
                texts = [str(t) if isinstance(t, str) else (tasks_dict.get(int(t), "") if tasks_dict and isinstance(t, (int, np.integer)) else str(t)) for t in task_data]
            else:
                texts = [str(task_data)] * batch_size if not isinstance(task_data, (list, torch.Tensor)) else [""] * batch_size
        else:
            texts = [""] * batch_size
        
        result = {
            "images": images_list,
            "text": texts,
            "action": actions,
        }
        
        if states is not None:
            result["state"] = states
        
        return result
    
    return collate_fn


def train_with_lerobot_dataset(config_path: str = "config.yaml"):
    """
    使用LeRobot数据集进行训练测试
    所有配置从config.yaml的lerobot_test部分读取
    
    Args:
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("LeRobot数据集训练测试")
    print("=" * 60)
    
    if not HAS_LEROBOT:
        raise ImportError(
            "lerobot library not installed. "
            "Install with: pip install lerobot==0.3.3"
        )
    
    # 1. 加载配置
    print(f"\n步骤1: 加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 获取lerobot_test配置
    lerobot_test_config = config.get("lerobot_test", {})
    if not lerobot_test_config:
        raise ValueError("config.yaml中缺少lerobot_test配置部分")
    
    dataset_config = lerobot_test_config.get("dataset", {})
    training_config_lerobot = lerobot_test_config.get("training", {})
    dataloader_config = lerobot_test_config.get("dataloader", {})
    task_description_config = lerobot_test_config.get("task_description", {})
    
    # 从配置中获取参数
    repo_id = dataset_config.get("repo_id", "k1000dai/libero-object-smolvla")
    root = dataset_config.get("root", None)
    local_path = dataset_config.get("local_path", None)
    action_horizon = dataset_config.get("action_horizon", 50)
    image_size = dataset_config.get("image_size", 224)
    
    max_steps = training_config_lerobot.get("max_steps", 100)
    batch_size = training_config_lerobot.get("batch_size", 2)
    
    use_batch_task = task_description_config.get("use_batch_task", True)
    use_tasks_jsonl = task_description_config.get("use_tasks_jsonl", True)
    
    # 获取模型和训练配置（用于模型初始化和优化器）
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    
    # 获取数据配置（用于获取默认的state_dim）
    data_config = get_data_config(config)
    default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
    
    # 合并训练配置（lerobot_test的training配置优先）
    for key, value in training_config_lerobot.items():
        if value is not None:
            training_config[key] = value
    
    # 2. 加载LeRobot数据集
    print(f"\n步骤2: 加载LeRobot数据集")
    print(f"  Repo ID: {repo_id}")
    if root:
        print(f"  Root: {root}")
    if local_path:
        print(f"  Local Path: {local_path}")
    
    # 2.1 确定数据集路径并读取info.json
    dataset_path = None
    dataset_info = None
    
    if local_path:
        dataset_path = Path(local_path).resolve()
        if not dataset_path.exists():
            raise ValueError(f"本地数据集路径不存在: {dataset_path}")
        print(f"  使用本地数据集路径: {dataset_path}")
        dataset_info = load_dataset_info(dataset_path)
    elif root:
        dataset_path = Path(root) / repo_id if repo_id else Path(root)
        if (dataset_path / "meta" / "info.json").exists():
            print(f"  使用本地数据集路径: {dataset_path}")
            dataset_info = load_dataset_info(dataset_path)
        else:
            print(f"  警告: 本地路径不存在info.json，将尝试从HF加载")
    else:
        # 使用HF数据集，尝试从缓存或下载
        print(f"  使用HuggingFace数据集: {repo_id}")
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        potential_paths = [
            cache_dir / repo_id.replace("/", "___"),
            Path("./cache") / "datasets" / repo_id.replace("/", "___"),
        ]
        for cache_path in potential_paths:
            if (cache_path / "meta" / "info.json").exists():
                dataset_path = cache_path
                dataset_info = load_dataset_info(dataset_path)
                print(f"  从缓存加载数据集信息: {dataset_path}")
                break
    
    # 2.2 从info.json获取信息
    fps = 10  # 默认值
    image_keys = None
    state_key = None
    state_dim = None
    
    if dataset_info:
        fps = dataset_info.get("fps", 10)
        print(f"  数据集FPS: {fps}")
        
        image_keys = get_image_keys_from_info(dataset_info)
        print(f"  图像键名（从info.json）: {image_keys}")
        
        state_key = get_state_key_from_info(dataset_info)
        print(f"  状态键名（从info.json）: {state_key}")
        
        state_dim = get_state_dim_from_info(dataset_info, default_state_dim=default_state_dim)
        print(f"  状态维度（从info.json）: {state_dim} (默认值: {default_state_dim})")
    else:
        print(f"  警告: 无法从本地获取info.json，将在加载数据集后获取信息")
    
    # 2.3 创建delta_timestamps
    print(f"  创建delta_timestamps (action_horizon={action_horizon}, fps={fps})...")
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    time_window = action_horizon / fps
    print(f"  时间窗口: {time_window}秒")
    print(f"  Delta timestamps (前5个): {delta_timestamps['action'][:5]}...")
    print(f"  Delta timestamps (后5个): {delta_timestamps['action'][-5:]}...")
    
    # 2.4 加载任务描述（备选方案）
    tasks_dict = {}
    if use_tasks_jsonl and dataset_path and (dataset_path / "meta" / "tasks.jsonl").exists():
        print(f"  加载任务描述（备选方案）...")
        tasks_dict = load_tasks_from_jsonl(dataset_path)
        if tasks_dict:
            print(f"  加载了{len(tasks_dict)}个任务描述")
    
    # 2.5 创建LeRobotDataset
    print(f"  创建LeRobotDataset...")
    try:
        if local_path:
            dataset_dir = Path(local_path).resolve()
            info_file = dataset_dir / "meta" / "info.json"
            
            if not info_file.exists():
                raise ValueError(f"本地数据集路径不存在或无效: {info_file}")
            
            dataset_name = dataset_dir.name
            root_path_str = str(dataset_dir)
            
            print(f"  本地数据集路径: {dataset_dir}")
            print(f"  数据集名称 (repo_id): {dataset_name}")
            print(f"  Root路径: {root_path_str}")
            
            lerobot_dataset = LeRobotDataset(
                repo_id=dataset_name,
                root=root_path_str,
                delta_timestamps=delta_timestamps
            )
            print(f"  ✓ 使用本地数据集创建成功: repo_id={dataset_name}, root={root_path_str}")
        elif root and repo_id:
            root_path = Path(root).resolve()
            dataset_full_path = root_path / repo_id
            info_file = dataset_full_path / "meta" / "info.json"
            
            if not info_file.exists():
                raise ValueError(f"本地数据集路径不存在: {info_file}")
            
            lerobot_dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(dataset_full_path),
                delta_timestamps=delta_timestamps
            )
            print(f"  ✓ 使用本地数据集创建成功: repo_id={repo_id}, root={dataset_full_path}")
        else:
            print(f"  尝试从HuggingFace加载数据集: {repo_id}")
            lerobot_dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)
            print(f"  ✓ 使用HuggingFace数据集创建成功: repo_id={repo_id}")
    except Exception as e:
        print(f"  ✗ 创建LeRobotDataset失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"  数据集长度: {len(lerobot_dataset)}")
    
    # 2.6 检查第一个batch以确认格式
    try:
        test_loader = DataLoader(lerobot_dataset, batch_size=1, shuffle=False)
        test_batch = next(iter(test_loader))
        print(f"  第一个batch的键: {list(test_batch.keys())[:10]}...")
        
        # 如果之前没有从info.json获取，现在从batch中获取
        if not image_keys:
            image_keys = [k for k in test_batch.keys() if 'observation.images' in k.lower()]
            print(f"  从batch中获取图像键名: {image_keys}")
        
        if not state_key:
            state_keys_candidates = [k for k in test_batch.keys() if 'observation.state' in k.lower() or ('state' in k.lower() and 'observation' in k.lower())]
            state_key = state_keys_candidates[0] if state_keys_candidates else "observation.state"
            print(f"  从batch中获取状态键名: {state_key}")
        
        # 从batch中获取状态维度（如果之前没有获取）
        if not state_dim and state_key in test_batch:
            state_shape = test_batch[state_key].shape
            if len(state_shape) >= 2:
                state_dim = int(state_shape[-1])  # 获取最后一维作为state_dim
                print(f"  从batch中获取状态维度: {state_dim}")
        
        # 检查action格式
        if "action" in test_batch:
            action_shape = test_batch["action"].shape
            print(f"    action形状: {action_shape}")
            if len(action_shape) == 3:
                actual_horizon = action_shape[1]
                if actual_horizon != action_horizon:
                    print(f"    警告: 期望action_horizon={action_horizon}，但实际为{actual_horizon}，使用实际值")
                    action_horizon = actual_horizon
    except Exception as e:
        print(f"  警告: 无法检查第一个batch: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 创建数据加载器
    print(f"\n步骤3: 创建数据加载器")
    print(f"  Batch size: {batch_size}")
    print(f"  最大步数: {max_steps}")
    
    # 限制数据集大小以确保不超过max_steps
    max_samples = max_steps * batch_size
    if len(lerobot_dataset) > max_samples:
        from torch.utils.data import Subset
        indices = list(range(min(max_samples, len(lerobot_dataset))))
        lerobot_dataset = Subset(lerobot_dataset, indices)
        print(f"  限制数据集大小为: {len(lerobot_dataset)} 样本")
    
    # 创建自定义collate_fn
    custom_collate_fn = create_collate_fn(
        image_keys=image_keys,
        state_key=state_key,
        tasks_dict=tasks_dict,
        image_size=image_size,
        use_batch_task=use_batch_task
    )
    
    train_loader = DataLoader(
        lerobot_dataset,
        batch_size=batch_size,
        shuffle=dataloader_config.get("shuffle", True),
        num_workers=dataloader_config.get("num_workers", 0),
        pin_memory=dataloader_config.get("pin_memory", False),
        collate_fn=custom_collate_fn
    )
    
    print(f"  数据加载器长度: {len(train_loader)} batches")
    print(f"  预计训练步数: {min(len(train_loader), max_steps)}")
    
    # 4. 创建模型
    print(f"\n步骤4: 创建模型")
    
    vlm_config = model_config.get("vlm", {})
    action_head_config = model_config.get("action_head", {})
    vla_config = model_config.get("vla", {})
    
    action_head_config = action_head_config.copy()
    action_head_config["action_horizon"] = action_horizon
    
    vla_config = vla_config.copy()
    vla_config["future_action_window_size"] = action_horizon - 1
    
    # 使用从数据集获取的state_dim，如果没有则使用配置中的值
    if state_dim is None:
        state_dim = vla_config.get("state_dim", 7)
        print(f"  使用配置中的状态维度: {state_dim}")
    else:
        print(f"  使用数据集的状态维度: {state_dim}")
    
    model = QwenGR00TVLAModel(
        vlm_config=vlm_config,
        action_head_config=action_head_config,
        use_state=vla_config.get("use_state", True),
        state_dim=state_dim,  # 使用从数据集获取的实际状态维度
        future_action_window_size=action_horizon - 1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  模型已移动到设备: {device}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  State dimension: {state_dim}")
    
    # 5. 创建优化器和调度器
    print(f"\n步骤5: 创建优化器和调度器")
    
    merged_training_config = {
        **training_config,
        "batch_size": batch_size,
        "num_epochs": 1
    }
    
    optimizer = create_optimizer(model, merged_training_config)
    num_training_steps = min(len(train_loader), max_steps)
    scheduler = create_scheduler(optimizer, merged_training_config, num_training_steps)
    print(f"  优化器: {type(optimizer).__name__}")
    print(f"  调度器: {type(scheduler).__name__ if scheduler else 'None'}")
    print(f"  训练步数: {num_training_steps}")
    
    # 6. 训练循环
    print(f"\n步骤6: 开始训练（运行{num_training_steps}步）")
    model.train()
    
    total_loss = 0.0
    losses = []
    
    progress_bar = tqdm(enumerate(train_loader), total=num_training_steps, desc="Training")
    
    for step, batch in progress_bar:
        if step >= max_steps:
            break
        
        # 准备输入
        images = batch["images"]
        texts = batch["text"]
        actions = batch["action"].to(device)
        
        # 验证actions维度
        if actions.ndim != 3:
            raise ValueError(f"意外的actions维度: {actions.shape}, 期望 [B, action_horizon, action_dim]")
        if actions.shape[1] != action_horizon:
            print(f"  警告: Step {step+1}: actions.shape[1]={actions.shape[1]}, 期望action_horizon={action_horizon}")
            if actions.shape[1] < action_horizon:
                last_action = actions[:, -1:, :]
                padding = last_action.repeat(1, action_horizon - actions.shape[1], 1)
                actions = torch.cat([actions, padding], dim=1)
            else:
                actions = actions[:, :action_horizon, :]
        
        # 处理状态（如果存在）
        states = None
        if "state" in batch:
            states = batch["state"].to(device)
        
        # 准备模型输入
        inputs = {
            "images": images,
            "instructions": texts,
            "actions": actions
        }
        if states is not None:
            inputs["states"] = states
        
        # 前向传播
        outputs = model(inputs=inputs)
        
        # 获取损失
        if "action_loss" in outputs:
            loss = outputs["action_loss"]
        elif "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError("模型输出中没有损失值")
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪和优化器步进
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            merged_training_config.get("max_grad_norm", 1.0)
        )
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        # 记录损失
        loss_value = loss.item()
        total_loss += loss_value
        losses.append(loss_value)
        
        # 更新进度条
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({
            "loss": f"{loss_value:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # 每10步打印一次详细损失
        if (step + 1) % 10 == 0:
            print(f"\n  Step {step + 1}/{num_training_steps}: "
                  f"Loss = {loss_value:.4f}, "
                  f"Avg Loss = {avg_loss:.4f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.2e}")
    
    # 7. 打印训练总结
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"总步数: {len(losses)}")
    print(f"平均损失: {sum(losses) / len(losses):.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"最小损失: {min(losses):.4f}")
    print(f"最大损失: {max(losses):.4f}")
    
    # 绘制损失曲线（如果可用）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        save_path = Path("test_lerobot_loss.png")
        plt.savefig(save_path)
        print(f"\n损失曲线已保存: {save_path}")
    except ImportError:
        print("\n提示: 安装matplotlib可以绘制损失曲线: pip install matplotlib")
    
    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VLA Training with LeRobot Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml). All configuration is read from config.yaml's lerobot_test section."
    )
    args = parser.parse_args()
    
    try:
        model, losses = train_with_lerobot_dataset(config_path=args.config)
        print("\n✓ 训练测试成功完成")
        exit(0)
    except Exception as e:
        print(f"\n✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
