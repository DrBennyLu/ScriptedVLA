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
VLA模型训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import json
import numpy as np
import random
from PIL import Image

from src.ScriptedVLA.model import QwenGR00TVLAModel
from src.ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    create_normalizer_from_dataset,
    Normalizer
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None


def set_seed(seed: int):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch的确定性模式（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset_info(dataset_path: Path) -> dict:
    """从数据集meta/info.json中加载信息"""
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"无法找到info.json文件: {info_file}")
    
    with open(info_file, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    return info


def create_delta_timestamps(action_horizon: int, fps: int) -> dict:
    """根据action_horizon和fps创建delta_timestamps"""
    return {"action": [t / fps for t in range(action_horizon)]}


def load_tasks_from_jsonl(dataset_path: Path) -> dict:
    """从meta/tasks.jsonl中加载任务描述（备选方案）"""
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        return {}
    
    tasks = {}
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            task_data = json.loads(line.strip())
            if "task_index" in task_data:
                task_idx = task_data["task_index"]
            else:
                task_idx = len(tasks)
            
            if "task" in task_data:
                task_desc = task_data["task"]
            elif "description" in task_data:
                task_desc = task_data["description"]
            elif "instruction" in task_data:
                task_desc = task_data["instruction"]
            else:
                non_meta_keys = [k for k in task_data.keys() if k != "task_index"]
                task_desc = str(task_data[non_meta_keys[0]]) if non_meta_keys else ""
            
            tasks[task_idx] = task_desc
    
    return tasks


def get_image_keys_from_info(info: dict) -> list:
    """
    从info.json的features.observation.images下获取图像键名
    
    注意：此函数已弃用，推荐从config.yaml的dataset.image_keys配置中直接读取。
    保留此函数仅用于向后兼容和测试目的。
    """
    image_keys = []
    if "features" in info:
        features = info["features"]
        for key in features.keys():
            if key.startswith("observation.images."):
                image_keys.append(key)
    
    return sorted(image_keys)


def get_state_key_from_info(info: dict) -> str:
    """
    从info.json的features.observation.state下获取状态键名
    
    注意：此函数已弃用，推荐从config.yaml的dataset.state_key配置中直接读取。
    保留此函数仅用于向后兼容和测试目的。
    """
    if "features" in info and "observation" in info["features"]:
        obs_features = info["features"]["observation"]
        if "state" in obs_features:
            return "observation.state"
    
    return "observation.state"


def get_state_dim_from_info(info: dict, default_state_dim: int = 7) -> int:
    """
    从info.json的features.observation.state下获取状态维度
    
    注意：此函数已弃用，推荐从config.yaml的data.robot_state.state_dim配置中直接读取。
    保留此函数仅用于向后兼容和测试目的。
    """
    if "features" in info:
        features = info["features"]
        if "observation.state" in features:
            state_shape = features["observation.state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                return int(state_shape[0])
        elif "observation" in features and "state" in features["observation"]:
            obs_features = features["observation"]
            state_shape = obs_features["state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                return int(state_shape[0])
    
    return default_state_dim


def _tensor_to_pil_image(img_tensor, image_size=None):
    """
    将tensor转换为PIL.Image
    LeRobot数据集返回的图像已经是归一化到0-1的tensor，需要转换为0-255的PIL.Image
    """
    # 确保tensor格式为 [C, H, W]
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
    elif img_tensor.dim() != 3:
        raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}, expected [C, H, W]")
    
    # 转换为numpy数组 [H, W, C]
    img_tensor = img_tensor.permute(1, 2, 0)
    img_array = img_tensor.cpu().numpy()
    
    # LeRobot数据集返回的图像已经是0-1归一化的，需要转换为0-255
    # 检查是否已经是0-1范围（lerobot数据集通常返回0-1的float tensor）
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0 and img_array.min() >= 0.0:
            # 0-1归一化的图像，转换为0-255
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.max() <= 255.0:
            # 已经是0-255范围，直接转换类型
            img_array = img_array.astype(np.uint8)
        else:
            # 其他情况，先clamp到0-255再转换
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # 确保是RGB格式
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=2)
    
    img_pil = Image.fromarray(img_array, mode='RGB')
    if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
        img_pil = img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img_pil


def create_collate_fn(image_keys, state_key, image_size=None, use_batch_task=True, normalizer=None):
    """
    创建collate函数，处理lerobot返回的batch格式
    
    Args:
        image_keys: 图像键名列表（从config.yaml读取，例如：["observation.images.wrist_image"]）
        state_key: 状态键名（从config.yaml读取，例如："observation.state"）
        image_size: 图像尺寸（可选）
        use_batch_task: 是否使用batch中的task字段
        normalizer: 归一化器（可选）
    """
    def collate_fn(batch_list):
        from torch.utils.data._utils.collate import default_collate
        batch_dict = default_collate(batch_list)
        batch_size = len(batch_list)
        
        # 验证图像键是否存在
        missing_keys = [k for k in image_keys if k not in batch_dict]
        if missing_keys:
            raise ValueError(f"配置的图像键不存在于batch中: {missing_keys}, 可用键: {list(batch_dict.keys())}")
        
        # 处理图像：根据config.yaml中的image_keys数量判断单相机/多相机
        # 单相机（len(image_keys)==1）：List[PIL.Image]
        # 多相机（len(image_keys)>1）：List[List[PIL.Image]]
        images_list = []
        for i in range(batch_size):
            if len(image_keys) == 1:
                img_tensor = batch_dict[image_keys[0]][i]
                images_list.append(_tensor_to_pil_image(img_tensor, image_size))
            else:
                camera_images = [_tensor_to_pil_image(batch_dict[key][i], image_size) 
                                for key in image_keys]
                images_list.append(camera_images)
        
        # 处理actions：归一化
        actions = batch_dict["action"]
        if normalizer is not None:
            actions = normalizer.normalize_action(actions)
        
        # 处理states：归一化（如果存在）
        states = None
        if state_key in batch_dict:
            states = batch_dict[state_key]
            if normalizer is not None:
                states = normalizer.normalize_state(states)
        
        # 处理文本任务描述
        # lerobot数据集中的task字段直接返回字符串列表
        if use_batch_task and "task" in batch_dict:
            task_data = batch_dict["task"]
            # task字段是字符串列表，直接使用
            texts = [str(t) for t in task_data] if isinstance(task_data, list) else [str(task_data)] * batch_size
        else:
            texts = [""] * batch_size
        
        # 构建结果
        result = {
            "images": images_list,
            "text": texts,
            "action": actions,
        }
        if states is not None:
            result["state"] = states
        
        return result
    
    return collate_fn


def create_optimizer(model, config):
    """创建优化器"""
    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("type", "adamw")
    
    # 确保学习率和权重衰减是数值类型
    learning_rate = config.get("learning_rate", 1e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    weight_decay = config.get("weight_decay", 0.01)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    if opt_type.lower() == "adamw":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=opt_config.get("betas", [0.9, 0.999]),
            eps=opt_config.get("eps", 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    sched_config = config.get("scheduler", {})
    sched_type = sched_config.get("type", "cosine")
    
    # 确保数值类型正确
    warmup_ratio = sched_config.get("warmup_ratio", 0.1)
    if isinstance(warmup_ratio, str):
        warmup_ratio = float(warmup_ratio)
    
    min_lr_ratio = sched_config.get("min_lr_ratio", 0.01)
    if isinstance(min_lr_ratio, str):
        min_lr_ratio = float(min_lr_ratio)
    
    learning_rate = config.get("learning_rate", 1e-4)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    if sched_type == "cosine":
        # Warmup + Cosine
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=min_lr_ratio,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=learning_rate * min_lr_ratio
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        return None


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    config,
    logger,
    epoch,
    start_step=0
):
    """
    训练一个epoch
    
    Args:
        start_step: 起始全局步数
        
    Returns:
        (avg_loss, last_step): 平均损失和最后一个步数
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    current_step = start_step
    
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}",
        unit="batch",
        leave=True
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # create_collate_fn已经返回处理好的PIL.Image列表格式
        # 单相机: List[PIL.Image]，多相机: List[List[PIL.Image]]
        images = batch["images"]
        texts = batch["text"]
        # create_collate_fn已经处理好了，actions格式为 [B, action_horizon, action_dim]
        actions = batch["action"].to(device)
        
        # 处理状态信息（如果存在）
        states = None
        if "state" in batch:
            states = batch["state"].to(device)
        
        # 前向传播（训练模式，提供actions以计算损失）
        # 使用统一输入格式
        inputs = {
            "images": images,
            "instructions": texts,
            "actions": actions
        }
        if states is not None:
            inputs["states"] = states
        outputs = model(inputs=inputs)
        
        # 模型在训练模式下固定返回action_loss
        loss = outputs["action_loss"]
        
        # 梯度累积
        loss = loss / config.get("gradient_accumulation_steps", 1)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪和优化器步进
        if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get("max_grad_norm", 1.0)
            )
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            # 只有在实际优化器步进时才增加步数
            current_step += 1
        
        total_loss += loss.item() * config.get("gradient_accumulation_steps", 1)
        num_batches += 1
        
        # 更新进度条，显示更多信息
        current_avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            "loss": f"{loss.item() * config.get('gradient_accumulation_steps', 1):.4f}",
            "avg_loss": f"{current_avg_loss:.4f}",
            "lr": f"{current_lr:.2e}"
        })
        
        # 日志记录（使用当前步数）
        if current_step % config.get("logging_steps", 100) == 0:
            # 获取批次中的层次化信息（用于日志）
            task_info = ""
            if "task_name" in batch:
                task_names = batch["task_name"]
                if isinstance(task_names, list) and len(task_names) > 0:
                    task_info = f", Task: {task_names[0]}"
                elif isinstance(task_names, str):
                    task_info = f", Task: {task_names}"
            
            logger.info(
                f"Step {current_step}: Loss = {loss.item() * config.get('gradient_accumulation_steps', 1):.4f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}{task_info}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, current_step


def evaluate(model, dataloader, criterion, device, logger, max_eval_batches=50):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数（未使用，保持兼容性）
        device: 设备
        logger: 日志记录器
        max_eval_batches: 最大评估批次数量，用于限制评估时间（默认50）
    """
    model.train()  # 设置为训练模式以计算损失（使用no_grad禁用梯度）
    total_loss = 0.0
    num_batches = 0
    
    # 限制评估批次数量
    total_batches = len(dataloader)
    eval_batches = min(max_eval_batches, total_batches)
    
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader),
            total=eval_batches,
            desc="评估中",
            unit="batch",
            leave=True
        )
        
        for batch_idx, batch in progress_bar:
            # 限制评估批次数量
            if batch_idx >= eval_batches:
                break
            
            # create_collate_fn已经返回处理好的格式
            images = batch["images"]
            texts = batch["text"]
            # create_collate_fn已经处理好了，actions格式为 [B, action_horizon, action_dim]
            actions = batch["action"].to(device)
            
            # 准备模型输入
            inputs = {
                "images": images,
                "instructions": texts,
                "actions": actions
            }
            if "state" in batch:
                inputs["states"] = batch["state"].to(device)
            
            outputs = model(inputs=inputs)
            
            # 模型在训练模式下固定返回action_loss
            loss = outputs["action_loss"]
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            current_avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{current_avg_loss:.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f} (评估了 {num_batches}/{total_batches} 个批次)")
    return avg_loss


def find_latest_checkpoint(checkpoint_dir: Path):
    """
    查找最新的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录路径
        
    Returns:
        (checkpoint_path, step): 最新检查点路径和对应的步数，如果没有找到则返回(None, 0)
    """
    if not checkpoint_dir.exists():
        return None, 0
    
    # 查找所有符合格式的检查点文件: checkpoint_step_*.pt
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    
    if not checkpoint_files:
        return None, 0
    
    # 从文件名中提取步数
    max_step = 0
    latest_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        try:
            # 从文件名中提取步数: checkpoint_step_12345.pt -> 12345
            filename = checkpoint_file.stem  # 去掉扩展名
            step_str = filename.replace("checkpoint_step_", "")
            step = int(step_str)
            
            if step > max_step:
                max_step = step
                latest_checkpoint = checkpoint_file
        except (ValueError, AttributeError):
            # 如果文件名格式不正确，跳过
            continue
    
    return latest_checkpoint, max_step


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, device):
    """
    加载检查点并恢复模型、优化器和调度器状态
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型对象
        optimizer: 优化器对象
        scheduler: 调度器对象（可以为None）
        device: 设备
        
    Returns:
        (start_step, loss, normalizer): 起始步数、损失值和归一化器
    """
    print(f"  加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ 模型状态已加载")
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  ✓ 优化器状态已加载")
    
    # 加载调度器状态（如果存在）
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"  ✓ 调度器状态已加载")
    
    # 加载归一化器（如果存在）
    normalizer = None
    if "normalizer" in checkpoint:
        normalizer = Normalizer.from_dict(checkpoint["normalizer"])
        print(f"  ✓ 归一化器状态已加载")
    
    # 获取步数和损失
    start_step = checkpoint.get("global_step", 0)
    loss = checkpoint.get("loss", 0.0)
    
    print(f"  ✓ 检查点加载完成")
    print(f"    起始步数: {start_step}")
    print(f"    检查点损失: {loss:.4f}")
    
    return start_step, loss, normalizer


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, global_step=None, normalizer=None):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if global_step is not None:
        checkpoint["global_step"] = global_step
    if normalizer is not None:
        checkpoint["normalizer"] = normalizer.to_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def train_with_lerobot_dataset(config_path: str = "config.yaml", dataset_path: str = "./dataset/libero_object", max_steps: int = 20000, save_steps: int = 5000):
    """
    使用LeRobot数据集进行训练
    
    Args:
        config_path: 配置文件路径
        dataset_path: 数据集路径（默认为 ./dataset/libero_object）
        max_steps: 最大训练步数（默认20000）
        save_steps: 保存检查点的间隔步数（默认5000）
    """
    print("=" * 60)
    print("LeRobot数据集训练")
    print("=" * 60)
    
    if not HAS_LEROBOT:
        raise ImportError(
            "lerobot library not installed. "
            "Install with: pip install lerobot==0.3.3"
        )
    
    # 1. 加载配置
    print(f"\n步骤1: 加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 设置随机种子
    seed = config.get("seed", 42)
    print(f"\n步骤0.5: 设置随机种子: {seed}")
    set_seed(seed)
    print(f"  ✓ 随机种子已设置")
    
    # 获取模型和训练配置
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
    
    # 获取数据集配置
    dataset_config = config.get("dataset", {})
    dataloader_config = dataset_config.get("dataloader", {})
    task_description_config = dataset_config.get("task_description", {})
    
    # 从dataset配置中获取数据集相关参数
    # 优先级：命令行参数 > 配置文件 > 默认值
    if dataset_path == "./dataset/libero_object":  # 如果使用的是默认值，从配置文件读取
        local_path = dataset_config.get("local_path", "./dataset/libero_object")
    else:
        local_path = dataset_path  # 使用命令行传入的路径
    
    action_horizon = dataset_config.get("action_horizon", 50)
    # 从model.vlm.image_size读取图像尺寸，而不是dataset.image_size
    vlm_config = model_config.get("vlm", {})
    image_size = vlm_config.get("image_size", 224)
    
    # 从task_description配置中获取参数
    use_batch_task = task_description_config.get("use_batch_task", True)
    
    # 使用config.training中的配置
    merged_training_config = training_config.copy()
    
    # 从config.training中读取batch_size
    batch_size = training_config.get("batch_size", 8)
    
    # 从配置文件中读取训练参数
    # 优先级：命令行参数 > 配置文件 > 默认值
    # 如果函数参数是默认值，说明用户没有通过命令行指定，则从配置文件读取
    if max_steps == 20000:  # 如果使用的是默认值，从配置文件读取
        max_steps = training_config.get("max_steps", 20000)
    
    if save_steps == 5000:  # 如果使用的是默认值，从配置文件读取
        save_steps = merged_training_config.get("save_steps", 5000)
    
    eval_steps = merged_training_config.get("eval_steps", 5000)
    logging_steps = merged_training_config.get("logging_steps", 100)
    
    # 打印训练参数信息
    print(f"\n训练参数（从配置文件读取）:")
    print(f"  save_steps: {save_steps}")
    print(f"  eval_steps: {eval_steps}")
    print(f"  logging_steps: {logging_steps}")
    
    # 2. 加载LeRobot数据集
    print(f"\n步骤2: 加载LeRobot数据集")
    dataset_path_obj = Path(local_path).resolve()
    if not dataset_path_obj.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path_obj}")
    
    print(f"  数据集路径: {dataset_path_obj}")
    
    # 2.5. 从配置文件读取数据集参数
    print(f"\n步骤2.5: 从配置文件读取数据集参数")
    
    # 从配置读取相机和维度配置
    image_keys = dataset_config.get("image_keys", ["observation.images.wrist_image"])
    if not isinstance(image_keys, list):
        raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
    print(f"  图像键名（从配置）: {image_keys}")
    
    state_key = dataset_config.get("state_key", "observation.state")
    print(f"  状态键名（从配置）: {state_key}")
    
    action_dim = dataset_config.get("action_dim", model_config.get("action_head", {}).get("action_dim", 7))
    print(f"  动作维度（从配置）: {action_dim}")
    
    state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
    print(f"  状态维度（从配置）: {state_dim}")
    
    # 从info.json获取fps（仅用于创建delta_timestamps）
    dataset_info = load_dataset_info(dataset_path_obj)
    fps = dataset_info.get("fps", 10)
    print(f"  数据集FPS（从info.json）: {fps}")
    
    # 创建归一化器
    print(f"\n步骤2.6: 创建数据归一化器")
    try:
        normalizer = create_normalizer_from_dataset(dataset_path_obj)
        print(f"  ✓ 归一化器创建成功")
        if normalizer.action_min is not None:
            print(f"  Action范围: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
        if normalizer.state_min is not None:
            print(f"  State范围: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
    except Exception as e:
        print(f"  ✗ 归一化器创建失败: {e}")
        print(f"  警告: 将不使用归一化，训练可能不稳定")
        normalizer = None
    
    # 创建delta_timestamps
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    print(f"  Action horizon: {action_horizon}")
    
    # 创建LeRobotDataset
    print(f"  创建LeRobotDataset...")
    # 验证info.json文件存在
    info_file = dataset_path_obj / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"本地数据集路径不存在或无效: {info_file}")
    
    dataset_name = dataset_path_obj.name
    root_path_str = str(dataset_path_obj)  # 使用数据集目录本身作为root，而不是父目录
    
    print(f"  本地数据集路径: {dataset_path_obj}")
    print(f"  数据集名称 (repo_id): {dataset_name}")
    print(f"  Root路径: {root_path_str}")
    
    try:
        lerobot_dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps
        )
        print(f"  ✓ 使用本地数据集创建成功: repo_id={dataset_name}, root={root_path_str}")
    except Exception as e:
        print(f"  ✗ 创建LeRobotDataset失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 3. 创建数据加载器
    print(f"\n步骤3: 创建数据加载器")
    print(f"  Batch size: {batch_size}")
    print(f"  最大训练步数: {max_steps}")
    
    custom_collate_fn = create_collate_fn(
        image_keys=image_keys,
        state_key=state_key,
        image_size=image_size,
        use_batch_task=use_batch_task,
        normalizer=normalizer
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
    
    # 4. 创建模型
    print(f"\n步骤4: 创建模型")
    
    vlm_config = model_config.get("vlm", {})
    action_head_config = model_config.get("action_head", {}).copy()
    action_head_config["action_horizon"] = action_horizon
    action_head_config["action_dim"] = action_dim  # 使用从配置读取的action_dim
    
    vla_config = model_config.get("vla", {}).copy()
    vla_config["future_action_window_size"] = action_horizon - 1
    
    model = QwenGR00TVLAModel(
        vlm_config=vlm_config,
        action_head_config=action_head_config,
        use_state=vla_config.get("use_state", True),
        state_dim=state_dim,
        future_action_window_size=action_horizon - 1
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  模型已移动到设备: {device}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  Action dimension: {action_dim}")
    print(f"  State dimension: {state_dim}")
    
    # 计算并打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    # 5. 创建优化器和调度器
    print(f"\n步骤5: 创建优化器和调度器")
    
    optimizer = create_optimizer(model, merged_training_config)
    scheduler = create_scheduler(optimizer, merged_training_config, max_steps)
    print(f"  优化器: {type(optimizer).__name__}")
    print(f"  调度器: {type(scheduler).__name__ if scheduler else 'None'}")
    
    # 6. 创建保存目录
    save_dir = Path(merged_training_config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  检查点保存目录: {save_dir}")
    
    # 6.5. 检查是否有可用的检查点并恢复
    print(f"\n步骤6.5: 检查断点续训")
    latest_checkpoint_path, latest_step_from_filename = find_latest_checkpoint(save_dir)
    
    start_step = 0
    if latest_checkpoint_path is not None:
        print(f"  发现检查点: {latest_checkpoint_path}")
        print(f"  文件名中的步数: {latest_step_from_filename}")
        
        # 加载检查点（会返回检查点中保存的实际步数）
        start_step, checkpoint_loss, loaded_normalizer = load_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler, device
        )
        
        # 如果检查点中有归一化器，使用它；否则使用新创建的
        if loaded_normalizer is not None:
            normalizer = loaded_normalizer
            print(f"  使用检查点中的归一化器")
            # 重新创建collate_fn以使用新的normalizer
            custom_collate_fn = create_collate_fn(
                image_keys=image_keys,
                state_key=state_key,
                image_size=image_size,
                use_batch_task=use_batch_task,
                normalizer=normalizer
            )
            # 重新创建DataLoader以使用新的collate_fn
            train_loader = DataLoader(
                lerobot_dataset,
                batch_size=batch_size,
                shuffle=dataloader_config.get("shuffle", True),
                num_workers=dataloader_config.get("num_workers", 0),
                pin_memory=dataloader_config.get("pin_memory", False),
                collate_fn=custom_collate_fn
            )
        
        # 使用检查点中保存的实际步数，而不是文件名中的步数
        # 因为检查点中的步数更准确
        if start_step >= max_steps:
            print(f"  警告: 检查点步数({start_step})已超过或等于最大训练步数({max_steps})")
            print(f"  训练已完成，无需继续训练")
            return model, []
        
        print(f"  将从步数 {start_step} 继续训练到 {max_steps}")
    else:
        print(f"  未找到检查点，将从步数 0 开始训练")
    
    # 7. 训练循环
    remaining_steps = max_steps - start_step
    print(f"\n步骤7: 开始训练（从步数 {start_step} 继续，剩余 {remaining_steps} 步，每{save_steps}步保存一次检查点）")
    model.train()
    
    losses = []
    
    # 创建数据加载器的迭代器，以便循环使用
    loader_iter = iter(train_loader)
    progress_bar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training")
    
    for step in progress_bar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        
        # create_collate_fn已经返回处理好的格式
        images = batch["images"]
        texts = batch["text"]
        # create_collate_fn已经处理好了，actions格式为 [B, action_horizon, action_dim]
        actions = batch["action"].to(device)
        
        # 准备模型输入
        inputs = {
            "images": images,
            "instructions": texts,
            "actions": actions
        }
        if "state" in batch:
            inputs["states"] = batch["state"].to(device)
        
        # 前向传播
        outputs = model(inputs=inputs)
        
        # 模型在训练模式下固定返回action_loss
        loss = outputs["action_loss"]
        
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
        losses.append(loss_value)
        
        # 更新进度条
        avg_loss = sum(losses) / len(losses)
        progress_bar.set_postfix({
            "loss": f"{loss_value:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # 每logging_steps步打印一次详细损失
        if (step + 1) % logging_steps == 0:
            print(f"\n  Step {step + 1}/{max_steps}: "
                  f"Loss = {loss_value:.4f}, "
                  f"Avg Loss = {avg_loss:.4f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存检查点（只在save_steps的倍数时保存，避免重复保存）
        if (step + 1) % save_steps == 0:
            checkpoint_path = save_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, 0, loss_value, checkpoint_path, 
                global_step=step + 1, normalizer=normalizer
            )
    
    # 8. 打印训练总结
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"总步数: {len(losses)}")
    print(f"平均损失: {sum(losses) / len(losses):.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"最小损失: {min(losses):.4f}")
    print(f"最大损失: {max(losses):.4f}")
    
    # 9. 绘制和保存损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(losses, linewidth=1, alpha=0.7)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss Curve", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加平滑曲线
        if len(losses) > 100:
            window_size = min(100, len(losses) // 20)
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(losses), i + window_size // 2 + 1)
                smoothed_losses.append(sum(losses[start_idx:end_idx]) / (end_idx - start_idx))
            plt.plot(smoothed_losses, linewidth=2, label='Smoothed', alpha=0.8)
            plt.legend()
        
        save_path = save_dir / "training_loss_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n损失曲线已保存: {save_path}")
        
        # 显示图像（如果可能）
        try:
            plt.show()
        except:
            print("提示: 无法显示图像，但已保存到文件")
        
        plt.close()
    except ImportError:
        print("\n提示: 安装matplotlib可以绘制损失曲线: pip install matplotlib")
    
    return model, losses


def main():
    parser = argparse.ArgumentParser(description="Train VLA Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/libero_object",
        help="Path to LeRobot dataset (default: ./dataset/libero_object)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="Maximum training steps (default: 20000)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Steps interval for saving checkpoints (default: 5000)"
    )
    args = parser.parse_args()
    
    # 使用LeRobot数据集训练
    try:
        model, losses = train_with_lerobot_dataset(
            config_path=args.config,
            dataset_path=args.dataset_path,
            max_steps=args.max_steps,
            save_steps=args.save_steps
        )
        print("\n✓ 训练成功完成")
    except Exception as e:
        print(f"\n✗ LeRobot训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

