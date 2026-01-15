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
from PIL import Image

from ScriptedVLA.model import QwenGR00TVLAModel
from ScriptedVLA.data import (
    VLADataset,
    LIBERODataset,
    ACTDataset,
    create_libero_dataset_from_config,
    create_act_dataset_from_config
)
from ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    setup_logger,
    log_model_info
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None


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
    """从info.json的features.observation.images下获取图像键名"""
    image_keys = []
    if "features" in info:
        features = info["features"]
        for key in features.keys():
            if key.startswith("observation.images."):
                image_keys.append(key)
    
    return sorted(image_keys)


def get_state_key_from_info(info: dict) -> str:
    """从info.json的features.observation.state下获取状态键名"""
    if "features" in info and "observation" in info["features"]:
        obs_features = info["features"]["observation"]
        if "state" in obs_features:
            return "observation.state"
    
    return "observation.state"


def get_state_dim_from_info(info: dict, default_state_dim: int = 7) -> int:
    """从info.json的features.observation.state下获取状态维度"""
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


def create_collate_fn(image_keys=None, state_key=None, tasks_dict=None, image_size=None, use_batch_task=True):
    """创建collate函数，处理lerobot返回的batch格式"""
    def collate_fn(batch_list):
        from torch.utils.data._utils.collate import default_collate
        batch_dict = default_collate(batch_list)
        batch_size = len(batch_list)
        
        if image_keys:
            available_image_keys = [k for k in image_keys if k in batch_dict]
            if not available_image_keys:
                available_image_keys = [k for k in batch_dict.keys() if 'observation.images' in k.lower()]
        else:
            available_image_keys = [k for k in batch_dict.keys() if 'observation.images' in k.lower()]
        
        if not available_image_keys:
            raise ValueError(f"无法找到图像数据，可用键: {list(batch_dict.keys())}")
        
        images_list = []
        if len(available_image_keys) == 1:
            # 单相机：List[PIL.Image]
            images_tensor = batch_dict[available_image_keys[0]]
            for i in range(batch_size):
                img_tensor = images_tensor[i]
                img_tensor = img_tensor.permute(1, 2, 0)
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
                if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
                    img_pil = img_pil.resize((image_size, image_size))
                images_list.append(img_pil)
        else:
            # 多相机：List[List[PIL.Image]]
            for i in range(batch_size):
                camera_images = []
                for key in sorted(available_image_keys):
                    img_tensor = batch_dict[key][i]
                    img_tensor = img_tensor.permute(1, 2, 0)
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
                    if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
                        img_pil = img_pil.resize((image_size, image_size))
                    camera_images.append(img_pil)
                images_list.append(camera_images)
        
        actions = batch_dict["action"]
        
        states = None
        state_key_to_use = state_key if state_key and state_key in batch_dict else "observation.state"
        if state_key_to_use in batch_dict:
            states = batch_dict[state_key_to_use]
        
        texts = []
        if use_batch_task and "task" in batch_dict:
            task_data = batch_dict["task"]
            if isinstance(task_data, torch.Tensor):
                if task_data.dtype == torch.int64:
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
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 处理多相机图像
        if "images" in batch:
            # 多相机模式：将字典中的每个图像移到设备
            images = {k: v.to(device) for k, v in batch["images"].items()}
        elif "image" in batch:
            # 单相机模式（向后兼容）
            images = batch["image"].to(device)
        else:
            raise ValueError("Batch must contain either 'images' (dict) or 'image' (tensor)")
        
        texts = batch["text"]
        actions = batch["action"].to(device)  # 应该是 [B, action_horizon, action_dim]（HDF5格式）或 [B, action_dim]（JSON格式）
        
        # 处理actions维度：检查是否是action chunk
        # 如果数据源是JSON格式，可能返回单个动作 [B, action_dim]，需要扩展
        if actions.dim() == 2:
            # JSON格式：只有单个动作，需要扩展为action chunk
            # 注意：这不如真正的action chunk准确，建议使用HDF5格式
            # 从模型配置获取action_horizon（需要从model中获取）
            if hasattr(model, 'action_head') and hasattr(model.action_head, 'action_horizon'):
                action_horizon = model.action_head.action_horizon
            else:
                # 如果无法从模型获取，使用默认值
                action_horizon = 4
            # [B, action_dim] -> [B, action_horizon, action_dim]
            actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
        elif actions.dim() == 3:
            # HDF5格式：已经是action chunk [B, action_horizon, action_dim]
            pass
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}, expected [B, action_dim] or [B, action_horizon, action_dim]")
        
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
        
            # 获取损失（QwenGR00TVLAModel在训练模式下直接返回loss）
        if "loss" in outputs:
            loss = outputs["loss"]
        elif "action_loss" in outputs:
            loss = outputs["action_loss"]
        else:
            # 如果没有损失，说明是推理模式，不应该发生
            raise ValueError("Model output does not contain loss. Did you forget to provide actions?")
        
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
        
        # 更新进度条
        progress_bar.set_postfix({"loss": f"{loss.item() * config.get('gradient_accumulation_steps', 1):.4f}"})
        
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


def evaluate(model, dataloader, criterion, device, logger):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 按任务和episode统计损失（如果可用）
    task_losses = {}
    episode_losses = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 处理多相机图像
            if "images" in batch:
                images = {k: v.to(device) for k, v in batch["images"].items()}
            elif "image" in batch:
                images = batch["image"].to(device)
            else:
                raise ValueError("Batch must contain either 'images' (dict) or 'image' (tensor)")
            
            texts = batch["text"]
            actions = batch["action"].to(device)  # 应该是 [B, action_horizon, action_dim]（HDF5格式）或 [B, action_dim]（JSON格式）
            
            # 处理actions维度：检查是否是action chunk
            if actions.dim() == 2:
                # JSON格式：只有单个动作，需要扩展为action chunk
                if hasattr(model, 'action_head') and hasattr(model.action_head, 'action_horizon'):
                    action_horizon = model.action_head.action_horizon
                else:
                    action_horizon = 4
                actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
            elif actions.dim() == 3:
                # HDF5格式：已经是action chunk
                pass
            else:
                raise ValueError(f"Unexpected action shape: {actions.shape}")
            
            # 处理状态信息（如果存在）
            states = None
            if "state" in batch:
                states = batch["state"].to(device)
            
            # 前向传播（评估模式，但提供actions以计算损失）
            # 注意：模型需要处于训练模式才能计算损失（QwenGR00TVLAModel的forward方法只在self.training=True时计算损失）
            # 使用torch.no_grad()禁用梯度，这样既不会更新参数，又能计算损失
            model.train()  # 设置为训练模式以计算损失
            # 使用统一输入格式
            inputs = {
                "images": images,
                "instructions": texts,
                "actions": actions
            }
            if states is not None:
                inputs["states"] = states
            outputs = model(inputs=inputs)
            
            # 获取损失（QwenGR00TVLAModel在训练模式下直接返回loss）
            if "loss" in outputs:
                loss = outputs["loss"]
            elif "action_loss" in outputs:
                loss = outputs["action_loss"]
            else:
                raise ValueError("Model output does not contain loss. Did you forget to provide actions?")
            total_loss += loss.item()
            num_batches += 1
            
            # 统计任务和episode级别的损失
            if "task_name" in batch:
                task_names = batch["task_name"]
                if isinstance(task_names, (list, tuple)):
                    for i, task_name in enumerate(task_names):
                        if task_name not in task_losses:
                            task_losses[task_name] = []
                        task_losses[task_name].append(loss.item())
                elif isinstance(task_names, str):
                    if task_names not in task_losses:
                        task_losses[task_names] = []
                    task_losses[task_names].append(loss.item())
            
            if "episode_id" in batch and "task_name" in batch:
                episode_ids = batch["episode_id"]
                task_names = batch["task_name"]
                if isinstance(episode_ids, (list, tuple)) and isinstance(task_names, (list, tuple)):
                    for i, (task_name, episode_id) in enumerate(zip(task_names, episode_ids)):
                        key = (task_name, episode_id)
                        if key not in episode_losses:
                            episode_losses[key] = []
                        episode_losses[key].append(loss.item())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    
    # 打印任务级别的统计（如果有）
    if task_losses:
        logger.info("Task-level losses:")
        for task_name, losses in sorted(task_losses.items()):
            avg_task_loss = sum(losses) / len(losses) if losses else 0.0
            logger.info(f"  {task_name}: {avg_task_loss:.4f} ({len(losses)} samples)")
    
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
        (start_step, loss): 起始步数和损失值
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
    
    # 获取步数和损失
    start_step = checkpoint.get("global_step", 0)
    loss = checkpoint.get("loss", 0.0)
    
    print(f"  ✓ 检查点加载完成")
    print(f"    起始步数: {start_step}")
    print(f"    检查点损失: {loss:.4f}")
    
    return start_step, loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, global_step=None):
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
    
    # 获取模型和训练配置（从config.training读取，而不是lerobot_test）
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
    
    # 获取lerobot_test配置（仅用于数据集相关配置，不用于训练配置）
    lerobot_test_config = config.get("lerobot_test", {})
    dataset_config = lerobot_test_config.get("dataset", {})
    dataloader_config = lerobot_test_config.get("dataloader", {})
    task_description_config = lerobot_test_config.get("task_description", {})
    
    # 从lerobot_test.dataset配置中获取数据集相关参数
    repo_id = dataset_config.get("repo_id", "k1000dai/libero-object-smolvla")
    local_path = dataset_path  # 使用传入的路径
    action_horizon = dataset_config.get("action_horizon", 50)
    image_size = dataset_config.get("image_size", 224)
    
    # 从task_description配置中获取参数
    use_batch_task = task_description_config.get("use_batch_task", True)
    use_tasks_jsonl = task_description_config.get("use_tasks_jsonl", True)
    
    # 使用config.training中的配置（不从lerobot_test.training读取）
    merged_training_config = training_config.copy()
    
    # 从config.training中读取batch_size
    batch_size = training_config.get("batch_size", 8)
    
    # 从配置文件中读取训练参数
    # 优先级：命令行参数 > 配置文件 > 默认值
    # 如果函数参数是默认值，说明用户没有通过命令行指定，则从配置文件读取
    if save_steps == 5000:  # 如果使用的是默认值，从配置文件读取
        save_steps = merged_training_config.get("save_steps", 5000)
    # 否则使用命令行参数传入的值（不改变save_steps）
    
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
    dataset_info = load_dataset_info(dataset_path_obj)   #TODO:在此处增加数据集归一化
    
    # 从info.json获取信息
    fps = dataset_info.get("fps", 10)
    print(f"  数据集FPS: {fps}")
    
    image_keys = get_image_keys_from_info(dataset_info)
    print(f"  图像键名: {image_keys}")
    
    state_key = get_state_key_from_info(dataset_info)
    print(f"  状态键名: {state_key}")
    
    state_dim = get_state_dim_from_info(dataset_info, default_state_dim=default_state_dim)
    print(f"  状态维度: {state_dim}")
    
    # 创建delta_timestamps
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    print(f"  Action horizon: {action_horizon}")
    
    # 加载任务描述
    tasks_dict = {}
    if use_tasks_jsonl and (dataset_path_obj / "meta" / "tasks.jsonl").exists():
        tasks_dict = load_tasks_from_jsonl(dataset_path_obj)
        print(f"  加载了{len(tasks_dict)}个任务描述")
    
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
    
    # 4. 创建模型
    print(f"\n步骤4: 创建模型")
    
    vlm_config = model_config.get("vlm", {})
    action_head_config = model_config.get("action_head", {}).copy()
    action_head_config["action_horizon"] = action_horizon
    
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
    print(f"  State dimension: {state_dim}")
    
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
        start_step, checkpoint_loss = load_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler, device
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
        
        # 准备输入
        images = batch["images"]
        texts = batch["text"]
        actions = batch["action"].to(device)
        
        # 验证actions维度
        if actions.ndim != 3:
            raise ValueError(f"意外的actions维度: {actions.shape}, 期望 [B, action_horizon, action_dim]")
        if actions.shape[1] != action_horizon:
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
                model, optimizer, scheduler, 0, loss_value, checkpoint_path, global_step=step + 1
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
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no_lerobot",
        action="store_true",
        help="Don't use LeRobot dataset (use original training logic)"
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
    
    # 默认使用lerobot数据集训练，除非用户指定--no_lerobot
    if not args.no_lerobot:
        # 使用LeRobot数据集训练
        try:
            model, losses = train_with_lerobot_dataset(
                config_path=args.config,
                dataset_path=args.dataset_path,
                max_steps=args.max_steps,
                save_steps=args.save_steps
            )
            print("\n✓ 训练成功完成")
            return
        except Exception as e:
            print(f"\n✗ LeRobot训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 原有的训练逻辑（保留向后兼容）
    
    # 加载配置
    config = load_config(args.config)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    
    # 设置日志
    logger = setup_logger(
        log_dir=config.get("logging", {}).get("log_dir", "./logs")
    )
    logger.info("Starting VLA training...")
    logger.info(f"Config: {config}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 获取相机和状态配置
    camera_config = data_config.get("cameras", {})
    camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    
    # 创建模型
    vla_config = model_config.get("vla", {})
    future_action_window_size = vla_config.get("future_action_window_size", 10)
    model = QwenGR00TVLAModel(
        vlm_config=model_config.get("vlm", {}),
        action_head_config=model_config.get("action_head", {}),
        camera_names=camera_names,
        use_state=use_state,
        state_dim=state_dim,
        future_action_window_size=future_action_window_size
    )
    model = model.to(device)
    log_model_info(logger, model)
    
    # 创建数据集和数据加载器
    dataset_type = data_config.get("dataset_type", "custom")
    image_size = data_config.get("image_size", model_config.get("vlm", {}).get("image_size", 224))
    
    logger.info(f"Using dataset type: {dataset_type}")
    logger.info(f"Cameras: {camera_names}, Use state: {use_state}, State dim: {state_dim}")
    
    # 从模型配置获取action_horizon（用于数据集初始化）
    action_head_config = model_config.get("action_head", {})
    action_horizon = action_head_config.get("action_horizon", 4)
    
    if dataset_type == "libero":
        # LIBERO数据集
        libero_config = data_config.get("libero", {})
        train_dataset = LIBERODataset(
            dataset_path=libero_config.get("dataset_path", "./data/libero"),
            task_names=libero_config.get("task_names", None),
            image_size=image_size,
            max_episode_length=libero_config.get("max_episode_length", 100)
        )
        # LIBERO通常不区分train/val，可以按比例分割
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    elif dataset_type == "act":
        # ACT数据集
        act_config = data_config.get("act", {})
        train_dataset = ACTDataset(
            dataset_path=act_config.get("dataset_path", "./data/act"),
            image_size=image_size,
            chunk_size=act_config.get("chunk_size", 1)
        )
        # ACT数据集也按比例分割
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    elif dataset_type == "lerobot":
        # LeRobot数据集（兼容HF开源数据集）
        if not HAS_LEROBOT:
            raise ImportError(
                "LeRobot library not installed. "
                "Install with: pip install lerobot datasets"
            )
        
        lerobot_config = data_config.get("lerobot", {})
        lerobot_action_horizon = lerobot_config.get("action_horizon") or action_horizon
        
        train_dataset = LeRobotDatasetAdapter(
            dataset_path=lerobot_config.get("dataset_path", "lerobot/pusht"),
            image_size=image_size,
            camera_names=lerobot_config.get("camera_names", camera_names),
            use_state=lerobot_config.get("use_state", use_state),
            state_dim=lerobot_config.get("state_dim", state_dim),
            action_horizon=lerobot_action_horizon,
            pad_action_chunk=lerobot_config.get("pad_action_chunk", True)
        )
        # LeRobot数据集也按比例分割
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    else:
        # 自定义数据集
        train_dataset = VLADataset(
            data_path=data_config.get("train_data_path", "./dataset/train"),
            image_size=image_size,
            camera_names=camera_names,
            use_state=use_state,
            state_dim=state_dim,
            action_horizon=action_horizon,
            pad_action_chunk=True  # 如果episode末尾不够action_horizon，使用最后一个动作填充
        )
        val_dataset = VLADataset(
            data_path=data_config.get("val_data_path", "./dataset/val"),
            image_size=image_size,
            camera_names=camera_names,
            use_state=use_state,
            state_dim=state_dim,
            action_horizon=action_horizon,
            pad_action_chunk=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        prefetch_factor=data_config.get("prefetch_factor", 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 统计任务和episode信息（如果可用）
    if len(train_dataset) > 0:
        try:
            sample = train_dataset[0]
            if "task_name" in sample:
                tasks = set()
                episodes = set()
                for s in train_dataset:
                    if "task_name" in s:
                        tasks.add(s["task_name"])
                    if "episode_id" in s:
                        episodes.add((s.get("task_name", "unknown"), s["episode_id"]))
                logger.info(f"Train tasks: {len(tasks)}, Episodes: {len(episodes)}")
        except Exception as e:
            # 如果统计失败，忽略
            pass
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, training_config)
    num_training_steps = len(train_loader) * training_config.get("num_epochs", 100)
    scheduler = create_scheduler(optimizer, training_config, num_training_steps)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 恢复训练（如果有）
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        start_step = checkpoint.get("global_step", start_epoch * len(train_loader))
        best_val_loss = checkpoint.get("loss", float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # 创建保存目录
    save_dir = Path(training_config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    num_epochs = training_config.get("num_epochs", 100)
    eval_steps = training_config.get("eval_steps", 500)
    save_steps = training_config.get("save_steps", 1000)
    
    # 使用恢复的步数或计算起始步数
    global_step = start_step
    last_eval_step = start_step - eval_steps  # 确保第一次评估在正确时机
    last_save_step = start_step - save_steps  # 确保第一次保存在正确时机
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, training_config, logger, epoch+1, start_step=global_step
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 验证（基于步数，而不是轮数）
        if (global_step - last_eval_step >= eval_steps or global_step % eval_steps == 0) or epoch == num_epochs - 1:
            val_loss = evaluate(model, val_loader, criterion, device, logger)
            last_eval_step = global_step
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_dir / "best_model.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, best_model_path, global_step=global_step
                )
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # 定期保存检查点（基于步数，而不是轮数）
        if global_step - last_save_step >= save_steps or global_step % save_steps == 0:
            checkpoint_path = save_dir / f"checkpoint_step_{global_step}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, checkpoint_path, global_step=global_step
            )
            last_save_step = global_step
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

