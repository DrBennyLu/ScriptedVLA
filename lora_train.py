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
VLA模型 LoRA 微调训练脚本
使用 LoRA 方法微调 QwenVLM，框架结构与 train.py 一致
"""

import os
import sys

# 若配置了 cache_dir 或 local_model_path，在 import transformers 之前设置离线模式
def _maybe_enable_offline():
    import yaml
    from pathlib import Path
    config_path = "config.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vlm = cfg.get("model", {}).get("vlm", {})
        if vlm.get("cache_dir") or vlm.get("local_model_path"):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"


_maybe_enable_offline()

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
    create_normalizer_from_lerobot_meta,
    Normalizer,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True

    class LeRobotDatasetSubset(LeRobotDataset):
        """
        LeRobotDataset 子类，修复使用 episodes=subset 时的索引越界问题。
        当传入 episodes 列表时，hf_dataset 中每行的 episode_index 仍是原始 episode 编号，
        而 episode_data_index 是按子集位置 (0..len(episodes)-1) 建的，需在 _get_query_indices
        中将原始 episode 编号映射为子集内位置。
        """

        def _get_query_indices(self, idx: int, ep_idx: int):
            if self.episodes is not None and ep_idx in self.episodes:
                ep_idx = self.episodes.index(ep_idx)
            return super()._get_query_indices(idx, ep_idx)

except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None
    LeRobotDatasetSubset = None


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


def _tensor_to_pil_image(img_tensor, image_size=None):
    """将tensor转换为PIL.Image"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    elif img_tensor.dim() != 3:
        raise ValueError(f"Unexpected tensor shape: {img_tensor.shape}, expected [C, H, W]")

    img_tensor = img_tensor.permute(1, 2, 0)
    img_array = img_tensor.cpu().numpy()

    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0 and img_array.min() >= 0.0:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.max() <= 255.0:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=2)

    img_pil = Image.fromarray(img_array, mode='RGB')
    if image_size and (img_pil.size[0] != image_size or img_pil.size[1] != image_size):
        img_pil = img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img_pil


def create_collate_fn(
    image_keys,
    state_key,
    image_size=None,
    use_batch_task=True,
    normalizer=None,
    normalize_action=True,
    normalize_state=True,
):
    """创建collate函数，处理lerobot返回的batch格式"""
    def collate_fn(batch_list):
        from torch.utils.data._utils.collate import default_collate

        if len(batch_list) > 0:
            first_sample = batch_list[0]
            if not isinstance(first_sample, dict):
                raise ValueError(f"batch_list中的样本应该是字典，但得到: {type(first_sample)}")

        batch_dict = default_collate(batch_list)
        batch_size = len(batch_list)

        if not hasattr(collate_fn, '_debug_printed'):
            print(f"[调试] collate_fn - batch_dict中的键: {list(batch_dict.keys())}")
            collate_fn._debug_printed = True

        missing_keys = [k for k in image_keys if k not in batch_dict]
        if missing_keys:
            raise ValueError(f"配置的图像键不存在于batch中: {missing_keys}, 可用键: {list(batch_dict.keys())}")

        images_list = []
        for i in range(batch_size):
            if len(image_keys) == 1:
                img_tensor = batch_dict[image_keys[0]][i]
                images_list.append(_tensor_to_pil_image(img_tensor, image_size))
            else:
                camera_images = [_tensor_to_pil_image(batch_dict[key][i], image_size)
                                for key in image_keys]
                images_list.append(camera_images)

        actions = batch_dict["action"]
        if normalize_action and normalizer is not None:
            actions = normalizer.normalize_action(actions)

        states = None
        if state_key in batch_dict:
            states = batch_dict[state_key]
            if isinstance(states, torch.Tensor):
                if states.numel() > 0 and states.abs().sum().item() > 1e-6:
                    if normalize_state and normalizer is not None:
                        states = normalizer.normalize_state(states)
                else:
                    import warnings
                    warnings.warn(f"状态数据全为0，可能数据提取有问题。batch_dict中的键: {list(batch_dict.keys())}")
                    states = None
            else:
                if normalize_state and normalizer is not None:
                    states = normalizer.normalize_state(states)
        else:
            possible_state_keys = [k for k in batch_dict.keys() if 'state' in k.lower()]
            if possible_state_keys:
                import warnings
                warnings.warn(f"配置的状态键 '{state_key}' 不在batch中。可用键: {list(batch_dict.keys())}")
                if len(possible_state_keys) > 0:
                    states = batch_dict[possible_state_keys[0]]
                    if normalize_state and normalizer is not None:
                        states = normalizer.normalize_state(states)

        if use_batch_task and "task" in batch_dict:
            task_data = batch_dict["task"]
            texts = [str(t) for t in task_data] if isinstance(task_data, list) else [str(task_data)] * batch_size
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


def apply_lora_to_vlm(vla_model: QwenGR00TVLAModel, lora_config: dict):
    """
    对 VLA 模型中的 QwenVLM 应用 LoRA 微调

    Args:
        vla_model: QwenGR00TVLAModel 实例
        lora_config: config.training.lora 配置字典

    Returns:
        应用 LoRA 后的模型（原地修改 vla_model.qwen_vl_interface.model）
    """
    if not HAS_PEFT:
        raise ImportError(
            "peft library not installed. "
            "Install with: pip install peft>=0.10.0"
        )

    if not lora_config.get("enabled", True):
        raise ValueError("LoRA 未启用，请设置 config.training.lora.enabled: true")

    # 获取 VLM 的底层模型（Qwen2VLForConditionalGeneration 等）
    base_model = vla_model.qwen_vl_interface.model

    # 如果已经冻结，需要先解冻以便 peft 能正确应用 LoRA
    for param in base_model.parameters():
        param.requires_grad = True

    # 构建 LoraConfig
    target_modules = lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=lora_config.get("modules_to_save", None),
        layers_to_transform=lora_config.get("layers_to_transform", None),
    )

    # 应用 LoRA
    peft_model = get_peft_model(base_model, peft_config)

    # 替换 VLM 中的模型引用
    vla_model.qwen_vl_interface.model = peft_model

    # 打印可训练参数统计
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA 可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of VLM)")

    return vla_model


def create_optimizer(model, config):
    """创建优化器"""
    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("type", "adamw")

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


def evaluate(
    model,
    dataloader,
    device,
    criterion=None,
    logger=None,
    max_eval_batches=50,
):
    """评估模型"""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    eval_batches = min(max_eval_batches, total_batches)

    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader),
            total=eval_batches,
            desc="Eval",
            unit="batch",
            leave=True,
        )
        for batch_idx, batch in progress_bar:
            if batch_idx >= eval_batches:
                break
            images = batch["images"]
            texts = batch["text"]
            actions = batch["action"].to(device)
            inputs = {
                "images": images,
                "instructions": texts,
                "actions": actions,
            }
            if "state" in batch:
                inputs["states"] = batch["state"].to(device)
            outputs = model(inputs=inputs)
            loss = outputs["action_loss"]
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    if was_training:
        model.train()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    msg = f"Eval Loss: {avg_loss:.4f} (batches {num_batches}/{total_batches})"
    if logger is not None:
        logger.info(msg)
    else:
        print(f"\n  [Eval] {msg}")
    return avg_loss


def find_latest_checkpoint(checkpoint_dir: Path):
    """查找最新的检查点文件"""
    if not checkpoint_dir.exists():
        return None, 0

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if not checkpoint_files:
        return None, 0

    max_step = 0
    latest_checkpoint = None

    for checkpoint_file in checkpoint_files:
        try:
            filename = checkpoint_file.stem
            step_str = filename.replace("checkpoint_step_", "")
            step = int(step_str)

            if step > max_step:
                max_step = step
                latest_checkpoint = checkpoint_file
        except (ValueError, AttributeError):
            continue

    return latest_checkpoint, max_step


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, device):
    """加载检查点并恢复模型、优化器和调度器状态"""
    print(f"  加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ 模型状态已加载")

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  ✓ 优化器状态已加载")

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"  ✓ 调度器状态已加载")

    normalizer = None
    if "normalizer" in checkpoint:
        normalizer = Normalizer.from_dict(checkpoint["normalizer"])
        print(f"  ✓ 归一化器状态已加载")

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
    使用 LeRobot 数据集进行 LoRA 微调训练

    Args:
        config_path: 配置文件路径
        dataset_path: 数据集路径
        max_steps: 最大训练步数
        save_steps: 保存检查点的间隔步数
    """
    print("=" * 60)
    print("LeRobot 数据集 LoRA 微调训练")
    print("=" * 60)

    if not HAS_LEROBOT:
        raise ImportError(
            "lerobot library not installed. "
            "Install with: pip install lerobot==0.3.3"
        )

    if not HAS_PEFT:
        raise ImportError(
            "peft library not installed. "
            "Install with: pip install peft>=0.10.0"
        )

    # 1. 加载配置
    print(f"\n步骤1: 加载配置文件: {config_path}")
    config = load_config(config_path)

    seed = config.get("seed", 42)
    print(f"\n步骤0.5: 设置随机种子: {seed}")
    set_seed(seed)
    print(f"  ✓ 随机种子已设置")

    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)

    # 获取 LoRA 配置
    lora_config = training_config.get("lora", {})
    if not lora_config.get("enabled", True):
        raise ValueError("LoRA 未启用。请在 config.training.lora 中设置 enabled: true")

    dataset_config = config.get("dataset", {})
    dataloader_config = dataset_config.get("dataloader", {})
    task_description_config = dataset_config.get("task_description", {})

    if dataset_path == "./dataset/libero_object":
        local_path = dataset_config.get("local_path", "./dataset/libero_object")
    else:
        local_path = dataset_path

    action_horizon = dataset_config.get("action_horizon", 50)
    vlm_config = model_config.get("vlm", {})
    image_size = vlm_config.get("image_size", 224)

    use_batch_task = task_description_config.get("use_batch_task", True)

    merged_training_config = training_config.copy()
    batch_size = training_config.get("batch_size", 8)

    if max_steps == 20000:
        max_steps = training_config.get("max_steps", 20000)

    if save_steps == 5000:
        save_steps = merged_training_config.get("save_steps", 5000)

    eval_steps = merged_training_config.get("eval_steps", 5000)
    max_eval_batches = merged_training_config.get("max_eval_batches", 50)
    logging_steps = merged_training_config.get("logging_steps", 100)

    print(f"\n训练参数（从配置文件读取）:")
    print(f"  save_steps: {save_steps}")
    print(f"  eval_steps: {eval_steps}")
    print(f"  max_eval_batches: {max_eval_batches}")
    print(f"  logging_steps: {logging_steps}")
    print(f"  LoRA r: {lora_config.get('r', 8)}, alpha: {lora_config.get('lora_alpha', 16)}")

    # 2. 加载 LeRobot 数据集
    print(f"\n步骤2: 加载 LeRobot 数据集")
    dataset_path_obj = Path(local_path).resolve()
    if not dataset_path_obj.exists():
        raise ValueError(f"数据集路径不存在: {dataset_path_obj}")

    print(f"  数据集路径: {dataset_path_obj}")

    print(f"\n步骤2.5: 从配置文件读取数据集参数")

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

    dataset_info = load_dataset_info(dataset_path_obj)
    fps = dataset_info.get("fps", 10)
    print(f"  数据集FPS（从info.json）: {fps}")

    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    print(f"  Action horizon: {action_horizon}")

    print(f"\n步骤2.6: 创建 LeRobotDataset 与归一化器")
    info_file = dataset_path_obj / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"本地数据集路径不存在或无效: {info_file}")

    dataset_name = dataset_path_obj.name
    root_path_str = str(dataset_path_obj)

    try:
        episode_slice = [0, 22, 25, 28, 30, 41, 47, 59, 63, 73, 91, 116, 119, 172, 206, 234, 236,
        237, 238, 239, 240, 242, 243, 266, 277, 286, 287, 307, 314, 315, 332, 339, 348, 350, 352,
        353, 365, 366, 368, 370, 390, 393, 400, 411, 420]
        lerobot_dataset = LeRobotDatasetSubset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps,
            episodes=episode_slice
        )
        print(f"  ✓ LeRobotDataset 创建成功: repo_id={dataset_name}, root={root_path_str}")
    except Exception as e:
        print(f"  ✗ 创建LeRobotDataset失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        normalizer = create_normalizer_from_lerobot_meta(
            lerobot_dataset,
            state_key=state_key,
            action_key="action",
        )
        print(f"  ✓ 归一化器已从 meta.episodes_stats 创建")
        if normalizer.action_min is not None:
            print(f"  Action 范围: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
        if normalizer.state_min is not None:
            print(f"  State 范围: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
    except Exception as e:
        print(f"  从 meta.episodes_stats 创建归一化器失败: {e}，尝试从 meta/episodes_stats.jsonl 创建")
        try:
            normalizer = create_normalizer_from_dataset(dataset_path_obj)
            print(f"  ✓ 归一化器已从 episodes_stats.jsonl 创建")
        except Exception as e2:
            print(f"  ✗ 归一化器创建失败: {e2}")
            print(f"  警告: 将不使用归一化，训练可能不稳定")
            normalizer = None

    # 3. 创建数据加载器
    print(f"\n步骤3: 创建数据加载器")
    print(f"  Batch size: {batch_size}")
    print(f"  最大训练步数: {max_steps}")

    normalize_action = data_config.get("normalize_action", False)
    normalize_state = data_config.get("normalize_state", False)
    custom_collate_fn = create_collate_fn(
        image_keys=image_keys,
        state_key=state_key,
        image_size=image_size,
        use_batch_task=use_batch_task,
        normalizer=normalizer,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
    )

    num_workers = dataloader_config.get("num_workers", 0)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": dataloader_config.get("shuffle", True),
        "num_workers": num_workers,
        "pin_memory": dataloader_config.get("pin_memory", False),
        "collate_fn": custom_collate_fn,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = dataloader_config.get("prefetch_factor", 2)

    train_loader = DataLoader(lerobot_dataset, **dataloader_kwargs)

    print(f"  数据加载器长度: {len(train_loader)} batches")

    # 4. 创建模型（LoRA 微调：VLM 不冻结，后续应用 LoRA）
    print(f"\n步骤4: 创建模型并应用 LoRA")

    vlm_config = model_config.get("vlm", {}).copy()
    # LoRA 模式下不冻结 VLM 基座（peft 会冻结基座并添加可训练的 LoRA 适配器）
    vlm_config["freeze_vlm"] = False

    action_head_config = model_config.get("action_head", {}).copy()
    action_head_config["action_horizon"] = action_horizon
    action_head_config["action_dim"] = action_dim

    vla_config = model_config.get("vla", {}).copy()
    vla_config["future_action_window_size"] = action_horizon - 1

    use_state_vlm = vla_config.get("use_state_vlm", vla_config.get("use_state", True))
    use_state_action_head = vla_config.get("use_state_action_head", vla_config.get("use_state", True))

    model = QwenGR00TVLAModel(
        vlm_config=vlm_config,
        action_head_config=action_head_config,
        use_state_vlm=use_state_vlm,
        use_state_action_head=use_state_action_head,
        state_dim=state_dim,
        future_action_window_size=action_horizon - 1
    )

    # 应用 LoRA 到 QwenVLM
    model = apply_lora_to_vlm(model, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  模型已移动到设备: {device}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  Action dimension: {action_dim}")
    print(f"  State dimension: {state_dim}")

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

    # 6.5. 检查断点续训
    print(f"\n步骤6.5: 检查断点续训")
    latest_checkpoint_path, latest_step_from_filename = find_latest_checkpoint(save_dir)

    start_step = 0
    if latest_checkpoint_path is not None:
        print(f"  发现检查点: {latest_checkpoint_path}")
        print(f"  文件名中的步数: {latest_step_from_filename}")

        start_step, checkpoint_loss, loaded_normalizer = load_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler, device
        )

        if loaded_normalizer is not None:
            normalizer = loaded_normalizer
            print(f"  使用检查点中的归一化器")
            custom_collate_fn = create_collate_fn(
                image_keys=image_keys,
                state_key=state_key,
                image_size=image_size,
                use_batch_task=use_batch_task,
                normalizer=normalizer,
                normalize_action=normalize_action,
                normalize_state=normalize_state,
            )
            dataloader_kwargs["collate_fn"] = custom_collate_fn
            train_loader = DataLoader(lerobot_dataset, **dataloader_kwargs)

        if start_step >= max_steps:
            print(f"  警告: 检查点步数({start_step})已超过或等于最大训练步数({max_steps})")
            print(f"  训练已完成，无需继续训练")
            return model, []

        print(f"  将从步数 {start_step} 继续训练到 {max_steps}")
    else:
        print(f"  未找到检查点，将从步数 0 开始训练")

    # 7. 训练循环
    remaining_steps = max_steps - start_step
    print(f"\n步骤7: 开始 LoRA 微调训练（从步数 {start_step} 继续，剩余 {remaining_steps} 步）")
    model.train()

    losses = []
    loader_iter = iter(train_loader)

    progress_bar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="LoRA Training")

    for step in progress_bar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        images = batch["images"]
        texts = batch["text"]
        actions = batch["action"].to(device)

        inputs = {
            "images": images,
            "instructions": texts,
            "actions": actions
        }
        if "state" in batch:
            inputs["states"] = batch["state"].to(device)

        outputs = model(inputs=inputs)
        loss = outputs["action_loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            merged_training_config.get("max_grad_norm", 1.0)
        )
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        loss_value = loss.item()
        losses.append(loss_value)

        avg_loss = sum(losses) / len(losses)
        progress_bar.set_postfix({
            "loss": f"{loss_value:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        if (step + 1) % logging_steps == 0:
            print(f"\n  Step {step + 1}/{max_steps}: "
                  f"Loss = {loss_value:.4f}, "
                  f"Avg Loss = {avg_loss:.4f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.2e}")

        if (step + 1) % eval_steps == 0:
            eval_batches_cap = min(max_eval_batches, 10)
            eval_loss = evaluate(
                model,
                train_loader,
                device,
                criterion=None,
                logger=None,
                max_eval_batches=eval_batches_cap,
            )
            progress_bar.write(f"  [Eval] Step {step + 1}: Eval Loss = {eval_loss:.4f} (batches={eval_batches_cap})")

        if (step + 1) % save_steps == 0:
            checkpoint_path = save_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(
                model, optimizer, scheduler, 0, loss_value, checkpoint_path,
                global_step=step + 1, normalizer=normalizer
            )

    # 8. 训练总结
    print("\n" + "=" * 60)
    print("LoRA 微调训练完成")
    print("=" * 60)
    print(f"总步数: {len(losses)}")
    print(f"平均损失: {sum(losses) / len(losses):.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"最小损失: {min(losses):.4f}")
    print(f"最大损失: {max(losses):.4f}")

    # 9. 保存损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(losses, linewidth=1, alpha=0.7)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("LoRA Training Loss Curve", fontsize=14)
        plt.grid(True, alpha=0.3)

        if len(losses) > 100:
            window_size = min(100, len(losses) // 20)
            smoothed_losses = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(losses), i + window_size // 2 + 1)
                smoothed_losses.append(sum(losses[start_idx:end_idx]) / (end_idx - start_idx))
            plt.plot(smoothed_losses, linewidth=2, label='Smoothed', alpha=0.8)
            plt.legend()

        save_path = save_dir / "lora_training_loss_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n损失曲线已保存: {save_path}")

        try:
            plt.show()
        except Exception:
            print("提示: 无法显示图像，但已保存到文件")

        plt.close()
    except ImportError:
        print("\n提示: 安装matplotlib可以绘制损失曲线: pip install matplotlib")

    return model, losses


def main():
    parser = argparse.ArgumentParser(description="Train VLA Model with LoRA")
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

    try:
        model, losses = train_with_lerobot_dataset(
            config_path=args.config,
            dataset_path=args.dataset_path,
            max_steps=args.max_steps,
            save_steps=args.save_steps
        )
        print("\n✓ LoRA 微调训练成功完成")
    except Exception as e:
        print(f"\n✗ LoRA 微调训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
