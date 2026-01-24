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
VLA模型推理脚本
从dataset/libero_object读取一帧数据，加载最新checkpoint，进行推理并对比GT
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.ScriptedVLA.model import QwenGR00TVLAModel
from src.ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_data_config,
    Normalizer
)
from train import load_dataset_info, get_state_dim_from_info, create_delta_timestamps

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """查找最新的检查点文件"""
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None
    
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
    
    return latest_checkpoint


def load_dataset_frame(
    dataset_path: str,
    frame_idx: int = 0,
    config_path: str = "config.yaml"
) -> Dict:
    """从数据集加载一帧数据"""
    if not HAS_LEROBOT:
        raise ImportError("lerobot library not installed. Install with: pip install lerobot==0.3.3")
    
    config = load_config(config_path)
    data_config = get_data_config(config)
    dataset_config = config.get("dataset", {})
    
    dataset_dir = Path(dataset_path).resolve()
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_dir}")
    
    dataset_info = load_dataset_info(dataset_dir)
    fps = dataset_info.get("fps", 10)
    action_horizon = dataset_config.get("action_horizon", 50)
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    
    dataset_name = dataset_dir.name
    root_path_str = str(dataset_dir)
    
    lerobot_dataset = LeRobotDataset(
        repo_id=dataset_name,
        root=root_path_str,
        delta_timestamps=delta_timestamps
    )
    
    if len(lerobot_dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    
    if frame_idx >= len(lerobot_dataset):
        frame_idx = 0
    
    sample = lerobot_dataset[frame_idx]
    
    # 转换样本格式
    result = {}
    
    # 处理图像
    images_dict = {}
    image_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
    
    if image_keys:
        for key in image_keys:
            camera_name = key.replace("observation.images.", "")
            img_tensor = sample[key]
            if isinstance(img_tensor, torch.Tensor):
                images_dict[camera_name] = img_tensor
        result["images"] = images_dict if images_dict else {}
    else:
        result["images"] = {}
    
    # 处理文本指令
    if "task" in sample:
        result["text"] = sample["task"]
    elif "instruction" in sample:
        result["text"] = sample["instruction"]
    else:
        result["text"] = "Perform the task"
    
    # 处理动作
    if "action" in sample:
        result["action"] = sample["action"]
    
    # 处理状态
    if "observation.state" in sample:
        result["state"] = sample["observation.state"]
    elif "observation" in sample and "state" in sample["observation"]:
        result["state"] = sample["observation"]["state"]
    elif "state" in sample:
        result["state"] = sample["state"]
    
    return result


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    device: Optional[str] = None
) -> Tuple[QwenGR00TVLAModel, Optional[Normalizer]]:
    """从checkpoint加载模型"""
    config = load_config(config_path)
    model_config = get_model_config(config)
    data_config = get_data_config(config)
    dataset_config = config.get("dataset", {})
    
    # 获取相机名称：从dataset.image_keys中提取
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    camera_names = []
    for key in image_keys:
        if key.startswith("observation.images."):
            camera_name = key.replace("observation.images.", "")
            camera_names.append(camera_name)
    
    if not camera_names:
        camera_names = ["global_img", "left_wrist_img"]  # 默认值
    
    # 获取状态配置
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
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
    
    # 加载检查点
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path_obj, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    
    # 加载归一化器
    normalizer = None
    if "normalizer" in checkpoint:
        normalizer = Normalizer.from_dict(checkpoint["normalizer"])
    
    return model, normalizer


def prepare_images_input(
    images: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """准备图像输入"""
    images_dict = {}
    for cam_name, img_tensor in images.items():
        if isinstance(img_tensor, torch.Tensor):
            if img_tensor.dim() == 3:  # [C, H, W]
                img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
            images_dict[cam_name] = img_tensor.to(device)
    return images_dict


def run_inference(
    model: QwenGR00TVLAModel,
    images: Dict[str, torch.Tensor],
    instruction: str,
    states: Optional[torch.Tensor] = None,
    normalizer: Optional[Normalizer] = None
) -> np.ndarray:
    """运行推理"""
    device = next(model.parameters()).device
    
    # 准备图像输入
    images_input = prepare_images_input(images, device)
    
    # 处理状态
    if states is not None:
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32)
        states = states.to(device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
    elif model.use_state:
        states = torch.zeros(1, model.state_dim, device=device)
    
    # 推理
    inputs = {
        "images": images_input,
        "instructions": [instruction],
        "states": states
    }
    
    with torch.no_grad():
        outputs = model.predict_action(inputs=inputs)
    
    # 提取动作
    if "normalized_actions" in outputs:
        actions = outputs["normalized_actions"]  # [B, T, action_dim]
    elif "actions" in outputs:
        actions = outputs["actions"]  # [B, T, action_dim]
    else:
        raise ValueError("Model output does not contain actions")
    
    # 转换为numpy
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # 反归一化
    if normalizer is not None:
        actions = normalizer.denormalize_action(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
    
    # 返回action chunk [T, action_dim]
    if len(actions.shape) == 3:  # [B, T, action_dim]
        return actions[0]  # [T, action_dim]
    elif len(actions.shape) == 2:  # [B, action_dim] 或 [T, action_dim]
        if actions.shape[0] == 1:  # [1, action_dim]
            return actions[0:1]  # [1, action_dim]
        else:
            return actions  # [T, action_dim]
    else:
        return actions.reshape(1, -1)  # [1, action_dim]


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VLA模型推理：从数据集读取一帧，推理并对比GT")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset/libero_object",
        help="数据集路径"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="检查点目录"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="要测试的帧索引"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（'cuda'或'cpu'），如果为None则自动选择"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLA模型推理")
    print("=" * 60)
    
    # 1. 查找最新checkpoint
    print("\n[Step 1] 查找最新checkpoint...")
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        print(f"  ✗ 未找到checkpoint文件在: {checkpoint_dir}")
        return
    
    print(f"  ✓ 找到最新checkpoint: {latest_checkpoint}")
    
    # 2. 加载模型
    print("\n[Step 2] 加载模型...")
    model, normalizer = load_model_from_checkpoint(
        str(latest_checkpoint),
        args.config,
        args.device
    )
    device = next(model.parameters()).device
    print(f"  ✓ 模型加载成功，设备: {device}")
    
    # 3. 加载数据集帧
    print("\n[Step 3] 加载数据集帧...")
    sample = load_dataset_frame(args.dataset, args.frame_idx, args.config)
    print(f"  ✓ 数据加载成功")
    print(f"    指令: {sample.get('text', 'N/A')}")
    
    # 4. 准备输入
    images = sample.get("images", {})
    if not images:
        print("  ✗ 样本中没有图像数据")
        return
    
    instruction = sample.get("text", "Perform the task")
    states = sample.get("state")
    if states is not None and isinstance(states, torch.Tensor):
        states = states.numpy()
    
    # 5. 运行推理
    print("\n[Step 4] 运行推理...")
    predicted_actions = run_inference(
        model=model,
        images=images,
        instruction=instruction,
        states=states,
        normalizer=normalizer
    )
    print(f"  ✓ 推理成功")
    print(f"    预测action chunk形状: {predicted_actions.shape}")
    
    # 6. 获取GT action chunk
    print("\n[Step 5] 对比GT和预测...")
    true_action = sample.get("action")
    if true_action is None:
        print("  ✗ 样本中没有action数据")
        return
    
    if isinstance(true_action, torch.Tensor):
        true_action = true_action.numpy()
    
    # 处理GT action维度
    if len(true_action.shape) == 2:  # [action_horizon, action_dim]
        true_actions = true_action
    elif len(true_action.shape) == 1:  # [action_dim]
        true_actions = true_action.reshape(1, -1)  # [1, action_dim]
    else:
        true_actions = true_action.reshape(-1, true_action.shape[-1])
    
    # 确保维度匹配
    min_len = min(len(predicted_actions), len(true_actions))
    predicted_actions = predicted_actions[:min_len]
    true_actions = true_actions[:min_len]
    
    # 7. 计算并显示对比结果
    print("\n" + "=" * 60)
    print("推理结果与GT对比")
    print("=" * 60)
    
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    action_dim = predicted_actions.shape[1]
    
    # 逐时间步对比
    print(f"\n{'Time':>6s} {'Action':>10s} {'Predicted':>12s} {'GT':>12s} {'Diff':>12s}")
    print("-" * 60)
    
    mae_per_dim = np.zeros(action_dim)
    for t in range(min_len):
        pred = predicted_actions[t]
        true = true_actions[t]
        diff = np.abs(pred - true)
        mae_per_dim += diff
        
        for i, name in enumerate(action_names[:action_dim]):
            print(f"{t:>6d} {name:>10s} {pred[i]:12.4f} {true[i]:12.4f} {diff[i]:12.4f}")
    
    mae_per_dim /= min_len
    mae_overall = np.mean(mae_per_dim)
    
    print("\n" + "-" * 60)
    print(f"{'MAE per dim':>18s}", end="")
    for i, name in enumerate(action_names[:action_dim]):
        print(f" {name:>10s}", end="")
    print()
    print(f"{'':>18s}", end="")
    for mae in mae_per_dim:
        print(f" {mae:10.4f}", end="")
    print()
    print(f"{'Overall MAE':>18s} {mae_overall:10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
