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
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Optional, Union, List

from ScriptedVLA.model import QwenGR00TVLAModel
from ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_inference_config,
    get_data_config,
    Normalizer
)


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    加载和预处理图像
    
    Args:
        image_path: 图像路径
        image_size: 图像尺寸
        
    Returns:
        预处理后的图像张量 [1, C, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    
    # 转换为张量并归一化
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    
    # 添加batch维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    device: Optional[str] = None
) -> QwenGR00TVLAModel:
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        device: 设备（"cuda"或"cpu"），如果为None则自动选择
        
    Returns:
        加载好的模型（已设置为eval模式）
    """
    # 加载配置
    config = load_config(config_path)
    model_config = get_model_config(config)
    data_config = get_data_config(config)
    inference_config = get_inference_config(config)
    
    # 获取相机和状态配置
    camera_config = data_config.get("cameras", {})
    camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    
    # 设置设备
    if device is None:
        device = inference_config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # 创建模型
    print("Loading model...")
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
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 加载归一化器（如果存在）
    normalizer = None
    if "normalizer" in checkpoint:
        normalizer = Normalizer.from_dict(checkpoint["normalizer"])
        print(f"Normalizer loaded from checkpoint")
    
    return model, normalizer


def prepare_images_input(
    images: Union[str, Dict[str, str], Dict[str, torch.Tensor], torch.Tensor],
    camera_names: List[str],
    image_size: int = 224,
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    准备图像输入（支持单相机和多相机）
    
    Args:
        images: 图像输入，可以是：
            - str: 单个图像路径（单相机模式）
            - Dict[str, str]: {camera_name: image_path}（多相机模式）
            - Dict[str, torch.Tensor]: {camera_name: tensor}（多相机模式，已处理）
            - torch.Tensor: 单个图像tensor（单相机模式）
        camera_names: 相机名称列表
        image_size: 图像尺寸
        device: 设备
        
    Returns:
        处理后的图像输入（单相机返回tensor，多相机返回dict）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果已经是tensor格式
    if isinstance(images, torch.Tensor):
        # 确保有batch维度
        if images.dim() == 3:  # [C, H, W]
            images = images.unsqueeze(0)  # [1, C, H, W]
        images = images.to(device)
        if len(camera_names) == 1:
            return images
        else:
            # 多相机模式，但只提供了一个tensor，复制到所有相机
            images_dict = {}
            for cam_name in camera_names:
                images_dict[cam_name] = images
            return images_dict
    
    # 如果是dict且值已经是tensor
    if isinstance(images, dict) and isinstance(list(images.values())[0], torch.Tensor):
        images_dict = {}
        for cam_name in camera_names:
            if cam_name in images:
                img_tensor = images[cam_name]
                # 确保有batch维度
                if img_tensor.dim() == 3:  # [C, H, W]
                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                images_dict[cam_name] = img_tensor.to(device)
            else:
                # 如果某个相机没有提供，使用第一个可用的
                first_tensor = list(images.values())[0]
                if first_tensor.dim() == 3:  # [C, H, W]
                    first_tensor = first_tensor.unsqueeze(0)  # [1, C, H, W]
                images_dict[cam_name] = first_tensor.to(device)
        return images_dict
    
    # 如果是字符串路径（单相机）
    if isinstance(images, str):
        image = load_image(images, image_size=image_size)
        image = image.to(device)
        if len(camera_names) == 1:
            return image
        else:
            # 多相机模式，但只提供了一个图像路径，复制到所有相机
            images_dict = {}
            for cam_name in camera_names:
                images_dict[cam_name] = image
            return images_dict
    
    # 如果是dict且值是路径（多相机）
    if isinstance(images, dict):
        images_dict = {}
        base_path = Path(list(images.values())[0]) if images else None
        for cam_name in camera_names:
            if cam_name in images:
                img_path = images[cam_name]
            else:
                # 尝试从基础路径推断
                if base_path:
                    possible_paths = [
                        base_path.parent / f"{cam_name}_{base_path.name}",
                        base_path.parent / f"{base_path.stem}_{cam_name}{base_path.suffix}",
                        base_path
                    ]
                    img_path = None
                    for p in possible_paths:
                        if p.exists():
                            img_path = str(p)
                            break
                    if img_path is None:
                        img_path = str(base_path)
                else:
                    raise ValueError(f"Cannot find image for camera {cam_name}")
            
            img_tensor = load_image(img_path, image_size=image_size)
            images_dict[cam_name] = img_tensor.to(device)
        return images_dict
    
    raise ValueError(f"Unsupported images type: {type(images)}")


def run_inference(
    model: QwenGR00TVLAModel,
    images: Union[str, Dict[str, str], Dict[str, torch.Tensor], torch.Tensor],
    instructions: Union[str, List[str]],
    states: Optional[Union[torch.Tensor, np.ndarray]] = None,
    camera_names: Optional[List[str]] = None,
    image_size: int = 224,
    return_full_output: bool = False,
    normalizer: Optional[Normalizer] = None
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    运行推理
    
    Args:
        model: 已加载的模型
        images: 图像输入（支持多种格式，见prepare_images_input）
        instructions: 文本指令（字符串或字符串列表）
        states: 机器人状态（可选）
        camera_names: 相机名称列表（如果为None，从模型获取）
        image_size: 图像尺寸
        return_full_output: 如果为True，返回完整的输出字典；否则只返回动作数组
        
    Returns:
        如果return_full_output=False: 动作数组 [action_dim] 或 [T, action_dim]
        如果return_full_output=True: 完整的输出字典
    """
    device = next(model.parameters()).device
    
    # 获取相机名称
    if camera_names is None:
        camera_names = model.camera_names
    
    # 准备图像输入
    images_input = prepare_images_input(images, camera_names, image_size, device)
    
    # 处理指令
    if isinstance(instructions, str):
        instructions = [instructions]
    
    # 处理状态
    if states is not None:
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32)
        states = states.to(device)
        # 确保有batch维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
    elif model.use_state:
        # 如果没有提供状态，使用零向量
        states = torch.zeros(1, model.state_dim, device=device)
    
    # 推理
    inputs = {
        "images": images_input,
        "instructions": instructions,
        "states": states
    }
    
    with torch.no_grad():
        outputs = model.predict_action(inputs=inputs)
    
    if return_full_output:
        return outputs
    
    # 提取动作
    if "normalized_actions" in outputs:
        actions = outputs["normalized_actions"]  # [B, T, action_dim]
    elif "actions" in outputs:
        actions = outputs["actions"]  # [B, T, action_dim]
    else:
        raise ValueError("Model output does not contain actions")
    
    # 如果是tensor，转换为numpy
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # 反归一化动作（如果提供了归一化器）
    if normalizer is not None:
        actions = normalizer.denormalize_action(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
    
    # 取第一个batch的动作
    if len(actions.shape) == 3:  # [B, T, action_dim]
        return actions[0, 0, :]  # [action_dim] - 取第一个batch的第一个时间步
    elif len(actions.shape) == 2:  # [B, action_dim]
        return actions[0, :]  # [action_dim]
    else:
        return actions[0]  # [action_dim]


def main():
    parser = argparse.ArgumentParser(description="VLA Model Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text instruction (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output action (optional)"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    inference_config = get_inference_config(config)
    model_config = get_model_config(config)
    
    # 加载模型
    checkpoint_path = args.checkpoint or inference_config.get("checkpoint_path", "./checkpoints/best_model.pt")
    model, normalizer = load_model_from_checkpoint(checkpoint_path, args.config)
    
    # 获取图像尺寸
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    
    # 加载图像
    print(f"Loading image: {args.image}")
    images_input = args.image
    
    # 运行推理
    print("Running inference...")
    actions = run_inference(
        model=model,
        images=images_input,
        instructions=args.text if args.text else "",
        image_size=image_size,
        normalizer=normalizer
    )
    
    # 打印结果
    print("\n" + "="*50)
    print("Predicted Actions:")
    print("="*50)
    action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    for name, value in zip(action_names, actions):
        print(f"  {name:10s}: {value:8.4f}")
    print("="*50)
    
    # 保存结果（如果需要）
    if args.output:
        np.save(args.output, actions)
        print(f"\nActions saved to: {args.output}")


if __name__ == "__main__":
    main()

