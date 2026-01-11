"""
VLA模型推理脚本
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from ScriptedVLA.model import QwenGR00TVLAModel
from ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_inference_config,
    get_data_config
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
    model_config = get_model_config(config)
    inference_config = get_inference_config(config)
    data_config = get_data_config(config)
    
    # 获取相机和状态配置
    camera_config = data_config.get("cameras", {})
    camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    
    # 设置设备
    device = torch.device(inference_config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
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
    checkpoint_path = args.checkpoint or inference_config.get("checkpoint_path", "./checkpoints/best_model.pt")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 加载图像（支持多相机）
    print(f"Loading image: {args.image}")
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    
    # 如果只有一个相机，使用单图像模式
    if len(camera_names) == 1:
        image = load_image(args.image, image_size=image_size)
        image = image.to(device)
        images_input = image
    else:
        # 多相机模式：尝试加载多个图像
        # 如果只提供了一个图像路径，使用第一个相机名称
        images_dict = {}
        base_path = Path(args.image)
        for cam_name in camera_names:
            # 尝试多种可能的文件名
            possible_paths = [
                base_path.parent / f"{cam_name}_{base_path.name}",
                base_path.parent / f"{base_path.stem}_{cam_name}{base_path.suffix}",
                base_path  # 如果找不到，使用原始路径
            ]
            found = False
            for img_path in possible_paths:
                if img_path.exists():
                    img_tensor = load_image(str(img_path), image_size=image_size)
                    images_dict[cam_name] = img_tensor.to(device)
                    found = True
                    break
            if not found:
                print(f"Warning: Image for camera {cam_name} not found, using provided image")
                img_tensor = load_image(args.image, image_size=image_size)
                images_dict[cam_name] = img_tensor.to(device)
        images_input = images_dict
    
    # 处理状态（如果使用）
    states = None
    if use_state:
        # 如果没有提供状态，使用零向量
        states = torch.zeros(1, state_dim, device=device)
    
    # 推理
    print("Running inference...")
    # 使用统一输入格式
    inputs = {
        "images": images_input,
        "instructions": [args.text] if args.text else [""],  # 至少需要一个指令
        "states": states
    }
    
    with torch.no_grad():
        outputs = model.predict_action(inputs=inputs)
        
        # 获取预测的动作
        if "normalized_actions" in outputs:
            actions = outputs["normalized_actions"]  # [B, T, action_dim]
        elif "actions" in outputs:
            actions = outputs["actions"]  # [B, T, action_dim]
        else:
            raise ValueError("Model output does not contain actions")
        
        # 如果是numpy数组，直接使用；如果是tensor，转换为numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # 取第一个batch的第一个动作（或平均多个时间步）
        if len(actions.shape) == 3:  # [B, T, action_dim]
            actions = actions[0, 0, :]  # 取第一个batch的第一个时间步
        elif len(actions.shape) == 2:  # [B, action_dim]
            actions = actions[0, :]  # 取第一个batch
        else:
            actions = actions[0]  # 取第一个元素
    
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

