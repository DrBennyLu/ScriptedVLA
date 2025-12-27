"""
VLA模型推理脚本
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from src.model import VLAModel
from src.utils import (
    load_config,
    get_model_config,
    get_inference_config
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
    
    # 设置设备
    device = torch.device(inference_config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    print("Loading model...")
    model = VLAModel(
        vlm_config=model_config.get("vlm", {}),
        action_head_config=model_config.get("action_head", {}),
        use_cross_attention=model_config.get("vla", {}).get("use_cross_attention", True),
        cross_attention_layers=model_config.get("vla", {}).get("cross_attention_layers", 3)
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
    
    # 加载图像
    print(f"Loading image: {args.image}")
    image = load_image(
        args.image,
        image_size=model_config.get("vlm", {}).get("image_size", 224)
    )
    image = image.to(device)
    
    # 推理
    print("Running inference...")
    with torch.no_grad():
        if args.text:
            outputs = model(image, texts=[args.text])
        else:
            outputs = model(image)
        
        actions = outputs["actions"]
        actions = actions.cpu().numpy()[0]  # 移除batch维度
    
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

