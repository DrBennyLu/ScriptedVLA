"""
测试推理脚本
从本地libero_object数据集读取一帧数据，加载训练好的模型checkpoint，完成动作生成的推理
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import load_model_from_checkpoint, run_inference
from ScriptedVLA.utils import load_config, get_data_config
from ScriptedVLA.data.lerobot_dataset_adapter import LeRobotDatasetAdapter


def load_dataset_frame(
    dataset_path: str,
    frame_idx: int = 0,
    config_path: str = "config.yaml"
) -> Dict:
    """
    从本地libero_object数据集加载一帧数据
    
    Args:
        dataset_path: 数据集路径（例如 "./dataset/libero_object"）
        frame_idx: 要加载的帧索引（默认0，即第一帧）
        config_path: 配置文件路径
        
    Returns:
        包含图像、指令、状态等信息的字典
    """
    # 加载配置
    config = load_config(config_path)
    data_config = get_data_config(config)
    lerobot_config = config.get("lerobot_test", {}).get("dataset", {})
    
    # 获取相机和状态配置
    camera_config = data_config.get("cameras", {})
    camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    image_size = lerobot_config.get("image_size", 224)
    
    # 创建数据集适配器
    print(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDatasetAdapter(
        dataset_path=dataset_path,
        image_size=image_size,
        camera_names=camera_names,
        use_state=use_state,
        state_dim=state_dim,
        action_horizon=lerobot_config.get("action_horizon", 50),
        pad_action_chunk=True
    )
    
    # 检查数据集长度
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    
    if frame_idx >= len(dataset):
        print(f"Warning: frame_idx {frame_idx} >= dataset length {len(dataset)}, using frame 0")
        frame_idx = 0
    
    # 获取指定帧的数据
    print(f"Loading frame {frame_idx} from dataset...")
    sample = dataset[frame_idx]
    
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample info:")
    if "episode_id" in sample:
        print(f"  Episode ID: {sample['episode_id']}")
    if "step_id" in sample:
        print(f"  Step ID: {sample['step_id']}")
    if "task_name" in sample:
        print(f"  Task name: {sample['task_name']}")
    
    return sample


def test_inference_from_dataset(
    dataset_path: str = "./dataset/libero_object",
    checkpoint_path: str = "./checkpoints/best_model.pt",
    config_path: str = "config.yaml",
    frame_idx: int = 0,
    device: Optional[str] = None
):
    """
    从数据集读取一帧数据并运行推理
    
    Args:
        dataset_path: 数据集路径
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        frame_idx: 要测试的帧索引
        device: 设备（"cuda"或"cpu"），如果为None则自动选择
    """
    print("=" * 60)
    print("VLA模型推理测试")
    print("=" * 60)
    
    # 1. 加载数据集中的一帧数据
    print("\n[Step 1] 加载数据集帧...")
    sample = load_dataset_frame(dataset_path, frame_idx, config_path)
    
    # 2. 加载模型checkpoint
    print("\n[Step 2] 加载模型checkpoint...")
    model = load_model_from_checkpoint(checkpoint_path, config_path, device)
    
    # 3. 准备输入数据
    print("\n[Step 3] 准备输入数据...")
    
    # 提取图像（可能是dict格式，每个相机一个tensor）
    images = sample.get("images", {})
    if isinstance(images, dict):
        # 多相机模式：转换为tensor dict
        images_dict = {}
        for cam_name, img_tensor in images.items():
            # 确保tensor在正确的设备上
            if isinstance(img_tensor, torch.Tensor):
                images_dict[cam_name] = img_tensor
            else:
                raise ValueError(f"Unexpected image type for camera {cam_name}: {type(img_tensor)}")
        images_input = images_dict
    else:
        # 单相机模式
        if isinstance(images, torch.Tensor):
            images_input = images
        else:
            raise ValueError(f"Unexpected images type: {type(images)}")
    
    # 提取指令
    instruction = sample.get("text", sample.get("instruction", ""))
    if not instruction:
        instruction = "Perform the task"  # 默认指令
    
    print(f"  Instruction: {instruction}")
    
    # 提取状态（如果使用）
    states = None
    if "state" in sample:
        states = sample["state"]
        if isinstance(states, torch.Tensor):
            states = states.numpy()
        print(f"  State shape: {states.shape if states is not None else None}")
    
    # 4. 运行推理
    print("\n[Step 4] 运行推理...")
    actions = run_inference(
        model=model,
        images=images_input,
        instructions=instruction,
        states=states,
        return_full_output=False
    )
    
    # 5. 显示结果
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    
    if len(actions.shape) == 1:
        # 单个动作 [action_dim]
        action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        print("预测的动作（第一个时间步）:")
        for name, value in zip(action_names[:len(actions)], actions):
            print(f"  {name:10s}: {value:8.4f}")
    elif len(actions.shape) == 2:
        # 动作序列 [T, action_dim]
        print(f"预测的动作序列: shape {actions.shape}")
        action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        print("\n前3个时间步的动作:")
        for t in range(min(3, actions.shape[0])):
            print(f"  Time step {t}:")
            for name, value in zip(action_names[:actions.shape[1]], actions[t]):
                print(f"    {name:10s}: {value:8.4f}")
        if actions.shape[0] > 3:
            print(f"  ... (共 {actions.shape[0]} 个时间步)")
    else:
        print(f"预测的动作: shape {actions.shape}")
        print(f"  Actions: {actions}")
    
    # 6. 如果有真实动作，进行比较
    if "action" in sample:
        true_action = sample["action"]
        if isinstance(true_action, torch.Tensor):
            true_action = true_action.numpy()
        
        print("\n" + "=" * 60)
        print("与真实动作比较（如果可用）")
        print("=" * 60)
        
        if len(true_action.shape) == 1:
            # 单个动作比较
            if len(actions.shape) == 1:
                action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
                print(f"{'Action':10s} {'Predicted':>12s} {'True':>12s} {'Diff':>12s}")
                print("-" * 50)
                for name, pred, true_val in zip(action_names[:len(actions)], actions, true_action):
                    diff = abs(pred - true_val)
                    print(f"{name:10s} {pred:12.4f} {true_val:12.4f} {diff:12.4f}")
                
                # 计算平均误差
                mae = np.mean(np.abs(actions - true_action))
                print(f"\n平均绝对误差 (MAE): {mae:.4f}")
        elif len(true_action.shape) == 2:
            # 动作序列比较
            print(f"真实动作序列: shape {true_action.shape}")
            if len(actions.shape) == 2:
                # 比较第一个时间步
                if actions.shape[0] > 0 and true_action.shape[0] > 0:
                    pred_first = actions[0]
                    true_first = true_action[0]
                    mae = np.mean(np.abs(pred_first - true_first))
                    print(f"第一个时间步的平均绝对误差 (MAE): {mae:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return actions


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试VLA模型推理")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset/libero_object",
        help="数据集路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="模型checkpoint路径"
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
        help="要测试的帧索引（默认0）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（'cuda'或'cpu'），如果为None则自动选择"
    )
    
    args = parser.parse_args()
    
    # 检查数据集路径
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print(f"请确保数据集已下载到: {dataset_path}")
        return
    
    # 检查checkpoint路径
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: Checkpoint文件不存在: {checkpoint_path}")
        print(f"请确保模型已训练并保存到: {checkpoint_path}")
        return
    
    # 运行测试
    try:
        actions = test_inference_from_dataset(
            dataset_path=str(dataset_path),
            checkpoint_path=str(checkpoint_path),
            config_path=args.config,
            frame_idx=args.frame_idx,
            device=args.device
        )
        print(f"\n推理成功！输出动作shape: {actions.shape}")
    except Exception as e:
        print(f"\n错误: 推理失败")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
