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
测试推理脚本
从本地libero_object数据集读取一帧数据，加载训练好的模型checkpoint，完成动作生成的推理
支持单帧测试和完整episode测试（包含3D可视化）
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import tempfile
import random
import gc
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import load_model_from_checkpoint, run_inference, prepare_images_input
from src.ScriptedVLA.utils import load_config, get_data_config, get_model_config, Normalizer
from src.ScriptedVLA.data.lerobot_dataset_adapter import LeRobotDatasetAdapter
from train import load_dataset_info, get_state_dim_from_info, create_delta_timestamps

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


def get_test_model_config(config_path: str = "config.yaml", dataset_path: Optional[str] = None, use_test_config: bool = False) -> str:
    """
    获取应用了测试配置后的临时配置文件路径
    与test_training.py使用相同的配置读取方式，包括从数据集信息中获取state_dim
    
    Args:
        config_path: 原始配置文件路径
        dataset_path: 数据集路径（用于获取state_dim等信息）
        use_test_config: 是否使用test配置（默认False，直接使用原始配置以确保与训练时一致）
        
    Returns:
        临时配置文件路径（如果使用了测试配置），否则返回原路径
    """
    config = load_config(config_path)
    test_config = config.get("test", {})
    test_model_config = test_config.get("model", {})
    data_config = get_data_config(config)
    
    # 只有在明确指定使用测试配置时才应用测试配置
    # 默认情况下，推理测试应该使用与训练时相同的配置
    if use_test_config and test_model_config:
        # 合并测试配置（测试配置优先）
        original_model_config = config.get("model", {})
        
        # 从数据集信息中获取state_dim（与test_training.py一致）
        state_dim = None
        if dataset_path:
            try:
                dataset_dir = Path(dataset_path).resolve()
                dataset_info = load_dataset_info(dataset_dir)
                default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
                state_dim = get_state_dim_from_info(dataset_info, default_state_dim=default_state_dim)
                print(f"  从数据集信息获取state_dim: {state_dim} (默认值: {default_state_dim})")
            except Exception as e:
                print(f"  警告: 无法从数据集获取state_dim: {e}，使用配置中的默认值")
        
        # 如果成功获取state_dim，更新配置
        if state_dim is not None:
            # 更新data配置中的state_dim
            if "data" not in config:
                config["data"] = {}
            if "robot_state" not in config["data"]:
                config["data"]["robot_state"] = {}
            config["data"]["robot_state"]["state_dim"] = state_dim
        
        config["model"] = {
            **original_model_config,
            **test_model_config,
            # 确保action_head配置正确合并
            "action_head": {
                **original_model_config.get("action_head", {}),
                **test_model_config.get("action_head", {})
            },
            # 确保vla配置正确合并
            "vla": {
                **original_model_config.get("vla", {}),
                **test_model_config.get("vla", {})
            }
        }
        
        # 创建临时配置文件
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config_file, default_flow_style=False, allow_unicode=True)
        temp_config_path = temp_config_file.name
        temp_config_file.close()
        
        return temp_config_path
    
    return config_path


def validate_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    device: Optional[str] = None,
    dataset_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    验证checkpoint文件是否正确
    
    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        device: 设备
        
    Returns:
        (is_valid, info_dict): 是否有效，以及信息字典
    """
    info = {
        "checkpoint_exists": False,
        "checkpoint_keys": [],
        "model_loaded": False,
        "normalizer_loaded": False,
        "model_type": None,
        "action_dim": None,
        "state_dim": None,
        "errors": []
    }
    
    try:
        # 检查文件是否存在
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            info["errors"].append(f"Checkpoint文件不存在: {checkpoint_path}")
            return False, info
        
        info["checkpoint_exists"] = True
        
        # 加载checkpoint（验证时使用CPU，避免占用GPU显存）
        # 注意：验证时使用CPU加载checkpoint，实际推理时会使用GPU
        validation_device = torch.device("cpu")
        
        # 在验证前清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 强制同步，确保显存释放
            torch.cuda.synchronize()
        
        checkpoint = torch.load(checkpoint_path, map_location=validation_device)
        info["checkpoint_keys"] = list(checkpoint.keys())
        
        # 验证必需的键
        required_keys = ["model_state_dict"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            info["errors"].append(f"Checkpoint缺少必需的键: {missing_keys}")
            return False, info
        
        # 尝试加载模型（使用测试配置，与test_training.py一致）
        # 注意：验证时使用CPU加载模型，避免占用GPU显存
        temp_config_path = None
        model = None
        try:
            # 获取应用了测试配置的配置文件路径（从数据集信息中获取state_dim）
            test_config_path = get_test_model_config(config_path, dataset_path=dataset_path, use_test_config=True)
            if test_config_path != config_path:
                temp_config_path = test_config_path
            
            # 验证时使用CPU加载模型，避免占用GPU显存
            # 这样可以确保验证过程不会影响后续的GPU推理
            validation_device = torch.device("cpu")
            
            # 加载模型以验证（使用torch.no_grad()和CPU节省显存）
            with torch.no_grad():
                model, normalizer = load_model_from_checkpoint(checkpoint_path, test_config_path, str(validation_device))
                info["model_loaded"] = True
                info["normalizer_loaded"] = normalizer is not None
                info["model_type"] = type(model).__name__
                
                # 获取模型维度信息
                if hasattr(model, 'action_head') and hasattr(model.action_head, 'action_dim'):
                    info["action_dim"] = model.action_head.action_dim
                elif hasattr(model, 'action_dim'):
                    info["action_dim"] = model.action_dim
                
                if hasattr(model, 'state_dim'):
                    info["state_dim"] = model.state_dim
                
        except Exception as e:
            info["errors"].append(f"模型加载失败: {str(e)}")
            return False, info
        finally:
            # 释放模型以节省显存
            if model is not None:
                # 确保模型在CPU上，然后删除
                try:
                    model = model.cpu()
                except:
                    pass
                del model
                # 清理CPU和GPU缓存
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # 强制同步，确保显存释放
                    torch.cuda.synchronize()
            # 清理临时配置文件
            if temp_config_path and Path(temp_config_path).exists():
                Path(temp_config_path).unlink()
        
        return True, info
        
    except Exception as e:
        info["errors"].append(f"验证过程中出错: {str(e)}")
        return False, info


def load_dataset_frame(
    dataset_path: str,
    frame_idx: int = 0,
    config_path: str = "config.yaml"
) -> Dict:
    """
    从本地libero_object数据集加载一帧数据
    使用lerobot的LeRobotDataset（与test_training.py一致）
    
    Args:
        dataset_path: 数据集路径（例如 "./dataset/libero_object"）
        frame_idx: 要加载的帧索引（默认0，即第一帧）
        config_path: 配置文件路径
        
    Returns:
        包含图像、指令、状态等信息的字典
    """
    if not HAS_LEROBOT:
        raise ImportError(
            "lerobot library not installed. "
            "Install with: pip install lerobot==0.3.3"
        )
    
    # 加载配置
    config = load_config(config_path)
    data_config = get_data_config(config)
    dataset_config = config.get("dataset", {})
    
    # 获取数据集信息
    dataset_dir = Path(dataset_path).resolve()
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_dir}")
    
    dataset_info = load_dataset_info(dataset_dir)
    fps = dataset_info.get("fps", 10)
    action_horizon = dataset_config.get("action_horizon", 50)
    
    # 创建delta_timestamps
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    
    # 创建LeRobotDataset（与test_training.py一致）
    dataset_name = dataset_dir.name
    root_path_str = str(dataset_dir)
    
    print(f"Loading dataset from: {dataset_dir}")
    print(f"  Dataset name (repo_id): {dataset_name}")
    print(f"  Root path: {root_path_str}")
    
    lerobot_dataset = LeRobotDataset(
        repo_id=dataset_name,
        root=root_path_str,
        delta_timestamps=delta_timestamps
    )
    
    # 检查数据集长度
    if len(lerobot_dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    
    if frame_idx >= len(lerobot_dataset):
        print(f"Warning: frame_idx {frame_idx} >= dataset length {len(lerobot_dataset)}, using frame 0")
        frame_idx = 0
    
    # 获取指定帧的数据
    print(f"Loading frame {frame_idx} from dataset...")
    sample = lerobot_dataset[frame_idx]
    
    # 打印原始sample的键，用于调试
    print(f"  Raw sample keys: {list(sample.keys())}")
    
    # 转换样本格式以匹配期望的格式
    # LeRobotDataset返回的格式：键名是 observation.images.{camera_name}
    result = {}
    
    # 处理图像（LeRobotDataset返回的格式）
    # LeRobotDataset返回的键名格式是 "observation.images.image", "observation.images.image2" 等
    images_dict = {}
    image_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
    
    if image_keys:
        print(f"  Found image keys: {image_keys}")
        for key in image_keys:
            # 提取相机名称（例如 "observation.images.image" -> "image"）
            camera_name = key.replace("observation.images.", "")
            img_tensor = sample[key]
            
            # LeRobotDataset返回的是 [C, H, W] 格式的tensor
            if isinstance(img_tensor, torch.Tensor):
                images_dict[camera_name] = img_tensor
            else:
                print(f"  Warning: Image at {key} is not a tensor: {type(img_tensor)}")
        
        if len(images_dict) > 0:
            result["images"] = images_dict
        else:
            print(f"  ✗ Warning: No valid images found in image_keys")
            result["images"] = {}
    else:
        # 尝试其他可能的格式
        if "observation" in sample and "images" in sample["observation"]:
            obs_images = sample["observation"]["images"]
            if isinstance(obs_images, dict) and len(obs_images) > 0:
                result["images"] = obs_images
            else:
                print(f"  ✗ Warning: observation.images is not a valid dict")
                result["images"] = {}
        elif "images" in sample:
            images = sample["images"]
            if isinstance(images, dict) and len(images) > 0:
                result["images"] = images
            else:
                print(f"  ✗ Warning: sample.images is not a valid dict")
                result["images"] = {}
        else:
            print(f"  ✗ Warning: No images found in sample")
            print(f"    Available keys: {list(sample.keys())}")
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
    # lerobot数据集的状态数据保存在observation.state键下
    if "observation.state" in sample:
        result["state"] = sample["observation.state"]
        print(f"  ✓ 从 'observation.state' 键提取状态数据")
    elif "observation" in sample and "state" in sample["observation"]:
        result["state"] = sample["observation"]["state"]
        print(f"  ✓ 从 'observation.state' 嵌套键提取状态数据")
    elif "state" in sample:
        result["state"] = sample["state"]
        print(f"  ✓ 从 'state' 键提取状态数据")
    else:
        print(f"  ⚠️  警告: 无法找到状态数据，可用键: {list(sample.keys())}")
    
    # 调试：打印状态数据信息
    if "state" in result:
        state_data = result["state"]
        if isinstance(state_data, torch.Tensor):
            print(f"  状态数据形状: {state_data.shape}")
            print(f"  状态数据范围: [{state_data.min().item():.4f}, {state_data.max().item():.4f}]")
            print(f"  状态数据均值: {state_data.mean().item():.4f}")
            if state_data.numel() > 0 and state_data.abs().sum().item() < 1e-6:
                print(f"  ⚠️  警告: 状态数据全为0，可能数据提取有问题")
        elif isinstance(state_data, np.ndarray):
            print(f"  状态数据形状: {state_data.shape}")
            print(f"  状态数据范围: [{state_data.min():.4f}, {state_data.max():.4f}]")
            print(f"  状态数据均值: {state_data.mean():.4f}")
            if state_data.size > 0 and np.abs(state_data).sum() < 1e-6:
                print(f"  ⚠️  警告: 状态数据全为0，可能数据提取有问题")
    
    # 处理episode和step信息
    if "episode_index" in sample:
        result["episode_id"] = sample["episode_index"]
    if "frame_index" in sample:
        result["step_id"] = sample["frame_index"]
    
    print(f"Sample keys: {list(result.keys())}")
    print(f"Sample info:")
    if "episode_id" in result:
        print(f"  Episode ID: {result['episode_id']}")
    if "step_id" in result:
        print(f"  Step ID: {result['step_id']}")
    
    return result


def test_inference_from_dataset(
    dataset_path: str = "./dataset/libero_object",
    checkpoint_path: str = "./checkpoints/best_model.pt",
    config_path: str = "config.yaml",
    frame_idx: int = 0,
    device: Optional[str] = None,
    validate_checkpoint_first: bool = True
):
    """
    从数据集读取一帧数据并运行推理，并与GT值进行比较
    
    Args:
        dataset_path: 数据集路径
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        frame_idx: 要测试的帧索引
        device: 设备（"cuda"或"cpu"），如果为None则自动选择
        validate_checkpoint_first: 是否先验证checkpoint
        
    Returns:
        Dict包含预测动作、真实动作、误差等信息
    """
    print("=" * 60)
    print("VLA模型单帧推理测试")
    print("=" * 60)
    
    # 加载配置并设置随机种子
    config = load_config(config_path)
    seed = config.get("seed", 42)
    print(f"\n设置随机种子: {seed}")
    set_seed(seed)
    print(f"  ✓ 随机种子已设置")
    
    # 从model.vlm.image_size读取图像尺寸，而不是dataset.image_size
    model_config = get_model_config(config)
    vlm_config = model_config.get("vlm", {})
    image_size = vlm_config.get("image_size", 224)
    print(f"\n图像尺寸配置: {image_size}x{image_size}")
    
    # 从dataset配置中读取image_keys，用于判断单相机/多相机
    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    if not isinstance(image_keys, list):
        raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
    print(f"图像键名配置: {image_keys} ({'单相机' if len(image_keys) == 1 else '多相机'})")
    
    result = {
        "success": False,
        "predicted_actions": None,
        "true_actions": None,
        "mae": None,
        "errors": []
    }
    
    try:
        # 0. 验证checkpoint（可选）
        if validate_checkpoint_first:
            print("\n[Step 0] 验证checkpoint文件...")
            is_valid, checkpoint_info = validate_checkpoint(checkpoint_path, config_path, device, dataset_path=dataset_path)
            if not is_valid:
                error_msg = f"Checkpoint验证失败: {checkpoint_info.get('errors', [])}"
                print(f"  ✗ {error_msg}")
                result["errors"].append(error_msg)
                return result
            
            print(f"  ✓ Checkpoint验证通过")
            print(f"    模型类型: {checkpoint_info.get('model_type', 'Unknown')}")
            print(f"    动作维度: {checkpoint_info.get('action_dim', 'Unknown')}")
            print(f"    状态维度: {checkpoint_info.get('state_dim', 'Unknown')}")
            print(f"    归一化器: {'已加载' if checkpoint_info.get('normalizer_loaded') else '未加载'}")
        
        # 1. 加载数据集中的一帧数据
        print("\n[Step 1] 加载数据集帧...")
        sample = load_dataset_frame(dataset_path, frame_idx, config_path)
        
        # 验证数据维度
        action = sample.get("action")
        if action is None:
            error_msg = "样本中没有action数据"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        # 处理action维度（可能是[action_dim]或[action_horizon, action_dim]）
        if len(action.shape) == 2:
            # [action_horizon, action_dim] - 取第一个时间步
            true_action = action[0]
        elif len(action.shape) == 1:
            # [action_dim]
            true_action = action
        else:
            error_msg = f"意外的action维度: {action.shape}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        print(f"  ✓ 数据加载成功")
        print(f"    Action维度: {true_action.shape}")
        if "state" in sample:
            state = sample["state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            print(f"    State维度: {state.shape if state is not None else None}")
        
        # 2. 加载模型checkpoint（使用测试配置，与test_training.py一致）
        print("\n[Step 2] 加载模型checkpoint...")
        # 获取应用了测试配置的配置文件路径（从数据集信息中获取state_dim）
        # 默认不使用测试配置，直接使用原始配置以确保与训练时一致
        test_config_path = get_test_model_config(config_path, dataset_path=dataset_path, use_test_config=False)
        temp_config_path = None
        if test_config_path != config_path:
            temp_config_path = test_config_path
            config = load_config(test_config_path)
            print(f"  使用测试配置: hidden_dim={config['model']['action_head'].get('hidden_dim', 'unknown')}, num_layers={config['model']['action_head'].get('num_layers', 'unknown')}")
            if "data" in config and "robot_state" in config["data"]:
                print(f"  state_dim={config['data']['robot_state'].get('state_dim', 'unknown')}")
            # 重新读取image_keys（使用测试配置）
            dataset_config = config.get("dataset", {})
            image_keys = dataset_config.get("image_keys", ["observation.images.image"])
            if not isinstance(image_keys, list):
                raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
        else:
            # 如果没有使用测试配置，确保使用原始config
            config = load_config(config_path)
            # 重新读取image_keys（使用原始配置）
            dataset_config = config.get("dataset", {})
            image_keys = dataset_config.get("image_keys", ["observation.images.image"])
            if not isinstance(image_keys, list):
                raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
        
        # 从model.vlm.image_size读取图像尺寸（优先使用测试配置中的，如果没有则使用原始配置）
        model_config = get_model_config(config)
        vlm_config = model_config.get("vlm", {})
        image_size = vlm_config.get("image_size", 224)
        
        # 从dataset配置中读取image_keys，用于判断单相机/多相机
        dataset_config = config.get("dataset", {})
        image_keys = dataset_config.get("image_keys", ["observation.images.image"])
        if not isinstance(image_keys, list):
            raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
        
        try:
            model, normalizer = load_model_from_checkpoint(checkpoint_path, test_config_path, device)
            # 确保模型处于eval模式并清理显存
            model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            # 清理临时配置文件
            if temp_config_path and Path(temp_config_path).exists():
                Path(temp_config_path).unlink()
        
        # 3. 准备输入数据
        print("\n[Step 3] 准备输入数据...")
        
        # 提取图像（可能是dict格式，每个相机一个tensor）
        images = sample.get("images", {})
        
        # 检查图像是否为空
        if not images or (isinstance(images, dict) and len(images) == 0):
            error_msg = "样本中没有图像数据"
            print(f"  ✗ {error_msg}")
            print(f"    Sample keys: {list(sample.keys())}")
            if "observation" in sample:
                print(f"    Observation keys: {list(sample['observation'].keys())}")
            result["errors"].append(error_msg)
            return result
        
        # 根据config.yaml中的image_keys处理图像
        # LeRobot返回的图像已经是0-1归一化的tensor，保持原样传递给run_inference
        images_dict = {}
        if isinstance(images, dict):
            # 从image_keys中提取相机名称，按顺序处理
            for key in image_keys:
                camera_name = key.replace("observation.images.", "")
                if camera_name in images:
                    img_tensor = images[camera_name]
                    if isinstance(img_tensor, torch.Tensor):
                        # 确保tensor格式正确 [C, H, W]
                        if img_tensor.dim() == 3:  # [C, H, W]
                            images_dict[camera_name] = img_tensor
                        elif img_tensor.dim() == 4:  # [1, C, H, W] 或 [B, C, H, W]
                            if img_tensor.shape[0] == 1:
                                images_dict[camera_name] = img_tensor.squeeze(0)  # [C, H, W]
                            else:
                                images_dict[camera_name] = img_tensor[0]  # [C, H, W]
                        else:
                            raise ValueError(f"Unexpected image tensor shape for camera {camera_name}: {img_tensor.shape}")
                    else:
                        raise ValueError(f"Unexpected image type for camera {camera_name}: {type(img_tensor)}")
        elif isinstance(images, torch.Tensor):
            # 如果images是tensor，且image_keys只有一个，则使用第一个相机名
            if len(image_keys) == 1:
                camera_name = image_keys[0].replace("observation.images.", "")
                if images.dim() == 3:  # [C, H, W]
                    images_dict[camera_name] = images
                elif images.dim() == 4:  # [1, C, H, W] 或 [B, C, H, W]
                    if images.shape[0] == 1:
                        images_dict[camera_name] = images.squeeze(0)  # [C, H, W]
                    else:
                        images_dict[camera_name] = images[0]  # [C, H, W]
                else:
                    raise ValueError(f"Unexpected images tensor shape: {images.shape}")
            else:
                raise ValueError(f"当image_keys有多个时，images必须是dict格式")
        else:
            raise ValueError(f"Unexpected images type: {type(images)}")
        
        images_input = images_dict
        
        # 提取指令
        instruction = sample.get("text", sample.get("instruction", ""))
        if not instruction:
            instruction = "Perform the task"  # 默认指令
        
        print(f"  Instruction: {instruction}")
        
        # 提取状态（如果使用）
        # lerobot数据集的状态数据保存在observation.state键下
        states = None
        if "state" in sample:
            states = sample["state"]
            if isinstance(states, torch.Tensor):
                states = states.numpy()
            print(f"  State shape: {states.shape if states is not None else None}")
            if states is not None:
                print(f"  State range: [{states.min():.4f}, {states.max():.4f}]")
                print(f"  State mean: {states.mean():.4f}")
                if states.size > 0 and np.abs(states).sum() < 1e-6:
                    print(f"  ⚠️  警告: 状态数据全为0，可能数据提取有问题")
        else:
            print(f"  ⚠️  警告: sample中没有'state'键，可用键: {list(sample.keys())}")
        
        # 4. 显示输入图像和打印chat内容（在输入模型之前）
        print("\n[Step 4] 显示输入图像和chat内容...")
        
        # 准备图像（转换为PIL Image）
        device_obj = next(model.parameters()).device
        images_pil_dict = prepare_images_input(images_input, device_obj, image_size=image_size)
        
        # 显示图像（最多2张）
        image_list = list(images_pil_dict.values())
        num_images_to_show = min(2, len(image_list))
        
        if num_images_to_show > 0:
            fig, axes = plt.subplots(1, num_images_to_show, figsize=(5 * num_images_to_show, 5))
            if num_images_to_show == 1:
                axes = [axes]
            
            for idx in range(num_images_to_show):
                img = image_list[idx]
                camera_name = list(images_pil_dict.keys())[idx]
                axes[idx].imshow(img)
                axes[idx].set_title(f"Input Image {idx+1}: {camera_name}", fontsize=10)
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
            print(f"  ✓ 已显示 {num_images_to_show} 张输入图像")
        
        # 获取并打印chat内容
        # 根据image_keys数量判断单相机/多相机
        if len(image_keys) == 1:
            # 单相机模式
            images_for_model = [list(images_pil_dict.values())[0]]
        else:
            # 多相机模式：按照image_keys的顺序排列图像
            camera_names = []
            for key in image_keys:
                camera_name = key.replace("observation.images.", "")
                if camera_name in images_pil_dict:
                    camera_names.append(camera_name)
            images_for_model = [[images_pil_dict[name] for name in camera_names]]
        
        # 准备状态（转换为tensor）
        states_tensor = None
        if states is not None:
            if not isinstance(states, torch.Tensor):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            else:
                states_tensor = states
            states_tensor = states_tensor.to(device_obj)
            if states_tensor.dim() == 1:
                states_tensor = states_tensor.unsqueeze(0)
        elif model.use_state:
            states_tensor = torch.zeros(1, model.state_dim, device=device_obj)
        
        # 调用build_qwenvl_inputs获取chat内容
        try:
            qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
                images=images_for_model,
                instructions=[instruction],
                states=states_tensor if model.use_state else None
            )
            
            # 获取处理后的文本（从processor.apply_chat_template的输出）
            # 我们需要重新构建messages来获取chat内容
            messages = []
            enhanced_instruction = instruction
            
            # 如果使用状态，需要构建增强的指令
            if model.use_state and states_tensor is not None:
                states_np = states_tensor.detach().cpu().numpy()
                if states_np.ndim == 3:
                    states_np = states_np[:, 0, :]
                state_values = states_np[0]
                state_str = ", ".join([f"{val:.3f}" for val in state_values])
                if instruction and instruction.strip():
                    enhanced_instruction = f"{instruction.strip()}\n[Robot State: {state_str}]"
                else:
                    enhanced_instruction = f"[Robot State: {state_str}]"
            
            # 构建message
            message = {
                "role": "user",
                "content": []
            }
            
            # 添加图像
            if images_for_model and len(images_for_model) > 0:
                img = images_for_model[0] if not isinstance(images_for_model[0], list) else images_for_model[0][0]
                if img is not None:
                    message["content"].append({
                        "type": "image",
                        "image": img
                    })
            
            # 添加文本指令
            instruction_clean = enhanced_instruction.strip() if enhanced_instruction else ""
            if not instruction_clean:
                instruction_clean = "Analyze the image and provide visual understanding."
            
            message["content"].append({
                "type": "text",
                "text": instruction_clean
            })
            
            messages.append(message)
            
            # 使用processor的apply_chat_template获取chat内容
            if hasattr(model.qwen_vl_interface.processor, 'apply_chat_template'):
                chat_text = model.qwen_vl_interface.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                print("\n" + "=" * 60)
                print("输入模型的Chat内容:")
                print("=" * 60)
                print(chat_text)
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("输入模型的指令内容:")
                print("=" * 60)
                print(f"Instruction: {instruction_clean}")
                print("=" * 60)
            
        except Exception as e:
            print(f"  警告: 无法获取chat内容: {e}")
            print(f"  使用原始指令: {instruction}")
        
        # 5. 运行推理
        print("\n[Step 5] 运行推理...")
        predicted_actions = run_inference(
            model=model,
            images=images_input,
            instruction=instruction,
            image_keys=image_keys,
            states=states,
            normalizer=normalizer,
            image_size=image_size
        )
        
        # 验证输出维度
        if len(predicted_actions.shape) not in [1, 2]:
            error_msg = f"意外的预测动作维度: {predicted_actions.shape}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        # 处理预测动作维度（确保是一维）
        if len(predicted_actions.shape) == 1:
            pred_action = predicted_actions
        elif len(predicted_actions.shape) == 2:
            # 如果是[T, action_dim]，取第一个时间步
            pred_action = predicted_actions[0] if predicted_actions.shape[0] > 0 else predicted_actions.flatten()
        else:
            pred_action = predicted_actions.flatten()
        
        # 验证维度匹配
        if len(pred_action) != len(true_action):
            error_msg = f"维度不匹配: 预测动作 {len(pred_action)} vs 真实动作 {len(true_action)}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        print(f"  ✓ 推理成功")
        print(f"    预测动作维度: {pred_action.shape}")
        
        # 6. 显示结果并比较
        print("\n" + "=" * 60)
        print("推理结果与GT比较")
        print("=" * 60)
        
        action_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        print(f"\n{'Action':10s} {'Predicted':>12s} {'True':>12s} {'Diff':>12s}")
        print("-" * 50)
        
        mae_per_dim = []
        for name, pred, true_val in zip(action_names[:len(pred_action)], pred_action, true_action):
            diff = abs(pred - true_val)
            mae_per_dim.append(diff)
            print(f"{name:10s} {pred:12.4f} {true_val:12.4f} {diff:12.4f}")
        
        # 计算总体MAE
        mae = np.mean(np.abs(pred_action - true_action))
        print(f"\n{'总体MAE':10s} {'':12s} {'':12s} {mae:12.4f}")
        
        result["success"] = True
        result["predicted_actions"] = pred_action
        result["true_actions"] = true_action
        result["mae"] = mae
        
        print("\n" + "=" * 60)
        print("✓ 单帧测试完成！")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        error_msg = f"测试过程中出错: {str(e)}"
        print(f"\n✗ {error_msg}")
        import traceback
        traceback.print_exc()
        result["errors"].append(error_msg)
        return result


def load_dataset_episode(
    dataset_path: str,
    episode_id: int = 0,
    config_path: str = "config.yaml"
) -> List[Dict]:
    """
    从数据集中加载一个完整的episode
    
    Args:
        dataset_path: 数据集路径
        episode_id: episode ID
        config_path: 配置文件路径
        
    Returns:
        包含该episode所有帧的列表，按step_id排序
    """
    if not HAS_LEROBOT:
        raise ImportError(
            "lerobot library not installed. "
            "Install with: pip install lerobot==0.3.3"
        )
    
    # 加载配置
    config = load_config(config_path)
    dataset_config = config.get("dataset", {})
    
    # 获取数据集信息
    dataset_dir = Path(dataset_path).resolve()
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_dir}")
    
    dataset_info = load_dataset_info(dataset_dir)
    fps = dataset_info.get("fps", 10)
    action_horizon = dataset_config.get("action_horizon", 50)
    
    # 创建delta_timestamps
    delta_timestamps = create_delta_timestamps(action_horizon, fps)
    
    # 创建LeRobotDataset（与test_training.py一致）
    dataset_name = dataset_dir.name
    root_path_str = str(dataset_dir)
    
    print(f"Loading dataset from: {dataset_dir}")
    lerobot_dataset = LeRobotDataset(
        repo_id=dataset_name,
        root=root_path_str,
        delta_timestamps=delta_timestamps
    )
    
    # 检查数据集长度
    if len(lerobot_dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    
    # 收集该episode的所有帧
    episode_frames = []
    print(f"Searching for frames in episode {episode_id}...")
    
    for idx in range(len(lerobot_dataset)):
        sample = lerobot_dataset[idx]
        
        # 获取episode_index
        sample_episode_id = sample.get("episode_index", -1)
        
        if sample_episode_id == episode_id:
            # 转换样本格式
            result = {}
            
            # 处理图像（LeRobotDataset返回的格式：键名是 observation.images.{camera_name}）
            images_dict = {}
            image_keys = [k for k in sample.keys() if k.startswith("observation.images.")]
            
            if image_keys:
                for key in image_keys:
                    # 提取相机名称（例如 "observation.images.image" -> "image"）
                    camera_name = key.replace("observation.images.", "")
                    img_tensor = sample[key]
                    
                    # LeRobotDataset返回的是 [C, H, W] 格式的tensor
                    if isinstance(img_tensor, torch.Tensor):
                        images_dict[camera_name] = img_tensor
                
                if len(images_dict) > 0:
                    result["images"] = images_dict
                else:
                    result["images"] = {}
            else:
                # 尝试其他可能的格式
                if "observation" in sample and "images" in sample["observation"]:
                    obs_images = sample["observation"]["images"]
                    if isinstance(obs_images, dict) and len(obs_images) > 0:
                        result["images"] = obs_images
                    else:
                        result["images"] = {}
                elif "images" in sample:
                    images = sample["images"]
                    if isinstance(images, dict) and len(images) > 0:
                        result["images"] = images
                    else:
                        result["images"] = {}
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
            
            # 处理episode和step信息
            result["episode_id"] = sample_episode_id
            if "frame_index" in sample:
                result["step_id"] = sample["frame_index"]
            else:
                result["step_id"] = idx  # 使用索引作为step_id
            
            episode_frames.append(result)
    
    # 按step_id排序
    episode_frames.sort(key=lambda x: x.get("step_id", 0))
    
    if not episode_frames:
        raise ValueError(f"No frames found for episode {episode_id}")
    
    print(f"Found {len(episode_frames)} frames in episode {episode_id}")
    return episode_frames


def test_episode_inference_with_3d_visualization(
    dataset_path: str = "./dataset/libero_object",
    checkpoint_path: str = "./checkpoints/best_model.pt",
    config_path: str = "config.yaml",
    episode_id: int = 0,
    device: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    测试一个episode的推理，并绘制3D轨迹对比图
    
    Args:
        dataset_path: 数据集路径
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径
        episode_id: 要测试的episode ID
        device: 设备
        output_path: 输出图像路径（如果为None，则显示但不保存）
        
    Returns:
        Dict包含预测轨迹、真实轨迹、误差等信息
    """
    print("=" * 60)
    print("VLA模型Episode推理测试（3D可视化）")
    print("=" * 60)
    
    # 加载配置并设置随机种子
    config = load_config(config_path)
    seed = config.get("seed", 42)
    print(f"\n设置随机种子: {seed}")
    set_seed(seed)
    print(f"  ✓ 随机种子已设置")
    
    # 从model.vlm.image_size读取图像尺寸，而不是dataset.image_size
    model_config = get_model_config(config)
    vlm_config = model_config.get("vlm", {})
    image_size = vlm_config.get("image_size", 224)
    print(f"\n图像尺寸配置: {image_size}x{image_size}")
    
    # 从dataset配置中读取image_keys，用于判断单相机/多相机
    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    if not isinstance(image_keys, list):
        raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
    print(f"图像键名配置: {image_keys} ({'单相机' if len(image_keys) == 1 else '多相机'})")
    
    result = {
        "success": False,
        "episode_id": episode_id,
        "num_frames": 0,
        "predicted_xyz": None,
        "true_xyz": None,
        "mae": None,
        "errors": []
    }
    
    try:
        # 1. 验证checkpoint
        print("\n[Step 1] 验证checkpoint文件...")
        is_valid, checkpoint_info = validate_checkpoint(checkpoint_path, config_path, device, dataset_path=dataset_path)
        if not is_valid:
            error_msg = f"Checkpoint验证失败: {checkpoint_info.get('errors', [])}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        print(f"  ✓ Checkpoint验证通过")
        
        # 2. 加载模型（使用测试配置，与test_training.py一致）
        print("\n[Step 2] 加载模型...")
        # 获取应用了测试配置的配置文件路径（从数据集信息中获取state_dim）
        # 默认不使用测试配置，直接使用原始配置以确保与训练时一致
        test_config_path = get_test_model_config(config_path, dataset_path=dataset_path, use_test_config=False)
        temp_config_path = None
        if test_config_path != config_path:
            temp_config_path = test_config_path
            config = load_config(test_config_path)
            print(f"  使用测试配置: hidden_dim={config['model']['action_head'].get('hidden_dim', 'unknown')}, num_layers={config['model']['action_head'].get('num_layers', 'unknown')}")
            if "data" in config and "robot_state" in config["data"]:
                print(f"  state_dim={config['data']['robot_state'].get('state_dim', 'unknown')}")
        else:
            # 如果没有使用测试配置，确保使用原始config
            config = load_config(config_path)
        
        # 从model.vlm.image_size读取图像尺寸（优先使用测试配置中的，如果没有则使用原始配置）
        model_config = get_model_config(config)
        vlm_config = model_config.get("vlm", {})
        image_size = vlm_config.get("image_size", 224)
        
        # 从dataset配置中读取image_keys，用于判断单相机/多相机
        dataset_config = config.get("dataset", {})
        image_keys = dataset_config.get("image_keys", ["observation.images.image"])
        if not isinstance(image_keys, list):
            raise ValueError(f"配置中的image_keys必须是列表，当前类型: {type(image_keys)}")
        
        try:
            model, normalizer = load_model_from_checkpoint(checkpoint_path, test_config_path, device)
            # 确保模型处于eval模式并清理显存
            model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            # 清理临时配置文件
            if temp_config_path and Path(temp_config_path).exists():
                Path(temp_config_path).unlink()
        print(f"  ✓ 模型加载成功")
        
        # 3. 加载episode数据
        print(f"\n[Step 3] 加载episode {episode_id}数据...")
        episode_frames = load_dataset_episode(dataset_path, episode_id, config_path)
        result["num_frames"] = len(episode_frames)
        print(f"  ✓ Episode包含 {len(episode_frames)} 帧")
        
        # 4. 对每一帧进行推理
        print(f"\n[Step 4] 对每一帧进行推理...")
        predicted_xyz_list = []
        true_xyz_list = []
        
        device_obj = next(model.parameters()).device
        
        for frame_idx, sample in enumerate(episode_frames):
            if frame_idx % 10 == 0:
                print(f"  处理帧 {frame_idx + 1}/{len(episode_frames)}...")
            
            # 根据config.yaml中的image_keys处理图像
            # LeRobot返回的图像已经是0-1归一化的tensor，保持原样传递给run_inference
            images = sample.get("images", {})
            images_dict = {}
            if isinstance(images, dict):
                # 从image_keys中提取相机名称，按顺序处理
                for key in image_keys:
                    camera_name = key.replace("observation.images.", "")
                    if camera_name in images:
                        img_tensor = images[camera_name]
                        if isinstance(img_tensor, torch.Tensor):
                            # 确保格式为 [C, H, W]
                            if img_tensor.dim() == 4:
                                img_tensor = img_tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
                            images_dict[camera_name] = img_tensor.cpu()  # 保持在CPU，run_inference会处理
            elif isinstance(images, torch.Tensor):
                # 如果images是tensor，且image_keys只有一个，则使用第一个相机名
                if len(image_keys) == 1:
                    camera_name = image_keys[0].replace("observation.images.", "")
                    if images.dim() == 4:
                        images = images.squeeze(0)  # [1, C, H, W] -> [C, H, W]
                    images_dict[camera_name] = images.cpu()  # 保持在CPU，run_inference会处理
                else:
                    continue  # 多相机模式但images是tensor，跳过
            else:
                continue  # 未知格式，跳过
            
            if not images_dict:
                continue  # 没有有效的图像，跳过
            
            images_input = images_dict
            
            # 提取指令
            instruction = sample.get("text", sample.get("instruction", "Perform the task"))
            
            # 提取状态
            states = None
            if "state" in sample:
                states = sample["state"]
                if isinstance(states, torch.Tensor):
                    states = states.numpy()
            
            # 显示输入图像和打印chat内容（仅对第一帧）
            if frame_idx == 0:
                print(f"\n  [帧 {frame_idx + 1}] 显示输入图像和chat内容...")
                
                # 准备图像（转换为PIL Image）
                images_pil_dict = prepare_images_input(images_input, device_obj, image_size=image_size)
                
                # 显示图像（最多2张）
                image_list = list(images_pil_dict.values())
                num_images_to_show = min(2, len(image_list))
                
                if num_images_to_show > 0:
                    fig, axes = plt.subplots(1, num_images_to_show, figsize=(5 * num_images_to_show, 5))
                    if num_images_to_show == 1:
                        axes = [axes]
                    
                    for idx in range(num_images_to_show):
                        img = image_list[idx]
                        camera_name = list(images_pil_dict.keys())[idx]
                        axes[idx].imshow(img)
                        axes[idx].set_title(f"Input Image {idx+1}: {camera_name}", fontsize=10)
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    print(f"    ✓ 已显示 {num_images_to_show} 张输入图像")
                
                # 获取并打印chat内容
                if len(image_keys) == 1:
                    images_for_model = [list(images_pil_dict.values())[0]]
                else:
                    camera_names = []
                    for key in image_keys:
                        camera_name = key.replace("observation.images.", "")
                        if camera_name in images_pil_dict:
                            camera_names.append(camera_name)
                    images_for_model = [[images_pil_dict[name] for name in camera_names]]
                
                # 准备状态（转换为tensor）
                states_tensor = None
                if states is not None:
                    if not isinstance(states, torch.Tensor):
                        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
                    else:
                        states_tensor = states
                    states_tensor = states_tensor.to(device_obj)
                    if states_tensor.dim() == 1:
                        states_tensor = states_tensor.unsqueeze(0)
                elif model.use_state:
                    states_tensor = torch.zeros(1, model.state_dim, device=device_obj)
                
                # 调用build_qwenvl_inputs获取chat内容
                try:
                    enhanced_instruction = instruction
                    if model.use_state and states_tensor is not None:
                        states_np = states_tensor.detach().cpu().numpy()
                        if states_np.ndim == 3:
                            states_np = states_np[:, 0, :]
                        state_values = states_np[0]
                        state_str = ", ".join([f"{val:.3f}" for val in state_values])
                        if instruction and instruction.strip():
                            enhanced_instruction = f"{instruction.strip()}\n[Robot State: {state_str}]"
                        else:
                            enhanced_instruction = f"[Robot State: {state_str}]"
                    
                    message = {
                        "role": "user",
                        "content": []
                    }
                    
                    if images_for_model and len(images_for_model) > 0:
                        img = images_for_model[0] if not isinstance(images_for_model[0], list) else images_for_model[0][0]
                        if img is not None:
                            message["content"].append({
                                "type": "image",
                                "image": img
                            })
                    
                    instruction_clean = enhanced_instruction.strip() if enhanced_instruction else ""
                    if not instruction_clean:
                        instruction_clean = "Analyze the image and provide visual understanding."
                    
                    message["content"].append({
                        "type": "text",
                        "text": instruction_clean
                    })
                    
                    messages = [message]
                    
                    if hasattr(model.qwen_vl_interface.processor, 'apply_chat_template'):
                        chat_text = model.qwen_vl_interface.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        print("\n    " + "=" * 56)
                        print("    输入模型的Chat内容:")
                        print("    " + "=" * 56)
                        print(f"    {chat_text}")
                        print("    " + "=" * 56)
                    else:
                        print("\n    " + "=" * 56)
                        print("    输入模型的指令内容:")
                        print("    " + "=" * 56)
                        print(f"    Instruction: {instruction_clean}")
                        print("    " + "=" * 56)
                except Exception as e:
                    print(f"    警告: 无法获取chat内容: {e}")
                    print(f"    使用原始指令: {instruction}")
            
            # 运行推理
            try:
                predicted_action = run_inference(
                    model=model,
                    images=images_input,
                    instruction=instruction,
                    image_keys=image_keys,
                    states=states,
                    normalizer=normalizer,
                    image_size=image_size
                )
                
                # 提取x, y, z（前三维）
                if len(predicted_action.shape) == 1:
                    pred_xyz = predicted_action[:3]  # [x, y, z]
                elif len(predicted_action.shape) == 2:
                    pred_xyz = predicted_action[0, :3]  # 取第一个时间步的x, y, z
                else:
                    pred_xyz = predicted_action.flatten()[:3]
                
                predicted_xyz_list.append(pred_xyz)
                
                # 提取真实动作的x, y, z
                true_action = sample.get("action")
                if isinstance(true_action, torch.Tensor):
                    true_action = true_action.numpy()
                
                if len(true_action.shape) == 1:
                    true_xyz = true_action[:3]  # [x, y, z]
                elif len(true_action.shape) == 2:
                    true_xyz = true_action[0, :3]  # 取第一个时间步的x, y, z
                else:
                    true_xyz = true_action.flatten()[:3]
                
                true_xyz_list.append(true_xyz)
                
            except Exception as e:
                print(f"  警告: 帧 {frame_idx} 推理失败: {e}")
                continue
        
        if not predicted_xyz_list or not true_xyz_list:
            error_msg = "没有成功推理的帧"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        # 转换为numpy数组
        predicted_xyz = np.array(predicted_xyz_list)  # [T, 3]
        true_xyz = np.array(true_xyz_list)  # [T, 3]
        
        # 验证维度
        if predicted_xyz.shape != true_xyz.shape:
            error_msg = f"维度不匹配: 预测 {predicted_xyz.shape} vs 真实 {true_xyz.shape}"
            print(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
            return result
        
        print(f"  ✓ 成功推理 {len(predicted_xyz_list)} 帧")
        
        # 5. 计算误差
        mae = np.mean(np.abs(predicted_xyz - true_xyz), axis=0)  # [3] - 每个维度的MAE
        mae_overall = np.mean(np.abs(predicted_xyz - true_xyz))  # 总体MAE
        
        print(f"\n  MAE (x, y, z): ({mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f})")
        print(f"  MAE (overall): {mae_overall:.4f}")
        
        result["predicted_xyz"] = predicted_xyz
        result["true_xyz"] = true_xyz
        result["mae"] = mae_overall
        
        # 6. 绘制3D轨迹对比图
        print(f"\n[Step 5] 绘制3D轨迹对比图...")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制真实轨迹（蓝色）
        ax.plot(
            true_xyz[:, 0], 
            true_xyz[:, 1], 
            true_xyz[:, 2], 
            'b-', 
            linewidth=2, 
            label='Ground Truth',
            alpha=0.7
        )
        ax.scatter(
            true_xyz[0, 0], 
            true_xyz[0, 1], 
            true_xyz[0, 2], 
            c='blue', 
            marker='o', 
            s=100, 
            label='GT Start'
        )
        ax.scatter(
            true_xyz[-1, 0], 
            true_xyz[-1, 1], 
            true_xyz[-1, 2], 
            c='blue', 
            marker='s', 
            s=100, 
            label='GT End'
        )
        
        # 绘制预测轨迹（红色）
        ax.plot(
            predicted_xyz[:, 0], 
            predicted_xyz[:, 1], 
            predicted_xyz[:, 2], 
            'r--', 
            linewidth=2, 
            label='Predicted',
            alpha=0.7
        )
        ax.scatter(
            predicted_xyz[0, 0], 
            predicted_xyz[0, 1], 
            predicted_xyz[0, 2], 
            c='red', 
            marker='o', 
            s=100, 
            label='Pred Start'
        )
        ax.scatter(
            predicted_xyz[-1, 0], 
            predicted_xyz[-1, 1], 
            predicted_xyz[-1, 2], 
            c='red', 
            marker='s', 
            s=100, 
            label='Pred End'
        )
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Episode {episode_id} - Action Trajectory Comparison (XYZ)\n'
                    f'MAE: ({mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}) | Overall: {mae_overall:.4f}', 
                    fontsize=14)
        ax.legend(loc='best')
        ax.grid(True)
        
        # 设置相等的坐标轴比例
        max_range = np.array([
            predicted_xyz[:, 0].max() - predicted_xyz[:, 0].min(),
            predicted_xyz[:, 1].max() - predicted_xyz[:, 1].min(),
            predicted_xyz[:, 2].max() - predicted_xyz[:, 2].min(),
            true_xyz[:, 0].max() - true_xyz[:, 0].min(),
            true_xyz[:, 1].max() - true_xyz[:, 1].min(),
            true_xyz[:, 2].max() - true_xyz[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (predicted_xyz[:, 0].max() + predicted_xyz[:, 0].min() + 
                true_xyz[:, 0].max() + true_xyz[:, 0].min()) / 4.0
        mid_y = (predicted_xyz[:, 1].max() + predicted_xyz[:, 1].min() + 
                true_xyz[:, 1].max() + true_xyz[:, 1].min()) / 4.0
        mid_z = (predicted_xyz[:, 2].max() + predicted_xyz[:, 2].min() + 
                true_xyz[:, 2].max() + true_xyz[:, 2].min()) / 4.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 图像已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        result["success"] = True
        
        print("\n" + "=" * 60)
        print("✓ Episode测试完成！")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        error_msg = f"测试过程中出错: {str(e)}"
        print(f"\n✗ {error_msg}")
        import traceback
        traceback.print_exc()
        result["errors"].append(error_msg)
        return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="测试VLA模型推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单帧测试
  python test/test_inference.py --mode single --dataset ./dataset/libero_object --checkpoint ./test_temp/test_checkpoint.pt
  
  # Episode测试（3D可视化）
  python test/test_inference.py --mode episode --dataset ./dataset/libero_object --checkpoint ./test_temp/test_checkpoint.pt --episode_id 0 --output episode_trajectory.png
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "episode"],
        help="测试模式: 'single' (单帧测试) 或 'episode' (episode测试)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset/libero_object",
        help="数据集路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./test_temp/test_checkpoint.pt",
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
        help="要测试的帧索引（仅用于single模式，默认0）"
    )
    parser.add_argument(
        "--episode_id",
        type=int,
        default=0,
        help="要测试的episode ID（仅用于episode模式，默认0）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（'cuda'或'cpu'），如果为None则自动选择"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图像路径（仅用于episode模式，如果为None则显示但不保存）"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="跳过checkpoint验证（仅用于single模式）"
    )
    
    args = parser.parse_args()
    
    # 加载配置并设置随机种子
    config = load_config(args.config)
    seed = config.get("seed", 42)
    print(f"设置随机种子: {seed}")
    set_seed(seed)
    print(f"  ✓ 随机种子已设置")
    
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
    
    # 根据模式运行测试
    try:
        if args.mode == "single":
            # 单帧测试
            result = test_inference_from_dataset(
            dataset_path=str(dataset_path),
            checkpoint_path=str(checkpoint_path),
            config_path=args.config,
            frame_idx=args.frame_idx,
                device=args.device,
                validate_checkpoint_first=not args.no_validate
            )
            
            if result["success"]:
                print(f"\n✓ 单帧测试成功！")
                print(f"  MAE: {result['mae']:.4f}")
            else:
                print(f"\n✗ 单帧测试失败")
                if result["errors"]:
                    for error in result["errors"]:
                        print(f"  - {error}")
                        
        elif args.mode == "episode":
            # Episode测试
            result = test_episode_inference_with_3d_visualization(
                dataset_path=str(dataset_path),
                checkpoint_path=str(checkpoint_path),
                config_path=args.config,
                episode_id=args.episode_id,
                device=args.device,
                output_path=args.output
            )
            
            if result["success"]:
                print(f"\n✓ Episode测试成功！")
                print(f"  处理的帧数: {result['num_frames']}")
                print(f"  MAE: {result['mae']:.4f}")
            else:
                print(f"\n✗ Episode测试失败")
                if result["errors"]:
                    for error in result["errors"]:
                        print(f"  - {error}")
                        
    except Exception as e:
        print(f"\n错误: 测试失败")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
