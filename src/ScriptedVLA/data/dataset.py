"""
VLA数据集处理模块
"""

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class VLADataset(Dataset):
    """
    VLA数据集
    支持多相机图像、机器人状态、文本指令和动作标签
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 224,
        transform: Optional[callable] = None,
        camera_names: Optional[List[str]] = None,
        use_state: bool = True,
        state_dim: int = 7
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据目录路径
            image_size: 图像尺寸
            transform: 图像变换（可选）
            camera_names: 相机名称列表，例如 ["global_img", "left_wrist_img"]
            use_state: 是否使用机器人状态信息
            state_dim: 机器人状态维度
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transform = transform
        self.camera_names = camera_names or ["global_img", "left_wrist_img"]
        self.use_state = use_state
        self.state_dim = state_dim
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
        print(f"Cameras: {self.camera_names}, Use state: {self.use_state}, State dim: {self.state_dim}")
    
    def _load_samples(self) -> List[Dict]:
        """
        加载数据样本
        期望的数据结构：
        - images/: 图像文件目录（每个相机一个子目录，或使用相机名称作为前缀）
        - annotations.json: 包含图像路径、文本指令、状态和动作标签的JSON文件
        
        或者：
        - 每个样本一个目录，包含各相机的图像文件和annotation.json
        """
        samples = []
        
        # 检查是否有统一的annotations.json文件
        annotations_file = self.data_path / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # 加载多相机图像路径
                    image_paths = {}
                    valid = True
                    
                    # 支持两种格式：
                    # 1. 每个相机一个路径：{"global_img": "images/global_img_0000.jpg", ...}
                    # 2. 单个image_path（向后兼容）
                    if "image_paths" in item:
                        # 多相机格式
                        for cam_name in self.camera_names:
                            if cam_name in item["image_paths"]:
                                img_path = self.data_path / item["image_paths"][cam_name]
                                if img_path.exists():
                                    image_paths[cam_name] = str(img_path)
                                else:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                    elif "image_path" in item:
                        # 单相机格式（向后兼容）
                        img_path = self.data_path / item["image_path"]
                        if img_path.exists():
                            # 如果只有一个相机名称，使用它
                            if len(self.camera_names) == 1:
                                image_paths[self.camera_names[0]] = str(img_path)
                            else:
                                # 多个相机但只有一张图，使用第一个相机名称
                                image_paths[self.camera_names[0]] = str(img_path)
                        else:
                            valid = False
                    else:
                        valid = False
                    
                    if valid:
                        sample = {
                            "image_paths": image_paths,
                            "text": item.get("text", ""),
                            "action": np.array(item["action"], dtype=np.float32),
                            # 层次化标识
                            "task_name": item.get("task_name", "default_task"),
                            "episode_id": item.get("episode_id", 0),
                            "step_id": item.get("step_id", 0)
                        }
                        
                        # 加载状态信息（如果存在）
                        if self.use_state:
                            if "state" in item:
                                sample["state"] = np.array(item["state"], dtype=np.float32)
                            else:
                                # 如果没有状态信息，创建零向量
                                sample["state"] = np.zeros(self.state_dim, dtype=np.float32)
                        
                        samples.append(sample)
        else:
            # 遍历子目录查找样本
            for sample_dir in self.data_path.iterdir():
                if sample_dir.is_dir():
                    annotation_path = sample_dir / "annotation.json"
                    
                    if annotation_path.exists():
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            ann = json.load(f)
                            
                            # 加载多相机图像
                            image_paths = {}
                            valid = True
                            
                            for cam_name in self.camera_names:
                                # 尝试多种可能的文件名
                                possible_names = [
                                    f"{cam_name}.jpg",
                                    f"{cam_name}.png",
                                    f"image_{cam_name}.jpg",
                                    "image.jpg"  # 向后兼容
                                ]
                                
                                found = False
                                for name in possible_names:
                                    img_path = sample_dir / name
                                    if img_path.exists():
                                        image_paths[cam_name] = str(img_path)
                                        found = True
                                        break
                                
                                if not found and len(self.camera_names) == 1:
                                    # 如果只有一个相机，尝试通用名称
                                    img_path = sample_dir / "image.jpg"
                                    if img_path.exists():
                                        image_paths[cam_name] = str(img_path)
                                        found = True
                                
                                if not found:
                                    valid = False
                                    break
                            
                            if valid:
                                sample = {
                                    "image_paths": image_paths,
                                    "text": ann.get("text", ""),
                                    "action": np.array(ann["action"], dtype=np.float32),
                                    # 层次化标识
                                    "task_name": ann.get("task_name", "default_task"),
                                    "episode_id": ann.get("episode_id", 0),
                                    "step_id": ann.get("step_id", 0)
                                }
                                
                                if self.use_state:
                                    if "state" in ann:
                                        sample["state"] = np.array(ann["state"], dtype=np.float32)
                                    else:
                                        sample["state"] = np.zeros(self.state_dim, dtype=np.float32)
                                
                                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            {
                "images": Dict[str, torch.Tensor],  # 多相机图像字典 {camera_name: [C, H, W]}
                "text": str,
                "state": torch.Tensor,  # [state_dim] (如果use_state=True)
                "action": torch.Tensor  # [action_dim]
            }
        """
        sample = self.samples[idx]
        
        # 加载多相机图像
        images_dict = {}
        for cam_name, img_path in sample["image_paths"].items():
            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            images_dict[cam_name] = image_tensor
        
        # 动作标签
        action = torch.from_numpy(sample["action"])
        
        result = {
            "images": images_dict,
            "text": sample["text"],
            "action": action,
            # 层次化标识
            "task_name": sample.get("task_name", "default_task"),
            "episode_id": sample.get("episode_id", 0),
            "step_id": sample.get("step_id", 0)
        }
        
        # 添加状态信息（如果使用）
        if self.use_state:
            result["state"] = torch.from_numpy(sample["state"])
        
        return result


def create_dummy_dataset(
    output_path: str,
    num_samples: int = 100,
    camera_names: Optional[List[str]] = None,
    use_state: bool = True,
    state_dim: int = 7,
    action_dim: int = 7,
    num_tasks: int = 3,
    episodes_per_task: int = 5,
    steps_per_episode: int = 10
):
    """
    创建虚拟数据集用于测试
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量（如果指定，将覆盖任务/episode/step配置）
        camera_names: 相机名称列表，例如 ["global_img", "left_wrist_img"]
        use_state: 是否生成状态信息
        state_dim: 状态维度
        action_dim: 动作维度
        num_tasks: 任务数量
        episodes_per_task: 每个任务的episode数量
        steps_per_episode: 每个episode的step数量
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认相机配置
    if camera_names is None:
        camera_names = ["global_img", "left_wrist_img"]
    
    # 创建images目录
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    samples = []
    sample_idx = 0
    
    # 如果指定了num_samples，使用简单的线性分配
    if num_samples > 0:
        # 计算每个层级的数量
        total_samples = num_tasks * episodes_per_task * steps_per_episode
        if num_samples < total_samples:
            # 如果指定的样本数少于总样本数，按比例分配
            scale = num_samples / total_samples
            num_tasks = max(1, int(num_tasks * scale))
            episodes_per_task = max(1, int(episodes_per_task * scale))
            steps_per_episode = max(1, int(steps_per_episode * scale))
    
    # 生成层次化数据
    for task_idx in range(num_tasks):
        task_name = f"task_{task_idx:03d}"
        
        for episode_idx in range(episodes_per_task):
            episode_id = episode_idx
            
            for step_idx in range(steps_per_episode):
                step_id = step_idx
                
                # 为每个相机创建图像
                image_paths = {}
                for cam_idx, cam_name in enumerate(camera_names):
                    # 为不同相机生成不同颜色的图像以便区分
                    color_offset = cam_idx * 50
                    image = Image.new(
                        'RGB',
                        (224, 224),
                        color=(
                            (sample_idx + color_offset) % 255,
                            ((sample_idx*2) + color_offset) % 255,
                            ((sample_idx*3) + color_offset) % 255
                        )
                    )
                    # 使用层次化命名
                    image_filename = f"{task_name}_ep{episode_id:03d}_step{step_id:03d}_{cam_name}.jpg"
                    image_path = images_dir / image_filename
                    image.save(image_path)
                    image_paths[cam_name] = f"images/{image_filename}"
                
                # 创建样本数据
                sample = {
                    "image_paths": image_paths,
                    "text": f"Task {task_name}, Episode {episode_id}, Step {step_id}",
                    "action": np.random.randn(action_dim).tolist(),
                    # 层次化标识
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "step_id": step_id
                }
                
                # 添加状态信息
                if use_state:
                    sample["state"] = np.random.randn(state_dim).tolist()
                
                samples.append(sample)
                sample_idx += 1
                
                # 如果达到指定的样本数，停止生成
                if num_samples > 0 and sample_idx >= num_samples:
                    break
            if num_samples > 0 and sample_idx >= num_samples:
                break
        if num_samples > 0 and sample_idx >= num_samples:
            break
    
    # 保存annotations.json
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    tasks = set(s["task_name"] for s in samples)
    episodes = {}
    for s in samples:
        task = s["task_name"]
        if task not in episodes:
            episodes[task] = set()
        episodes[task].add(s["episode_id"])
    
    print(f"Created dummy dataset with {len(samples)} samples at {output_path}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Episodes per task: ~{len(episodes[list(tasks)[0]]) if tasks else 0}")
    print(f"  Steps per episode: ~{len(samples) // (len(tasks) * len(episodes[list(tasks)[0]]) if tasks and episodes else 1)}")
    print(f"  Cameras: {camera_names}")
    print(f"  State: {'Yes' if use_state else 'No'} (dim={state_dim})")
    print(f"  Action dim: {action_dim}")


def filter_dataset_by_hierarchy(
    dataset: VLADataset,
    task_names: Optional[List[str]] = None,
    episode_ids: Optional[List[int]] = None,
    step_ids: Optional[List[int]] = None
) -> "Subset":
    """
    根据层次结构筛选数据集
    
    Args:
        dataset: VLADataset实例
        task_names: 要包含的任务名称列表，None表示不过滤
        episode_ids: 要包含的episode编号列表，None表示不过滤
        step_ids: 要包含的step编号列表，None表示不过滤
        
    Returns:
        筛选后的数据集子集
    """
    from torch.utils.data import Subset
    
    indices = []
    
    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        
        # 检查任务名称
        if task_names is not None:
            if sample.get("task_name") not in task_names:
                continue
        
        # 检查episode编号
        if episode_ids is not None:
            if sample.get("episode_id") not in episode_ids:
                continue
        
        # 检查step编号
        if step_ids is not None:
            if sample.get("step_id") not in step_ids:
                continue
        
        indices.append(idx)
    
    return Subset(dataset, indices)


def get_dataset_statistics(dataset: VLADataset) -> Dict:
    """
    获取数据集的统计信息
    
    Args:
        dataset: VLADataset实例
        
    Returns:
        包含统计信息的字典
    """
    stats = {
        "total_samples": len(dataset),
        "tasks": set(),
        "episodes": set(),
        "steps": set(),
        "task_episode_map": {},
        "episode_step_map": {}
    }
    
    for sample in dataset.samples:
        task_name = sample.get("task_name", "unknown")
        episode_id = sample.get("episode_id", -1)
        step_id = sample.get("step_id", -1)
        
        stats["tasks"].add(task_name)
        stats["episodes"].add((task_name, episode_id))
        stats["steps"].add((task_name, episode_id, step_id))
        
        if task_name not in stats["task_episode_map"]:
            stats["task_episode_map"][task_name] = set()
        stats["task_episode_map"][task_name].add(episode_id)
        
        episode_key = (task_name, episode_id)
        if episode_key not in stats["episode_step_map"]:
            stats["episode_step_map"][episode_key] = set()
        stats["episode_step_map"][episode_key].add(step_id)
    
    # 转换为列表以便JSON序列化
    stats["tasks"] = sorted(list(stats["tasks"]))
    stats["episodes"] = sorted(list(stats["episodes"]))
    stats["task_episode_map"] = {
        k: sorted(list(v)) for k, v in stats["task_episode_map"].items()
    }
    stats["episode_step_map"] = {
        f"{k[0]}_ep{k[1]}": sorted(list(v)) 
        for k, v in stats["episode_step_map"].items()
    }
    
    return stats

