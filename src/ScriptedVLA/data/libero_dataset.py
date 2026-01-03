"""
LIBERO数据集适配器
将LIBERO数据集转换为VLA训练格式
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import json


class LIBERODataset(Dataset):
    """
    LIBERO数据集适配器
    将LIBERO任务数据转换为VLA训练格式
    """
    
    def __init__(
        self,
        dataset_path: str,
        task_names: Optional[List[str]] = None,
        image_size: int = 224,
        max_episode_length: int = 100,
        transform: Optional[callable] = None
    ):
        """
        初始化LIBERO数据集
        
        Args:
            dataset_path: LIBERO数据集路径
            task_names: 要使用的任务名称列表，None表示使用所有任务
            image_size: 图像尺寸
            max_episode_length: 最大episode长度
            transform: 图像变换
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.max_episode_length = max_episode_length
        self.transform = transform
        
        # 加载LIBERO任务
        try:
            from libero.libero import benchmark
        except ImportError:
            raise ImportError(
                "LIBERO包未安装。请运行: pip install libero"
            )
        
        # 获取数据集名称（从路径推断或使用默认值）
        dataset_name = self._infer_dataset_name()
        
        benchmark_dict = benchmark.get_benchmark_dict()
        if dataset_name not in benchmark_dict:
            # 尝试使用libero_spatial作为默认
            dataset_name = "libero_spatial"
            if dataset_name not in benchmark_dict:
                raise ValueError(f"无法找到LIBERO数据集: {dataset_name}")
        
        task_suite = benchmark_dict[dataset_name]()
        all_tasks = task_suite.get_tasks()
        
        # 过滤任务
        if task_names:
            self.tasks = [task for task in all_tasks if task.name in task_names]
        else:
            self.tasks = all_tasks
        
        # 加载数据样本
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from LIBERO dataset")
        print(f"Tasks: {[task.name for task in self.tasks]}")
    
    def _infer_dataset_name(self) -> str:
        """从路径推断数据集名称"""
        path_str = str(self.dataset_path)
        if "spatial" in path_str:
            return "libero_spatial"
        elif "object" in path_str:
            return "libero_object"
        elif "goal" in path_str:
            return "libero_goal"
        elif "100" in path_str:
            return "libero_100"
        else:
            return "libero_spatial"  # 默认
    
    def _load_samples(self) -> List[Dict]:
        """
        加载LIBERO数据样本
        LIBERO数据集通常包含演示数据（demonstrations）
        """
        samples = []
        
        # LIBERO数据集结构：
        # 每个任务包含多个演示（demonstrations）
        # 每个演示包含图像序列和动作序列
        
        for task in self.tasks:
            # 获取任务的演示数据
            # 注意：LIBERO的实际数据加载可能需要特定的API
            # 这里提供一个通用的适配接口
            
            # 尝试从文件系统加载数据
            task_data_path = self.dataset_path / task.name
            
            if task_data_path.exists():
                # 如果数据已经下载到本地
                demos = self._load_task_demos(task_data_path)
                samples.extend(demos)
            else:
                # 如果数据需要从LIBERO API加载
                # 这里创建一个占位符，实际使用时需要根据LIBERO的API调整
                print(f"Warning: Task {task.name} data not found at {task_data_path}")
                print("LIBERO数据可能需要通过其API加载，请参考LIBERO文档")
        
        return samples
    
    def _load_task_demos(self, task_path: Path) -> List[Dict]:
        """从文件系统加载任务演示数据"""
        samples = []
        
        # 查找演示文件
        demo_files = list(task_path.glob("demo_*.npz")) + list(task_path.glob("*.h5"))
        
        for demo_file in demo_files:
            try:
                if demo_file.suffix == ".npz":
                    data = np.load(demo_file, allow_pickle=True)
                    
                    # 提取图像和动作
                    # LIBERO数据格式可能因版本而异，这里提供通用处理
                    images = data.get("images", data.get("obs", None))
                    actions = data.get("actions", data.get("action", None))
                    
                    if images is not None and actions is not None:
                        # 将序列数据转换为样本
                        for i in range(len(images) - 1):
                            samples.append({
                                "image": images[i],
                                "text": "",  # LIBERO可能没有文本指令
                                "action": actions[i],
                                "task_name": task_path.name
                            })
                
            except Exception as e:
                print(f"Warning: Failed to load {demo_file}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个样本"""
        sample = self.samples[idx]
        
        # 处理图像
        if isinstance(sample["image"], np.ndarray):
            image = Image.fromarray(sample["image"])
        else:
            image = sample["image"]
        
        # 调整大小
        image = image.resize((self.image_size, self.image_size))
        
        # 转换为张量
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        # 应用变换
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # 处理动作
        action = torch.from_numpy(np.array(sample["action"], dtype=np.float32))
        
        # 确保动作维度正确（如果是高维动作，可能需要处理）
        if len(action.shape) > 1:
            action = action.flatten()
        
        return {
            "image": image_tensor,
            "text": sample.get("text", ""),
            "action": action,
            "task_name": sample.get("task_name", "")
        }


def create_libero_dataset_from_config(config: Dict) -> LIBERODataset:
    """
    从配置创建LIBERO数据集
    
    Args:
        config: 配置字典，包含dataset_path等参数
        
    Returns:
        LIBERODataset实例
    """
    return LIBERODataset(
        dataset_path=config.get("dataset_path", "./data/libero"),
        task_names=config.get("task_names", None),
        image_size=config.get("image_size", 224),
        max_episode_length=config.get("max_episode_length", 100)
    )

