"""
VLA数据集处理模块
"""

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import numpy as np
import json
import os
import h5py
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
        state_dim: int = 7,
        action_horizon: int = 4,
        pad_action_chunk: bool = True
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
            action_horizon: 动作序列长度（action chunk大小），即从当前时间步开始的未来N步动作
            pad_action_chunk: 如果当前step + action_horizon超过episode长度，是否使用最后一个动作填充
                            如果为False，则跳过这些样本
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transform = transform
        self.camera_names = camera_names or ["global_img", "left_wrist_img"]
        self.use_state = use_state
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.pad_action_chunk = pad_action_chunk
        
        # 加载数据索引
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
        print(f"Cameras: {self.camera_names}, Use state: {self.use_state}, State dim: {self.state_dim}")
        print(f"Action horizon (chunk size): {self.action_horizon}")
    
    def _load_samples(self) -> List[Dict]:
        """
        加载数据样本
        支持的数据格式：
        1. HDF5格式（推荐）：每个episode一个.h5文件
           - observation/image: [T, num_cameras, H, W, C]
           - action: [T, action_dim]
           - state: [T, state_dim] (可选)
           - instruction: [T] 字符串数组
        2. JSON格式（向后兼容）：
           - annotations.json: 包含图像路径、文本指令、状态和动作标签的JSON文件
           - 或每个样本一个目录，包含各相机的图像文件和annotation.json
        """
        samples = []
        
        # 首先检查是否有HDF5文件
        h5_files = sorted(list(self.data_path.glob("*.h5")))
        if h5_files:
            # 从HDF5文件加载
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        # 读取元数据
                        task_name = f.attrs.get('task_name', 'unknown')
                        episode_id = f.attrs.get('episode_id', 0)
                        num_steps = f.attrs.get('num_steps', 0)
                        camera_names_h5 = json.loads(f.attrs.get('camera_names', '[]'))
                        
                        # 读取数据
                        obs_group = f['observation']
                        images = obs_group['image']  # [T, num_cameras, H, W, C]
                        actions = f['action']  # [T, action_dim] - 保存完整episode的动作序列
                        instructions = f['instruction']  # [T] 字符串数组
                        
                        # 读取状态（如果存在）
                        states = None
                        if 'state' in f:
                            states = f['state']  # [T, state_dim]
                        
                        # 将actions转换为numpy数组（如果还不是）
                        actions_array = np.array(actions)  # [T, action_dim]
                        states_array = np.array(states) if states is not None else None  # [T, state_dim] or None
                        
                        # 为每个step创建样本（但跳过无法形成完整action chunk的样本）
                        for step_idx in range(num_steps):
                            # 检查是否可以形成完整的action chunk
                            if step_idx + self.action_horizon > num_steps:
                                if not self.pad_action_chunk:
                                    # 如果不需要填充，跳过这个样本
                                    continue
                                # 如果需要填充，会在__getitem__中处理
                            
                            # 提取当前step的数据
                            step_images = images[step_idx]  # [num_cameras, H, W, C]
                            step_instruction = instructions[step_idx].decode('utf-8') if isinstance(instructions[step_idx], bytes) else instructions[step_idx]
                            
                            # 将图像转换为PIL Image并保存路径（或直接存储numpy数组）
                            # 为了兼容性，我们创建一个临时路径标识
                            image_paths = {}
                            for cam_idx, cam_name in enumerate(camera_names_h5):
                                # 使用HDF5文件路径和step索引作为标识
                                image_paths[cam_name] = {
                                    'h5_file': str(h5_file),
                                    'step_idx': step_idx,
                                    'camera_idx': cam_idx,
                                    'type': 'hdf5'
                                }
                            
                            sample = {
                                "image_paths": image_paths,
                                "h5_data": {
                                    "h5_file": str(h5_file),
                                    "step_idx": step_idx,
                                    "num_steps": num_steps,
                                    "images": step_images,  # [num_cameras, H, W, C]
                                    # 保存完整的episode actions以便在__getitem__中提取chunk
                                    "episode_actions": actions_array,  # [T, action_dim]
                                    "instruction": step_instruction
                                },
                                "text": step_instruction,
                                # 向后兼容：保留单个动作（但将在__getitem__中使用action chunk）
                                "action": actions_array[step_idx],  # [action_dim] numpy array
                                "task_name": task_name,
                                "episode_id": episode_id,
                                "step_id": step_idx
                            }
                            
                            # 添加状态信息
                            if self.use_state:
                                if states_array is not None:
                                    sample["state"] = states_array[step_idx]  # [state_dim]
                                    sample["h5_data"]["episode_states"] = states_array  # [T, state_dim]
                                    sample["h5_data"]["state"] = states_array[step_idx]  # 向后兼容
                                else:
                                    # 如果没有状态信息，创建零向量
                                    sample["state"] = np.zeros(self.state_dim, dtype=np.float32)
                            
                            samples.append(sample)
                except Exception as e:
                    print(f"Warning: Failed to load {h5_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return samples
        
        # 如果没有HDF5文件，尝试加载JSON格式（向后兼容）
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
        获取一个样本，包含action chunk（动作序列）
        
        Returns:
            {
                "images": Dict[str, torch.Tensor],  # 多相机图像字典 {camera_name: [C, H, W]}
                "text": str,
                "state": torch.Tensor,  # [state_dim] (如果use_state=True)
                "action": torch.Tensor  # [action_horizon, action_dim] - action chunk
            }
        """
        sample = self.samples[idx]
        
        # 检查是否来自HDF5文件
        if "h5_data" in sample:
            # 从HDF5数据加载
            h5_data = sample["h5_data"]
            images_dict = {}
            
            # 从HDF5数据中提取图像
            step_images = h5_data["images"]  # [num_cameras, H, W, C]
            camera_names_h5 = list(sample["image_paths"].keys())
            
            for cam_idx, cam_name in enumerate(camera_names_h5):
                # 提取当前相机的图像 [H, W, C]
                image_array = step_images[cam_idx]
                
                # 转换为PIL Image并调整大小
                image = Image.fromarray(image_array)
                image = image.resize((self.image_size, self.image_size))
                
                # 转换为tensor [C, H, W]
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                
                if self.transform:
                    image_tensor = self.transform(image_tensor)
                
                images_dict[cam_name] = image_tensor
            
            # 提取action chunk：从当前step开始，提取action_horizon个动作
            step_idx = h5_data["step_idx"]
            num_steps = h5_data["num_steps"]
            episode_actions = h5_data["episode_actions"]  # [T, action_dim]
            
            # 计算action chunk的结束索引
            end_idx = step_idx + self.action_horizon
            
            if end_idx <= num_steps:
                # 可以提取完整的action chunk
                action_chunk = episode_actions[step_idx:end_idx]  # [action_horizon, action_dim]
            else:
                # 需要填充：使用最后一个动作填充
                if self.pad_action_chunk:
                    action_chunk = np.zeros((self.action_horizon, episode_actions.shape[1]), dtype=episode_actions.dtype)
                    available_steps = num_steps - step_idx
                    action_chunk[:available_steps] = episode_actions[step_idx:]
                    # 使用最后一个动作填充剩余部分
                    if available_steps > 0:
                        last_action = episode_actions[-1]
                        action_chunk[available_steps:] = last_action
                    else:
                        # 如果step_idx已经超出范围，使用最后一个动作
                        action_chunk[:] = episode_actions[-1]
                else:
                    # 这种情况不应该发生，因为已经在_load_samples中过滤了
                    raise ValueError(f"Cannot form action chunk: step_idx={step_idx}, num_steps={num_steps}, action_horizon={self.action_horizon}")
            
            # 转换为torch tensor
            action = torch.from_numpy(action_chunk)  # [action_horizon, action_dim]
            
            result = {
                "images": images_dict,
                "text": h5_data["instruction"],
                "action": action,  # [action_horizon, action_dim] - action chunk
                # 层次化标识
                "task_name": sample.get("task_name", "default_task"),
                "episode_id": sample.get("episode_id", 0),
                "step_id": sample.get("step_id", 0)
            }
            
            # 添加状态信息（如果使用）
            if self.use_state:
                if "episode_states" in h5_data:
                    # 提取当前step的状态
                    episode_states = h5_data["episode_states"]  # [T, state_dim]
                    result["state"] = torch.from_numpy(episode_states[step_idx])  # [state_dim]
                elif "state" in h5_data:
                    result["state"] = torch.from_numpy(h5_data["state"])
                else:
                    result["state"] = torch.zeros(self.state_dim, dtype=torch.float32)
            
            return result
        
        # 从文件路径加载（JSON格式，向后兼容）
        # 注意：JSON格式不支持action chunk，只能返回单个动作
        # 这种情况下，需要通过expand创建action chunk（在训练代码中处理）
        # 或者建议使用HDF5格式以获得完整的episode数据
        images_dict = {}
        for cam_name, img_path in sample["image_paths"].items():
            # 检查是否是HDF5格式的路径标识
            if isinstance(img_path, dict) and img_path.get("type") == "hdf5":
                # 这不应该发生，因为h5_data应该已经处理了
                raise ValueError("Unexpected HDF5 path format in non-h5_data sample")
            
            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            images_dict[cam_name] = image_tensor
        
        # 动作标签（JSON格式只支持单个动作，需要通过expand创建chunk）
        # 为了向后兼容，返回单个动作 [action_dim]
        # 在训练代码中需要检查并处理这种情况
        action = torch.from_numpy(sample["action"])  # [action_dim]
        
        result = {
            "images": images_dict,
            "text": sample["text"],
            "action": action,  # [action_dim] - JSON格式只支持单个动作
            # 层次化标识
            "task_name": sample.get("task_name", "default_task"),
            "episode_id": sample.get("episode_id", 0),
            "step_id": sample.get("step_id", 0)
        }
        
        # 添加状态信息（如果使用）
        if self.use_state:
            if "state" in sample:
                result["state"] = torch.from_numpy(sample["state"])
            else:
                result["state"] = torch.zeros(self.state_dim, dtype=torch.float32)
        
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
    steps_per_episode: int = 10,
    image_size: int = 224
):
    """
    创建虚拟数据集（HDF5格式）用于测试
    每个episode保存为一个HDF5文件
    
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
        image_size: 图像尺寸（默认224）
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认相机配置
    if camera_names is None:
        camera_names = ["global_img", "left_wrist_img"]
    
    num_cameras = len(camera_names)
    
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
    
    episode_count = 0
    total_episodes = num_tasks * episodes_per_task
    
    # 生成层次化数据，每个episode一个HDF5文件
    for task_idx in range(num_tasks):
        task_name = f"task_{task_idx:03d}"
        
        for episode_idx in range(episodes_per_task):
            episode_id = episode_idx
            
            # 为当前episode准备数据
            episode_images = []  # [T, num_cameras, H, W, C]
            episode_actions = []  # [T, action_dim]
            episode_states = []  # [T, state_dim] (如果use_state)
            episode_instructions = []  # [T] 字符串列表
            
            for step_idx in range(steps_per_episode):
                step_id = step_idx
                
                # 为每个相机创建图像
                step_images = []  # [num_cameras, H, W, C]
                for cam_idx, cam_name in enumerate(camera_names):
                    # 为不同相机生成不同颜色的图像以便区分
                    color_offset = cam_idx * 50
                    base_value = (episode_count * steps_per_episode + step_idx + color_offset) % 255
                    image = Image.new(
                        'RGB',
                        (image_size, image_size),
                        color=(
                            base_value,
                            (base_value * 2) % 255,
                            (base_value * 3) % 255
                        )
                    )
                    # 转换为numpy数组 [H, W, C]
                    image_array = np.array(image, dtype=np.uint8)
                    step_images.append(image_array)
                
                # 堆叠所有相机的图像 [num_cameras, H, W, C]
                step_images = np.stack(step_images, axis=0)
                episode_images.append(step_images)
                
                # 生成动作
                action = np.random.randn(action_dim).astype(np.float32)
                episode_actions.append(action)
                
                # 生成指令
                instruction = f"Task {task_name}, Episode {episode_id}, Step {step_id}"
                episode_instructions.append(instruction)
                
                # 生成状态（如果使用）
                if use_state:
                    state = np.random.randn(state_dim).astype(np.float32)
                    episode_states.append(state)
            
            # 转换为numpy数组
            # episode_images: [T, num_cameras, H, W, C]
            episode_images = np.stack(episode_images, axis=0).astype(np.uint8)
            # episode_actions: [T, action_dim]
            episode_actions = np.stack(episode_actions, axis=0).astype(np.float32)
            # episode_states: [T, state_dim] (如果use_state)
            if use_state:
                episode_states = np.stack(episode_states, axis=0).astype(np.float32)
            
            # 创建HDF5文件
            h5_filename = f"{task_name}_ep{episode_id:03d}.h5"
            h5_path = output_path / h5_filename
            
            with h5py.File(h5_path, 'w') as f:
                # 保存图像数据
                # observation/image: [T, num_cameras, H, W, C]
                obs_group = f.create_group('observation')
                obs_group.create_dataset('image', data=episode_images, compression='gzip', compression_opts=4)
                
                # 保存动作数据
                # action: [T, action_dim]
                f.create_dataset('action', data=episode_actions, compression='gzip', compression_opts=4)
                
                # 保存状态数据（如果使用）
                if use_state:
                    # state: [T, state_dim]
                    f.create_dataset('state', data=episode_states, compression='gzip', compression_opts=4)
                
                # 保存指令数据
                # instruction: 字符串数组，需要特殊处理
                instruction_dtype = h5py.special_dtype(vlen=str)
                instruction_dataset = f.create_dataset('instruction', (len(episode_instructions),), dtype=instruction_dtype)
                instruction_dataset[:] = episode_instructions
                
                # 保存元数据
                f.attrs['task_name'] = task_name
                f.attrs['episode_id'] = episode_id
                f.attrs['num_steps'] = steps_per_episode
                f.attrs['num_cameras'] = num_cameras
                f.attrs['camera_names'] = json.dumps(camera_names)
                f.attrs['action_dim'] = action_dim
                if use_state:
                    f.attrs['state_dim'] = state_dim
                f.attrs['image_size'] = image_size
            
            episode_count += 1
            
            # 如果达到指定的样本数，停止生成
            if num_samples > 0:
                total_samples_created = episode_count * steps_per_episode
                if total_samples_created >= num_samples:
                    break
        if num_samples > 0 and episode_count * steps_per_episode >= num_samples:
            break
    
    # 统计信息
    h5_files = list(output_path.glob("*.h5"))
    tasks = set()
    episodes_per_task_dict = {}
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            task_name = f.attrs.get('task_name', 'unknown')
            episode_id = f.attrs.get('episode_id', -1)
            tasks.add(task_name)
            if task_name not in episodes_per_task_dict:
                episodes_per_task_dict[task_name] = []
            episodes_per_task_dict[task_name].append(episode_id)
    
    print(f"Created dummy dataset with {len(h5_files)} episodes at {output_path}")
    print(f"  Format: HDF5 (one file per episode)")
    print(f"  Tasks: {len(tasks)}")
    if tasks:
        avg_episodes = sum(len(episodes) for episodes in episodes_per_task_dict.values()) / len(episodes_per_task_dict)
        print(f"  Episodes per task: ~{avg_episodes:.1f}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Cameras: {camera_names} ({num_cameras} cameras)")
    print(f"  State: {'Yes' if use_state else 'No'} (dim={state_dim})")
    print(f"  Action dim: {action_dim}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Total samples: {len(h5_files) * steps_per_episode}")


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

