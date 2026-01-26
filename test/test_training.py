"""
训练流程测试脚本
创建虚拟数据集，运行几个训练迭代循环，验证训练流程是否正常
所有配置信息从config.yaml文件中读取
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
import shutil
import numpy as np
from PIL import Image
import argparse

from src.ScriptedVLA.data.dataset import create_dummy_dataset, VLADataset
from src.ScriptedVLA.model.vla_qwen_groot import QwenGR00TVLAModel
from src.ScriptedVLA.utils import (
    load_config, 
    get_model_config, 
    get_training_config, 
    get_data_config,
    create_normalizer_from_dataset,
    Normalizer
)
from train import (
    create_optimizer, 
    create_scheduler, 
    train_epoch, 
    evaluate, 
    save_checkpoint,
    create_collate_fn,
    load_dataset_info,
    get_image_keys_from_info,
    get_state_key_from_info,
    get_state_dim_from_info,
    create_delta_timestamps,
    load_tasks_from_jsonl
)
import h5py
import json
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None


def create_episodes_stats_jsonl(dataset_dir: Path, use_state: bool = True):
    """
    为数据集创建episodes_stats.jsonl文件（用于归一化）
    支持LeRobot格式（parquet文件）和HDF5格式（episode_*.hdf5或*.h5）
    
    Args:
        dataset_dir: 数据集目录
        use_state: 是否使用状态
    """
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    
    # 检查文件是否已存在
    if episodes_stats_path.exists():
        print(f"  episodes_stats.jsonl已存在，跳过创建")
        return
    
    # 尝试使用LeRobotDataset来读取数据并创建统计信息
    if HAS_LEROBOT:
        try:
            print(f"  使用LeRobotDataset读取数据创建统计信息...")
            # 读取info.json获取配置
            info_file = meta_dir / "info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                fps = info.get("fps", 10)
                action_horizon = info.get("action_horizon", 50)
            else:
                fps = 10
                action_horizon = 50
            
            delta_timestamps = create_delta_timestamps(action_horizon, fps)
            dataset_name = dataset_dir.name
            root_path_str = str(dataset_dir)
            
            lerobot_dataset = LeRobotDataset(
                repo_id=dataset_name,
                root=root_path_str,
                delta_timestamps=delta_timestamps
            )
            
            # 从数据集读取数据并计算统计信息
            print(f"  正在从数据集计算统计信息（这可能需要一些时间）...")
            
            # 按episode分组收集数据
            episode_data = {}
            dataset_len = len(lerobot_dataset) if hasattr(lerobot_dataset, '__len__') else None
            
            # 限制处理的样本数，避免处理时间过长
            max_samples = min(5000, dataset_len if dataset_len else 5000)
            
            for idx in range(max_samples):
                try:
                    sample = lerobot_dataset[idx]
                    # 从sample中获取episode_index
                    episode_idx = sample.get("episode_index", None)
                    if episode_idx is None:
                        # 尝试从index推断
                        episode_idx = idx // 100  # 假设每个episode大约100步
                    
                    if episode_idx not in episode_data:
                        episode_data[episode_idx] = {
                            "actions": [],
                            "states": []
                        }
                    
                    # 收集action数据
                    if "action" in sample:
                        action = sample["action"]
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        # action可能是 [action_horizon, action_dim] 或 [action_dim]
                        if len(action.shape) == 2 and action.shape[0] > 1:
                            # [action_horizon, action_dim] - 取第一个时间步
                            episode_data[episode_idx]["actions"].append(action[0])
                        elif len(action.shape) == 1:
                            # [action_dim]
                            episode_data[episode_idx]["actions"].append(action)
                    
                    # 收集state数据
                    if use_state:
                        # 尝试不同的state键名
                        state = None
                        for key in ["state", "observation.state", "observation_state"]:
                            if key in sample:
                                state = sample[key]
                                break
                        
                        if state is not None:
                            if isinstance(state, torch.Tensor):
                                state = state.cpu().numpy()
                            episode_data[episode_idx]["states"].append(state)
                except Exception as e:
                    # 跳过出错的样本
                    continue
            
            # 为每个episode计算统计信息
            with open(episodes_stats_path, 'w', encoding='utf-8') as f:
                for episode_idx in sorted(episode_data.keys()):
                    data = episode_data[episode_idx]
                    
                    if not data["actions"]:
                        continue
                    
                    actions = np.array(data["actions"])  # [T, action_dim]
                    
                    stats = {
                        "action": {
                            "min": actions.min(axis=0).tolist(),
                            "max": actions.max(axis=0).tolist(),
                            "mean": actions.mean(axis=0).tolist(),
                            "std": actions.std(axis=0).tolist(),
                            "count": [len(actions)]
                        }
                    }
                    
                    if use_state and data["states"]:
                        states = np.array(data["states"])  # [T, state_dim]
                        stats["observation.state"] = {
                            "min": states.min(axis=0).tolist(),
                            "max": states.max(axis=0).tolist(),
                            "mean": states.mean(axis=0).tolist(),
                            "std": states.std(axis=0).tolist(),
                            "count": [len(states)]
                        }
                    
                    episode_stats = {
                        "episode_index": episode_idx,
                        "stats": stats
                    }
                    
                    f.write(json.dumps(episode_stats) + "\n")
            
            print(f"  ✓ episodes_stats.jsonl创建完成: {episodes_stats_path} (处理了 {len(episode_data)} 个episode)")
            return
        except Exception as e:
            print(f"  警告: 使用LeRobotDataset创建统计信息失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 回退方法：查找HDF5文件
    h5_files = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not h5_files:
        h5_files = sorted(dataset_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"  警告: 在 {dataset_dir} 中未找到HDF5文件，且无法使用LeRobotDataset")
        print(f"  请确保数据集格式正确或episodes_stats.jsonl已存在")
        return
    
    print(f"  正在从 {len(h5_files)} 个HDF5文件创建episodes_stats.jsonl...")
    
    with open(episodes_stats_path, 'w', encoding='utf-8') as f:
        for episode_idx, h5_file in enumerate(h5_files):
            try:
                with h5py.File(h5_file, 'r') as hf:
                    # 查找action键
                    actions_key = None
                    for key in hf.keys():
                        if 'action' in key.lower():
                            actions_key = key
                            break
                    
                    if actions_key is None:
                        print(f"  警告: {h5_file} 中未找到action数据")
                        continue
                    
                    actions = np.array(hf[actions_key])  # [T, action_dim]
                    
                    stats = {
                        "action": {
                            "min": actions.min(axis=0).tolist(),
                            "max": actions.max(axis=0).tolist(),
                            "mean": actions.mean(axis=0).tolist(),
                            "std": actions.std(axis=0).tolist(),
                            "count": [len(actions)]
                        }
                    }
                    
                    # 查找state键
                    state_key = None
                    if use_state:
                        for key in hf.keys():
                            if 'state' in key.lower() and 'observation' in key.lower():
                                state_key = key
                                break
                        if state_key is None:
                            if 'observation' in hf and 'state' in hf['observation']:
                                state_key = 'observation/state'
                            elif 'state' in hf:
                                state_key = 'state'
                    
                    if state_key:
                        if '/' in state_key:
                            parts = state_key.split('/')
                            state_data = hf
                            for part in parts:
                                state_data = state_data[part]
                            states = np.array(state_data)
                        else:
                            states = np.array(hf[state_key])
                        
                        stats["observation.state"] = {
                            "min": states.min(axis=0).tolist(),
                            "max": states.max(axis=0).tolist(),
                            "mean": states.mean(axis=0).tolist(),
                            "std": states.std(axis=0).tolist(),
                            "count": [len(states)]
                        }
                    
                    episode_stats = {
                        "episode_index": episode_idx,
                        "stats": stats
                    }
                    
                    f.write(json.dumps(episode_stats) + "\n")
            except Exception as e:
                print(f"  警告: 处理 {h5_file} 时出错: {e}")
                continue
    
    print(f"  ✓ episodes_stats.jsonl创建完成: {episodes_stats_path}")


def setup_dataset(dataset_path: str = "./dataset/libero_object", use_state: bool = True):
    """
    设置数据集，如果episodes_stats.jsonl不存在则创建
    
    Args:
        dataset_path: 数据集路径
        use_state: 是否使用状态
    
    Returns:
        dataset_dir: 数据集目录Path对象
    """
    dataset_dir = Path(dataset_path).resolve()
    
    if not dataset_dir.exists():
        raise ValueError(f"数据集路径不存在: {dataset_dir}")
    
    print(f"使用数据集: {dataset_dir}")
    
    # 检查并创建episodes_stats.jsonl
    meta_dir = dataset_dir / "meta"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    
    if not episodes_stats_path.exists():
        print(f"episodes_stats.jsonl不存在，正在创建...")
        create_episodes_stats_jsonl(dataset_dir, use_state=use_state)
    else:
        print(f"episodes_stats.jsonl已存在: {episodes_stats_path}")
    
    return dataset_dir


def test_training_loop(config_path: str = "config.yaml"):
    """
    测试训练循环
    
    Args:
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("训练流程测试")
    print("=" * 60)
    
    try:
        # 加载配置
        print(f"\n加载配置文件: {config_path}")
        config = load_config(config_path)
        test_config = config.get("test", {})
        data_config = get_data_config(config)
        model_config = get_model_config(config)
        training_config = get_training_config(config)
        
        # 获取测试配置
        test_model_config = test_config.get("model", {})
        test_training_config = test_config.get("training", {})
        test_dataloader_config = test_config.get("dataloader", {})
        
        # 创建临时目录
        temp_dir_config = test_config.get("temp_dir")
        if temp_dir_config is None:
            temp_dir = Path(tempfile.mkdtemp())
        else:
            temp_dir = Path(temp_dir_config)
            temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"临时目录: {temp_dir}")
        
        try:
            # 1. 设置数据集（使用dataset/libero_object）
            print("\n步骤1: 设置数据集")
            dataset_path = test_config.get("dataset_path", "./dataset/libero_object")
            robot_state_config = data_config.get("robot_state", {})
            use_state = robot_state_config.get("use_state", True)
            
            dataset_dir = setup_dataset(dataset_path, use_state=use_state)
            
            # 2. 加载数据集信息
            print("\n步骤3: 加载数据集信息")
            if not HAS_LEROBOT:
                raise ImportError(
                    "lerobot library not installed. "
                    "Install with: pip install lerobot==0.3.3"
                )
            
            dataset_info = load_dataset_info(dataset_dir)
            fps = dataset_info.get("fps", 10)
            image_keys = get_image_keys_from_info(dataset_info)
            state_key = get_state_key_from_info(dataset_info)
            default_state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
            state_dim = get_state_dim_from_info(dataset_info, default_state_dim=default_state_dim)
            
            # 从配置中获取action_horizon
            dataset_config = config.get("dataset", {})
            action_horizon = dataset_config.get("action_horizon", 50)
            image_size = dataset_config.get("image_size", 224)
            task_description_config = dataset_config.get("task_description", {})
            
            # 创建delta_timestamps
            delta_timestamps = create_delta_timestamps(action_horizon, fps)
            
            # 创建LeRobotDataset
            dataset_name = dataset_dir.name
            root_path_str = str(dataset_dir)
            
            lerobot_dataset = LeRobotDataset(
                repo_id=dataset_name,
                root=root_path_str,
                delta_timestamps=delta_timestamps
            )
            
            print(f"  数据集FPS: {fps}")
            print(f"  图像键名: {image_keys}")
            print(f"  状态键名: {state_key}")
            print(f"  状态维度: {state_dim}")
            print(f"  Action horizon: {action_horizon}")
            print(f"  数据集样本数: {len(lerobot_dataset)}")
            
            # 3.5. 创建归一化器
            print("\n步骤3.5: 创建数据归一化器")
            try:
                normalizer = create_normalizer_from_dataset(dataset_dir)
                print(f"  ✓ 归一化器创建成功")
                if normalizer.action_min is not None:
                    print(f"  Action范围: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
                if normalizer.state_min is not None:
                    print(f"  State范围: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
            except Exception as e:
                print(f"  ✗ 归一化器创建失败: {e}")
                print(f"  警告: 将不使用归一化，训练可能不稳定")
                normalizer = None
            
            # 4. 创建数据加载器
            print("\n步骤4: 创建数据加载器")
            batch_size = test_training_config.get("batch_size", 2)
            num_workers = test_dataloader_config.get("num_workers", 0)
            pin_memory = test_dataloader_config.get("pin_memory", False)
            shuffle = test_dataloader_config.get("shuffle", True)
            
            # 从配置中获取task_description配置
            dataset_config = config.get("dataset", {})
            task_description_config = dataset_config.get("task_description", {})
            use_batch_task = task_description_config.get("use_batch_task", True)
            
            custom_collate_fn = create_collate_fn(
                image_keys=image_keys,
                state_key=state_key,
                image_size=image_size,
                use_batch_task=use_batch_task,
                normalizer=normalizer
            )
            
            train_loader = DataLoader(
                lerobot_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
            
            # 验证时使用相同的数据集（不shuffle）
            val_loader = DataLoader(
                lerobot_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
            
            print(f"  训练数据加载器长度: {len(train_loader)} batches")
            print(f"  验证数据加载器长度: {len(val_loader)} batches")
            
            # 调试：检查第一个batch的状态数据
            print(f"\n[调试] 检查第一个batch的状态数据...")
            try:
                test_batch = next(iter(train_loader))
                print(f"  Batch中的键: {list(test_batch.keys())}")
                if "state" in test_batch:
                    state_data = test_batch["state"]
                    print(f"  状态数据形状: {state_data.shape}")
                    print(f"  状态数据类型: {type(state_data)}")
                    if isinstance(state_data, torch.Tensor):
                        print(f"  状态数据范围: [{state_data.min().item():.4f}, {state_data.max().item():.4f}]")
                        print(f"  状态数据均值: {state_data.mean().item():.4f}")
                        print(f"  状态数据前3个样本的前5个值:")
                        for i in range(min(3, state_data.shape[0])):
                            print(f"    样本{i}: {state_data[i, :min(5, state_data.shape[1])].tolist()}")
                        if state_data.numel() > 0 and state_data.abs().sum().item() < 1e-6:
                            print(f"  ⚠️  警告: 状态数据全为0，可能数据提取有问题")
                else:
                    print(f"  ⚠️  警告: batch中没有'state'键！")
                    print(f"  这可能导致模型无法使用状态信息。")
                    print(f"  配置的状态键名: {state_key}")
            except Exception as e:
                print(f"  ⚠️  无法检查第一个batch: {e}")
                import traceback
                traceback.print_exc()
            
            # 5. 创建模型
            print("\n步骤5: 创建模型")
            # 合并模型配置（测试配置优先）
            test_vlm_config = test_model_config.get("vlm", {})
            test_action_head_config = test_model_config.get("action_head", {})
            test_vla_config = test_model_config.get("vla", {})
            
            merged_vlm_config = {
                **model_config.get("vlm", {}),
                **test_vlm_config
            }
            merged_action_head_config = {
                **model_config.get("action_head", {}),
                **test_action_head_config
            }
            merged_vla_config = {
                **model_config.get("vla", {}),
                **test_vla_config
            }
            
            # 设置action_horizon和future_action_window_size
            merged_action_head_config["action_horizon"] = action_horizon
            future_action_window_size = merged_vla_config.get("future_action_window_size", action_horizon - 1)
            
            # 使用从数据集获取的state_dim和use_state
            use_state = merged_vla_config.get("use_state", True)
            state_dim = merged_vla_config.get("state_dim", state_dim)
            
            model = QwenGR00TVLAModel(
                vlm_config=merged_vlm_config,
                action_head_config=merged_action_head_config,
                camera_names=None,  # 使用默认值
                use_state=use_state,
                state_dim=state_dim,
                future_action_window_size=future_action_window_size
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"  模型已移动到设备: {device}")
            
            # 6. 创建优化器和调度器
            # 合并训练配置（测试配置优先）
            merged_training_config = {
                **training_config,
                **test_training_config
            }
            
            # 从配置中读取训练参数（直接使用，无需判断）
            max_steps = merged_training_config.get("max_steps")
            save_steps = merged_training_config.get("save_steps")
            eval_steps = merged_training_config.get("eval_steps")
            logging_steps = merged_training_config.get("logging_steps")
            save_dir = merged_training_config.get("save_dir") or training_config.get("save_dir", "./checkpoints")
            
            print(f"  训练参数:")
            print(f"    max_steps: {max_steps}")
            print(f"    save_steps: {save_steps}")
            print(f"    eval_steps: {eval_steps}")
            print(f"    logging_steps: {logging_steps}")
            print(f"    save_dir: {save_dir}")
            
            optimizer = create_optimizer(model, merged_training_config)
            scheduler = create_scheduler(optimizer, merged_training_config, max_steps)
            print(f"  优化器: {type(optimizer).__name__}")
            print(f"  调度器: {type(scheduler).__name__ if scheduler else 'None'}")
            
            # 7. 创建损失函数
            criterion = nn.MSELoss()
            
            # 8. 创建简单的logger
            class SimpleLogger:
                def info(self, msg):
                    print(f"  [INFO] {msg}")
            
            logger = SimpleLogger()
            
            # 9. 运行训练迭代（使用step-based训练，参考train.py）
            print(f"\n步骤7: 开始训练（最大步数: {max_steps}，每{save_steps}步保存一次）")
            model.train()
            global_step = 0
            
            # 创建保存目录
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            
            progress_bar = tqdm(range(max_steps), desc="Training", unit="step")
            
            for step in progress_bar:
                # 获取一个batch
                try:
                    batch = next(iter(train_loader))
                except StopIteration:
                    # 如果数据加载器耗尽，重新创建
                    train_loader = DataLoader(
                        lerobot_dataset,
                        batch_size=merged_training_config.get("batch_size", 2),
                        shuffle=True,
                        num_workers=test_dataloader_config.get("num_workers", 0),
                        pin_memory=test_dataloader_config.get("pin_memory", False),
                        collate_fn=custom_collate_fn
                    )
                    batch = next(iter(train_loader))
                
                # 准备输入
                images_list = batch["images"]
                texts = batch["text"]
                actions = batch["action"].to(device)
                
                # 处理actions维度
                if actions.dim() == 2:
                    action_horizon = merged_action_head_config.get("action_horizon", 50)
                    actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
                
                inputs = {
                    "images": images_list,
                    "instructions": texts,
                    "actions": actions,
                }
                
                if "state" in batch:
                    inputs["states"] = batch["state"].to(device)
                
                # 前向传播
                outputs = model(inputs=inputs)
                
                # 获取损失
                loss = outputs.get("action_loss") or outputs.get("loss")
                if loss is None:
                    raise ValueError("模型输出中没有损失值")
                
                # 梯度累积
                loss = loss / merged_training_config.get("gradient_accumulation_steps", 1)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪和优化器步进
                if (step + 1) % merged_training_config.get("gradient_accumulation_steps", 1) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        merged_training_config.get("max_grad_norm", 1.0)
                    )
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # 日志记录
                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step}/{max_steps}: Loss = {loss.item() * merged_training_config.get('gradient_accumulation_steps', 1):.4f}, "
                        f"LR = {optimizer.param_groups[0]['lr']:.2e}"
                    )
                
                # 评估
                if global_step > 0 and global_step % eval_steps == 0:
                    model.train()  # 评估时需要训练模式以计算损失
                    val_loss = evaluate_with_unified_input(
                        model,
                        val_loader,
                        criterion,
                        device,
                        logger,
                        normalizer=normalizer
                    )
                    logger.info(f"Step {global_step}: 验证损失 = {val_loss:.4f}")
                    model.train()  # 恢复训练模式
                
                # 保存检查点
                if global_step > 0 and global_step % save_steps == 0:
                    checkpoint_path = save_dir_path / f"checkpoint_step_{global_step}.pt"
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch=0,  # step-based训练不使用epoch
                        loss=loss.item(),
                        save_path=str(checkpoint_path),
                        global_step=global_step,
                        normalizer=normalizer
                    )
                    logger.info(f"检查点已保存: {checkpoint_path}")
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": global_step
                })
                
                # 达到最大步数时停止
                if global_step >= max_steps:
                    break
            
            progress_bar.close()
            
            # 10. 最终保存检查点
            print("\n步骤8: 保存最终检查点")
            final_checkpoint_path = save_dir_path / "test_checkpoint_final.pt"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=0,
                loss=loss.item() if 'loss' in locals() else 0.0,
                save_path=str(final_checkpoint_path),
                global_step=global_step,
                normalizer=normalizer
            )
            print(f"  最终检查点已保存: {final_checkpoint_path}")
            
            print("\n" + "=" * 60)
            print("✓ 训练流程测试成功")
            print("=" * 60)
            return True
            
        finally:
            # 清理临时目录（如果使用自动创建的临时目录）
            print(f"暂时不清理临时目录: {temp_dir}")
            # if temp_dir_config is None and temp_dir.exists():
            #     shutil.rmtree(temp_dir)
            #     print(f"\n已清理临时目录: {temp_dir}")
    
    except Exception as e:
        print(f"\n✗ 训练流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_epoch_with_unified_input(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    config,
    logger,
    epoch,
    start_step=0,
    normalizer=None
):
    """
    适配统一输入格式的训练epoch函数
    基于train.py中的train_epoch，使用create_collate_fn处理后的batch格式
    注意：create_collate_fn已经处理了归一化和图像格式转换
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    current_step = start_step
    
    for batch_idx, batch in enumerate(dataloader):
        # create_collate_fn已经返回处理好的格式：
        # - images: List[PIL.Image] 或 List[List[PIL.Image]]（已转换）
        # - action: [B, action_horizon, action_dim]（已归一化）
        # - state: [B, state_dim]（已归一化，如果存在）
        # - text: List[str]
        images_list = batch["images"]
        texts = batch["text"]
        actions = batch["action"].to(device)  # 已经是 [B, action_horizon, action_dim]，已归一化
        
        # 处理actions维度：确保是3维
        if actions.dim() == 2:
            # 如果只有 [B, action_dim]，需要扩展为action chunk
            if hasattr(model, 'action_head') and hasattr(model.action_head, 'action_horizon'):
                action_horizon = model.action_head.action_horizon
            elif hasattr(model, 'action_model') and hasattr(model.action_model, 'action_horizon'):
                action_horizon = model.action_model.action_horizon
            else:
                action_horizon = 4
            actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
        elif actions.dim() == 3:
            # 已经是action chunk [B, action_horizon, action_dim]
            pass
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}, expected [B, action_dim] or [B, action_horizon, action_dim]")
        
        # 准备输入字典
        inputs = {
            "images": images_list,
            "instructions": texts,
            "actions": actions,
        }
        
        # 添加状态信息（如果存在，已归一化）
        if "state" in batch:
            inputs["states"] = batch["state"].to(device)
        
        # 前向传播（使用统一输入格式）
        outputs = model(inputs=inputs)
        
        # 获取损失
        if "action_loss" in outputs:
            loss = outputs["action_loss"]
        elif "loss" in outputs:
            loss = outputs["loss"]
        else:
            # 如果没有损失，需要计算（这种情况不应该发生）
            raise ValueError("模型输出中没有损失值")
        
        # 梯度累积
        loss = loss / config.get("gradient_accumulation_steps", 1)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪和优化器步进
        if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get("max_grad_norm", 1.0)
            )
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            current_step += 1
        
        total_loss += loss.item() * config.get("gradient_accumulation_steps", 1)
        num_batches += 1
        
        # 日志记录
        if current_step % config.get("logging_steps", 10) == 0:
            logger.info(
                f"Step {current_step}: Loss = {loss.item() * config.get('gradient_accumulation_steps', 1):.4f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, current_step


def evaluate_with_unified_input(model, dataloader, criterion, device, logger, normalizer=None, max_eval_batches=10):
    """
    适配统一输入格式的评估函数
    基于train.py中的evaluate，使用create_collate_fn处理后的batch格式
    注意：create_collate_fn已经处理了归一化和图像格式转换
    
    注意：模型需要处于训练模式（model.train()）才能计算损失，
    因为QwenGR00TVLAModel的forward方法只在self.training=True且actions提供时才计算损失。
    我们使用torch.no_grad()来禁用梯度计算，这样既不会更新参数，又能计算损失。
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数（未使用，保持兼容性）
        device: 设备
        logger: 日志记录器
        normalizer: 归一化器（未使用，保持兼容性）
        max_eval_batches: 最大评估批次数量，用于限制评估时间（默认10）
    """
    # 设置为训练模式以计算损失，但使用no_grad禁用梯度
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 限制评估批次数量，避免评估时间过长
    total_batches = len(dataloader)
    eval_batches = min(max_eval_batches, total_batches)
    
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        pbar = tqdm(
            enumerate(dataloader),
            total=eval_batches,
            desc="评估中",
            unit="batch",
            leave=False
        )
        
        for batch_idx, batch in pbar:
            # 限制评估批次数量
            if batch_idx >= eval_batches:
                break
            
            # create_collate_fn已经返回处理好的格式：
            # - images: List[PIL.Image] 或 List[List[PIL.Image]]（已转换）
            # - action: [B, action_horizon, action_dim]（已归一化）
            # - state: [B, state_dim]（已归一化，如果存在）
            # - text: List[str]
            images_list = batch["images"]
            texts = batch["text"]
            actions = batch["action"].to(device)  # 已经是 [B, action_horizon, action_dim]，已归一化
            
            # 处理actions维度：确保是3维
            if actions.dim() == 2:
                # 如果只有 [B, action_dim]，需要扩展为action chunk
                if hasattr(model, 'action_head') and hasattr(model.action_head, 'action_horizon'):
                    action_horizon = model.action_head.action_horizon
                elif hasattr(model, 'action_model') and hasattr(model.action_model, 'action_horizon'):
                    action_horizon = model.action_model.action_horizon
                else:
                    action_horizon = 4
                actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
            elif actions.dim() == 3:
                # 已经是action chunk [B, action_horizon, action_dim]
                pass
            else:
                raise ValueError(f"Unexpected action shape: {actions.shape}, expected [B, action_dim] or [B, action_horizon, action_dim]")
            
            inputs = {
                "images": images_list,
                "instructions": texts,
                "actions": actions,  # [B, action_horizon, action_dim]
            }
            
            # 添加状态信息（如果存在，已归一化）
            if "state" in batch:
                inputs["states"] = batch["state"].to(device)
            
            # 前向传播
            outputs = model(inputs=inputs)
            
            # 获取损失
            if "action_loss" in outputs:
                loss = outputs["action_loss"]
            elif "loss" in outputs:
                loss = outputs["loss"]
            else:
                # 如果没有损失，需要计算（这种情况不应该发生）
                raise ValueError("模型输出中没有损失值")
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前损失
            current_avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{current_avg_loss:.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f} (评估了 {num_batches}/{total_batches} 个批次)")
    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VLA Training Loop")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    success = test_training_loop(config_path=args.config)
    exit(0 if success else 1)

