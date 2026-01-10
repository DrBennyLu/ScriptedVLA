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

from ScriptedVLA.data.dataset import create_dummy_dataset, VLADataset
from ScriptedVLA.model.vla_qwen_groot import QwenGR00TVLAModel
from ScriptedVLA.utils import load_config, get_model_config, get_training_config, get_data_config
from train import create_optimizer, create_scheduler, train_epoch, evaluate, save_checkpoint


def create_test_datasets(temp_dir: Path, test_config: dict, data_config: dict):
    """
    创建测试用的虚拟数据集（HDF5格式）
    
    Args:
        temp_dir: 临时目录路径
        test_config: 测试配置字典
        data_config: 数据配置字典
    
    Returns:
        train_dir, val_dir: 训练和验证数据集目录
    """
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    
    # 从配置中获取参数
    dataset_config = test_config.get("dataset", {})
    num_episodes = dataset_config.get("num_episodes", 2)
    steps_per_episode = dataset_config.get("steps_per_episode", 5)
    num_tasks = dataset_config.get("num_tasks", 1)
    val_episodes = dataset_config.get("val_episodes")
    if val_episodes is None:
        val_episodes = max(1, num_episodes // 2)
    
    camera_config = data_config.get("cameras", {})
    camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
    
    robot_state_config = data_config.get("robot_state", {})
    use_state = robot_state_config.get("use_state", True)
    state_dim = robot_state_config.get("state_dim", 7)
    
    action_config = data_config.get("action", {})
    action_dim = action_config.get("action_dim", 7)
    
    image_size = data_config.get("image_size", 224)
    
    # 创建训练数据集
    print(f"创建训练数据集: {train_dir}")
    create_dummy_dataset(
        output_path=str(train_dir),
        num_samples=0,  # 使用任务/episode/step配置
        camera_names=camera_names,
        use_state=use_state,
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        episodes_per_task=num_episodes,
        steps_per_episode=steps_per_episode,
        image_size=image_size
    )
    
    # 创建验证数据集
    print(f"创建验证数据集: {val_dir}")
    create_dummy_dataset(
        output_path=str(val_dir),
        num_samples=0,
        camera_names=camera_names,
        use_state=use_state,
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        episodes_per_task=val_episodes,
        steps_per_episode=steps_per_episode,
        image_size=image_size
    )
    
    return train_dir, val_dir


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
            # 1. 创建虚拟数据集
            print("\n步骤1: 创建虚拟数据集")
            train_dir, val_dir = create_test_datasets(
                temp_dir, 
                test_config,
                data_config
            )
            
            # 2. 获取模型配置（需要在创建数据集之前，因为需要action_horizon）
            print("\n步骤2: 获取模型配置")
            test_vlm_config = test_model_config.get("vlm", {})
            test_action_head_config = test_model_config.get("action_head", {})
            test_vla_config = test_model_config.get("vla", {})
            
            # 合并测试配置和默认配置（测试配置优先）
            vlm_config = {
                **model_config.get("vlm", {}),
                **test_vlm_config
            }
            action_head_config = {
                **model_config.get("action_head", {}),
                **test_action_head_config
            }
            
            # 3. 加载数据集
            print("\n步骤3: 加载数据集")
            camera_config = data_config.get("cameras", {})
            camera_names = camera_config.get("names", ["global_img", "left_wrist_img"])
            robot_state_config = data_config.get("robot_state", {})
            use_state = robot_state_config.get("use_state", True)
            state_dim = robot_state_config.get("state_dim", 7)
            image_size = data_config.get("image_size", 224)
            
            # 从模型配置中获取action_horizon（动作序列长度）
            action_horizon = action_head_config.get("action_horizon", 4)
            
            train_dataset = VLADataset(
                data_path=str(train_dir),
                image_size=image_size,
                camera_names=camera_names,
                use_state=use_state,
                state_dim=state_dim,
                action_horizon=action_horizon,
                pad_action_chunk=True  # 如果episode末尾不够action_horizon，使用最后一个动作填充
            )
            val_dataset = VLADataset(
                data_path=str(val_dir),
                image_size=image_size,
                camera_names=camera_names,
                use_state=use_state,
                state_dim=state_dim,
                action_horizon=action_horizon,
                pad_action_chunk=True
            )
            
            print(f"  训练样本数: {len(train_dataset)}")
            print(f"  验证样本数: {len(val_dataset)}")
            
            # 4. 创建数据加载器
            print("\n步骤4: 创建数据加载器")
            batch_size = test_training_config.get("batch_size", 2)
            num_workers = test_dataloader_config.get("num_workers", 0)
            pin_memory = test_dataloader_config.get("pin_memory", False)
            shuffle = test_dataloader_config.get("shuffle", True)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            # 5. 创建模型
            print("\n步骤5: 创建模型")
            
            # 合并测试配置和默认配置（测试配置优先）
            vlm_config = {
                **model_config.get("vlm", {}),
                **test_vlm_config
            }
            action_head_config = {
                **model_config.get("action_head", {}),
                **test_action_head_config
            }
            
            model = QwenGR00TVLAModel(
                vlm_config=vlm_config,
                action_head_config=action_head_config,
                use_state=test_vla_config.get("use_state", True),
                state_dim=test_vla_config.get("state_dim", 7),
                future_action_window_size=test_vla_config.get("future_action_window_size", 3)
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"  模型已移动到设备: {device}")
            
            # 5. 创建优化器和调度器
            print("\n步骤5: 创建优化器和调度器")
            # 合并训练配置（测试配置优先）
            merged_training_config = {
                **training_config,
                **test_training_config
            }
            
            optimizer = create_optimizer(model, merged_training_config)
            num_epochs = merged_training_config.get("num_epochs", 2)
            num_training_steps = len(train_loader) * num_epochs
            scheduler = create_scheduler(optimizer, merged_training_config, num_training_steps)
            print(f"  优化器: {type(optimizer).__name__}")
            print(f"  调度器: {type(scheduler).__name__ if scheduler else 'None'}")
            
            # 7. 创建损失函数
            criterion = nn.MSELoss()
            
            # 8. 创建简单的logger
            class SimpleLogger:
                def info(self, msg):
                    print(f"  [INFO] {msg}")
            
            logger = SimpleLogger()
            
            # 9. 运行训练迭代
            print(f"\n步骤7: 运行训练迭代（{num_epochs}个epoch）")
            model.train()
            global_step = 0
            
            for epoch in range(num_epochs):
                print(f"\n  Epoch {epoch + 1}/{num_epochs}")
                
                # 使用train.py中的train_epoch函数
                # 注意：需要适配新的统一输入格式
                train_loss, global_step = train_epoch_with_unified_input(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    device,
                    merged_training_config,
                    logger,
                    epoch + 1,
                    start_step=global_step
                )
                print(f"  Epoch {epoch + 1} 训练损失: {train_loss:.4f}")
            
            # 10. 运行验证
            print("\n步骤8: 运行验证")
            # 注意：验证时模型需要处于训练模式才能计算损失
            # 因为模型的forward方法只在 self.training=True 且 actions 提供时才计算损失
            model.train()  # 设置为训练模式以计算损失
            val_loss = evaluate_with_unified_input(
                model,
                val_loader,
                criterion,
                device,
                logger
            )
            print(f"  验证损失: {val_loss:.4f}")
            
            # 11. 测试保存检查点
            print("\n步骤9: 测试保存检查点")
            checkpoint_path = temp_dir / "test_checkpoint.pt"
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=num_epochs - 1,
                loss=val_loss,
                save_path=str(checkpoint_path),
                global_step=global_step
            )
            print(f"  检查点已保存: {checkpoint_path}")
            
            # 验证检查点文件存在
            assert checkpoint_path.exists(), "检查点文件不存在"
            print("  ✓ 检查点文件验证成功")
            
            print("\n" + "=" * 60)
            print("✓ 训练流程测试成功")
            print("=" * 60)
            return True
            
        finally:
            # 清理临时目录（如果使用自动创建的临时目录）
            if temp_dir_config is None and temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"\n已清理临时目录: {temp_dir}")
    
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
    start_step=0
):
    """
    适配统一输入格式的训练epoch函数
    基于train.py中的train_epoch，但使用新的统一输入格式
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    current_step = start_step
    
    for batch_idx, batch in enumerate(dataloader):
        # 准备统一输入格式
        # 将多相机图像字典转换为列表格式
        images_dict = batch["images"]  # Dict[str, Tensor] {camera_name: [B, C, H, W]}
        batch_size = len(batch["text"])
        camera_names = sorted(images_dict.keys())  # 保持顺序一致
        
        # 转换为PIL Image列表格式：List[List[PIL.Image]]（多相机）或 List[PIL.Image]（单相机）
        # build_qwenvl_inputs期望的格式是：
        # - List[Image.Image] - 单相机
        # - List[List[Image.Image]] - 多相机（每个样本一个图像列表）
        images_list = []
        for i in range(batch_size):
            sample_images = []
            for cam_name in camera_names:
                cam_tensor = images_dict[cam_name][i].cpu()  # [C, H, W]
                # 转换为PIL Image
                # 从 [C, H, W] 转换为 [H, W, C]，然后转换为numpy，再转换为PIL
                cam_array = cam_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                cam_array = (cam_array * 255).astype(np.uint8)  # 反归一化
                cam_image = Image.fromarray(cam_array)
                sample_images.append(cam_image)
            
            # 如果是多相机，使用列表的列表；如果是单相机，直接使用列表
            if len(camera_names) > 1:
                images_list.append(sample_images)  # List[List[PIL.Image]]
            else:
                images_list.append(sample_images[0])  # List[PIL.Image]
        
        # 处理actions维度：数据集应该返回action chunk [B, action_horizon, action_dim]
        # 但如果数据源是JSON格式，可能返回单个动作 [B, action_dim]，需要扩展
        actions = batch["action"].to(device)
        if actions.dim() == 2:
            # JSON格式：只有单个动作 [B, action_dim]，需要扩展为action chunk
            # 注意：这不如真正的action chunk准确，建议使用HDF5格式
            action_horizon = model.action_model.action_horizon
            # [B, action_dim] -> [B, action_horizon, action_dim]
            actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
        elif actions.dim() == 3:
            # HDF5格式：已经是action chunk [B, action_horizon, action_dim]
            pass
        else:
            raise ValueError(f"Unexpected action shape: {actions.shape}, expected [B, action_dim] or [B, action_horizon, action_dim]")
        
        # 准备输入字典
        inputs = {
            "images": images_list,  # List[Dict[str, PIL.Image]]
            "instructions": batch["text"],  # List[str]
            "actions": actions,  # [B, action_horizon, action_dim]
        }
        
        # 添加状态信息（如果存在）
        if "state" in batch:
            inputs["states"] = batch["state"].to(device)  # [B, state_dim]
        
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


def evaluate_with_unified_input(model, dataloader, criterion, device, logger):
    """
    适配统一输入格式的评估函数
    基于train.py中的evaluate，但使用新的统一输入格式
    
    注意：模型需要处于训练模式（model.train()）才能计算损失，
    因为QwenGR00TVLAModel的forward方法只在self.training=True且actions提供时才计算损失。
    我们使用torch.no_grad()来禁用梯度计算，这样既不会更新参数，又能计算损失。
    """
    # 设置为训练模式以计算损失，但使用no_grad禁用梯度
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 准备统一输入格式
            images_dict = batch["images"]  # Dict[str, Tensor] {camera_name: [B, C, H, W]}
            batch_size = len(batch["text"])
            camera_names = sorted(images_dict.keys())
            
            # 转换为PIL Image列表格式：List[List[PIL.Image]]（多相机）或 List[PIL.Image]（单相机）
            images_list = []
            for i in range(batch_size):
                sample_images = []
                for cam_name in camera_names:
                    cam_tensor = images_dict[cam_name][i].cpu()  # [C, H, W]
                    # 转换为PIL Image
                    cam_array = cam_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                    cam_array = (cam_array * 255).astype(np.uint8)  # 反归一化
                    cam_image = Image.fromarray(cam_array)
                    sample_images.append(cam_image)
                
                # 如果是多相机，使用列表的列表；如果是单相机，直接使用列表
                if len(camera_names) > 1:
                    images_list.append(sample_images)  # List[List[PIL.Image]]
                else:
                    images_list.append(sample_images[0])  # List[PIL.Image]
            
            # 处理actions维度：数据集应该返回action chunk [B, action_horizon, action_dim]
            # 但如果数据源是JSON格式，可能返回单个动作 [B, action_dim]，需要扩展
            actions = batch["action"].to(device)
            if actions.dim() == 2:
                # JSON格式：只有单个动作 [B, action_dim]，需要扩展为action chunk
                # 注意：这不如真正的action chunk准确，建议使用HDF5格式
                action_horizon = model.action_model.action_horizon
                # [B, action_dim] -> [B, action_horizon, action_dim]
                actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)
            elif actions.dim() == 3:
                # HDF5格式：已经是action chunk [B, action_horizon, action_dim]
                pass
            else:
                raise ValueError(f"Unexpected action shape: {actions.shape}, expected [B, action_dim] or [B, action_horizon, action_dim]")
            
            inputs = {
                "images": images_list,
                "instructions": batch["text"],
                "actions": actions,  # [B, action_horizon, action_dim]
            }
            
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
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
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

