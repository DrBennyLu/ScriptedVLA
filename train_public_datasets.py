"""
在公开数据集（LIBERO/ACT）上训练VLA模型的示例脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
from pathlib import Path
from tqdm import tqdm
import os

from src.model import VLAModel
from src.data import (
    download_dataset,
    LIBERODataset,
    ACTDataset
)
from src.utils import (
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    setup_logger,
    log_model_info
)


def create_optimizer(model, config):
    """创建优化器"""
    opt_config = config.get("optimizer", {})
    opt_type = opt_config.get("type", "adamw")
    
    if opt_type.lower() == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=opt_config.get("betas", [0.9, 0.999]),
            eps=opt_config.get("eps", 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    sched_config = config.get("scheduler", {})
    sched_type = sched_config.get("type", "cosine")
    warmup_ratio = sched_config.get("warmup_ratio", 0.1)
    min_lr_ratio = sched_config.get("min_lr_ratio", 0.01)
    
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    if sched_type == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=min_lr_ratio,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=config.get("learning_rate", 1e-4) * min_lr_ratio
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        return None


def train_epoch(
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
    训练一个epoch
    
    Args:
        start_step: 起始全局步数
        
    Returns:
        (avg_loss, last_step): 平均损失和最后一个步数
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    current_step = start_step
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        texts = batch["text"]
        actions = batch["action"].to(device)
        
        outputs = model(images, texts)
        pred_actions = outputs["actions"]
        
        loss = criterion(pred_actions, actions)
        loss = loss / config.get("gradient_accumulation_steps", 1)
        
        loss.backward()
        
        if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get("max_grad_norm", 1.0)
            )
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            # 只有在实际优化器步进时才增加步数
            current_step += 1
        
        total_loss += loss.item() * config.get("gradient_accumulation_steps", 1)
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item() * config.get('gradient_accumulation_steps', 1):.4f}"})
        
        # 日志记录（使用当前步数）
        if current_step % config.get("logging_steps", 100) == 0:
            logger.info(
                f"Step {current_step}: Loss = {loss.item() * config.get('gradient_accumulation_steps', 1):.4f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.2e}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, current_step


def evaluate(model, dataloader, criterion, device, logger):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            texts = batch["text"]
            actions = batch["action"].to(device)
            
            outputs = model(images, texts)
            pred_actions = outputs["actions"]
            
            loss = criterion(pred_actions, actions)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, global_step=None):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if global_step is not None:
        checkpoint["global_step"] = global_step
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train VLA Model on Public Datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["libero", "act"],
        help="Dataset type: libero or act"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (for LIBERO: libero_spatial, libero_object, etc.)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset if not exists"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = get_data_config(config)
    
    # 设置日志
    logger = setup_logger(
        log_dir=config.get("logging", {}).get("log_dir", "./logs")
    )
    logger.info("Starting VLA training on public dataset...")
    logger.info(f"Dataset: {args.dataset}")
    
    # 下载数据集（如果需要）
    if args.download:
        logger.info(f"Downloading {args.dataset} dataset...")
        dataset_path = download_dataset(
            dataset_type=args.dataset,
            dataset_name=args.dataset_name,
            output_dir=None
        )
        logger.info(f"Dataset downloaded to: {dataset_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建模型
    model = VLAModel(
        vlm_config=model_config.get("vlm", {}),
        action_head_config=model_config.get("action_head", {}),
        use_cross_attention=model_config.get("vla", {}).get("use_cross_attention", True),
        cross_attention_layers=model_config.get("vla", {}).get("cross_attention_layers", 3)
    )
    model = model.to(device)
    log_model_info(logger, model)
    
    # 创建数据集
    image_size = data_config.get("image_size", model_config.get("vlm", {}).get("image_size", 224))
    
    if args.dataset == "libero":
        libero_config = data_config.get("libero", {})
        dataset_name = args.dataset_name or libero_config.get("dataset_name", "libero_spatial")
        dataset_path = libero_config.get("dataset_path", f"./data/libero_{dataset_name}")
        
        logger.info(f"Loading LIBERO dataset: {dataset_name}")
        full_dataset = LIBERODataset(
            dataset_path=dataset_path,
            task_names=libero_config.get("task_names", None),
            image_size=image_size,
            max_episode_length=libero_config.get("max_episode_length", 100)
        )
    elif args.dataset == "act":
        act_config = data_config.get("act", {})
        dataset_path = act_config.get("dataset_path", "./data/act")
        
        logger.info(f"Loading ACT dataset from: {dataset_path}")
        full_dataset = ACTDataset(
            dataset_path=dataset_path,
            image_size=image_size,
            chunk_size=act_config.get("chunk_size", 1)
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 分割训练集和验证集
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        prefetch_factor=data_config.get("prefetch_factor", 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 8),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, training_config)
    num_training_steps = len(train_loader) * training_config.get("num_epochs", 100)
    scheduler = create_scheduler(optimizer, training_config, num_training_steps)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 恢复训练（如果有）
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        start_step = checkpoint.get("global_step", start_epoch * len(train_loader))
        best_val_loss = checkpoint.get("loss", float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # 创建保存目录
    save_dir = Path(training_config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    num_epochs = training_config.get("num_epochs", 100)
    eval_steps = training_config.get("eval_steps", 500)
    save_steps = training_config.get("save_steps", 1000)
    
    # 使用恢复的步数或计算起始步数
    global_step = start_step
    last_eval_step = start_step - eval_steps  # 确保第一次评估在正确时机
    last_save_step = start_step - save_steps  # 确保第一次保存在正确时机
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, training_config, logger, epoch+1, start_step=global_step
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 验证（基于步数，而不是轮数）
        if (global_step - last_eval_step >= eval_steps or global_step % eval_steps == 0) or epoch == num_epochs - 1:
            val_loss = evaluate(model, val_loader, criterion, device, logger)
            last_eval_step = global_step
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_dir / f"best_model_{args.dataset}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, best_model_path, global_step=global_step
                )
                logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # 定期保存检查点（基于步数，而不是轮数）
        if global_step - last_save_step >= save_steps or global_step % save_steps == 0:
            checkpoint_path = save_dir / f"checkpoint_{args.dataset}_step_{global_step}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, checkpoint_path, global_step=global_step
            )
            last_save_step = global_step
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

