"""
公共 CLI 参数：--config, --device, --seed, --checkpoint, --dataset 等。
各脚本通过 add_common_args(parser) 复用，parse_common_args(args) 提取为 CommonArgs。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CommonArgs:
    """公共命令行参数，优先级高于 config.yaml。"""

    config_path: str = "config.yaml"
    device: Optional[str] = None
    seed: Optional[int] = 42
    checkpoint: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    dataset_path: Optional[str] = None


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_config: bool = True,
    include_device: bool = True,
    include_seed: bool = True,
    include_checkpoint: bool = False,
    include_checkpoint_dir: bool = False,
    include_dataset: bool = False,
) -> None:
    """
    向 parser 添加公共参数。按需开关各参数。
    """
    if include_config:
        parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="配置文件路径",
        )
    if include_device:
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="cuda 或 cpu，未指定时自动选择",
        )
    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="随机种子",
        )
    if include_checkpoint:
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="checkpoint 文件或目录路径",
        )
    if include_checkpoint_dir:
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="checkpoint 目录路径",
        )
    if include_dataset:
        parser.add_argument(
            "--dataset",
            "--dataset_path",
            type=str,
            dest="dataset_path",
            default=None,
            help="数据集路径",
        )


def parse_common_args(args: argparse.Namespace) -> CommonArgs:
    """从已解析的 args 中提取公共参数。"""
    return CommonArgs(
        config_path=getattr(args, "config", "config.yaml"),
        device=getattr(args, "device", None),
        seed=getattr(args, "seed", 42),
        checkpoint=getattr(args, "checkpoint", None),
        checkpoint_dir=getattr(args, "checkpoint_dir", None),
        dataset_path=getattr(args, "dataset_path", None),
    )
