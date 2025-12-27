"""
配置加载工具
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取模型配置"""
    return config.get("model", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取训练配置"""
    return config.get("training", {})


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取数据配置"""
    return config.get("data", {})


def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取推理配置"""
    return config.get("inference", {})

