"""
配置加载工具
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def _normalize_numeric_value(value: Any) -> Any:
    """
    规范化数值类型，确保字符串形式的数字被转换为正确的数值类型
    
    Args:
        value: 待规范化的值
        
    Returns:
        规范化后的值
    """
    if isinstance(value, str):
        # 尝试转换为浮点数
        try:
            # 尝试解析为浮点数（包括科学计数法）
            float_val = float(value)
            # 如果是整数形式，返回整数
            if '.' not in value.lower() and 'e' not in value.lower():
                return int(float_val)
            return float_val
        except (ValueError, TypeError):
            # 如果无法转换，返回原值
            return value
    return value


def _normalize_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归规范化配置字典中的数值类型
    
    Args:
        config: 配置字典
        
    Returns:
        规范化后的配置字典
    """
    normalized = {}
    for key, value in config.items():
        if isinstance(value, dict):
            normalized[key] = _normalize_config_dict(value)
        elif isinstance(value, list):
            normalized[key] = [_normalize_numeric_value(item) for item in value]
        else:
            normalized[key] = _normalize_numeric_value(value)
    return normalized


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典（已规范化数值类型）
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 规范化数值类型
    config = _normalize_config_dict(config)
    
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

