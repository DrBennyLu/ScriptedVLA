"""
配置加载工具
"""

import os
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_offline_mode_if_needed(config_path: Optional[str] = None) -> None:
    """
    若配置了 model.vlm.cache_dir 或 model.vlm.local_model_path，则设置离线环境变量。
    必须在 import transformers 之前调用；各脚本应在入口最顶部调用此函数。
    config_path 未指定时从 sys.argv 解析 --config。
    """
    if config_path is None:
        config_path = _get_config_path_from_argv()
    path = Path(config_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vlm = cfg.get("model", {}) or {}
    vlm = vlm.get("vlm", {}) or {}
    if vlm.get("cache_dir") or vlm.get("local_model_path"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


def _get_config_path_from_argv() -> str:
    """从 sys.argv 中解析 --config 参数，用于脚本启动时的早期调用。"""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "config.yaml"


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


@dataclass
class ScriptConfig:
    """
    一次性解析后的通用配置，供 train/inference/collection 等脚本共享。
    变量名清晰，避免重复解析。
    """

    # 路径
    config_path: str
    dataset_path: str
    checkpoint_dir: str

    # 模型与数据（统一来源）
    image_size: int
    image_keys: List[str]
    state_key: str
    action_horizon: int
    action_dim: int
    state_dim: int

    # 归一化
    use_normalizer: bool
    normalize_action: bool
    normalize_state: bool

    # 通用
    seed: int = 42

    # 训练专用（ScriptConfig 用于训练时有效）
    batch_size: int = 8
    max_steps: int = 20000
    save_steps: int = 5000
    eval_steps: int = 5000
    logging_steps: int = 100
    max_eval_batches: int = 50
    save_dir: str = "./checkpoints"

    # 原始 config 字典（供需要访问完整配置的逻辑使用）
    raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)


def load_script_config(
    config_path: str = "config.yaml",
    *,
    dataset_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    save_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **cli_overrides: Any,
) -> ScriptConfig:
    """
    加载 config，一次性解析为 ScriptConfig。
    cli_overrides 覆盖解析出的同名字段（优先级：cli_overrides > 显参 > config.yaml）。
    image_size 统一取自 model.vlm.image_size，dataset.image_size 作 fallback。
    """
    raw = load_config(config_path)
    model_cfg = raw.get("model", {})
    vlm_cfg = model_cfg.get("vlm", {})
    action_head_cfg = model_cfg.get("action_head", {})
    dataset_cfg = raw.get("dataset", {})
    data_cfg = raw.get("data", {})
    training_cfg = raw.get("training", {})
    robot_state_cfg = data_cfg.get("robot_state", {})

    # image_size：优先 model.vlm.image_size
    image_size = vlm_cfg.get("image_size") or dataset_cfg.get("image_size", 224)

    image_keys = dataset_cfg.get("image_keys", ["observation.images.wrist_image"])
    if not isinstance(image_keys, list):
        image_keys = [image_keys] if image_keys else ["observation.images.wrist_image"]

    state_key = dataset_cfg.get("state_key", "observation.state")
    action_horizon = dataset_cfg.get("action_horizon", 50)
    action_dim = dataset_cfg.get("action_dim") or action_head_cfg.get("action_dim", 4)
    state_dim = robot_state_cfg.get("state_dim", 9)

    use_normalizer = data_cfg.get("use_normalizer", True)
    normalize_action = data_cfg.get("normalize_action", False) if use_normalizer else False
    normalize_state = data_cfg.get("normalize_state", False) if use_normalizer else False

    dataset_path_val = dataset_path or dataset_cfg.get("local_path", "./dataset/libero_object")
    inf_cfg = raw.get("inference", {})
    ckpt_path = inf_cfg.get("checkpoint_path")
    if checkpoint_dir is not None:
        checkpoint_dir_val = checkpoint_dir
    elif ckpt_path:
        checkpoint_dir_val = str(Path(ckpt_path).parent)
    else:
        checkpoint_dir_val = "./checkpoints"

    cfg = ScriptConfig(
        config_path=config_path,
        dataset_path=dataset_path_val,
        checkpoint_dir=checkpoint_dir_val,
        image_size=image_size,
        image_keys=image_keys,
        state_key=state_key,
        action_horizon=action_horizon,
        action_dim=action_dim,
        state_dim=state_dim,
        use_normalizer=use_normalizer,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        seed=raw.get("seed", 42),
        batch_size=training_cfg.get("batch_size", 8),
        max_steps=training_cfg.get("max_steps", 20000),
        save_steps=training_cfg.get("save_steps", 5000),
        eval_steps=training_cfg.get("eval_steps", 5000),
        logging_steps=training_cfg.get("logging_steps", 100),
        max_eval_batches=training_cfg.get("max_eval_batches", 50),
        save_dir=training_cfg.get("save_dir", "./checkpoints"),
        raw_config=raw,
    )

    # 显参覆盖
    if dataset_path is not None:
        cfg.dataset_path = dataset_path
    if checkpoint_dir is not None:
        cfg.checkpoint_dir = checkpoint_dir
    if max_steps is not None:
        cfg.max_steps = max_steps
    if save_steps is not None:
        cfg.save_steps = save_steps
    if seed is not None:
        cfg.seed = seed

    # cli_overrides 覆盖
    for k, v in cli_overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    return cfg

