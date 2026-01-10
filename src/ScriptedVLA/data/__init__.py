"""
数据处理模块
"""

from .dataset import (
    VLADataset,
    create_dummy_dataset,
    filter_dataset_by_hierarchy,
    get_dataset_statistics
)
from .download_datasets import download_dataset, download_libero_dataset, download_act_dataset
from .libero_dataset import LIBERODataset, create_libero_dataset_from_config
from .act_dataset import ACTDataset, create_act_dataset_from_config
try:
    from .lerobot_dataset_adapter import LeRobotDatasetAdapter, create_lerobot_dataset_from_config
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDatasetAdapter = None
    create_lerobot_dataset_from_config = None

__all__ = [
    "VLADataset",
    "create_dummy_dataset",
    "filter_dataset_by_hierarchy",
    "get_dataset_statistics",
    "download_dataset",
    "download_libero_dataset",
    "download_act_dataset",
    "LIBERODataset",
    "create_libero_dataset_from_config",
    "ACTDataset",
    "create_act_dataset_from_config",
]

if HAS_LEROBOT:
    __all__.extend([
        "LeRobotDatasetAdapter",
        "create_lerobot_dataset_from_config"
    ])

