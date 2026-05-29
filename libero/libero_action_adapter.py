"""Action helpers for LIBERO WebSocket evaluation."""

from __future__ import annotations

from typing import List

import numpy as np

ACTION_DIM = 7


def validate_action(action, action_dim: int = ACTION_DIM) -> List[float]:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != action_dim:
        raise ValueError(f"Expected action dim {action_dim}, got {arr.size}")
    return arr.tolist()


def zero_action(action_dim: int = ACTION_DIM) -> List[float]:
    return [0.0] * action_dim


def random_action(scale: float = 0.05, action_dim: int = ACTION_DIM) -> List[float]:
    arr = np.random.uniform(-scale, scale, size=action_dim).astype(np.float32)
    arr[-1] = float(np.clip(arr[-1], -1.0, 1.0))
    return arr.tolist()


def model_action_to_libero(action: np.ndarray, action_dim: int = ACTION_DIM) -> List[float]:
    """Pass through 7-dim OSC_POSE actions from model or dataset."""
    return validate_action(action, action_dim=action_dim)
