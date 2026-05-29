"""LIBERO action layout helpers (LeRobot libero-object OSC_POSE convention)."""

from __future__ import annotations

ACTION_DIM = 7

# 7-dim OSC_POSE delta: dx, dy, dz, droll, dpitch, dyaw, gripper (all in [-1, 1] in dataset)
ACTION_DIM_LABELS = [
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
]


def action_dim_labels() -> list[str]:
    return list(ACTION_DIM_LABELS)
