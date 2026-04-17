"""
Lightweight RL environment wrapper for Panda reaching task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .pick_place_env import PickPlaceEnv


@dataclass
class RLArmEnvConfig:
    """Panda 到达任务 RL 环境配置。"""

    use_gui: bool = False
    render: bool = False
    max_steps: int = 80
    target_tolerance: float = 0.05
    delta_scale: float = 0.03
    sim_steps_per_call: int = 24
    eef_z_min: float = 0.52
    eef_z_max: float = 0.95
    target_height_offset: float = 0.10
    gripper_width: float = 0.04
    seed: int | None = None


class RLArmEnv:
    """
    Panda 末端到达任务环境：
    - 观测：concat([joint_pos(9), target_xyz(3)])
    - 动作：[-1, 1] 范围的 delta xyz，再按 delta_scale 缩放
    - 奖励：成功为 1，否则为 0
    """

    def __init__(self, cfg: RLArmEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.base_env = PickPlaceEnv(
            render=cfg.render,
            use_gui=cfg.use_gui,
            seed=cfg.seed,
        )
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.step_count = 0

    @property
    def obs_dim(self) -> int:
        return 12

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.base_env.reset()
        self.step_count = 0
        self.target_pos = self._sample_target_pos()
        return self._build_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.shape[0] != 3:
            raise ValueError(f"action dim must be 3, got {action_arr.shape[0]}")
        action_arr = np.clip(action_arr, -1.0, 1.0)

        ee_now = self.base_env._get_ee_pos()
        target_ee = ee_now + action_arr * self.cfg.delta_scale
        target_ee[2] = float(np.clip(target_ee[2], self.cfg.eef_z_min, self.cfg.eef_z_max))
        cmd = np.array(
            [target_ee[0], target_ee[1], target_ee[2], self.cfg.gripper_width],
            dtype=np.float64,
        )
        self.base_env.step(cmd, sim_steps_per_call=self.cfg.sim_steps_per_call, step_delay=0.0)
        self.step_count += 1

        ee_after = self.base_env._get_ee_pos()
        dist = float(np.linalg.norm(ee_after - self.target_pos))
        success = dist < self.cfg.target_tolerance
        done = bool(success or self.step_count >= self.cfg.max_steps)
        reward = 1.0 if success else 0.0
        info = {
            "distance_to_target": dist,
            "success": success,
            "step_count": self.step_count,
            "target_pos": self.target_pos.copy(),
            "ee_pos": ee_after.copy(),
        }
        return self._build_obs(), reward, done, info

    def close(self) -> None:
        self.base_env.close()

    def _sample_target_pos(self) -> np.ndarray:
        """
        采样可达目标点。

        x/y 复用红色方块生成范围；
        z 采用与 pick-place 抓取目标相近的“抬高”策略，提升可达性。
        """
        z = self.base_env.table_height + self.base_env.cube_size + 0.001 + self.cfg.target_height_offset
        z = float(np.clip(z, self.cfg.eef_z_min, self.cfg.eef_z_max))
        x = self.rng.uniform(*self.base_env.cube_spawn_range_x)
        y = self.rng.uniform(*self.base_env.cube_spawn_range_y)
        return np.array([x, y, z], dtype=np.float64)

    def _build_obs(self) -> np.ndarray:
        joint_pos = self.base_env.get_joint_positions().astype(np.float32)
        target = self.target_pos.astype(np.float32)
        return np.concatenate([joint_pos, target], axis=0)
