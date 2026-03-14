# MIT License
#
# Copyright (c) 2026 ScriptedVLA Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Benny Lu
"""
在线仿真推理脚本（Online Simulation Inference）

加载训练好的 VLA 模型，在 PickPlace 仿真环境中进行实时闭环推理：
1. 加载 checkpoint，创建仿真环境并 reset
2. 获取当前观测（双相机图像 + 机器人状态，state 为 9 维）
3. 模型推理得到 50 步 action chunk（4 维：Δx, Δy, Δz, gripper，相对位移）
4. 转为绝对目标后逐步执行 action chunk 驱动机械臂
5. 执行完一个 chunk 后再次采集观测，重复推理，直到红色方块被放入盒子。

命令行使用示例:

  # 使用默认 config 和 checkpoint 目录运行
  python online_simulation_inference.py

  # 指定 checkpoint 目录与 config
  python online_simulation_inference.py --checkpoint_dir ./checkpoints --config config.yaml

  # 无 GUI 运行（DIRECT 仿真，便于批量评估）
  python online_simulation_inference.py --no_gui

  # 指定任务指令（抓取蓝方块）
  python online_simulation_inference.py --instruction "Pick up the blue cube and place it in the box."

  # 调整推理参数：最大 30 轮、步间延时 0.01s
  python online_simulation_inference.py --max_rounds 30 --step_delay 0.01

  # 禁用第一步平滑、使用完整预测
  python online_simulation_inference.py --no_smooth_first_step

  # Receding horizon：每轮只执行 5 步
  python online_simulation_inference.py --chunk_steps 5

  # 调试：打印 state/action 范围
  python online_simulation_inference.py --debug_ranges
"""

import sys

from src.ScriptedVLA.utils import ensure_offline_mode_if_needed

ensure_offline_mode_if_needed()

import argparse
from pathlib import Path

import numpy as np
import torch

from src.ScriptedVLA.utils import load_script_config
from src.ScriptedVLA.cli import add_common_args, parse_common_args
from inference import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    run_inference,
)
from simulator.pick_place_env import PickPlaceEnv


# 仿真环境固定返回: observation.images.top（俯视）, observation.images.wrist（夹爪）
ENV_SNAPSHOT_TOP = "observation.images.top"
ENV_SNAPSHOT_WRIST = "observation.images.wrist"


def _camera_name_to_snapshot_key(camera_name: str) -> str:
    """
    根据 config image_keys 中的相机名，自动匹配仿真器 snapshot 的键。
    - 含 wrist -> 夹爪相机 (observation.images.wrist)
    - 含 top 或 其他（image/base_image 等）-> 俯视/主相机 (observation.images.top)
    """
    cam_lower = camera_name.lower()
    if "wrist" in cam_lower:
        return ENV_SNAPSHOT_WRIST
    return ENV_SNAPSHOT_TOP


def snapshot_to_model_observation(snapshot: dict, image_keys: list):
    """
    将仿真环境 collect_snapshot 的一帧转为模型推理所需的 images 与 state。

    - 图像：按 image_keys 自动匹配 env 的 top/wrist，camera_name -> tensor (C,H,W)，0~1 归一化。
    - 状态：observation.state 原样使用，不截断。
    """
    images = {}
    for key in image_keys:
        camera_name = key.replace("observation.images.", "")
        snap_key = _camera_name_to_snapshot_key(camera_name)
        if snap_key not in snapshot:
            continue
        rgb = snapshot[snap_key]  # (H, W, 3) uint8
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=0)
        else:
            arr = np.transpose(arr, (2, 0, 1))
        images[camera_name] = torch.from_numpy(arr).float()
    state = snapshot.get("observation.state")
    if state is not None:
        state = np.asarray(state, dtype=np.float64)
    return images, state


def run_online_simulation(
    cfg,
    device: str | None = None,
    use_gui: bool = True,
    seed: int | None = 42,
    instruction: str = "Pick up the red cube and place it in the box.",
    sim_steps_per_call: int = 24,
    step_delay: float = 0.02,
    max_inference_rounds: int = 50,
    smooth_first_step: bool = True,
    first_step_alpha: float = 0.3,
    chunk_execution_steps: int | None = None,
    debug_print_ranges: bool = False,
    quiet: bool = False,
):
    """
    在线仿真推理主流程：加载模型 -> 打开环境 -> 观测-推理-执行循环，直到红色方块入盒。
    cfg: ScriptConfig，由 load_script_config 提供。
    """
    image_keys = cfg.image_keys
    action_horizon = cfg.action_horizon
    image_size = cfg.image_size
    use_normalizer = cfg.use_normalizer
    normalize_action = cfg.normalize_action
    normalize_state = cfg.normalize_state
    checkpoint_dir = cfg.checkpoint_dir
    config_path = cfg.config_path

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 加载模型
    ckpt_dir = Path(checkpoint_dir)
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    print(f"Loading checkpoint: {latest_ckpt}")
    model, normalizer = load_model_from_checkpoint(str(latest_ckpt), config_path, device)
    normalizer_to_use = normalizer if use_normalizer else None
    model.eval()

    # 每轮实际执行的步数（receding horizon）
    steps_per_round = chunk_execution_steps if chunk_execution_steps is not None else action_horizon
    steps_per_round = min(steps_per_round, action_horizon)

    # 创建仿真环境
    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=seed)
    try:
        obs = env.reset()
        # 首次观测：采集图像与状态供模型输入
        snapshot = env.collect_snapshot(None, image_width=image_size, image_height=image_size)
        round_count = 0
        done = False
        last_action_chunk = None  # 上一轮的 action chunk，用于第一步平滑的 gripper 与 debug

        while round_count < max_inference_rounds:
            round_count += 1
            if not quiet:
                print(f"\n--- Inference round {round_count} ---")

            # 当前 EE 位置（用于本轮推理后的第一步平滑）
            debug_positions = env.get_debug_positions()
            current_ee_pos = np.array(debug_positions["ee_pos"], dtype=np.float64) if "ee_pos" in debug_positions else None
            # 仿真器 action 空间：target = action[:3] + ee_to_grasp_offset，故保持位姿需 action[:3] = ee_pos - offset
            if current_ee_pos is not None and hasattr(env, "_ee_to_grasp_offset_xyz"):
                current_action_pos = current_ee_pos - np.asarray(env._ee_to_grasp_offset_xyz, dtype=np.float64)
            else:
                current_action_pos = current_ee_pos

            images_dict, state = snapshot_to_model_observation(snapshot, image_keys)

            if not images_dict:
                print("No images in snapshot, abort.")
                break

            if debug_print_ranges and state is not None:
                state_arr = np.asarray(state)
                print(f"  [debug] state range: min={state_arr.min():.4f} max={state_arr.max():.4f} mean={state_arr.mean():.4f}")
                if normalizer_to_use is not None and normalizer_to_use.state_min is not None:
                    print(f"  [debug] normalizer state: min={normalizer_to_use.state_min} max={normalizer_to_use.state_max}")

            # 模型推理，得到 (50, 4) action chunk（Δx, Δy, Δz, gripper）相对位移
            action_chunk = run_inference(
                model=model,
                images=images_dict,
                instruction=instruction,
                image_keys=image_keys,
                states=state,
                normalizer=normalizer_to_use,
                image_size=image_size,
                normalize_action=normalize_action,
                normalize_state=normalize_state,
            )
            action_chunk = np.asarray(action_chunk, dtype=np.float64)

            if debug_print_ranges:
                print(f"  [debug] action_chunk (denorm) range: min={action_chunk.min():.4f} max={action_chunk.max():.4f}")
                if normalizer_to_use is not None and normalizer_to_use.action_min is not None:
                    print(f"  [debug] normalizer action: min={normalizer_to_use.action_min} max={normalizer_to_use.action_max}")

            # 用 alpha 缩放 chunk 第一步的相对位移，减轻 chunk 边界突变
            if smooth_first_step and action_chunk.size > 0 and current_action_pos is not None and len(current_action_pos) >= 3:
                gripper = float(action_chunk[0, 3])
                if last_action_chunk is not None and last_action_chunk.shape[0] > 0:
                    gripper = float(last_action_chunk[-1, 3])
                rel_first = action_chunk[0, :3].copy()
                action_chunk[0, :3] = first_step_alpha * rel_first
                action_chunk[0, 3] = gripper

            steps_to_execute = min(steps_per_round, len(action_chunk))

            for i in range(steps_to_execute):
                # 模型输出为相对位移，转为绝对目标后执行
                ee = env._get_ee_pos()
                if hasattr(env, "_ee_to_grasp_offset_xyz"):
                    ee_action = ee - np.asarray(env._ee_to_grasp_offset_xyz, dtype=np.float64)
                else:
                    ee_action = np.asarray(ee, dtype=np.float64)
                abs_action = np.array(
                    [
                        ee_action[0] + action_chunk[i, 0],
                        ee_action[1] + action_chunk[i, 1],
                        ee_action[2] + action_chunk[i, 2],
                        action_chunk[i, 3],
                    ],
                    dtype=np.float64,
                )
                env.step(
                    abs_action,
                    sim_steps_per_call=sim_steps_per_call,
                    step_delay=step_delay,
                )

            last_action_chunk = action_chunk

            # 根据 instruction 推断目标颜色，检查对应红/蓝方块是否已入盒
            target_cube_id = (
                env._blue_cube_id if "blue" in instruction.lower() else env._red_cube_id
            )
            if target_cube_id is not None and env._is_cube_in_box(target_cube_id):
                if not quiet:
                    print("Target cube is in the box. Task success.")
                done = True
                break

            # 下一轮观测
            snapshot = env.collect_snapshot(None, image_width=image_size, image_height=image_size)

        if not done and not quiet:
            print(f"Reached max inference rounds ({max_inference_rounds}) without success.")
    finally:
        env.close()

    return done, round_count


def main():
    parser = argparse.ArgumentParser(description="Online simulation inference: run trained VLA in pick-place sim until red cube in box.")
    add_common_args(
        parser,
        include_config=True,
        include_device=True,
        include_seed=True,
        include_checkpoint_dir=True,
    )
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI (DIRECT simulation)")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube and place it in the box.", help="Task instruction")
    parser.add_argument("--max_rounds", type=int, default=50, help="Max inference rounds per episode")
    parser.add_argument("--step_delay", type=float, default=0.02, help="Delay per sim step (seconds)")
    parser.add_argument("--no_smooth_first_step", action="store_true", help="Disable first-step EE smoothing")
    parser.add_argument("--first_step_alpha", type=float, default=0.3, help="First-step blend: (1-alpha)*ee + alpha*pred (0=strong smooth)")
    parser.add_argument("--chunk_steps", type=int, default=None, help="Steps to execute per round (default: full horizon); receding horizon")
    parser.add_argument("--debug_ranges", action="store_true", help="Print state/action ranges for normalizer check")
    args = parser.parse_args()

    common = parse_common_args(args)
    cfg = load_script_config(common.config_path, checkpoint_dir=common.checkpoint_dir)

    done, rounds = run_online_simulation(
        cfg,
        device=common.device,
        use_gui=not args.no_gui,
        seed=common.seed,
        instruction=args.instruction,
        max_inference_rounds=args.max_rounds,
        step_delay=args.step_delay,
        smooth_first_step=not args.no_smooth_first_step,
        first_step_alpha=args.first_step_alpha,
        chunk_execution_steps=args.chunk_steps,
        debug_print_ranges=args.debug_ranges,
    )
    if done:
        print(f"Episode finished in {rounds} inference rounds.")


if __name__ == "__main__":
    main()
