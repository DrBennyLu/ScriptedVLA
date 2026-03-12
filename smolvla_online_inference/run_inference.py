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
"""
SmolVLA 在线仿真推理

使用 lerobot 官方 SmolVLA 模型在 PickPlace 仿真中进行实时闭环推理。
与 online_simulation_inference.py 架构一致，但加载 lerobot 训练的 checkpoint。

运行示例:
  python smolvla_online_inference/run_inference.py
  python smolvla_online_inference/run_inference.py --checkpoint_dir outputs/train/my_smolvla
  python smolvla_online_inference/run_inference.py --instruction "Pick up the blue cube and place it in the box."
  python smolvla_online_inference/run_inference.py --no_gui --chunk_steps 5
"""

import sys
from pathlib import Path

# 添加父项目根目录以导入 PickPlaceEnv
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import argparse
import numpy as np
import torch

from simulator.pick_place_env import PickPlaceEnv


# collect_snapshot 键名 -> lerobot 训练数据集键名
SNAPSHOT_TO_LEROBOT = {
    "observation.images.top": "observation.images.top_image",
    "observation.images.wrist": "observation.images.wrist_image",
}


def _find_checkpoint_path(checkpoint_dir: Path) -> Path:
    """查找 lerobot 训练输出中的 checkpoint 路径。"""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def _has_model_files(p: Path) -> bool:
        return (
            (p / "config.yaml").exists()
            or (p / "config.json").exists()
            or (p / "model.safetensors").exists()
            or list(p.glob("*.safetensors"))
        )

    if _has_model_files(checkpoint_dir):
        return checkpoint_dir

    # lerobot-train 可能输出: checkpoints/last/ 或 checkpoints/last/pretrained_model/
    for sub in ["checkpoints/last/pretrained_model", "checkpoints/last", "checkpoints"]:
        cand = checkpoint_dir / sub
        if cand.exists() and _has_model_files(cand):
            return cand

    subdirs = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if subdirs and _has_model_files(subdirs[-1]):
        return subdirs[-1]

    return checkpoint_dir


def _snapshot_to_lerobot_frame(snapshot: dict, task: str, frame_idx: int = 0) -> dict:
    """
    将 PickPlaceEnv collect_snapshot 输出转为 lerobot preprocess 期望的 frame 格式。
    """
    frame = {}
    for snap_key, lerobot_key in SNAPSHOT_TO_LEROBOT.items():
        if snap_key in snapshot:
            img = snapshot[snap_key]  # (H, W, 3) uint8
            arr = np.asarray(img, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            frame[lerobot_key] = arr

    if "observation.state" in snapshot:
        state = np.asarray(snapshot["observation.state"], dtype=np.float32)
        frame["observation.state"] = state

    frame["task"] = task
    frame["index"] = frame_idx
    frame["episode_index"] = 0
    frame["frame_index"] = frame_idx
    frame["task_index"] = 0
    frame["timestamp"] = np.array([float(frame_idx) / 10.0], dtype=np.float32)
    frame["action"] = np.zeros(4, dtype=np.float32)

    return frame


def run_online_simulation(
    checkpoint_dir: str = "outputs/train/my_smolvla",
    device: str | None = None,
    use_gui: bool = True,
    seed: int | None = 42,
    instruction: str = "Pick up the red cube and place it in the box.",
    image_size: int = 224,
    sim_steps_per_call: int = 24,
    step_delay: float = 0.02,
    max_inference_rounds: int = 50,
    smooth_first_step: bool = True,
    first_step_alpha: float = 0.3,
    chunk_execution_steps: int | None = None,
    quiet: bool = False,
) -> tuple[bool, int]:
    """
    在线仿真推理主流程：加载 SmolVLA -> 打开 PickPlace 环境 -> 观测-推理-执行循环。
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as e:
        raise ImportError(
            "lerobot[smolvla] is required. Install with: pip install \"lerobot[smolvla]\""
        ) from e

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    ckpt_path = _find_checkpoint_path(Path(checkpoint_dir))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    print(f"Loading SmolVLA from: {ckpt_path}")
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path)).to(device_obj).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(ckpt_path),
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )

    cfg = policy.config
    if hasattr(cfg, "n_action_steps") and cfg.n_action_steps:
        action_horizon = int(cfg.n_action_steps)
    elif hasattr(cfg, "chunk_size") and cfg.chunk_size:
        action_horizon = int(cfg.chunk_size)
    elif isinstance(cfg, dict):
        action_horizon = int(cfg.get("n_action_steps") or cfg.get("chunk_size") or 50)
    else:
        action_horizon = 50
    steps_per_round = (
        chunk_execution_steps if chunk_execution_steps is not None else action_horizon
    )
    steps_per_round = min(steps_per_round, action_horizon)

    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=seed)
    try:
        env.reset()
        snapshot = env.collect_snapshot(
            None, image_width=image_size, image_height=image_size
        )
        round_count = 0
        done = False
        last_action_chunk = None

        while round_count < max_inference_rounds:
            round_count += 1
            if not quiet:
                print(f"\n--- Inference round {round_count} ---")

            debug_positions = env.get_debug_positions()
            current_ee_pos = (
                np.array(debug_positions["ee_pos"], dtype=np.float64)
                if "ee_pos" in debug_positions
                else None
            )
            if current_ee_pos is not None and hasattr(env, "_ee_to_grasp_offset_xyz"):
                current_action_pos = current_ee_pos - np.asarray(
                    env._ee_to_grasp_offset_xyz, dtype=np.float64
                )
            else:
                current_action_pos = current_ee_pos

            frame = _snapshot_to_lerobot_frame(
                snapshot, task=instruction, frame_idx=round_count - 1
            )
            batch = preprocess(frame)

            with torch.inference_mode():
                action_chunk = policy.select_action(batch)
            action_chunk = postprocess(action_chunk)

            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()
            action_chunk = np.asarray(action_chunk, dtype=np.float64)

            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)

            if smooth_first_step and action_chunk.size > 0 and current_action_pos is not None and len(current_action_pos) >= 3:
                gripper = float(action_chunk[0, 3])
                if last_action_chunk is not None and last_action_chunk.shape[0] > 0:
                    gripper = float(last_action_chunk[-1, 3])
                rel_first = action_chunk[0, :3].copy()
                action_chunk[0, :3] = first_step_alpha * rel_first
                action_chunk[0, 3] = gripper

            steps_to_execute = min(steps_per_round, len(action_chunk))

            for i in range(steps_to_execute):
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

            target_cube_id = (
                env._blue_cube_id if "blue" in instruction.lower() else env._red_cube_id
            )
            if target_cube_id is not None and env._is_cube_in_box(target_cube_id):
                if not quiet:
                    print("Target cube is in the box. Task success.")
                done = True
                break

            snapshot = env.collect_snapshot(
                None, image_width=image_size, image_height=image_size
            )

        if not done and not quiet:
            print(f"Reached max inference rounds ({max_inference_rounds}) without success.")
    finally:
        env.close()

    return done, round_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA online simulation inference: run trained SmolVLA in pick-place sim."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/train/my_smolvla",
        help="Checkpoint directory (lerobot training output)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI (DIRECT simulation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (None to disable)")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the red cube and place it in the box.",
        help="Task instruction",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=50,
        help="Max inference rounds per episode",
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.02,
        help="Delay per sim step (seconds)",
    )
    parser.add_argument(
        "--no_smooth_first_step",
        action="store_true",
        help="Disable first-step EE smoothing",
    )
    parser.add_argument(
        "--first_step_alpha",
        type=float,
        default=0.3,
        help="First-step blend: (1-alpha)*ee + alpha*pred",
    )
    parser.add_argument(
        "--chunk_steps",
        type=int,
        default=None,
        help="Steps to execute per round (default: full chunk); receding horizon",
    )
    args = parser.parse_args()

    done, rounds = run_online_simulation(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_gui=not args.no_gui,
        seed=args.seed,
        instruction=args.instruction,
        max_inference_rounds=args.max_rounds,
        step_delay=args.step_delay,
        smooth_first_step=not args.no_smooth_first_step,
        first_step_alpha=args.first_step_alpha,
        chunk_execution_steps=args.chunk_steps,
    )
    if done:
        print(f"Episode finished in {rounds} inference rounds.")


if __name__ == "__main__":
    main()
