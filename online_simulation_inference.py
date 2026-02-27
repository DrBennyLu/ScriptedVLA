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
3. 模型推理得到 50 步 action chunk（4 维：x, y, z, gripper），与仿真器一致
4. 逐步执行 action chunk 驱动机械臂
5. 执行完一个 chunk 后再次采集观测，重复推理，直到红色方块被放入盒子。
"""

import os
import sys


def _maybe_enable_offline():
    """若配置了 cache_dir 或 local_model_path，在 import transformers 之前设置离线模式"""
    import yaml
    from pathlib import Path
    config_path = "config.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vlm = cfg.get("model", {}).get("vlm", {})
        if vlm.get("cache_dir") or vlm.get("local_model_path"):
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"


_maybe_enable_offline()

import argparse
from pathlib import Path

import numpy as np
import torch

from src.ScriptedVLA.utils import load_config, get_data_config, get_model_config
from inference import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    run_inference,
)
from simulator.pick_place_env import PickPlaceEnv


# 仿真环境 collect_snapshot 返回的键名与 config dataset.image_keys 的对应关系
# config: observation.images.image, observation.images.wrist_image
# env:    observation.images.top,   observation.images.wrist
SNAPSHOT_KEY_TO_CAMERA_NAME = {
    "observation.images.top": "image",
    "observation.images.wrist": "wrist_image",
}


def snapshot_to_model_observation(snapshot: dict, image_keys: list):
    """
    将仿真环境 collect_snapshot 的一帧转为模型推理所需的 images 与 state。

    - 图像：snapshot 的 top/wrist 按 image_keys 映射为 camera_name -> tensor (C,H,W)，0~1 归一化。
    - 状态：observation.state 按 config 的 state_dim（9 维）原样使用，不截断。
    """
    images = {}
    for key in image_keys:
        camera_name = key.replace("observation.images.", "")
        for snap_key, cam in SNAPSHOT_KEY_TO_CAMERA_NAME.items():
            if cam == camera_name and snap_key in snapshot:
                rgb = snapshot[snap_key]  # (H, W, 3) uint8
                arr = np.asarray(rgb, dtype=np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=0)
                else:
                    arr = np.transpose(arr, (2, 0, 1))
                images[camera_name] = torch.from_numpy(arr).float()
                break
    state = snapshot.get("observation.state")
    if state is not None:
        state = np.asarray(state, dtype=np.float64)
    return images, state


def run_online_simulation(
    checkpoint_dir: str = "./checkpoints",
    config_path: str = "config.yaml",
    device: str | None = None,
    use_gui: bool = True,
    seed: int | None = 42,
    instruction: str = "Pick up the red cube and place it in the box.",
    action_horizon: int = 50,
    sim_steps_per_call: int = 24,
    step_delay: float = 0.02,
    max_inference_rounds: int = 50,
):
    """
    在线仿真推理主流程：加载模型 -> 打开环境 -> 观测-推理-执行循环，直到红色方块入盒。
    """
    config = load_config(config_path)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    dataset_config = config.get("dataset", {})
    data_config = get_data_config(config)
    model_config = get_model_config(config)
    image_keys = dataset_config.get("image_keys", ["observation.images.image", "observation.images.wrist_image"])
    if not isinstance(image_keys, list):
        image_keys = [image_keys]
    action_horizon = dataset_config.get("action_horizon", action_horizon)
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    use_normalizer = data_config.get("use_normalizer", True)
    normalize_action = data_config.get("normalize_action", True) if use_normalizer else False
    normalize_state = data_config.get("normalize_state", True) if use_normalizer else False

    # 加载模型
    ckpt_dir = Path(checkpoint_dir)
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    print(f"Loading checkpoint: {latest_ckpt}")
    model, normalizer = load_model_from_checkpoint(str(latest_ckpt), config_path, device)
    normalizer_to_use = normalizer if use_normalizer else None
    model.eval()

    # 创建仿真环境
    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=seed)
    try:
        obs = env.reset()
        # 首次观测：采集图像与状态供模型输入
        snapshot = env.collect_snapshot(None, image_width=image_size, image_height=image_size)
        round_count = 0
        done = False

        while round_count < max_inference_rounds:
            round_count += 1
            print(f"\n--- Inference round {round_count} ---")

            images_dict, state = snapshot_to_model_observation(snapshot, image_keys)

            if not images_dict:
                print("No images in snapshot, abort.")
                break

            # 模型推理，得到 (50, 4) action chunk（x, y, z, gripper），与仿真器一致
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
            steps_to_execute = min(action_horizon, len(action_chunk))

            for i in range(steps_to_execute):
                env.step(
                    action_chunk[i],
                    sim_steps_per_call=sim_steps_per_call,
                    step_delay=step_delay,
                )

            # 检查红色方块是否已入盒
            if env._red_cube_id is not None and env._is_cube_in_box(env._red_cube_id):
                print("Red cube is in the box. Task success.")
                done = True
                break

            # 下一轮观测
            snapshot = env.collect_snapshot(None, image_width=image_size, image_height=image_size)

        if not done:
            print(f"Reached max inference rounds ({max_inference_rounds}) without success.")
    finally:
        env.close()

    return done


def main():
    parser = argparse.ArgumentParser(description="Online simulation inference: run trained VLA in pick-place sim until red cube in box.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI (DIRECT simulation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (None to disable)")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube and place it in the box.", help="Task instruction")
    parser.add_argument("--max_rounds", type=int, default=50, help="Max inference rounds per episode")
    parser.add_argument("--step_delay", type=float, default=0.02, help="Delay per sim step (seconds)")
    args = parser.parse_args()

    run_online_simulation(
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        device=args.device,
        use_gui=not args.no_gui,
        seed=args.seed,
        instruction=args.instruction,
        max_inference_rounds=args.max_rounds,
        step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
