from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator.rl_env import RLArmEnv, RLArmEnvConfig

from td3_reach_common import act_with_agent, load_agent


def parse_args() -> argparse.Namespace:
    """
    解析推理阶段的命令行参数。

    Returns:
        argparse.Namespace: 推理参数对象。
    """
    parser = argparse.ArgumentParser(description="GUI inference for Panda naive TD3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--sleep_sec", type=float, default=0.02)
    parser.add_argument("--checkpoint", type=str, default="rl_test/checkpoints/online_best.pt")
    parser.add_argument("--headless", action="store_true", help="Run inference without GUI")
    return parser.parse_args()


def main() -> None:
    """
    执行 TD3 推理流程：
    1) 加载训练好的 checkpoint；
    2) 在仿真环境中逐回合执行确定性策略；
    3) 输出成功率与平均步数。

    Returns:
        None.
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = RLArmEnv(
        RLArmEnvConfig(
            use_gui=not args.headless,
            render=not args.headless,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    agent = load_agent(ckpt_path, obs_dim=env.obs_dim, action_dim=env.action_dim, device=device)
    print(f"[load] checkpoint={ckpt_path}")

    success_count = 0
    total_steps = 0
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_steps = 0
            ep_reward = 0.0
            ep_success = 0.0
            while not done:
                action = act_with_agent(agent, obs, device=device, noise_std=0.0).astype(np.float32)
                obs, reward, done, info = env.step(action)
                ep_steps += 1
                ep_reward += reward
                ep_success = max(ep_success, float(info["success"]))
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
            success_count += int(ep_success > 0.5)
            total_steps += ep_steps
            print(
                f"[infer] ep={ep + 1}/{args.episodes} reward={ep_reward:.1f} "
                f"success={ep_success:.0f} steps={ep_steps}"
            )

        success_rate = success_count / max(1, args.episodes)
        avg_steps = total_steps / max(1, args.episodes)
        print(f"[result] success_rate={success_rate:.3f} avg_steps={avg_steps:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
