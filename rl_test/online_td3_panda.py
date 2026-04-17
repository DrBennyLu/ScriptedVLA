from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from simulator.rl_env import RLArmEnv, RLArmEnvConfig

from td3_reach_common import (
    ReplayBuffer,
    Transition,
    act_with_agent,
    export_q_logs,
    load_agent,
    save_agent,
    train_step_from_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online TD3 training from warmup checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_random_steps", type=int, default=1000)
    parser.add_argument("--explore_noise", type=float, default=0.2)
    parser.add_argument("--warmup_ckpt", type=str, default="rl_test/checkpoints/warmup_checkpoint.pt")
    parser.add_argument("--save_dir", type=str, default="rl_test/checkpoints")
    parser.add_argument("--log_dir", type=str, default="rl_test/logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = RLArmEnvConfig(
        use_gui=not args.headless,
        render=not args.headless,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    env = RLArmEnv(env_cfg)
    replay = ReplayBuffer(capacity=args.buffer_size)
    logs = []
    global_step = 0
    train_updates = 0
    best_success_rate = 0.0

    try:
        agent = load_agent(Path(args.warmup_ckpt), obs_dim=env.obs_dim, action_dim=env.action_dim, device=device)
        print(f"[load] warmup checkpoint: {args.warmup_ckpt}")

        recent_success = []
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            success = 0.0
            while not done:
                if global_step < args.start_random_steps:
                    action = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
                else:
                    action = act_with_agent(agent, obs, device=device, noise_std=args.explore_noise).astype(np.float32)
                next_obs, reward, done, info = env.step(action)
                replay.add(Transition(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=float(done)))
                obs = next_obs
                ep_reward += reward
                success = max(success, float(info["success"]))
                global_step += 1

                if len(replay) >= args.batch_size:
                    for _ in range(args.updates_per_step):
                        metrics = train_step_from_replay(agent, replay, args.batch_size, device)
                        train_updates += 1
                        row = {
                            "phase": "online",
                            "global_step": global_step,
                            "episode": ep + 1,
                            "reward": ep_reward,
                            "success": success,
                        }
                        row.update(metrics)
                        logs.append(row)

            recent_success.append(success)
            if len(recent_success) > 20:
                recent_success.pop(0)
            success_rate = float(np.mean(recent_success))
            print(
                f"[online] ep={ep + 1}/{args.episodes} reward={ep_reward:.1f} success={success:.0f} "
                f"recent_success_rate={success_rate:.2f} replay={len(replay)} updates={train_updates}"
            )

            save_dir = Path(args.save_dir)
            save_agent(save_dir / "online_latest.pt", agent, extra={"global_step": float(global_step)})
            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                save_agent(save_dir / "online_best.pt", agent, extra={"global_step": float(global_step)})

        log_dir = Path(args.log_dir)
        export_q_logs(logs, log_dir, base_name="online_q_curve")
        print(f"[done] online latest: {Path(args.save_dir) / 'online_latest.pt'}")
        print(f"[done] online best: {Path(args.save_dir) / 'online_best.pt'}")
        print(f"[done] online logs: {log_dir / 'online_q_curve.csv'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
