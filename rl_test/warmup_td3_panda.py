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
    build_td3_agent,
    export_q_logs,
    save_agent,
    train_step_from_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup TD3 for Panda reaching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Run with PyBullet DIRECT")
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--prefill_episodes", type=int, default=120)
    parser.add_argument("--warmup_updates", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--heuristic_ratio", type=float, default=0.35)
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="rl_test/checkpoints")
    parser.add_argument("--log_dir", type=str, default="rl_test/logs")
    return parser.parse_args()


def rollout_prefill(env: RLArmEnv, replay: ReplayBuffer, episodes: int, heuristic_ratio: float) -> None:
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            target = obs[-3:]
            ee = env.base_env._get_ee_pos().astype(np.float32)
            if random.random() < heuristic_ratio:
                direction = target - ee
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    action = direction / norm
                else:
                    action = np.zeros(3, dtype=np.float32)
            else:
                action = np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
            next_obs, reward, done, _ = env.step(action)
            replay.add(Transition(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=float(done)))
            obs = next_obs
            ep_reward += reward
        if (ep + 1) % 20 == 0:
            print(f"[prefill] episode={ep + 1}/{episodes} reward={ep_reward:.1f} replay={len(replay)}")


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

    try:
        rollout_prefill(env, replay, args.prefill_episodes, args.heuristic_ratio)
        if len(replay) < args.batch_size:
            raise RuntimeError(f"replay too small after prefill: {len(replay)} < {args.batch_size}")

        agent = build_td3_agent(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=device,
            max_action=args.max_action,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            hidden_dim=args.hidden_dim,
        )

        for i in range(args.warmup_updates):
            metrics = train_step_from_replay(agent, replay, args.batch_size, device)
            global_step += 1
            row = {"phase": "warmup", "global_step": global_step, "episode": -1, "reward": 0.0, "success": 0.0}
            row.update(metrics)
            logs.append(row)
            if (i + 1) % 100 == 0:
                print(
                    f"[warmup] update={i + 1}/{args.warmup_updates} "
                    f"q1={metrics['q1_mean']:.4f} q2={metrics['q2_mean']:.4f} tq={metrics['target_q_mean']:.4f}"
                )

        save_dir = Path(args.save_dir)
        log_dir = Path(args.log_dir)
        save_agent(
            save_dir / "warmup_checkpoint.pt",
            agent,
            extra={"global_step": float(global_step), "replay_size": float(len(replay))},
        )
        export_q_logs(logs, log_dir, base_name="warmup_q_curve")
        print(f"[done] warmup checkpoint: {save_dir / 'warmup_checkpoint.pt'}")
        print(f"[done] warmup logs: {log_dir / 'warmup_q_curve.csv'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
