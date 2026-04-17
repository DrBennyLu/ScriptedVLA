from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 运行参数对象。
    """
    parser = argparse.ArgumentParser(description="Warmup TD3 for Panda reaching")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", help="Run with PyBullet DIRECT")
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--prefill_episodes", type=int, default=120)
    parser.add_argument("--prefill_max_attempts", type=int, default=600)
    parser.add_argument("--prefill_episode_max_steps", type=int, default=400)
    parser.add_argument("--warmup_updates", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--planner_step_ratio", type=float, default=0.8)
    parser.add_argument("--planner_steps_per_phase", type=int, default=60)
    parser.add_argument("--planner_wait_steps", type=int, default=50)
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


def _delta_action_to_waypoint(env: RLArmEnv, waypoint: np.ndarray) -> np.ndarray:
    """
    将 EE 目标点转换为环境使用的 delta 动作。

    Args:
        env: RL 环境实例。
        waypoint: 世界坐标系下的 EE 目标位置，形状为 (3,)。

    Returns:
        np.ndarray: 裁剪到 [-1, 1] 的 delta 动作，形状为 (3,)。
    """
    ee = env.base_env._get_ee_pos().astype(np.float64)
    desired_delta = waypoint - ee
    action = desired_delta / env.cfg.delta_scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _record_step(
    env: RLArmEnv,
    obs: np.ndarray,
    action: np.ndarray,
    traj: list[Transition],
) -> tuple[np.ndarray, bool, bool]:
    """
    执行一步环境交互，并将 transition 追加到当前轨迹。

    Args:
        env: RL 环境。
        obs: 当前观测向量。
        action: delta xyz 动作，形状为 (3,)。
        traj: 当前轨迹的 transition 列表。

    Returns:
        tuple[np.ndarray, bool, bool]:
            - 下一时刻观测
            - done 标记
            - success 标记（来自 env info）
    """
    next_obs, reward, done, info = env.step(action)
    traj.append(
        Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=float(done),
        )
    )
    return next_obs, bool(done), bool(info.get("success", False))


def _interpolate_and_step(
    env: RLArmEnv,
    obs: np.ndarray,
    traj: list[Transition],
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    wait_steps: int,
) -> tuple[np.ndarray, bool]:
    """
    在 EE 空间执行一个“插值 + 到位等待”的运动阶段。

    Args:
        env: RL 环境。
        obs: 当前观测。
        traj: 当前尝试的 transition 列表。
        start: 起始 EE 位置，形状为 (3,)。
        target: 目标 EE 位置，形状为 (3,)。
        steps: 插值步数。
        wait_steps: 到位后额外等待步数。

    Returns:
        tuple[np.ndarray, bool]:
            - 最新观测
            - 是否成功
    """
    done = False
    success = False
    for k in range(1, max(1, steps) + 1):
        t = k / float(max(1, steps))
        waypoint = start + t * (target - start)
        action = _delta_action_to_waypoint(env, waypoint)
        obs, done, success = _record_step(env, obs, action, traj)
        if done:
            return obs, success

    # 到位等待：重复给目标位置命令，直到命中 success 或超时
    for _ in range(max(0, wait_steps)):
        action = _delta_action_to_waypoint(env, target)
        obs, done, success = _record_step(env, obs, action, traj)
        if done:
            return obs, success
    return obs, success


def _stabilize(
    env: RLArmEnv,
    obs: np.ndarray,
    traj: list[Transition],
    target_pos: np.ndarray,
    duration_steps: int,
) -> tuple[np.ndarray, bool]:
    """
    在目标点附近保持若干步，用于稳定控制。

    Args:
        env: RL 环境。
        obs: 当前观测。
        traj: 当前尝试的 transition 列表。
        target_pos: 需要保持的 EE 点，形状为 (3,)。
        duration_steps: 稳定阶段步数。

    Returns:
        tuple[np.ndarray, bool]:
            - 最新观测
            - 是否成功
    """
    done = False
    success = False
    for _ in range(max(0, duration_steps)):
        action = _delta_action_to_waypoint(env, target_pos)
        obs, done, success = _record_step(env, obs, action, traj)
        if done:
            return obs, success
    return obs, success


def _export_demo_stats(stats: list[dict[str, float]], log_dir: Path) -> None:
    """
    将成功 demo 统计导出为 CSV。

    Args:
        stats: 每条 demo 的统计信息列表。
        log_dir: 日志输出目录。

    Returns:
        None.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "prefill_demo_stats.csv"
    fields = ["demo_id", "attempt_id", "traj_len", "terminal_error", "success"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow({k: row.get(k, "") for k in fields})


def rollout_prefill_success(
    env: RLArmEnv,
    replay: ReplayBuffer,
    target_episodes: int,
    max_attempts: int,
    planner_step_ratio: float,
    planner_steps_per_phase: int,
    planner_wait_steps: int,
    prefill_episode_max_steps: int,
) -> list[dict[str, float]]:
    """
    采集成功的规划轨迹，并填充到 replay buffer。

    Args:
        env: RL 环境。
        replay: 待填充的 replay buffer。
        target_episodes: 需要收集的成功 demo 数量。
        max_attempts: 最大尝试次数，超过后报错。
        planner_step_ratio: 运动步长比例，范围 (0, 1]。
        planner_steps_per_phase: 每个插值阶段的最大步数。
        planner_wait_steps: 到位后等待收敛的步数。
        prefill_episode_max_steps: prefill 阶段临时放宽的单回合最大步数。

    Returns:
        list[dict[str, float]]: 成功 demo 统计列表。

    Raises:
        RuntimeError: 在最大尝试次数内无法收集足够成功 demo 时抛出。
    """
    success_eps = 0
    attempts = 0
    stats: list[dict[str, float]] = []
    original_max_steps = env.cfg.max_steps
    env.cfg.max_steps = max(original_max_steps, prefill_episode_max_steps)
    try:
        attempt_bar = tqdm(total=max_attempts, desc="prefill_attempts", leave=True)
        success_bar = tqdm(total=target_episodes, desc="successful_demos", leave=True)
        while success_eps < target_episodes and attempts < max_attempts:
            attempts += 1
            attempt_bar.update(1)
            obs = env.reset()
            done = False
            traj = []
            success = False
            target = obs[-3:].astype(np.float64)
            init_ee = env.base_env._get_ee_pos().astype(np.float64)

            # 先稳定到当前位姿，贴近 pick_place 里的 _stabilize
            obs, success = _stabilize(
                env=env,
                obs=obs,
                traj=traj,
                target_pos=init_ee,
                duration_steps=20,
            )
            if not success:
                # 单阶段直线插值到目标点，保持与 pick_place 的 _interpolate_and_step 风格一致
                start_ee = env.base_env._get_ee_pos().astype(np.float64)
                distance = float(np.linalg.norm(target - start_ee))
                step_len = env.cfg.delta_scale * max(0.1, min(1.0, planner_step_ratio))
                dynamic_steps = int(np.ceil(distance / max(step_len, 1e-6)))
                interp_steps = int(np.clip(dynamic_steps, 10, planner_steps_per_phase))
                obs, success = _interpolate_and_step(
                    env=env,
                    obs=obs,
                    traj=traj,
                    start=start_ee,
                    target=target,
                    steps=interp_steps,
                    wait_steps=planner_wait_steps,
                )

            terminal_err = float(np.linalg.norm(env.base_env._get_ee_pos().astype(np.float64) - target))

            if success:
                for tr in traj:
                    replay.add(tr)
                success_eps += 1
                success_bar.update(1)
                stats.append(
                    {
                        "demo_id": float(success_eps),
                        "attempt_id": float(attempts),
                        "traj_len": float(len(traj)),
                        "terminal_error": terminal_err,
                        "success": 1.0,
                    }
                )
                if success_eps % 20 == 0 or success_eps == target_episodes:
                    print(
                        f"[prefill-success] success_eps={success_eps}/{target_episodes} "
                        f"attempts={attempts} replay={len(replay)} terminal_err={terminal_err:.5f}"
                    )
            else:
                if attempts <= 10 or attempts % 25 == 0:
                    target_dbg = target.tolist()
                    ee_dbg = env.base_env._get_ee_pos().astype(np.float64).tolist()
                    print(
                        f"[prefill-debug] attempts={attempts} success_eps={success_eps} "
                        f"terminal_err={terminal_err:.5f} traj_len={len(traj)} "
                        f"target=({target_dbg[0]:.3f},{target_dbg[1]:.3f},{target_dbg[2]:.3f}) "
                        f"ee_end=({ee_dbg[0]:.3f},{ee_dbg[1]:.3f},{ee_dbg[2]:.3f})"
                    )

        attempt_bar.close()
        success_bar.close()

        if success_eps < target_episodes:
            raise RuntimeError(
                f"only collected {success_eps}/{target_episodes} successful demos "
                f"within {attempts} attempts"
            )
        return stats
    finally:
        env.cfg.max_steps = original_max_steps


def main() -> None:
    """
    执行 warmup 全流程：
    1) 采集成功规划轨迹并填充 replay；
    2) 基于 replay 执行 TD3 warmup 更新；
    3) 保存 checkpoint 与日志文件。

    Returns:
        None.
    """
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
        prefill_stats = rollout_prefill_success(
            env=env,
            replay=replay,
            target_episodes=args.prefill_episodes,
            max_attempts=args.prefill_max_attempts,
            planner_step_ratio=args.planner_step_ratio,
            planner_steps_per_phase=args.planner_steps_per_phase,
            planner_wait_steps=args.planner_wait_steps,
            prefill_episode_max_steps=args.prefill_episode_max_steps,
        )
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

        warmup_bar = tqdm(range(args.warmup_updates), desc="warmup_updates", leave=True)
        for i in warmup_bar:
            metrics = train_step_from_replay(agent, replay, args.batch_size, device)
            global_step += 1
            row = {"phase": "warmup", "global_step": global_step, "episode": -1, "reward": 0.0, "success": 0.0}
            row.update(metrics)
            logs.append(row)
            warmup_bar.set_postfix(
                q1=f"{metrics['q1_mean']:.3f}",
                q2=f"{metrics['q2_mean']:.3f}",
                tq=f"{metrics['target_q_mean']:.3f}",
            )

        save_dir = Path(args.save_dir)
        log_dir = Path(args.log_dir)
        _export_demo_stats(prefill_stats, log_dir)
        save_agent(
            save_dir / "warmup_checkpoint.pt",
            agent,
            extra={"global_step": float(global_step), "replay_size": float(len(replay))},
        )
        export_q_logs(logs, log_dir, base_name="warmup_q_curve")
        print(f"[done] prefill demo stats: {log_dir / 'prefill_demo_stats.csv'}")
        print(f"[done] warmup checkpoint: {save_dir / 'warmup_checkpoint.pt'}")
        print(f"[done] warmup logs: {log_dir / 'warmup_q_curve.csv'}")
        print(f"[done] warmup q curve: {log_dir / 'warmup_q_curve.png'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
