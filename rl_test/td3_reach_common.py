from __future__ import annotations

import csv
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from td3_naive_agent import TD3Agent, TD3Config


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def add(self, tr: Transition) -> None:
        self.buf.append(tr)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buf, batch_size)


def build_td3_agent(
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    *,
    max_action: float,
    gamma: float,
    tau: float,
    actor_lr: float,
    critic_lr: float,
    policy_noise: float,
    noise_clip: float,
    policy_delay: int,
    hidden_dim: int,
) -> TD3Agent:
    cfg = TD3Config(
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        max_action=max_action,
        hidden_dim=hidden_dim,
    )
    return TD3Agent(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg, device=device)


def train_step_from_replay(
    agent: TD3Agent,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    batch = replay.sample(batch_size)
    obs = torch.tensor(np.stack([b.obs for b in batch]), dtype=torch.float32, device=device)
    action = torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32, device=device)
    reward = torch.tensor([[b.reward] for b in batch], dtype=torch.float32, device=device)
    next_obs = torch.tensor(np.stack([b.next_obs for b in batch]), dtype=torch.float32, device=device)
    done = torch.tensor([[b.done] for b in batch], dtype=torch.float32, device=device)

    return agent.train_step(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done,
    )


def act_with_agent(
    agent: TD3Agent,
    obs: np.ndarray,
    device: torch.device,
    noise_std: float = 0.1,
) -> np.ndarray:
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        deterministic = noise_std <= 0
        action = agent.act(state, deterministic=deterministic, noise_std=noise_std).squeeze(0)
    action_np = action.detach().cpu().numpy()
    return np.clip(action_np, -agent.cfg.max_action, agent.cfg.max_action)


def save_agent(path: Path, agent: TD3Agent, extra: Dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "actor": agent.actor.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "total_updates": int(agent.total_updates),
        "cfg": {
            "gamma": agent.cfg.gamma,
            "tau": agent.cfg.tau,
            "actor_lr": agent.cfg.actor_lr,
            "critic_lr": agent.cfg.critic_lr,
            "policy_noise": agent.cfg.policy_noise,
            "noise_clip": agent.cfg.noise_clip,
            "policy_delay": agent.cfg.policy_delay,
            "max_action": agent.cfg.max_action,
            "hidden_dim": int(agent.cfg.hidden_dim),
        },
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_agent(path: Path, obs_dim: int, action_dim: int, device: torch.device) -> TD3Agent:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["cfg"]
    hidden_dim = int(cfg.get("hidden_dim", 256))
    agent = build_td3_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        max_action=float(cfg["max_action"]),
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        policy_noise=float(cfg["policy_noise"]),
        noise_clip=float(cfg["noise_clip"]),
        policy_delay=int(cfg["policy_delay"]),
        hidden_dim=hidden_dim,
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.actor_target.load_state_dict(ckpt["actor_target"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.total_updates = int(ckpt.get("total_updates", 0))
    return agent


def export_q_logs(logs: List[Dict[str, float]], output_dir: Path, base_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base_name}.csv"
    fields = [
        "phase",
        "global_step",
        "episode",
        "reward",
        "success",
        "critic_loss",
        "actor_loss",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in logs:
            writer.writerow({k: row.get(k, "") for k in fields})

    try:
        # Windows + scientific stack may load duplicated OpenMP runtimes.
        # Set this before importing matplotlib to avoid hard abort on some setups.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(logs))
        q1 = np.array([float(row.get("q1_mean", 0.0)) for row in logs], dtype=np.float32)
        q2 = np.array([float(row.get("q2_mean", 0.0)) for row in logs], dtype=np.float32)
        tq = np.array([float(row.get("target_q_mean", 0.0)) for row in logs], dtype=np.float32)
        plt.figure(figsize=(9, 5))
        plt.plot(x, q1, label="q1_mean")
        plt.plot(x, q2, label="q2_mean")
        plt.plot(x, tq, label="target_q_mean")
        plt.xlabel("Train Update")
        plt.ylabel("Q Value")
        plt.title("Panda TD3 Q Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[warn] failed to plot q curve: {exc}")
