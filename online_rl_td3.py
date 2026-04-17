"""
Online TD3 training using RL token z_rl as state.

author: Benny Lu
license: MIT
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.ScriptedVLA.model import QwenGR00TVLAModel, RLTokenBottleneck
from src.ScriptedVLA.utils import ensure_offline_mode_if_needed, load_script_config

ensure_offline_mode_if_needed()


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def add(self, tr: Transition) -> None:
        self.buf.append(tr)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.buf, batch_size)
        return {
            "state": torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32),
            "action": torch.tensor(np.stack([b.action for b in batch]), dtype=torch.float32),
            "reward": torch.tensor([[b.reward] for b in batch], dtype=torch.float32),
            "next_state": torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32),
            "done": torch.tensor([[b.done] for b in batch], dtype=torch.float32),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, cfg: dict, device: torch.device):
        self.device = device
        self.max_action = max_action
        hidden = cfg.get("hidden_dim", 256)
        self.actor = MLP(state_dim, action_dim, hidden).to(device)
        self.actor_t = MLP(state_dim, action_dim, hidden).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.q1 = MLP(state_dim + action_dim, 1, hidden).to(device)
        self.q2 = MLP(state_dim + action_dim, 1, hidden).to(device)
        self.q1_t = MLP(state_dim + action_dim, 1, hidden).to(device)
        self.q2_t = MLP(state_dim + action_dim, 1, hidden).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.get("actor_lr", 1e-4))
        self.critic_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=cfg.get("critic_lr", 1e-3),
        )
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.policy_noise = cfg.get("policy_noise", 0.2)
        self.noise_clip = cfg.get("noise_clip", 0.5)
        self.policy_delay = cfg.get("policy_delay", 2)
        self.total_updates = 0

    def act(self, state: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tanh(self.actor(s)).detach().cpu().numpy()[0] * self.max_action
        if noise_std > 0:
            a = a + np.random.normal(0, noise_std, size=a.shape)
        return np.clip(a, -self.max_action, self.max_action)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch["state"].to(self.device)
        a = batch["action"].to(self.device)
        r = batch["reward"].to(self.device)
        ns = batch["next_state"].to(self.device)
        d = batch["done"].to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (torch.tanh(self.actor_t(ns)) * self.max_action + noise).clamp(-self.max_action, self.max_action)
            nsa = torch.cat([ns, na], dim=-1)
            tq = torch.min(self.q1_t(nsa), self.q2_t(nsa))
            y = r + (1.0 - d) * self.gamma * tq

        sa = torch.cat([s, a], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        self.total_updates += 1
        if self.total_updates % self.policy_delay == 0:
            pa = torch.tanh(self.actor(s)) * self.max_action
            actor_loss = -self.q1(torch.cat([s, pa], dim=-1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update()

        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def _soft_update(self):
        for t, s in zip(self.actor_t.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.q1_t.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
        for t, s in zip(self.q2_t.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)


def build_vla_and_rl(cfg) -> Tuple[QwenGR00TVLAModel, RLTokenBottleneck, torch.device]:
    raw = cfg.raw_config
    model_cfg = raw.get("model", {})
    vla_cfg = model_cfg.get("vla", {})
    action_head_cfg = model_cfg.get("action_head", {}).copy()
    action_head_cfg["action_horizon"] = cfg.action_horizon
    action_head_cfg["action_dim"] = cfg.action_dim
    vla = QwenGR00TVLAModel(
        vlm_config=model_cfg.get("vlm", {}),
        action_head_config=action_head_cfg,
        use_state_vlm=vla_cfg.get("use_state_vlm", vla_cfg.get("use_state", True)),
        use_state_action_head=vla_cfg.get("use_state_action_head", vla_cfg.get("use_state", True)),
        state_dim=cfg.state_dim,
        future_action_window_size=cfg.action_horizon - 1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vla.to(device).eval()
    for p in vla.parameters():
        p.requires_grad = False

    rlc = model_cfg.get("rl_token", {})
    rl = RLTokenBottleneck(
        input_dim=vla.qwen_vl_interface.get_hidden_dim(),
        model_dim=rlc.get("model_dim"),
        num_encoder_layers=rlc.get("num_encoder_layers", 2),
        num_decoder_layers=rlc.get("num_decoder_layers", 2),
        num_heads=rlc.get("num_heads", 8),
        ffn_dim=rlc.get("ffn_dim"),
        dropout=rlc.get("dropout", 0.1),
        rl_token_dim=rlc.get("rl_token_dim"),
    ).to(device).eval()
    for p in rl.parameters():
        p.requires_grad = False
    ckpt = rlc.get("checkpoint")
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        rl.load_state_dict(state["rl_token_state_dict"])
    return vla, rl, device


def encode_state(vla, rl, obs: dict, device: torch.device) -> np.ndarray:
    inputs = {"images": obs["images"], "instructions": obs["instructions"]}
    if "states" in obs:
        inputs["states"] = torch.tensor(obs["states"], dtype=torch.float32, device=device)
    with torch.no_grad():
        zrl = vla.extract_rl_token(inputs, rl)
    return zrl.squeeze(0).detach().cpu().numpy()


def run_online_td3(cfg):
    """
    Env requirement:
    - reset() -> obs(dict)
    - step(action) -> next_obs(dict), reward(float), done(bool), info(dict)
    obs must include images/instructions and optional states.
    """
    from simulator.pick_place_env import PickPlaceEnv

    vla, rl, device = build_vla_and_rl(cfg)
    td3_cfg = cfg.raw_config.get("training", {}).get("online_rl", {}).get("td3", {})
    env = PickPlaceEnv()
    action_dim = cfg.action_dim
    state_dim = cfg.raw_config.get("model", {}).get("rl_token", {}).get("rl_token_dim") or vla.qwen_vl_interface.get_hidden_dim()
    agent = TD3Agent(state_dim, action_dim, td3_cfg.get("max_action", 1.0), td3_cfg, device)
    replay = ReplayBuffer(td3_cfg.get("buffer_size", 100000))

    episodes = td3_cfg.get("episodes", 100)
    start_steps = td3_cfg.get("start_steps", 1000)
    batch_size = td3_cfg.get("batch_size", 128)
    explore_noise = td3_cfg.get("explore_noise", 0.1)
    total_steps = 0

    for ep in range(episodes):
        obs = env.reset()
        state = encode_state(vla, rl, obs, device)
        done = False
        ep_reward = 0.0
        while not done:
            if total_steps < start_steps:
                action = np.random.uniform(-agent.max_action, agent.max_action, size=(action_dim,))
            else:
                action = agent.act(state, noise_std=explore_noise)
            next_obs, reward, done, _ = env.step(action)
            next_state = encode_state(vla, rl, next_obs, device)
            replay.add(Transition(state, action, float(reward), next_state, float(done)))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                agent.train_step(batch)
        print(f"episode={ep+1}/{episodes} reward={ep_reward:.3f} replay={len(replay)}")

    out_dir = Path(td3_cfg.get("save_dir", "./checkpoints/td3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": agent.actor.state_dict()}, out_dir / "td3_actor_final.pt")
    print(f"saved td3 actor at {out_dir / 'td3_actor_final.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Online TD3 with RL token state")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", "--dataset_path", dest="dataset_path", type=str, default=None)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_script_config(args.config, dataset_path=args.dataset_path, seed=args.seed)
    run_online_td3(cfg)


if __name__ == "__main__":
    main()
