from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, max_action: float):
        super().__init__()
        self.max_action = float(max_action)
        self.mlp = _build_mlp(obs_dim, action_dim, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(obs)) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = _build_mlp(in_dim, 1, hidden_dim)
        self.q2 = _build_mlp(in_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_action: float = 1.0
    hidden_dim: int = 256


class TD3Agent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: TD3Config, device: torch.device):
        self.device = device
        self.cfg = cfg

        self.actor = Actor(obs_dim, action_dim, cfg.hidden_dim, cfg.max_action).to(device)
        self.actor_target = Actor(obs_dim, action_dim, cfg.hidden_dim, cfg.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.total_updates = 0

    def act(self, obs: torch.Tensor, deterministic: bool = True, noise_std: float = 0.0) -> torch.Tensor:
        obs = obs.to(self.device)
        action = self.actor(obs)
        if not deterministic and noise_std > 0.0:
            action = action + torch.randn_like(action) * float(noise_std)
        return action.clamp(-self.cfg.max_action, self.cfg.max_action)

    def train_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Dict[str, float]:
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            noise = (torch.randn_like(next_action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_action = (next_action + noise).clamp(-self.cfg.max_action, self.cfg.max_action)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.total_updates += 1
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_updates % self.cfg.policy_delay == 0:
            pred_action = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, pred_action).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update_targets()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
