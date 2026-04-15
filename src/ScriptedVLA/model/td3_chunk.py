"""
TD3 model for chunked action optimization.

This module provides a Gaussian actor and double-Q critic specialized for:
- RL token state (z_rl)
- proprioceptive state
- reference action chunks (e.g. predicted by VLA)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation_cls=nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(activation_cls())
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class TD3GaussianChunkActor(nn.Module):
    """
    Gaussian actor that predicts action chunks.
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: Optional[List[int]] = None,
        fixed_std: float = 0.1,
        max_action: float = 1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.rl_token_dim = rl_token_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.fixed_std = fixed_std
        self.max_action = max_action
        input_dim = rl_token_dim + state_dim + action_dim * chunk_size
        output_dim = action_dim * chunk_size
        self.mlp = _build_mlp(input_dim, output_dim, hidden_dims)

    def forward(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z_rl.shape[0]
        x = torch.cat([z_rl, state, ref_actions.reshape(batch_size, -1)], dim=-1)
        out = self.mlp(x).reshape(batch_size, self.chunk_size, self.action_dim)
        return torch.tanh(out) * self.max_action

    def sample(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_actions: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mean_action = self.forward(z_rl, state, ref_actions)
        if deterministic:
            return mean_action
        noise = torch.randn_like(mean_action) * self.fixed_std
        return (mean_action + noise).clamp(-self.max_action, self.max_action)


class TD3DoubleQCritic(nn.Module):
    """
    Double-Q critic operating on chunked actions.
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.rl_token_dim = rl_token_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        input_dim = rl_token_dim + state_dim + action_dim * chunk_size
        self.q1 = _build_mlp(input_dim, 1, hidden_dims)
        self.q2 = _build_mlp(input_dim, 1, hidden_dims)

    def forward(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = z_rl.shape[0]
        x = torch.cat([z_rl, state, actions.reshape(batch_size, -1)], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z_rl.shape[0]
        x = torch.cat([z_rl, state, actions.reshape(batch_size, -1)], dim=-1)
        return self.q1(x)


@dataclass
class TD3ChunkConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    fixed_std: float = 0.1
    max_action: float = 1.0
    ref_mask_prob: float = 0.5
    hidden_dims: Optional[List[int]] = None


class TD3ChunkAgent:
    """
    TD3 agent for chunked action optimization.
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        cfg: TD3ChunkConfig,
        device: torch.device,
    ):
        self.device = device
        self.cfg = cfg
        hidden_dims = cfg.hidden_dims if cfg.hidden_dims is not None else [256, 256]
        self.actor = TD3GaussianChunkActor(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            fixed_std=cfg.fixed_std,
            max_action=cfg.max_action,
        ).to(device)
        self.actor_target = TD3GaussianChunkActor(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            fixed_std=cfg.fixed_std,
            max_action=cfg.max_action,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3DoubleQCritic(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        ).to(device)
        self.critic_target = TD3DoubleQCritic(
            rl_token_dim=rl_token_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.total_updates = 0

    def mask_reference_actions(self, ref_actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.ref_mask_prob <= 0:
            return ref_actions
        mask = (torch.rand_like(ref_actions[..., :1]) >= self.cfg.ref_mask_prob).to(ref_actions.dtype)
        return ref_actions * mask

    def act(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_actions: torch.Tensor,
        deterministic: bool = True,
        apply_ref_mask: bool = False,
    ) -> torch.Tensor:
        z_rl = z_rl.to(self.device)
        state = state.to(self.device)
        ref_actions = ref_actions.to(self.device)
        if apply_ref_mask:
            ref_actions = self.mask_reference_actions(ref_actions)
        return self.actor.sample(z_rl, state, ref_actions, deterministic=deterministic)

    def train_step(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        ref_actions: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_z_rl: torch.Tensor,
        next_state: torch.Tensor,
        next_ref_actions: torch.Tensor,
        done: torch.Tensor,
        apply_ref_mask: bool = True,
    ) -> Dict[str, float]:
        z_rl = z_rl.to(self.device)
        state = state.to(self.device)
        ref_actions = ref_actions.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_z_rl = next_z_rl.to(self.device)
        next_state = next_state.to(self.device)
        next_ref_actions = next_ref_actions.to(self.device)
        done = done.to(self.device)

        masked_ref_actions = self.mask_reference_actions(ref_actions) if apply_ref_mask else ref_actions
        masked_next_ref_actions = self.mask_reference_actions(next_ref_actions) if apply_ref_mask else next_ref_actions

        with torch.no_grad():
            next_action = self.actor_target.sample(next_z_rl, next_state, masked_next_ref_actions, deterministic=False)
            noise = (torch.randn_like(next_action) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + noise).clamp(-self.cfg.max_action, self.cfg.max_action)
            target_q1, target_q2 = self.critic_target(next_z_rl, next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(z_rl, state, action)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.total_updates += 1
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_updates % self.cfg.policy_delay == 0:
            pred_action = self.actor(z_rl, state, masked_ref_actions)
            actor_loss = -self.critic.q1_only(z_rl, state, pred_action).mean()
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
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
