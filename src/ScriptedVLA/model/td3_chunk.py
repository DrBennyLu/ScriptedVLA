"""
TD3 model for chunked action optimization.

This module provides a Gaussian actor and double-Q critic specialized for:
- RL token state (z_rl)
- proprioceptive state
- reference action chunks (e.g. predicted by VLA)

author: Benny Lu
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
    """
    构建一个按给定隐藏层配置堆叠的 MLP 网络。

    Args:
        input_dim: 输入特征维度。
        output_dim: 输出特征维度。
        hidden_dims: 隐藏层维度列表，按顺序堆叠。
        activation_cls: 隐藏层激活函数类型，默认 ReLU。

    Returns:
        nn.Sequential: 构建完成的前馈网络。
    """
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
    TD3 的高斯 Actor，用于一次预测一个动作 chunk。
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
        """
        初始化 Actor 网络。

        Args:
            rl_token_dim: RL token 维度。
            state_dim: 低维状态向量维度。
            action_dim: 单步动作维度。
            chunk_size: 一次输出的动作步数 C。
            hidden_dims: MLP 隐藏层维度列表。
            fixed_std: 采样时使用的固定高斯噪声标准差。
            max_action: 动作绝对值上限。

        Returns:
            None.
        """
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
        """
        前向推理，输出动作均值（经 tanh 限幅）。

        Args:
            z_rl: RL token，形状一般为 [B, rl_token_dim]。
            state: 状态向量，形状一般为 [B, state_dim]。
            ref_actions: 参考动作 chunk，形状一般为 [B, C, action_dim]。

        Returns:
            torch.Tensor: 预测动作 chunk，形状 [B, C, action_dim]。
        """
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
        """
        基于前向均值动作进行采样，支持确定性/随机性输出。

        Args:
            z_rl: RL token，形状一般为 [B, rl_token_dim]。
            state: 状态向量，形状一般为 [B, state_dim]。
            ref_actions: 参考动作 chunk，形状一般为 [B, C, action_dim]。
            deterministic: 为 True 时直接返回均值动作；否则加高斯噪声。

        Returns:
            torch.Tensor: 采样后的动作 chunk，形状 [B, C, action_dim]。
        """
        mean_action = self.forward(z_rl, state, ref_actions)
        if deterministic:
            return mean_action
        noise = torch.randn_like(mean_action) * self.fixed_std
        return (mean_action + noise).clamp(-self.max_action, self.max_action)


class TD3DoubleQCritic(nn.Module):
    """
    TD3 的双 Q Critic，对 chunk 动作估计 Q 值。
    """

    def __init__(
        self,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        初始化双 Q 网络。

        Args:
            rl_token_dim: RL token 维度。
            state_dim: 状态向量维度。
            action_dim: 单步动作维度。
            chunk_size: 动作 chunk 长度 C。
            hidden_dims: MLP 隐藏层维度列表。

        Returns:
            None.
        """
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
        """
        同时计算 Q1 与 Q2。

        Args:
            z_rl: RL token，形状一般为 [B, rl_token_dim]。
            state: 状态向量，形状一般为 [B, state_dim]。
            actions: 动作 chunk，形状一般为 [B, C, action_dim]。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Q1: 形状 [B, 1]
                - Q2: 形状 [B, 1]
        """
        batch_size = z_rl.shape[0]
        x = torch.cat([z_rl, state, actions.reshape(batch_size, -1)], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(
        self,
        z_rl: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        仅计算 Q1（常用于 actor loss）。

        Args:
            z_rl: RL token，形状一般为 [B, rl_token_dim]。
            state: 状态向量，形状一般为 [B, state_dim]。
            actions: 动作 chunk，形状一般为 [B, C, action_dim]。

        Returns:
            torch.Tensor: Q1 值，形状 [B, 1]。
        """
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
    policy_constraint_beta: float = 0.1
    use_chunk_return_target: bool = True
    hidden_dims: Optional[List[int]] = None


class TD3ChunkAgent:
    """
    面向 chunk 动作优化的 TD3 Agent。
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
        """
        初始化 TD3 Agent，包括 actor/critic 与目标网络、优化器。

        Args:
            rl_token_dim: RL token 维度。
            state_dim: 状态向量维度。
            action_dim: 单步动作维度。
            chunk_size: 动作 chunk 长度 C。
            cfg: TD3 超参数配置。
            device: 训练与推理设备。

        Returns:
            None.
        """
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
        """
        按配置概率对参考动作进行随机 mask（置零）。

        Args:
            ref_actions: 参考动作 chunk，形状一般为 [B, C, action_dim]。

        Returns:
            torch.Tensor: mask 后的参考动作，形状与输入一致。
        """
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
        """
        对外动作接口：根据当前输入生成动作 chunk。

        Args:
            z_rl: RL token，形状一般为 [B, rl_token_dim]。
            state: 状态向量，形状一般为 [B, state_dim]。
            ref_actions: 参考动作 chunk，形状一般为 [B, C, action_dim]。
            deterministic: 是否使用确定性输出。
            apply_ref_mask: 是否先对参考动作做随机 mask。

        Returns:
            torch.Tensor: 生成动作 chunk，形状 [B, C, action_dim]。
        """
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
        chunk_return: Optional[torch.Tensor],
        next_z_rl: torch.Tensor,
        next_state: torch.Tensor,
        next_ref_actions: torch.Tensor,
        done: torch.Tensor,
        apply_ref_mask: bool = True,
    ) -> Dict[str, float]:
        """
        执行一次 TD3 训练更新（critic 必更新，actor 按 policy_delay 更新）。

        Args:
            z_rl: 当前时刻 RL token，形状 [B, rl_token_dim]。
            state: 当前时刻状态向量，形状 [B, state_dim]。
            ref_actions: 当前时刻参考动作 chunk，形状 [B, C, action_dim]。
            action: 当前监督动作 chunk（GT 或行为策略动作），形状 [B, C, action_dim]。
            reward: 单步奖励，形状一般为 [B, 1]。
            chunk_return: chunk 折扣回报，形状一般为 [B, 1]；为 None 时退回单步 TD 目标。
            next_z_rl: 下一时刻 RL token，形状 [B, rl_token_dim]。
            next_state: 下一时刻状态向量，形状 [B, state_dim]。
            next_ref_actions: 下一时刻参考动作 chunk，形状 [B, C, action_dim]。
            done: 终止标记，形状一般为 [B, 1]。
            apply_ref_mask: 是否对当前/下一时刻 reference 做 mask。

        Returns:
            Dict[str, float]: 训练指标字典，包含 critic/actor 损失及 Q 统计等。
        """
        z_rl = z_rl.to(self.device)
        state = state.to(self.device)
        ref_actions = ref_actions.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        chunk_return_tensor = None if chunk_return is None else chunk_return.to(self.device)
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
            if self.cfg.use_chunk_return_target and chunk_return_tensor is not None:
                gamma_bootstrap = self.cfg.gamma ** self.actor.chunk_size
                y = chunk_return_tensor + (1.0 - done) * gamma_bootstrap * target_q
            else:
                y = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(z_rl, state, action)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.total_updates += 1
        actor_loss = torch.tensor(0.0, device=self.device)
        actor_constraint_loss = torch.tensor(0.0, device=self.device)
        pred_vs_ref_mse = torch.tensor(0.0, device=self.device)
        if self.total_updates % self.cfg.policy_delay == 0:
            pred_action = self.actor(z_rl, state, masked_ref_actions)
            actor_constraint_loss = F.mse_loss(pred_action, masked_ref_actions)
            pred_vs_ref_mse = actor_constraint_loss.detach()
            actor_loss = -self.critic.q1_only(z_rl, state, pred_action).mean()
            actor_loss = actor_loss + self.cfg.policy_constraint_beta * actor_constraint_loss
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update_targets()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "actor_constraint_loss": float(actor_constraint_loss.item()),
            "pred_vs_ref_mse": float(pred_vs_ref_mse.item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "chunk_return_mean": float(chunk_return_tensor.mean().item()) if chunk_return_tensor is not None else 0.0,
        }

    def _soft_update_targets(self) -> None:
        """
        按 tau 对 actor/critic 目标网络执行软更新。

        Args:
            None.

        Returns:
            None.
        """
        tau = self.cfg.tau
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
