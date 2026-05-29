"""
Offline TD3 training with frozen VLA and RL token extractor.

在 libero/config_libero_object.yaml 中编辑顶层 ``train_rl_td3`` 后执行::

    python train_rl_td3.py

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from libero.libero_dataset_replay import resolve_training_episodes
from libero.rl_td3_replay import (
    OnlineFeatureProvider,
    ReplayBuffer,
    build_episode_index,
    build_replay_from_dataset,
    export_training_curves,
    load_frozen_vla_and_rl_token,
    load_lerobot_dataset,
    save_td3_agent,
    set_seed,
    train_from_batch,
)
from src.ScriptedVLA.model import TD3ChunkAgent, TD3ChunkConfig
from src.ScriptedVLA.utils import ScriptConfig, ensure_offline_mode_if_needed, load_script_config

DEFAULT_CONFIG_PATH = "libero/config_libero_object.yaml"

ensure_offline_mode_if_needed(DEFAULT_CONFIG_PATH)


@dataclass
class TrainRLTD3DatasetSettings:
    local_path: str
    task_index: Optional[int]
    episode_slice: Optional[List[int]]


@dataclass
class TrainRLTD3ReplaySettings:
    capacity: int
    stride: int


@dataclass
class TrainRLTD3TrainingSettings:
    batch_size: int
    num_updates: int
    update_ratio: int
    logging_steps: int
    save_steps: int


@dataclass
class TrainRLTD3Settings:
    config_path: str
    vla_checkpoint: Path
    rl_token_checkpoint: Path
    rl_chunk_size: int
    dataset: TrainRLTD3DatasetSettings
    replay: TrainRLTD3ReplaySettings
    training: TrainRLTD3TrainingSettings
    td3_cfg: TD3ChunkConfig
    save_dir: Path
    device: str
    seed: int
    validate_vla_checkpoint: bool
    resolved_episodes: Optional[List[int]]
    rl_token_network_cfg: Dict[str, Any]


def load_train_rl_td3_settings(raw: Dict[str, Any], cfg: ScriptConfig) -> TrainRLTD3Settings:
    if "train_rl_td3" not in raw:
        raise KeyError(
            f"missing top-level 'train_rl_td3' in {cfg.config_path}; "
            "add train_rl_td3 section (see libero/config_libero_object.yaml)"
        )
    block = raw["train_rl_td3"]
    token_block = raw.get("train_rl_token") or {}

    vla_ckpt = block.get("vla_checkpoint") or token_block.get("vla_checkpoint")
    if not vla_ckpt:
        raise ValueError("train_rl_td3.vla_checkpoint must be set in config")
    rl_token_ckpt = block.get("rl_token_checkpoint")
    if not rl_token_ckpt:
        raise ValueError("train_rl_td3.rl_token_checkpoint must be set in config")

    ds_cfg = block.get("dataset") or {}
    token_ds = token_block.get("dataset") or {}
    local_path = ds_cfg.get("local_path") or token_ds.get("local_path") or cfg.dataset_path
    task_index = ds_cfg.get("task_index", token_ds.get("task_index"))
    episode_slice = ds_cfg.get("episode_slice", token_ds.get("episode_slice"))

    dataset_path = Path(local_path).resolve()
    resolved = resolve_training_episodes(str(dataset_path), task_index, episode_slice)

    model_cfg = raw.get("model", {})
    action_head = model_cfg.get("action_head", {})
    default_chunk = int(block.get("rl_chunk_size", action_head.get("action_horizon", 10)))

    replay_cfg = block.get("replay") or {}
    train_cfg = block.get("training") or {}
    td3_raw = block.get("td3") or {}
    hidden_dims = td3_raw.get("hidden_dims", [256, 256])

    seed = block.get("seed")
    if seed is None:
        seed = cfg.seed

    td3_cfg = TD3ChunkConfig(
        gamma=float(td3_raw.get("gamma", 0.99)),
        tau=float(td3_raw.get("tau", 0.005)),
        actor_lr=float(td3_raw.get("actor_lr", 1e-4)),
        critic_lr=float(td3_raw.get("critic_lr", 1e-3)),
        policy_noise=float(td3_raw.get("policy_noise", 0.2)),
        noise_clip=float(td3_raw.get("noise_clip", 0.5)),
        policy_delay=int(td3_raw.get("policy_delay", 2)),
        fixed_std=float(td3_raw.get("actor_std", 0.1)),
        max_action=float(td3_raw.get("max_action", 1.0)),
        ref_mask_prob=float(td3_raw.get("ref_mask_prob", 0.5)),
        policy_constraint_beta=float(td3_raw.get("policy_constraint_beta", 0.1)),
        use_chunk_return_target=bool(td3_raw.get("use_chunk_return_target", True)),
        hidden_dims=list(hidden_dims),
    )

    device = block.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainRLTD3Settings(
        config_path=cfg.config_path,
        vla_checkpoint=Path(vla_ckpt).expanduser().resolve(),
        rl_token_checkpoint=Path(rl_token_ckpt).expanduser().resolve(),
        rl_chunk_size=int(default_chunk),
        dataset=TrainRLTD3DatasetSettings(
            local_path=str(dataset_path),
            task_index=task_index,
            episode_slice=episode_slice,
        ),
        replay=TrainRLTD3ReplaySettings(
            capacity=int(replay_cfg.get("capacity", 200000)),
            stride=int(replay_cfg.get("stride", 2)),
        ),
        training=TrainRLTD3TrainingSettings(
            batch_size=int(train_cfg.get("batch_size", 32)),
            num_updates=int(train_cfg.get("num_updates", 0)),
            update_ratio=int(train_cfg.get("update_ratio", 3)),
            logging_steps=int(train_cfg.get("logging_steps", 50)),
            save_steps=int(train_cfg.get("save_steps", 0)),
        ),
        td3_cfg=td3_cfg,
        save_dir=Path(block.get("save_dir", "./checkpoints/rl_td3")).expanduser().resolve(),
        device=str(device),
        seed=int(seed),
        validate_vla_checkpoint=bool(block.get("validate_vla_checkpoint", True)),
        resolved_episodes=resolved,
        rl_token_network_cfg=dict(token_block.get("network") or {}),
    )


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _resolve_num_updates(num_updates: int, replay_len: int, batch_size: int, update_ratio: int) -> int:
    if num_updates > 0:
        return num_updates
    return max(1, int(math.ceil(replay_len / max(batch_size, 1))) * update_ratio)


def train_rl_td3(cfg: ScriptConfig, settings: TrainRLTD3Settings) -> None:
    set_seed(settings.seed)
    device = torch.device(settings.device)
    settings.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen VLA: {settings.vla_checkpoint}")
    print(f"Loading frozen RL token: {settings.rl_token_checkpoint}")
    cfg, vla_model, rl_encoder = load_frozen_vla_and_rl_token(
        settings.config_path,
        settings.dataset.local_path,
        str(settings.vla_checkpoint),
        str(settings.rl_token_checkpoint),
        device,
        validate_vla=settings.validate_vla_checkpoint,
        rl_token_network_cfg=settings.rl_token_network_cfg,
    )

    vla_trainable = _count_trainable_params(vla_model)
    rl_trainable = _count_trainable_params(rl_encoder)
    if vla_trainable > 0 or rl_trainable > 0:
        raise RuntimeError(
            f"expected frozen VLA/RL token, got trainable params: vla={vla_trainable}, rl={rl_trainable}"
        )
    print(f"  VLA trainable params: {vla_trainable}, RL token trainable params: {rl_trainable}")

    vla_action_horizon = int(cfg.action_horizon)
    print(
        f"  state_dim={cfg.state_dim}, action_dim={cfg.action_dim}, "
        f"vla_action_horizon={vla_action_horizon}, rl_chunk_size={settings.rl_chunk_size}"
    )

    episodes = settings.resolved_episodes
    if episodes is not None:
        print(f"  task_index={settings.dataset.task_index}, episodes={len(episodes)}")
    dataset = load_lerobot_dataset(
        Path(settings.dataset.local_path),
        vla_action_horizon,
        episodes=episodes,
    )

    ep_map = build_episode_index(dataset)
    replay = build_replay_from_dataset(
        dataset,
        ep_map,
        cfg.state_key,
        settings.rl_chunk_size,
        settings.replay.stride,
        settings.td3_cfg.gamma,
        device,
        settings.replay.capacity,
    )
    print(f"  replay size={len(replay)}")

    provider = OnlineFeatureProvider(
        dataset,
        cfg.image_keys,
        cfg.state_key,
        vla_model,
        rl_encoder,
        device,
        settings.rl_chunk_size,
    )

    rl_token_dim = rl_encoder.rl_token_dim
    agent = TD3ChunkAgent(
        rl_token_dim=rl_token_dim,
        state_dim=int(cfg.state_dim),
        action_dim=int(cfg.action_dim),
        chunk_size=settings.rl_chunk_size,
        cfg=settings.td3_cfg,
        device=device,
    )
    td3_trainable = _count_trainable_params(agent.actor) + _count_trainable_params(agent.critic)
    print(f"  TD3 trainable params: {td3_trainable}")

    tr = settings.training
    num_updates = _resolve_num_updates(tr.num_updates, len(replay), tr.batch_size, tr.update_ratio)
    print(f"  training for {num_updates} updates (batch_size={tr.batch_size})")

    logs: List[Dict[str, float]] = []
    pbar = tqdm(range(num_updates), desc="rl_td3", total=num_updates)
    for step in pbar:
        batch = replay.sample(tr.batch_size)
        metrics = train_from_batch(agent, provider, batch, apply_ref_mask=True)
        metrics["step"] = float(step + 1)
        logs.append(metrics)

        if (step + 1) % tr.logging_steps == 0:
            pbar.set_postfix(
                q_loss=f"{metrics['critic_loss']:.4f}",
                pi_loss=f"{metrics['actor_loss']:.4f}",
                bc=f"{metrics['actor_constraint_loss']:.4f}",
            )

        if tr.save_steps > 0 and (step + 1) % tr.save_steps == 0:
            ckpt_path = settings.save_dir / f"td3_agent_step_{step + 1}.pt"
            save_td3_agent(
                ckpt_path,
                agent,
                rl_token_dim=rl_token_dim,
                state_dim=int(cfg.state_dim),
                action_dim=int(cfg.action_dim),
                chunk_size=settings.rl_chunk_size,
                td3_cfg=settings.td3_cfg,
            )

    final_path = settings.save_dir / "td3_agent_final.pt"
    save_td3_agent(
        final_path,
        agent,
        rl_token_dim=rl_token_dim,
        state_dim=int(cfg.state_dim),
        action_dim=int(cfg.action_dim),
        chunk_size=settings.rl_chunk_size,
        td3_cfg=settings.td3_cfg,
    )
    export_training_curves(logs, settings.save_dir)
    print(f"done. checkpoint={final_path}, curves in {settings.save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline TD3 with frozen VLA and RL token (config: train_rl_td3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            f"  python train_rl_td3.py\n"
            f"  python train_rl_td3.py --config {DEFAULT_CONFIG_PATH}\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}）",
    )
    args = parser.parse_args()

    cfg = load_script_config(args.config, dataset_path=None)
    settings = load_train_rl_td3_settings(cfg.raw_config, cfg)
    cfg.dataset_path = settings.dataset.local_path
    cfg.seed = settings.seed
    train_rl_td3(cfg, settings)


if __name__ == "__main__":
    main()
