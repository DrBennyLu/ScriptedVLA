"""
Offline TD3 training with frozen VLA and RL token extractor.

Edit ``train_rl_td3`` in libero/config_libero_object.yaml, then::

    python train_rl_td3.py
    python train_rl_td3.py --build-replay-only --replay-cache-dir ./data/replay_buffers/task6
    python train_rl_td3.py --replay-cache-dir ./data/replay_buffers/task6
    python train_rl_td3.py --config libero/config_libero_object.yaml

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from libero.libero_dataset_replay import resolve_training_episodes
from libero.rl_td3_replay import (
    ReplayBuffer,
    build_episode_index,
    build_replay_cache_meta,
    build_replay_from_dataset,
    export_training_curves,
    load_frozen_vla_and_rl_token,
    load_lerobot_dataset,
    load_replay_buffer,
    replay_cache_matches,
    resolve_rl_token_dim_for_meta,
    save_replay_buffer,
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
    cache_path: Optional[Path]
    rebuild_cache: bool
    feature_batch_size: int


@dataclass
class TrainRLTD3CheckpointSettings:
    save_dir: Path
    final_name: str
    step_name_template: str


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
    checkpoint: TrainRLTD3CheckpointSettings
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
    ckpt_cfg = block.get("checkpoint") or {}
    td3_raw = block.get("td3") or {}
    hidden_dims = td3_raw.get("hidden_dims", [256, 256])

    seed = block.get("seed")
    if seed is None:
        seed = cfg.seed

    legacy_save_dir = Path(block.get("save_dir", "./checkpoints/rl_td3")).expanduser().resolve()
    checkpoint_save_dir = Path(ckpt_cfg.get("save_dir", legacy_save_dir)).expanduser().resolve()

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
            cache_path=(
                Path(replay_cfg["cache_path"]).expanduser().resolve()
                if replay_cfg.get("cache_path")
                else None
            ),
            rebuild_cache=bool(replay_cfg.get("rebuild_cache", False)),
            feature_batch_size=int(replay_cfg.get("feature_batch_size", 8)),
        ),
        checkpoint=TrainRLTD3CheckpointSettings(
            save_dir=checkpoint_save_dir,
            final_name=str(ckpt_cfg.get("final_name", "td3_agent_final.pt")),
            step_name_template=str(ckpt_cfg.get("step_name_template", "td3_agent_step_{step}.pt")),
        ),
        training=TrainRLTD3TrainingSettings(
            batch_size=int(train_cfg.get("batch_size", 32)),
            num_updates=int(train_cfg.get("num_updates", 0)),
            update_ratio=int(train_cfg.get("update_ratio", 3)),
            logging_steps=int(train_cfg.get("logging_steps", 50)),
            save_steps=int(train_cfg.get("save_steps", 0)),
        ),
        td3_cfg=td3_cfg,
        save_dir=checkpoint_save_dir,
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


def _step_checkpoint_path(settings: TrainRLTD3Settings, step: int) -> Path:
    name = settings.checkpoint.step_name_template.format(step=step)
    return settings.checkpoint.save_dir / name


def _final_checkpoint_path(settings: TrainRLTD3Settings) -> Path:
    return settings.checkpoint.save_dir / settings.checkpoint.final_name


def resolve_replay_cache_path(
    settings: TrainRLTD3Settings,
    *,
    replay_cache_dir: Optional[Path] = None,
    replay_cache_file: Optional[Path] = None,
) -> Path:
    if replay_cache_file is not None:
        return replay_cache_file.expanduser().resolve()
    if replay_cache_dir is not None:
        return replay_cache_dir.expanduser().resolve() / "replay_cache.pt"
    if settings.replay.cache_path is not None:
        return settings.replay.cache_path
    raise ValueError(
        "replay cache path not set: configure train_rl_td3.replay.cache_path "
        "or pass --replay-cache / --replay-cache-dir"
    )


def build_or_load_replay_buffer(
    cfg: ScriptConfig,
    settings: TrainRLTD3Settings,
    device: torch.device,
    *,
    replay_cache_dir: Optional[Path] = None,
    replay_cache_file: Optional[Path] = None,
    rebuild_cache: Optional[bool] = None,
) -> tuple[ReplayBuffer, int, Path]:
    """
    Build offline replay from dataset (with precomputed z_rl) or load from disk cache.

    Returns:
        (replay, rl_token_dim, cache_path)
    """
    vla_action_horizon = int(cfg.action_horizon)
    episodes = settings.resolved_episodes
    cache_path = resolve_replay_cache_path(
        settings,
        replay_cache_dir=replay_cache_dir,
        replay_cache_file=replay_cache_file,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rl_token_dim_meta = resolve_rl_token_dim_for_meta(
        settings.rl_token_checkpoint,
        cfg.raw_config,
        settings.rl_token_network_cfg,
    )
    cache_meta_expected = build_replay_cache_meta(
        chunk_len=settings.rl_chunk_size,
        stride=settings.replay.stride,
        gamma=settings.td3_cfg.gamma,
        state_dim=int(cfg.state_dim),
        rl_token_dim=rl_token_dim_meta,
        dataset_path=settings.dataset.local_path,
        episodes=episodes,
        vla_checkpoint=str(settings.vla_checkpoint),
        rl_token_checkpoint=str(settings.rl_token_checkpoint),
        config_path=settings.config_path,
        task_index=settings.dataset.task_index,
    )

    do_rebuild = settings.replay.rebuild_cache if rebuild_cache is None else rebuild_cache
    replay = None
    rl_token_dim = int(cache_meta_expected["rl_token_dim"])

    if cache_path.is_file() and not do_rebuild:
        loaded_replay, cached_meta = load_replay_buffer(cache_path, settings.replay.capacity)
        ok, reason = replay_cache_matches(cache_meta_expected, cached_meta)
        if ok:
            replay = loaded_replay
            rl_token_dim = int(cached_meta["rl_token_dim"])
            print(f"  replay cache hit: {cache_path} (size={len(replay)})")
        else:
            print(f"  replay cache stale ({reason}), rebuilding...")

    if replay is None:
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
            vla_model,
            rl_encoder,
            cfg.image_keys,
            settings.replay.feature_batch_size,
        )
        rl_token_dim = int(rl_encoder.rl_token_dim)
        cache_meta_expected["rl_token_dim"] = rl_token_dim
        print(f"  replay size={len(replay)}")
        save_replay_buffer(cache_path, replay, cache_meta_expected)
    else:
        print(f"  replay size={len(replay)} (from cache, skipping VLA load)")

    tr0 = next(iter(replay.buf))
    if tr0.z_rl.numel() > 0 and int(tr0.z_rl.shape[0]) != rl_token_dim:
        raise RuntimeError(
            f"replay z_rl dim {tr0.z_rl.shape[0]} != expected rl_token_dim {rl_token_dim}"
        )

    return replay, rl_token_dim, cache_path


def build_replay_only(
    cfg: ScriptConfig,
    settings: TrainRLTD3Settings,
    *,
    replay_cache_dir: Optional[Path] = None,
    replay_cache_file: Optional[Path] = None,
    rebuild_cache: Optional[bool] = None,
) -> Path:
    """Run dataset -> replay buffer pipeline and save cache; no TD3 training."""
    set_seed(settings.seed)
    device = torch.device(settings.device)
    print(
        f"[build_replay] state_dim={cfg.state_dim}, action_dim={cfg.action_dim}, "
        f"rl_chunk_size={settings.rl_chunk_size}, stride={settings.replay.stride}"
    )
    if settings.resolved_episodes is not None:
        print(f"[build_replay] task_index={settings.dataset.task_index}, episodes={len(settings.resolved_episodes)}")

    replay, rl_token_dim, cache_path = build_or_load_replay_buffer(
        cfg,
        settings,
        device,
        replay_cache_dir=replay_cache_dir,
        replay_cache_file=replay_cache_file,
        rebuild_cache=rebuild_cache,
    )
    tr0 = next(iter(replay.buf))
    print(
        f"[build_replay] done: {len(replay)} transitions, rl_token_dim={rl_token_dim}, "
        f"cache={cache_path}"
    )
    print(
        f"[build_replay] sample: idx {tr0.sample_index} -> {tr0.next_sample_index}, "
        f"z_rl={tuple(tr0.z_rl.shape)}, action={tuple(tr0.action.shape)}"
    )
    return cache_path


def train_rl_td3(
    cfg: ScriptConfig,
    settings: TrainRLTD3Settings,
    *,
    replay_cache_dir: Optional[Path] = None,
    replay_cache_file: Optional[Path] = None,
    rebuild_cache: Optional[bool] = None,
) -> None:
    set_seed(settings.seed)
    device = torch.device(settings.device)
    settings.checkpoint.save_dir.mkdir(parents=True, exist_ok=True)

    vla_action_horizon = int(cfg.action_horizon)
    if settings.rl_chunk_size != vla_action_horizon:
        warnings.warn(
            f"rl_chunk_size ({settings.rl_chunk_size}) != vla action_horizon ({vla_action_horizon})",
            stacklevel=2,
        )
    print(
        f"  state_dim={cfg.state_dim}, action_dim={cfg.action_dim}, "
        f"vla_action_horizon={vla_action_horizon}, rl_chunk_size={settings.rl_chunk_size}"
    )
    print(f"  TD3 checkpoint dir: {settings.checkpoint.save_dir}")

    episodes = settings.resolved_episodes
    if episodes is not None:
        print(f"  task_index={settings.dataset.task_index}, episodes={len(episodes)}")

    replay, rl_token_dim, cache_path = build_or_load_replay_buffer(
        cfg,
        settings,
        device,
        replay_cache_dir=replay_cache_dir,
        replay_cache_file=replay_cache_file,
        rebuild_cache=rebuild_cache,
    )
    print(f"  using replay cache: {cache_path}")

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
        metrics = train_from_batch(agent, batch, apply_ref_mask=True)
        metrics["step"] = float(step + 1)
        logs.append(metrics)

        if (step + 1) % tr.logging_steps == 0:
            pbar.set_postfix(
                q_loss=f"{metrics['critic_loss']:.4f}",
                pi_loss=f"{metrics['actor_loss']:.4f}",
                bc=f"{metrics['actor_constraint_loss']:.4f}",
            )

        if tr.save_steps > 0 and (step + 1) % tr.save_steps == 0:
            ckpt_path = _step_checkpoint_path(settings, step + 1)
            save_td3_agent(
                ckpt_path,
                agent,
                rl_token_dim=rl_token_dim,
                state_dim=int(cfg.state_dim),
                action_dim=int(cfg.action_dim),
                chunk_size=settings.rl_chunk_size,
                td3_cfg=settings.td3_cfg,
            )

    final_path = _final_checkpoint_path(settings)
    save_td3_agent(
        final_path,
        agent,
        rl_token_dim=rl_token_dim,
        state_dim=int(cfg.state_dim),
        action_dim=int(cfg.action_dim),
        chunk_size=settings.rl_chunk_size,
        td3_cfg=settings.td3_cfg,
    )
    export_training_curves(logs, settings.checkpoint.save_dir)
    print(f"done. checkpoint={final_path}, curves in {settings.checkpoint.save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline TD3 with frozen VLA and RL token (config: train_rl_td3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            f"  python train_rl_td3.py\n"
            f"  python train_rl_td3.py --config {DEFAULT_CONFIG_PATH}\n"
            f"  python train_rl_td3.py --build-replay-only --replay-cache-dir ./data/replay_buffers/task6\n"
            f"  python train_rl_td3.py --build-replay-only --replay-cache ./data/replay_buffers/task6/replay_cache.pt --rebuild-cache\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--build-replay-only",
        action="store_true",
        help="仅构建并保存 replay buffer（预计算 z_rl），不进行 TD3 训练",
    )
    parser.add_argument(
        "--replay-cache-dir",
        type=str,
        default=None,
        help="replay 缓存目录，保存为 <dir>/replay_cache.pt（覆盖 config 中 replay.cache_path）",
    )
    parser.add_argument(
        "--replay-cache",
        type=str,
        default=None,
        help="replay 缓存文件完整路径（优先级高于 --replay-cache-dir）",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="强制重建 replay 缓存（忽略已有文件）",
    )
    args = parser.parse_args()

    cfg = load_script_config(args.config, dataset_path=None)
    settings = load_train_rl_td3_settings(cfg.raw_config, cfg)
    cfg.dataset_path = settings.dataset.local_path
    cfg.seed = settings.seed

    replay_cache_dir = Path(args.replay_cache_dir).expanduser() if args.replay_cache_dir else None
    replay_cache_file = Path(args.replay_cache).expanduser() if args.replay_cache else None
    rebuild_cache = True if args.rebuild_cache else None

    if args.build_replay_only:
        build_replay_only(
            cfg,
            settings,
            replay_cache_dir=replay_cache_dir,
            replay_cache_file=replay_cache_file,
            rebuild_cache=rebuild_cache,
        )
        return

    train_rl_td3(
        cfg,
        settings,
        replay_cache_dir=replay_cache_dir,
        replay_cache_file=replay_cache_file,
        rebuild_cache=rebuild_cache,
    )


if __name__ == "__main__":
    main()
