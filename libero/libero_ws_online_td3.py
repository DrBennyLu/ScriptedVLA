#!/usr/bin/env python3
"""
Online WS TD3 training with frozen VLA + RL token.

python -m libero.libero_ws_online_td3 --config libero/config_libero_object.yaml --task-id 6 --chunk-steps 10

Recommended (aligned with eval, conservative updates)::

  python -m libero.libero_ws_online_td3 --config libero/config_libero_object.yaml \\
    --rollout-deterministic --no-rollout-ref-mask \\
    --train-updates-per-step 1 --actor-lr-scale 0.1 --critic-lr-scale 0.1 \\
    --online-sample-ratio 0.1

"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

_entry_path = Path(__file__).resolve().parent / "_entry.py"
_spec = importlib.util.spec_from_file_location("libero_entry", _entry_path)
_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_entry)
_entry.maybe_reroute_main(__name__, __package__, __file__)

from train_rl_td3 import load_train_rl_td3_settings

from .libero_action_adapter import model_action_to_libero
from .libero_task_mapping import add_task_id_cli_arguments, resolve_task_ids_from_args
from .libero_ws_client import LiberoWSClient
from .libero_ws_td3_eval_core import ws_obs_to_vla_model_inputs
from .rl_td3_replay import (
    ReplayBuffer,
    ReplayTransition,
    apply_online_td3_lr_scale,
    export_training_curves,
    load_frozen_vla_and_rl_token,
    load_replay_buffer,
    load_td3_agent,
    online_chunk_reward_fields,
    prune_old_buffer_files,
    replay_positive_reward_stats,
    sample_mixed_batch_with_sources,
    save_online_step_buffer,
    save_td3_agent,
    set_seed,
    summarize_replay_buffer,
    train_from_batch_with_diagnostics,
)
from src.ScriptedVLA.utils import load_config, load_script_config


@dataclass
class OnlineWSRLSettings:
    max_train_steps: int
    save_every_steps: int
    train_updates_per_step: int
    replay_capacity: int
    buffer_dir: Path
    buffer_max_files: int
    batch_size: int
    td3_checkpoint: Path
    checkpoint_dir: Path
    online_sample_ratio: float
    replay_cache_path: Optional[Path]
    offline_replay_capacity: int
    rollout_deterministic: bool
    rollout_apply_ref_mask: bool
    actor_lr_scale: float
    critic_lr_scale: float
    align_online_rewards: bool
    logging_steps: int


def _resolve_td3_checkpoint_path(config: dict, args) -> Path:
    block = config.get("train_rl_td3") or {}
    ckpt_block = block.get("checkpoint") or {}
    eval_block = block.get("eval") or {}
    ckpt_dir = ckpt_block.get("save_dir") or block.get("save_dir")
    step_template = ckpt_block.get("step_name_template", "td3_agent_step_{step}.pt")
    final_name = ckpt_block.get("final_name", "td3_agent_final.pt")

    if args.td3_checkpoint and args.td3_step is not None:
        raise ValueError("use only one of --td3-checkpoint and --td3-step")

    if args.td3_checkpoint:
        path = Path(args.td3_checkpoint).expanduser()
    elif args.td3_step is not None:
        if not ckpt_dir:
            raise ValueError("--td3-step requires train_rl_td3.checkpoint.save_dir in config")
        path = Path(ckpt_dir) / step_template.format(step=int(args.td3_step))
    elif eval_block.get("td3_checkpoint"):
        path = Path(eval_block["td3_checkpoint"]).expanduser()
    elif eval_block.get("td3_step") is not None:
        if not ckpt_dir:
            raise ValueError("eval.td3_step requires train_rl_td3.checkpoint.save_dir in config")
        path = Path(ckpt_dir) / step_template.format(step=int(eval_block["td3_step"]))
    elif ckpt_dir:
        path = Path(ckpt_dir) / final_name
    else:
        raise ValueError("missing TD3 checkpoint path")

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TD3 checkpoint not found: {path}")
    return path


def _resolve_online_ws_settings(raw_config: dict, args) -> OnlineWSRLSettings:
    block = raw_config.get("Online_ws_rl_training") or {}
    td3_block = raw_config.get("train_rl_td3") or {}
    ckpt_block = td3_block.get("checkpoint") or {}
    default_save_dir = Path(
        ckpt_block.get("save_dir", td3_block.get("save_dir", "./checkpoints/rl_td3"))
    ).expanduser().resolve()

    def choose(key: str, cli_val, default):
        if cli_val is not None:
            return cli_val
        return block.get(key, default)

    td3_path_arg = argparse.Namespace(
        td3_checkpoint=choose("td3_checkpoint", args.td3_checkpoint, None),
        td3_step=choose("td3_step", args.td3_step, None),
    )
    td3_checkpoint = _resolve_td3_checkpoint_path(raw_config, td3_path_arg)

    replay_cfg = td3_block.get("replay") or {}
    default_cache = replay_cfg.get("cache_path")
    replay_cache_raw = choose("replay_cache_path", args.replay_cache_path, default_cache)
    replay_cache_path = (
        Path(replay_cache_raw).expanduser().resolve() if replay_cache_raw else None
    )
    default_offline_cap = int(replay_cfg.get("capacity", 4_000_000))

    rollout_deterministic = choose(
        "rollout_deterministic",
        args.rollout_deterministic if args.rollout_deterministic is not None else None,
        True,
    )
    if args.no_rollout_ref_mask:
        rollout_apply_ref_mask = False
    else:
        rollout_apply_ref_mask = choose(
            "rollout_apply_ref_mask",
            args.rollout_apply_ref_mask if args.rollout_apply_ref_mask is not None else None,
            False,
        )

    align_online_rewards = choose(
        "align_online_rewards",
        args.align_online_rewards if args.align_online_rewards is not None else None,
        True,
    )

    return OnlineWSRLSettings(
        max_train_steps=int(choose("max_train_steps", args.max_train_steps, 1000)),
        save_every_steps=int(choose("save_every_steps", args.save_every_steps, 200)),
        train_updates_per_step=int(choose("train_updates_per_step", args.train_updates_per_step, 1)),
        replay_capacity=int(choose("replay_capacity", args.replay_capacity, 200000)),
        buffer_dir=Path(choose("buffer_dir", args.buffer_dir, "./data/replay_buffers/online_ws")).expanduser().resolve(),
        buffer_max_files=int(choose("buffer_max_files", args.buffer_max_files, 2000)),
        batch_size=int(choose("batch_size", args.batch_size, 32)),
        td3_checkpoint=td3_checkpoint,
        checkpoint_dir=Path(
            choose("online_checkpoint_dir", args.online_checkpoint_dir, str(default_save_dir))
        ).expanduser().resolve(),
        online_sample_ratio=float(choose("online_sample_ratio", args.online_sample_ratio, 0.1)),
        replay_cache_path=replay_cache_path,
        offline_replay_capacity=int(
            choose("offline_replay_capacity", args.offline_replay_capacity, default_offline_cap)
        ),
        rollout_deterministic=bool(rollout_deterministic),
        rollout_apply_ref_mask=bool(rollout_apply_ref_mask),
        actor_lr_scale=float(choose("actor_lr_scale", args.actor_lr_scale, 0.1)),
        critic_lr_scale=float(choose("critic_lr_scale", args.critic_lr_scale, 0.1)),
        align_online_rewards=bool(align_online_rewards),
        logging_steps=int(choose("logging_steps", args.logging_steps, 50)),
    )


def _load_offline_replay_cache(
    cache_path: Optional[Path],
    capacity: int,
    online_sample_ratio: float,
) -> Optional[ReplayBuffer]:
    if online_sample_ratio >= 1.0:
        return None
    if cache_path is None:
        print("[online_ws_td3] offline replay disabled (no replay_cache_path)")
        return None
    if not cache_path.is_file():
        print(f"[online_ws_td3] warn: replay cache not found: {cache_path}, using online only")
        return None
    offline_replay, meta = load_replay_buffer(cache_path, capacity=capacity)
    stats = replay_positive_reward_stats(offline_replay)
    print(
        f"[online_ws_td3] loaded offline replay: {cache_path} "
        f"(size={int(stats['size'])}, positive_reward_rate={stats['positive_reward_rate']:.2%})"
    )
    print(f"[online_ws_td3] offline cache meta: chunk_len={meta.get('chunk_len')}, stride={meta.get('stride')}")
    return offline_replay


@torch.no_grad()
def _extract_features(vla_model, rl_encoder, obs_msg, image_keys, image_size, device, instruction, chunk_size):
    inputs, state_raw = ws_obs_to_vla_model_inputs(
        obs_msg,
        image_keys=image_keys,
        image_size=image_size,
        device=device,
        instruction=instruction,
    )
    z_tokens = vla_model.extract_vla_tokens(inputs)
    z_rl = rl_encoder.encode(z_tokens).float().squeeze(0).detach().cpu()
    pred = vla_model.predict_action(inputs)
    ref_actions = pred["normalized_actions"]
    if isinstance(ref_actions, np.ndarray):
        ref_tensor = torch.as_tensor(ref_actions, dtype=torch.float32, device=device)
    else:
        ref_tensor = ref_actions.to(device=device, dtype=torch.float32)
    if ref_tensor.dim() == 2:
        ref_tensor = ref_tensor.unsqueeze(0)
    ref_chunk = ref_tensor[:, :chunk_size, :].squeeze(0).detach().cpu()

    if state_raw is None:
        state_tensor = torch.zeros((chunk_size * 0,), dtype=torch.float32)
    else:
        state_tensor = torch.as_tensor(np.asarray(state_raw, dtype=np.float32).reshape(-1), dtype=torch.float32).detach().cpu()
    return z_rl, ref_chunk, state_tensor


async def _run_online_training(args) -> None:
    raw_config = load_config(args.config)
    cfg = load_script_config(args.config, dataset_path=None)
    td3_train_settings = load_train_rl_td3_settings(raw_config, cfg)
    settings = _resolve_online_ws_settings(raw_config, args)
    dataset_config = raw_config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    image_size = raw_config.get("model", {}).get("vlm", {}).get("image_size", 224)
    dataset_path = td3_train_settings.dataset.local_path

    set_seed(td3_train_settings.seed)
    device = torch.device(td3_train_settings.device)
    cfg, vla_model, rl_encoder = load_frozen_vla_and_rl_token(
        args.config,
        dataset_path,
        str(td3_train_settings.vla_checkpoint),
        str(td3_train_settings.rl_token_checkpoint),
        device,
        validate_vla=not args.skip_vla_validation,
        rl_token_network_cfg=td3_train_settings.rl_token_network_cfg,
    )
    td3_agent = load_td3_agent(settings.td3_checkpoint, device)
    apply_online_td3_lr_scale(td3_agent, settings.actor_lr_scale, settings.critic_lr_scale)
    online_replay = ReplayBuffer(capacity=settings.replay_capacity)
    offline_replay = _load_offline_replay_cache(
        settings.replay_cache_path,
        settings.offline_replay_capacity,
        settings.online_sample_ratio,
    )
    chunk_size = int(td3_agent.actor.chunk_size)
    gamma = float(td3_agent.cfg.gamma)

    if args.chunk_steps != chunk_size:
        print(
            f"[online_ws_td3] warn: chunk_steps={args.chunk_steps} != td3 chunk_size={chunk_size}; "
            "next-state alignment may differ from offline replay"
        )

    task_ids = resolve_task_ids_from_args(args, dataset_path=dataset_path, config=raw_config)
    init_ids = args.init_ids if args.init_ids is not None else list(range(args.num_rollouts))
    if not task_ids:
        raise RuntimeError("empty task_ids")
    if not init_ids:
        raise RuntimeError("empty init_ids")

    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings.buffer_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = settings.checkpoint_dir / f"online_ws_summary_{run_ts}.json"
    curves_dir = settings.checkpoint_dir / f"online_ws_curves_{run_ts}"

    print(f"[online_ws_td3] td3 init checkpoint: {settings.td3_checkpoint}")
    print(f"[online_ws_td3] online checkpoint dir: {settings.checkpoint_dir}")
    print(f"[online_ws_td3] max_train_steps={settings.max_train_steps}, updates_per_step={settings.train_updates_per_step}")
    print(
        f"[online_ws_td3] rollout: deterministic={settings.rollout_deterministic}, "
        f"apply_ref_mask={settings.rollout_apply_ref_mask}"
    )
    print(
        f"[online_ws_td3] lr scale: actor={settings.actor_lr_scale}, critic={settings.critic_lr_scale} "
        f"(base actor_lr={td3_agent.cfg.actor_lr}, critic_lr={td3_agent.cfg.critic_lr})"
    )
    print(
        f"[online_ws_td3] batch mix: online_sample_ratio={settings.online_sample_ratio:.2f} "
        f"(offline={1.0 - settings.online_sample_ratio:.2f})"
    )
    print(f"[online_ws_td3] align_online_rewards={settings.align_online_rewards}")
    if settings.replay_cache_path is not None:
        print(f"[online_ws_td3] replay_cache_path={settings.replay_cache_path}")

    if offline_replay is not None:
        audit_path = settings.checkpoint_dir / f"replay_audit_{run_ts}.json"
        audit = {
            "offline": summarize_replay_buffer(offline_replay, label="offline_cache"),
        }
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        print(f"[online_ws_td3] offline replay audit: {audit_path}")

    finished_episodes = 0
    success_episodes = 0
    task_ptr = 0
    init_ptr = 0
    global_step = 0
    logs = []

    pbar = tqdm(total=settings.max_train_steps, desc="online_ws_td3_train", dynamic_ncols=True)
    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        while global_step < settings.max_train_steps:
            task_id = task_ids[task_ptr % len(task_ids)]
            init_id = init_ids[init_ptr % len(init_ids)]
            task_ptr += 1
            if task_ptr % len(task_ids) == 0:
                init_ptr += 1

            created = await client.create_episode(task_id=task_id, init_id=init_id, max_steps=args.max_steps)
            episode_id = created["episode_id"]
            instruction = created.get("instruction", "")
            obs_msg = created
            episode_done = False
            episode_success = False
            episode_steps = 0

            try:
                while not episode_done and episode_steps < args.max_steps and global_step < settings.max_train_steps:
                    z_rl, ref_chunk, state = _extract_features(
                        vla_model, rl_encoder, obs_msg, image_keys, image_size, device, instruction, chunk_size
                    )
                    state_dev = state.unsqueeze(0).to(device=device, dtype=torch.float32)
                    if state_dev.shape[-1] == 0:
                        state_dev = torch.zeros((1, td3_agent.actor.state_dim), dtype=torch.float32, device=device)
                        state = state_dev.squeeze(0).detach().cpu()
                    ref_dev = ref_chunk.unsqueeze(0).to(device=device, dtype=torch.float32)
                    z_dev = z_rl.unsqueeze(0).to(device=device, dtype=torch.float32)
                    pred_chunk = td3_agent.act(
                        z_dev,
                        state_dev,
                        ref_dev,
                        deterministic=settings.rollout_deterministic,
                        apply_ref_mask=settings.rollout_apply_ref_mask,
                    )
                    action_chunk = pred_chunk.squeeze(0).detach().cpu()

                    action_buffer = [model_action_to_libero(action_chunk[i].numpy()) for i in range(len(action_chunk))]
                    steps_this_round = min(args.chunk_steps, len(action_buffer))
                    for i in range(steps_this_round):
                        obs_msg = await client.step(episode_id, action_buffer[i], include_images=True)
                        episode_steps += 1
                        episode_done = bool(obs_msg.get("done"))
                        episode_success = bool(obs_msg.get("success"))
                        if episode_done or episode_steps >= args.max_steps:
                            break

                    next_z_rl, next_ref_chunk, next_state = _extract_features(
                        vla_model, rl_encoder, obs_msg, image_keys, image_size, device, instruction, chunk_size
                    )
                    if next_state.numel() == 0:
                        next_state = torch.zeros((td3_agent.actor.state_dim,), dtype=torch.float32)

                    if settings.align_online_rewards:
                        reward, chunk_return, done_flag = online_chunk_reward_fields(
                            episode_done=episode_done,
                            episode_success=episode_success,
                            gamma=gamma,
                            chunk_size=chunk_size,
                        )
                    else:
                        reward = 1.0 if episode_success else 0.0
                        chunk_return = reward
                        done_flag = 1.0 if episode_done else 0.0

                    transition = ReplayTransition(
                        sample_index=global_step,
                        next_sample_index=global_step + 1,
                        episode_id=episode_id,
                        z_rl=z_rl,
                        state=state,
                        action=action_chunk,
                        ref_action=ref_chunk,
                        reward=reward,
                        chunk_return=chunk_return,
                        done=done_flag,
                        next_z_rl=next_z_rl,
                        next_state=next_state,
                        next_ref_action=next_ref_chunk,
                    )
                    online_replay.add(transition)
                    global_step += 1
                    save_online_step_buffer(settings.buffer_dir, global_step, transition)
                    prune_old_buffer_files(settings.buffer_dir, settings.buffer_max_files)

                    last_metrics = {"critic_loss": 0.0, "actor_constraint_loss": 0.0, "actor_loss": 0.0}
                    for _ in range(settings.train_updates_per_step):
                        batch, sources = sample_mixed_batch_with_sources(
                            online_replay,
                            offline_replay,
                            settings.batch_size,
                            settings.online_sample_ratio,
                        )
                        if not batch:
                            continue
                        last_metrics = train_from_batch_with_diagnostics(
                            td3_agent, batch, sources=sources, apply_ref_mask=True
                        )
                        last_metrics["global_step"] = float(global_step)
                        logs.append(last_metrics)

                    if settings.save_every_steps > 0 and global_step % settings.save_every_steps == 0:
                        ckpt_path = settings.checkpoint_dir / f"td3_agent_step_{global_step}.pt"
                        save_td3_agent(
                            ckpt_path,
                            td3_agent,
                            rl_token_dim=int(td3_agent.actor.rl_token_dim),
                            state_dim=int(td3_agent.actor.state_dim),
                            action_dim=int(td3_agent.actor.action_dim),
                            chunk_size=int(td3_agent.actor.chunk_size),
                            td3_cfg=td3_agent.cfg,
                        )

                    success_rate = (success_episodes / finished_episodes) if finished_episodes > 0 else 0.0
                    pbar.update(1)
                    online_stats = replay_positive_reward_stats(online_replay)
                    pbar.set_postfix(
                        step=global_step,
                        q_loss=f"{last_metrics.get('critic_loss', 0.0):.6f}",
                        bc_loss=f"{last_metrics.get('actor_constraint_loss', 0.0):.6f}",
                        total_loss=f"{last_metrics.get('actor_loss', 0.0):.6f}",
                        online_pos=f"{online_stats['positive_reward_rate']:.1%}",
                        ep_sr=f"{success_rate:.2%}",
                        on_mse=f"{last_metrics.get('online_action_ref_mse', 0.0):.4f}",
                    )

                    if settings.logging_steps > 0 and global_step % settings.logging_steps == 0 and logs:
                        export_training_curves(logs, curves_dir)
            finally:
                closed = await client.close_episode(episode_id)
                episode_success = episode_success or bool(closed.get("success", False))
                finished_episodes += 1
                if episode_success:
                    success_episodes += 1

    pbar.close()

    if logs:
        export_training_curves(logs, curves_dir)
        print(f"[online_ws_td3] training curves: {curves_dir}")

    final_path = settings.checkpoint_dir / "td3_agent_final.pt"
    save_td3_agent(
        final_path,
        td3_agent,
        rl_token_dim=int(td3_agent.actor.rl_token_dim),
        state_dim=int(td3_agent.actor.state_dim),
        action_dim=int(td3_agent.actor.action_dim),
        chunk_size=int(td3_agent.actor.chunk_size),
        td3_cfg=td3_agent.cfg,
    )

    online_audit = summarize_replay_buffer(online_replay, label="online_end")
    audit_path = settings.checkpoint_dir / f"replay_audit_{run_ts}.json"
    audit_payload = {"online_end": online_audit}
    if offline_replay is not None:
        audit_payload["offline"] = summarize_replay_buffer(offline_replay, label="offline_cache")
    if audit_path.is_file():
        with open(audit_path, encoding="utf-8") as f:
            audit_payload = {**json.load(f), **audit_payload}
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)

    summary = {
        "timestamp": run_ts,
        "max_train_steps": settings.max_train_steps,
        "final_global_step": global_step,
        "train_updates_per_step": settings.train_updates_per_step,
        "rollout_deterministic": settings.rollout_deterministic,
        "rollout_apply_ref_mask": settings.rollout_apply_ref_mask,
        "actor_lr_scale": settings.actor_lr_scale,
        "critic_lr_scale": settings.critic_lr_scale,
        "align_online_rewards": settings.align_online_rewards,
        "finished_episodes": finished_episodes,
        "success_episodes": success_episodes,
        "success_rate": (success_episodes / finished_episodes) if finished_episodes > 0 else 0.0,
        "final_checkpoint": str(final_path),
        "buffer_dir": str(settings.buffer_dir),
        "online_sample_ratio": settings.online_sample_ratio,
        "replay_cache_path": str(settings.replay_cache_path) if settings.replay_cache_path else None,
        "offline_replay_size": len(offline_replay) if offline_replay is not None else 0,
        "curves_dir": str(curves_dir),
        "replay_audit": str(audit_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[online_ws_td3] done. final checkpoint: {final_path}")
    print(f"[online_ws_td3] summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online WS TD3 training with frozen VLA/RL token")
    parser.add_argument("--config", type=str, default="libero/config_libero_object.yaml")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--init-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--skip-vla-validation", action="store_true")
    add_task_id_cli_arguments(parser)
    parser.set_defaults(task_id=6)

    parser.add_argument("--td3-checkpoint", type=str, default=None)
    parser.add_argument("--td3-step", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--train-updates-per-step", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--buffer-dir", type=str, default=None)
    parser.add_argument("--buffer-max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--online-checkpoint-dir", type=str, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument(
        "--online-sample-ratio",
        type=float,
        default=None,
        help="Per-sample probability of drawing from online replay (else offline replay_cache.pt)",
    )
    parser.add_argument(
        "--replay-cache-path",
        type=str,
        default=None,
        help="Offline replay_cache.pt path (default: train_rl_td3.replay.cache_path)",
    )
    parser.add_argument(
        "--offline-replay-capacity",
        type=int,
        default=None,
        help="Capacity when loading offline replay cache (default: train_rl_td3.replay.capacity)",
    )
    parser.add_argument(
        "--rollout-deterministic",
        action="store_true",
        default=None,
        help="Rollout with deterministic TD3 act (matches eval; default True in config)",
    )
    parser.add_argument(
        "--rollout-stochastic",
        action="store_true",
        help="Rollout with Gaussian exploration noise (legacy online behavior)",
    )
    parser.add_argument(
        "--rollout-apply-ref-mask",
        action="store_true",
        default=None,
        help="Apply ref dropout during rollout collection",
    )
    parser.add_argument(
        "--no-rollout-ref-mask",
        action="store_true",
        help="Disable ref dropout during rollout (matches eval)",
    )
    parser.add_argument("--actor-lr-scale", type=float, default=None)
    parser.add_argument("--critic-lr-scale", type=float, default=None)
    parser.add_argument(
        "--align-online-rewards",
        action="store_true",
        default=None,
        help="Terminal-only success reward (aligned with offline replay)",
    )
    parser.add_argument(
        "--legacy-online-rewards",
        action="store_true",
        help="Use legacy reward=success flag on every chunk after success",
    )

    args = parser.parse_args()
    if args.rollout_stochastic:
        args.rollout_deterministic = False
    if args.legacy_online_rewards:
        args.align_online_rewards = False
    asyncio.run(_run_online_training(args))


if __name__ == "__main__":
    main()
