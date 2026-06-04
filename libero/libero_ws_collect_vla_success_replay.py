#!/usr/bin/env python3
"""
Collect successful VLA WebSocket rollouts and build a TD3 replay cache.

Example::

  python -m libero.libero_ws_collect_vla_success_replay \\
    --config libero/config_libero_object.yaml \\
    --task-id 6 \\
    --target-success-episodes 400

Resume is automatic when save_path / progress_path already exist.

接下来离线训练td3
python train_rl_td3.py --config libero/config_libero_object.yaml \
  --vla-success-sample-ratio 0.7

"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

_entry_path = Path(__file__).resolve().parent / "_entry.py"
_spec = importlib.util.spec_from_file_location("libero_entry", _entry_path)
_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_entry)
_entry.maybe_reroute_main(__name__, __package__, __file__)

from train_rl_td3 import load_train_rl_td3_settings

from .libero_task_mapping import add_task_id_cli_arguments, resolve_task_ids_from_args
from .libero_ws_client import LiberoWSClient
from .libero_ws_vla_collect_core import (
    run_vla_collect_episode,
    vla_ws_inference_settings_from_config,
)
from .rl_td3_replay import (
    ReplayBuffer,
    append_ws_episode_to_replay,
    build_vla_success_replay_meta,
    load_frozen_vla_and_rl_token,
    load_normalizer_from_vla_checkpoint,
    load_replay_buffer,
    replay_positive_reward_stats,
    resolve_rl_token_dim_for_meta,
    save_replay_buffer,
    set_seed,
)
from src.ScriptedVLA.utils import load_config, load_script_config


@dataclass
class VlaSuccessCollectSettings:
    target_success_episodes: int
    save_path: Path
    progress_path: Path
    checkpoint_every_successes: int
    stride: int
    capacity: int
    ws_url: str
    max_steps: int
    chunk_steps: int
    summary_dir: Path


def _resolve_collect_settings(raw_config: dict, args) -> VlaSuccessCollectSettings:
    block = raw_config.get("vla_success_replay") or {}
    td3_block = raw_config.get("train_rl_td3") or {}
    replay_cfg = td3_block.get("replay") or {}

    def choose(key: str, cli_val, default):
        if cli_val is not None:
            return cli_val
        return block.get(key, default)

    save_path = Path(
        choose("save_path", args.save_path, "./data/replay_buffers/libero_object_task6/vla_success_replay_cache.pt")
    ).expanduser().resolve()
    progress_path = Path(
        choose(
            "progress_path",
            args.progress_path,
            str(save_path.parent / "vla_success_collect_progress.json"),
        )
    ).expanduser().resolve()

    return VlaSuccessCollectSettings(
        target_success_episodes=int(
            choose("target_success_episodes", args.target_success_episodes, 400)
        ),
        save_path=save_path,
        progress_path=progress_path,
        checkpoint_every_successes=int(
            choose("checkpoint_every_successes", args.checkpoint_every_successes, 10)
        ),
        stride=int(choose("stride", args.stride, replay_cfg.get("stride", 2))),
        capacity=int(choose("capacity", args.capacity, 500000)),
        ws_url=str(choose("ws_url", args.ws_url, "ws://127.0.0.1:8765")),
        max_steps=int(choose("max_steps", args.max_steps, 600)),
        chunk_steps=int(choose("chunk_steps", args.chunk_steps, td3_block.get("rl_chunk_size", 10))),
        summary_dir=save_path.parent,
    )


def _load_progress(path: Path) -> dict:
    if not path.is_file():
        return {
            "success_episodes": 0,
            "total_attempts": 0,
            "next_episode_id": 0,
            "init_ptr": 0,
        }
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def _load_or_create_replay(save_path: Path, capacity: int) -> ReplayBuffer:
    if save_path.is_file():
        replay, _ = load_replay_buffer(save_path, capacity=capacity)
        return replay
    return ReplayBuffer(capacity=capacity)


async def _run_collection(args) -> None:
    raw_config = load_config(args.config)
    cfg = load_script_config(args.config, dataset_path=None)
    td3_settings = load_train_rl_td3_settings(raw_config, cfg)
    settings = _resolve_collect_settings(raw_config, args)
    dataset_config = raw_config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    image_size = raw_config.get("model", {}).get("vlm", {}).get("image_size", 224)
    dataset_path = td3_settings.dataset.local_path
    gamma = float(td3_settings.td3_cfg.gamma)
    chunk_size = int(td3_settings.rl_chunk_size)

    if settings.chunk_steps != chunk_size:
        print(
            f"[vla_success_collect] warn: chunk_steps={settings.chunk_steps} != rl_chunk_size={chunk_size}"
        )

    set_seed(td3_settings.seed)
    device = torch.device(td3_settings.device)
    cfg, vla_model, rl_encoder = load_frozen_vla_and_rl_token(
        args.config,
        dataset_path,
        str(td3_settings.vla_checkpoint),
        str(td3_settings.rl_token_checkpoint),
        device,
        validate_vla=not args.skip_vla_validation,
        rl_token_network_cfg=td3_settings.rl_token_network_cfg,
    )

    normalizer = load_normalizer_from_vla_checkpoint(str(td3_settings.vla_checkpoint))
    ws_infer = vla_ws_inference_settings_from_config(raw_config, normalizer)
    print(
        f"[vla_success_collect] ws_infer: normalize_state={ws_infer.normalize_state}, "
        f"normalize_action={ws_infer.normalize_action}, "
        f"normalizer={'yes' if ws_infer.normalizer else 'no'}"
    )
    if ws_infer.normalize_state and ws_infer.normalizer is None:
        print(
            "[vla_success_collect] warn: normalize_state=true but checkpoint has no normalizer"
        )

    rl_token_dim = int(rl_encoder.rl_token_dim)
    replay = _load_or_create_replay(settings.save_path, settings.capacity)
    progress = _load_progress(settings.progress_path)
    success_episodes = int(progress.get("success_episodes", 0))
    total_attempts = int(progress.get("total_attempts", 0))
    next_episode_id = int(progress.get("next_episode_id", 0))
    init_ptr = int(progress.get("init_ptr", 0))

    task_ids = resolve_task_ids_from_args(args, dataset_path=dataset_path, config=raw_config)
    init_ids = args.init_ids if args.init_ids is not None else list(range(args.num_inits))
    if not task_ids or not init_ids:
        raise RuntimeError("empty task_ids or init_ids")

    settings.save_path.parent.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[vla_success_collect] save_path={settings.save_path}")
    print(f"[vla_success_collect] target={settings.target_success_episodes}, resume success={success_episodes}")
    print(f"[vla_success_collect] replay size={len(replay)}, stride={settings.stride}, chunk={chunk_size}")

    successes_since_ckpt = 0
    pbar = tqdm(
        total=settings.target_success_episodes,
        initial=success_episodes,
        desc="vla_success_episodes",
        dynamic_ncols=True,
    )

    async with LiberoWSClient(settings.ws_url) as client:
        await client.ping()
        while success_episodes < settings.target_success_episodes:
            task_id = task_ids[total_attempts % len(task_ids)]
            init_id = init_ids[init_ptr % len(init_ids)]
            init_ptr += 1
            total_attempts += 1

            label = f"attempt={total_attempts} task={task_id} init={init_id}"
            result, frames = await run_vla_collect_episode(
                client=client,
                vla_model=vla_model,
                image_keys=image_keys,
                image_size=image_size,
                chunk_size=chunk_size,
                task_id=task_id,
                init_id=init_id,
                max_steps=settings.max_steps,
                chunk_steps=settings.chunk_steps,
                device=device,
                ws_infer=ws_infer,
                rollout_label=label,
            )

            if result["success"] and frames:
                global_base = next_episode_id * (settings.max_steps + 1)
                n_added = append_ws_episode_to_replay(
                    replay,
                    frames,
                    episode_id=next_episode_id,
                    chunk_len=chunk_size,
                    stride=settings.stride,
                    gamma=gamma,
                    state_dim=int(cfg.state_dim),
                    global_index_base=global_base,
                    vla_model=vla_model,
                    rl_encoder=rl_encoder,
                    image_keys=image_keys,
                    image_size=image_size,
                    device=device,
                    instruction=result.get("instruction", ""),
                    ws_infer=ws_infer,
                )
                next_episode_id += 1
                success_episodes += 1
                successes_since_ckpt += 1
                pbar.update(1)
                stats = replay_positive_reward_stats(replay)
                pbar.set_postfix(
                    attempts=total_attempts,
                    transitions=len(replay),
                    pos_rate=f"{stats['positive_reward_rate']:.1%}",
                    added=n_added,
                )

                progress.update(
                    {
                        "success_episodes": success_episodes,
                        "total_attempts": total_attempts,
                        "next_episode_id": next_episode_id,
                        "init_ptr": init_ptr,
                    }
                )
                _save_progress(settings.progress_path, progress)

                if successes_since_ckpt >= settings.checkpoint_every_successes:
                    meta = build_vla_success_replay_meta(
                        chunk_len=chunk_size,
                        stride=settings.stride,
                        gamma=gamma,
                        state_dim=int(cfg.state_dim),
                        rl_token_dim=rl_token_dim,
                        dataset_path=dataset_path,
                        vla_checkpoint=str(td3_settings.vla_checkpoint),
                        rl_token_checkpoint=str(td3_settings.rl_token_checkpoint),
                        config_path=td3_settings.config_path,
                        task_index=td3_settings.dataset.task_index,
                        num_success_episodes=success_episodes,
                        total_attempts=total_attempts,
                    )
                    save_replay_buffer(settings.save_path, replay, meta)
                    successes_since_ckpt = 0
            else:
                pbar.set_postfix(
                    attempts=total_attempts,
                    success=success_episodes,
                    last="fail",
                )

    pbar.close()

    meta = build_vla_success_replay_meta(
        chunk_len=chunk_size,
        stride=settings.stride,
        gamma=gamma,
        state_dim=int(cfg.state_dim),
        rl_token_dim=rl_token_dim,
        dataset_path=dataset_path,
        vla_checkpoint=str(td3_settings.vla_checkpoint),
        rl_token_checkpoint=str(td3_settings.rl_token_checkpoint),
        config_path=td3_settings.config_path,
        task_index=td3_settings.dataset.task_index,
        num_success_episodes=success_episodes,
        total_attempts=total_attempts,
    )
    save_replay_buffer(settings.save_path, replay, meta)

    summary = {
        "timestamp": run_ts,
        "save_path": str(settings.save_path),
        "target_success_episodes": settings.target_success_episodes,
        "success_episodes": success_episodes,
        "total_attempts": total_attempts,
        "success_rate": success_episodes / total_attempts if total_attempts else 0.0,
        "transitions": len(replay),
        "positive_reward_rate": replay_positive_reward_stats(replay)["positive_reward_rate"],
    }
    summary_path = settings.summary_dir / f"vla_success_collect_summary_{run_ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[vla_success_collect] done. cache={settings.save_path} transitions={len(replay)}")
    print(f"[vla_success_collect] summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VLA-success WS episodes into TD3 replay cache")
    parser.add_argument("--config", type=str, default="libero/config_libero_object.yaml")
    parser.add_argument("--ws-url", type=str, default=None)
    add_task_id_cli_arguments(parser)
    parser.set_defaults(task_id=6)
    parser.add_argument("--target-success-episodes", type=int, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--progress-path", type=str, default=None)
    parser.add_argument("--checkpoint-every-successes", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--num-inits", type=int, default=20, help="init_id pool size 0..N-1")
    parser.add_argument("--init-ids", type=int, nargs="*", default=None)
    parser.add_argument("--skip-vla-validation", action="store_true")
    asyncio.run(_run_collection(parser.parse_args()))


if __name__ == "__main__":
    main()
