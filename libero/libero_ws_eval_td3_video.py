#!/usr/bin/env python3
"""
WebSocket eval with VLA + RL token + TD3 actor and per-rollout MP4 recording.

示例::

  python -m libero.libero_ws_eval_td3_video \\
    --config libero/config_libero_object.yaml \\
    --task-id 6 \\
    --num-rollouts 5 \\
    --video-dir ./results/rl_td3_rollout_videos \\
    --camera both \\
    --chunk-steps 10

TD3 checkpoint 解析优先级（见 ``_resolve_td3_checkpoint_path``）::

  1. --td3-checkpoint <完整路径>
  2. --td3-step <步数>  例如 8000 -> td3_agent_step_8000.pt
  3. config.train_rl_td3.eval.td3_checkpoint / eval.td3_step
  4. checkpoint.save_dir + final_name（默认 td3_agent_final.pt）


python -m libero.libero_ws_eval_td3_video \
  --config libero/config_libero_object.yaml \
  --td3-checkpoint ./checkpoints/libero_object_rl_td3_task6_0602/td3_agent_step_8000.pt \
  --task-id 6 --num-rollouts 5 --video-dir ./results/rl_td3_rollout_videos_0602 --camera both --chunk-steps 10



python -m libero.libero_ws_eval_td3_video \
  --config libero/config_libero_object.yaml \
  --td3-checkpoint ./checkpoints/libero_object_online_ws_td3_task6_0603/td3_agent_step_10000.pt \
  --task-id 6 --num-rollouts 5 --video-dir ./results/rl_td3_rollout_videos_0603 --camera both --chunk-steps 10

"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_entry_path = Path(__file__).resolve().parent / "_entry.py"
_spec = importlib.util.spec_from_file_location("libero_entry", _entry_path)
_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_entry)
_entry.maybe_reroute_main(__name__, __package__, __file__)

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import torch

from .libero_rollout_video import RolloutVideoRecorder, rollout_video_path
from .libero_task_mapping import add_task_id_cli_arguments, resolve_task_ids_from_args
from .libero_ws_client import LiberoWSClient
from .libero_ws_td3_eval_core import run_td3_eval_episode
from .rl_td3_replay import load_frozen_vla_and_rl_token, load_td3_agent
from src.ScriptedVLA.utils import load_config

# 仅作为回退：如果 config 中未提供 checkpoint.save_dir/final_name，
# 则退回到旧路径（兼容历史脚本调用）
DEFAULT_TD3_CHECKPOINT = "./checkpoints/libero_object_rl_td3_task6/td3_agent_step_10000.pt"


def _resolve_td3_checkpoint_path(config: dict, args) -> Path:
    """
  Resolve TD3 checkpoint for rollout.

  Priority:
    1. args.td3_checkpoint (full path)
    2. args.td3_step (step index -> step_name_template)
    3. train_rl_td3.eval.td3_checkpoint / eval.td3_step
    4. checkpoint.save_dir + final_name
    5. DEFAULT_TD3_CHECKPOINT
    """
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
        path = Path(DEFAULT_TD3_CHECKPOINT)

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TD3 checkpoint not found: {path}")
    return path


def _resolve_train_rl_td3_paths(config: dict, args) -> tuple[str, str, Path]:
    block = config.get("train_rl_td3") or {}
    token_block = config.get("train_rl_token") or {}

    vla_ckpt = args.vla_checkpoint or block.get("vla_checkpoint") or token_block.get(
        "vla_checkpoint"
    )
    rl_token_ckpt = args.rl_token_checkpoint or block.get("rl_token_checkpoint")
    td3_ckpt = _resolve_td3_checkpoint_path(config, args)

    if not vla_ckpt:
        raise ValueError("vla checkpoint required: set train_rl_td3.vla_checkpoint or --vla-checkpoint")
    if not rl_token_ckpt:
        raise ValueError(
            "rl_token checkpoint required: set train_rl_td3.rl_token_checkpoint or --rl-token-checkpoint"
        )
    return str(vla_ckpt), str(rl_token_ckpt), td3_ckpt


async def main_async(args):
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    image_size = config.get("model", {}).get("vlm", {}).get("image_size", 224)
    dataset_path = dataset_config.get("local_path", "./dada/libero-object")

    vla_ckpt, rl_token_ckpt, td3_ckpt = _resolve_train_rl_td3_paths(config, args)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[eval_td3_video] loading VLA: {vla_ckpt}")
    print(f"[eval_td3_video] loading RL token: {rl_token_ckpt}")
    print(f"[eval_td3_video] loading TD3: {td3_ckpt}")

    cfg, vla_model, rl_encoder = load_frozen_vla_and_rl_token(
        args.config,
        dataset_path,
        vla_ckpt,
        rl_token_ckpt,
        device,
        validate_vla=not args.skip_vla_validation,
        rl_token_network_cfg=(config.get("train_rl_token") or {}).get("network"),
    )
    td3_agent = load_td3_agent(td3_ckpt, device)
    chunk_size = int(td3_agent.actor.chunk_size)
    print(f"[eval_td3_video] chunk_size={chunk_size} (from TD3 checkpoint)")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = Path(args.video_dir) / run_ts
    video_root.mkdir(parents=True, exist_ok=True)
    print(f"[eval_td3_video] saving videos under: {video_root}")

    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        task_ids = resolve_task_ids_from_args(
            args, dataset_path=dataset_path, config=config
        )
        if args.init_ids is not None:
            init_ids = args.init_ids
        else:
            init_ids = list(range(args.num_rollouts))

        results = []
        rollout_counter = 0
        total = len(task_ids) * len(init_ids)
        for tid in task_ids:
            for init_id in init_ids:
                rollout_counter += 1
                recorder = RolloutVideoRecorder(fps=args.fps, camera=args.camera)
                label = (
                    f"rollout {rollout_counter}/{total} task_id={tid} init_id={init_id}"
                )
                result = await run_td3_eval_episode(
                    client=client,
                    vla_model=vla_model,
                    rl_encoder=rl_encoder,
                    td3_agent=td3_agent,
                    image_keys=image_keys,
                    image_size=image_size,
                    chunk_size=chunk_size,
                    task_id=tid,
                    init_id=init_id,
                    max_steps=args.max_steps,
                    chunk_steps=args.chunk_steps,
                    debug_ranges=args.debug_ranges,
                    video_recorder=recorder,
                    rollout_label=label,
                )

                out_path = rollout_video_path(
                    video_dir=video_root,
                    task_id=tid,
                    rollout_index=rollout_counter,
                    init_id=init_id,
                    success=result["success"],
                    task_name=result.get("task_name", ""),
                )
                saved = recorder.save(out_path)
                result["video_path"] = str(saved) if saved else None
                result["rollout_index"] = rollout_counter
                result["policy"] = "td3"
                results.append(result)

                print(
                    f"[eval_td3_video] {label} steps={result['steps']} "
                    f"success={result['success']} frames={result['num_video_frames']} "
                    f"video={result['video_path']}"
                )

        summary_path = video_root / "summary.json"
        summary = {
            "timestamp": run_ts,
            "policy": "td3",
            "vla_checkpoint": vla_ckpt,
            "rl_token_checkpoint": rl_token_ckpt,
            "td3_checkpoint": str(td3_ckpt),
            "chunk_size": chunk_size,
            "ws_url": args.ws_url,
            "num_rollouts": len(results),
            "success_rate": (
                sum(r["success"] for r in results) / len(results) if results else 0.0
            ),
            "results": results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if results:
            rate = summary["success_rate"]
            print(
                f"[eval_td3_video] summary: success_rate={rate:.2%} ({len(results)} rollouts)"
            )
            print(f"[eval_td3_video] summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO WebSocket TD3 eval (VLA+RL token+TD3) with rollout videos"
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--config", default="libero/config_libero_object.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--vla-checkpoint",
        default=None,
        help="VLA checkpoint (default: train_rl_td3.vla_checkpoint in config)",
    )
    parser.add_argument(
        "--rl-token-checkpoint",
        default=None,
        help="RL token checkpoint (default: train_rl_td3.rl_token_checkpoint)",
    )
    parser.add_argument(
        "--td3-checkpoint",
        default=None,
        help="TD3 权重完整路径（优先级最高，覆盖 config 默认 final）",
    )
    parser.add_argument(
        "--td3-step",
        type=int,
        default=None,
        help="训练步数 checkpoint，如 8000 -> <checkpoint.save_dir>/td3_agent_step_8000.pt",
    )
    parser.add_argument(
        "--skip-vla-validation",
        action="store_true",
        help="Skip VLA checkpoint validation against dataset",
    )
    add_task_id_cli_arguments(parser)
    parser.set_defaults(task_id=6)
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=3,
        help="Number of rollouts per task using init_id=0..num_rollouts-1",
    )
    parser.add_argument(
        "--init-ids",
        type=int,
        nargs="*",
        default=None,
        help="Explicit init_id list (overrides --num-rollouts)",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--debug-ranges", action="store_true")
    parser.add_argument(
        "--video-dir",
        default="./results/rl_td3_rollout_videos",
        help="Root directory; each run creates a timestamped subfolder",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument(
        "--camera",
        choices=["agentview", "wrist", "both"],
        default="agentview",
        help="Which camera view(s) to record",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
