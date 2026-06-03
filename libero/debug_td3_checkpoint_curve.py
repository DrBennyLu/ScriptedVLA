#!/usr/bin/env python3
"""
Batch-eval TD3 checkpoints on LIBERO WS (no video) to locate online-training degradation.

Example::

  python -m libero.debug_td3_checkpoint_curve \\
    --config libero/config_libero_object.yaml \\
    --task-id 6 --init-ids 0 1 2 \\
    --checkpoints \\
      ./checkpoints/libero_object_rl_td3_task6_0602/td3_agent_step_10000.pt \\
      ./checkpoints/libero_object_online_ws_td3_task6_0603/td3_agent_step_1000.pt \\
      ./checkpoints/libero_object_online_ws_td3_task6_0603/td3_agent_step_5000.pt \\
      ./checkpoints/libero_object_online_ws_td3_task6_0603/td3_agent_step_10000.pt
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import torch

_entry_path = Path(__file__).resolve().parent / "_entry.py"
_spec = importlib.util.spec_from_file_location("libero_entry", _entry_path)
_entry = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_entry)
_entry.maybe_reroute_main(__name__, __package__, __file__)

from libero.libero_task_mapping import add_task_id_cli_arguments, resolve_task_ids_from_args
from libero.libero_ws_client import LiberoWSClient
from libero.libero_ws_td3_eval_core import run_td3_eval_episode
from libero.rl_td3_replay import load_frozen_vla_and_rl_token, load_td3_agent
from src.ScriptedVLA.utils import load_config


async def _eval_checkpoint(
    *,
    client: LiberoWSClient,
    vla_model,
    rl_encoder,
    td3_ckpt: Path,
    device: torch.device,
    image_keys,
    image_size: int,
    task_ids,
    init_ids,
    max_steps: int,
    chunk_steps: int,
) -> dict:
    td3_agent = load_td3_agent(td3_ckpt, device)
    chunk_size = int(td3_agent.actor.chunk_size)
    results = []
    for tid in task_ids:
        for init_id in init_ids:
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
                max_steps=max_steps,
                chunk_steps=chunk_steps,
                debug_ranges=False,
            )
            results.append(result)
    success_rate = sum(r["success"] for r in results) / len(results) if results else 0.0
    return {
        "td3_checkpoint": str(td3_ckpt.resolve()),
        "chunk_size": chunk_size,
        "num_rollouts": len(results),
        "success_rate": success_rate,
        "results": results,
    }


async def main_async(args) -> None:
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    image_size = config.get("model", {}).get("vlm", {}).get("image_size", 224)
    dataset_path = dataset_config.get("local_path", "./dada/libero-object")
    token_block = config.get("train_rl_token") or {}
    td3_block = config.get("train_rl_td3") or {}

    vla_ckpt = args.vla_checkpoint or td3_block.get("vla_checkpoint")
    rl_token_ckpt = args.rl_token_checkpoint or td3_block.get("rl_token_checkpoint")
    if not vla_ckpt or not rl_token_ckpt:
        raise ValueError("vla_checkpoint and rl_token_checkpoint required")

    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    for ckpt in checkpoints:
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[checkpoint_curve] loading VLA once on {device}")
    cfg, vla_model, rl_encoder = load_frozen_vla_and_rl_token(
        args.config,
        dataset_path,
        str(vla_ckpt),
        str(rl_token_ckpt),
        device,
        validate_vla=not args.skip_vla_validation,
        rl_token_network_cfg=token_block.get("network"),
    )

    task_ids = resolve_task_ids_from_args(args, dataset_path=dataset_path, config=config)
    init_ids = args.init_ids if args.init_ids is not None else list(range(args.num_rollouts))

    curve = []
    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        for ckpt in checkpoints:
            print(f"[checkpoint_curve] eval {ckpt.name} ...")
            row = await _eval_checkpoint(
                client=client,
                vla_model=vla_model,
                rl_encoder=rl_encoder,
                td3_ckpt=ckpt,
                device=device,
                image_keys=image_keys,
                image_size=image_size,
                task_ids=task_ids,
                init_ids=init_ids,
                max_steps=args.max_steps,
                chunk_steps=args.chunk_steps,
            )
            print(
                f"[checkpoint_curve] {ckpt.name}: success_rate={row['success_rate']:.2%} "
                f"({sum(r['success'] for r in row['results'])}/{row['num_rollouts']})"
            )
            curve.append(row)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"checkpoint_curve_{run_ts}.json"
    payload = {
        "timestamp": run_ts,
        "task_ids": task_ids,
        "init_ids": init_ids,
        "ws_url": args.ws_url,
        "vla_checkpoint": str(vla_ckpt),
        "rl_token_checkpoint": str(rl_token_ckpt),
        "curve": curve,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[checkpoint_curve] saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TD3 checkpoint eval (no video)")
    parser.add_argument("--config", type=str, default="libero/config_libero_object.yaml")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--init-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--skip-vla-validation", action="store_true")
    parser.add_argument("--vla-checkpoint", type=str, default=None)
    parser.add_argument("--rl-token-checkpoint", type=str, default=None)
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="TD3 .pt paths to evaluate in order",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/checkpoint_curves",
    )
    add_task_id_cli_arguments(parser)
    parser.set_defaults(task_id=6)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
