#!/usr/bin/env python3
"""WebSocket client: load ScriptedVLA checkpoint and control LIBERO simulation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from inference import find_latest_checkpoint, load_model_from_checkpoint
from libero_task_mapping import add_task_id_cli_arguments, resolve_task_ids_from_args
from libero_ws_client import LiberoWSClient
from libero_ws_eval_core import run_eval_episode
from src.ScriptedVLA.utils import get_data_config, load_config


async def main_async(args):
    config = load_config(args.config)
    data_config = get_data_config(config)
    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    image_size = config.get("model", {}).get("vlm", {}).get("image_size", 224)
    use_normalizer = data_config.get("use_normalizer", True)
    normalize_action = data_config.get("normalize_action", True) if use_normalizer else False
    normalize_state = data_config.get("normalize_state", True) if use_normalizer else False

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        latest = find_latest_checkpoint(Path(args.checkpoint_dir))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in {args.checkpoint_dir}")
        ckpt_path = str(latest)
    print(f"[eval] loading checkpoint: {ckpt_path}")

    model, normalizer = load_model_from_checkpoint(
        ckpt_path, args.config, device=args.device
    )

    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        dataset_path = dataset_config.get("local_path", "./dada/libero-object")
        task_ids = resolve_task_ids_from_args(
            args, dataset_path=dataset_path, config=config
        )
        init_ids = args.init_ids if args.init_ids else list(range(args.num_rollouts))
        results = []
        for tid in task_ids:
            for init_id in init_ids:
                result = await run_eval_episode(
                    client=client,
                    model=model,
                    normalizer=normalizer,
                    image_keys=image_keys,
                    image_size=image_size,
                    normalize_action=normalize_action,
                    normalize_state=normalize_state,
                    task_id=tid,
                    init_id=init_id,
                    max_steps=args.max_steps,
                    chunk_steps=args.chunk_steps,
                    debug_ranges=args.debug_ranges,
                )
                results.append(result)
                print(
                    f"[eval] task_id={tid} init_id={init_id} "
                    f"steps={result['steps']} success={result['success']}"
                )

        if results:
            rate = sum(r["success"] for r in results) / len(results)
            print(f"[eval] summary: success_rate={rate:.2%} ({len(results)} rollouts)")


def main():
    parser = argparse.ArgumentParser(description="LIBERO WebSocket VLA eval client")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--config", default="config_libero_object.yaml")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/libero_object")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    add_task_id_cli_arguments(parser)
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per task (init_id 0..N-1)")
    parser.add_argument("--init-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--debug-ranges", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
