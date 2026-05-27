#!/usr/bin/env python3
"""
WebSocket eval client with configurable rollouts and per-rollout MP4 recording.

Same closed-loop inference as libero_ws_eval.py, plus:
  - --num-rollouts: how many episodes to run per task
  - --video-dir: save one MP4 per rollout for visual debugging


python libero_ws_eval_video.py \
  --config config_libero_object.yaml \
  --checkpoint-dir ./checkpoints/libero_object_task0_posttrain \
  --task-id 0 \
  --num-rollouts 5 \
  --video-dir ./results/rollout_videos \
  --camera both \
  --chunk-steps 10
  
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from inference import find_latest_checkpoint, load_model_from_checkpoint
from libero_rollout_video import RolloutVideoRecorder, rollout_video_path
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
    align_joint_angles = data_config.get("align_joint_angles", True) if use_normalizer else False
    clip_normalized_state = data_config.get("clip_normalized_state", True) if use_normalizer else False

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        latest = find_latest_checkpoint(Path(args.checkpoint_dir))
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in {args.checkpoint_dir}")
        ckpt_path = str(latest)
    print(f"[eval_video] loading checkpoint: {ckpt_path}")

    model, normalizer = load_model_from_checkpoint(
        ckpt_path, args.config, device=args.device
    )

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_root = Path(args.video_dir) / run_ts
    video_root.mkdir(parents=True, exist_ok=True)
    print(f"[eval_video] saving videos under: {video_root}")

    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        dataset_path = dataset_config.get("local_path", "./dada/libero-object")
        task_ids = resolve_task_ids_from_args(
            args, dataset_path=dataset_path, config=config
        )
        if args.init_ids is not None:
            init_ids = args.init_ids
        else:
            init_ids = list(range(args.num_rollouts))

        results = []
        rollout_counter = 0
        for tid in task_ids:
            for init_id in init_ids:
                rollout_counter += 1
                recorder = RolloutVideoRecorder(fps=args.fps, camera=args.camera)
                label = f"rollout {rollout_counter}/{len(task_ids) * len(init_ids)} task_id={tid} init_id={init_id}"
                result = await run_eval_episode(
                    client=client,
                    model=model,
                    normalizer=normalizer,
                    image_keys=image_keys,
                    image_size=image_size,
                    normalize_action=normalize_action,
                    normalize_state=normalize_state,
                    align_joint_angles=align_joint_angles,
                    clip_normalized_state=clip_normalized_state,
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
                results.append(result)

                print(
                    f"[eval_video] {label} steps={result['steps']} "
                    f"success={result['success']} frames={result['num_video_frames']} "
                    f"video={result['video_path']}"
                )

        summary_path = video_root / "summary.json"
        summary = {
            "timestamp": run_ts,
            "checkpoint": ckpt_path,
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
                f"[eval_video] summary: success_rate={rate:.2%} ({len(results)} rollouts)"
            )
            print(f"[eval_video] summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO WebSocket VLA eval with rollout videos"
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--config", default="config_libero_object.yaml")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/libero_object")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    add_task_id_cli_arguments(parser)
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
        default="./results/rollout_videos",
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
