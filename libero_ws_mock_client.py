#!/usr/bin/env python3
"""Mock WebSocket client: drive LIBERO sim with zero/random/GT action replay."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from libero_action_adapter import random_action, validate_action, zero_action
from libero_dataset_replay import (
    find_episode_by_instruction,
    find_first_episode_for_task,
    load_episode_actions,
)
from libero_task_mapping import add_task_id_cli_arguments, resolve_benchmark_task_ids
from libero_rollout_video import RolloutVideoRecorder, rollout_video_path
from libero_ws_client import LiberoWSClient


async def run_episode(
    client: LiberoWSClient,
    benchmark_task_id: int,
    init_id: int,
    max_steps: int,
    mode: str,
    dataset_path: Optional[str],
    episode_index: Optional[int],
    chunk_steps: int,
    skip_obs_decode: bool,
    video_recorder: Optional[RolloutVideoRecorder] = None,
    dataset_task_index: Optional[int] = None,
) -> dict:
    created = await client.create_episode(
        task_id=benchmark_task_id, init_id=init_id, max_steps=max_steps
    )
    episode_id = created["episode_id"]
    task_name = created.get("task_name", "")
    instruction = created.get("instruction", "")
    print(
        f"[mock] episode={episode_id} benchmark_task_id={benchmark_task_id} init_id={init_id} "
        f"instruction={instruction!r}"
    )

    gt_actions = None
    replay_episode_index = None
    if mode == "replay":
        if dataset_path is None:
            raise ValueError("--dataset-path required for replay mode")
        if episode_index is not None:
            replay_episode_index = episode_index
        elif instruction:
            replay_episode_index = find_episode_by_instruction(dataset_path, instruction)
        elif dataset_task_index is not None:
            replay_episode_index = find_first_episode_for_task(
                dataset_path, dataset_task_index
            )
        else:
            raise ValueError(
                "replay mode requires --episode-index, matching instruction, or dataset task_index"
            )
        gt_actions, _instr, _ = load_episode_actions(dataset_path, replay_episode_index)
        print(
            f"[mock] replay episode_index={replay_episode_index} "
            f"actions={gt_actions.shape}"
        )

    step_count = 0
    done = False
    success = False
    gt_idx = 0
    record_video = video_recorder is not None
    include_images = record_video or not skip_obs_decode

    if video_recorder is not None:
        video_recorder.reset()
        video_recorder.append_obs(created)

    try:
        while not done and step_count < max_steps:
            chunk: List[List[float]] = []
            for _ in range(chunk_steps):
                if mode == "zero":
                    chunk.append(zero_action())
                elif mode == "random":
                    chunk.append(random_action())
                elif mode == "replay":
                    if gt_idx >= len(gt_actions):
                        break
                    chunk.append(validate_action(gt_actions[gt_idx]))
                    gt_idx += 1
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            if not chunk:
                break

            for action in chunk:
                obs_msg = await client.step(
                    episode_id, action, include_images=include_images
                )
                if video_recorder is not None:
                    video_recorder.append_obs(obs_msg)
                step_count += 1
                done = bool(obs_msg.get("done"))
                success = bool(obs_msg.get("success"))
                if done:
                    break
                if step_count >= max_steps:
                    break

            if mode != "replay" or gt_idx >= len(gt_actions):
                if done:
                    break
    finally:
        closed = await client.close_episode(episode_id)

    final_success = success or closed.get("success", False)
    return {
        "benchmark_task_id": benchmark_task_id,
        "dataset_task_index": dataset_task_index,
        "init_id": init_id,
        "task_name": task_name,
        "instruction": instruction,
        "steps": step_count,
        "success": final_success,
        "mode": mode,
        "replay_episode_index": replay_episode_index,
        "num_video_frames": len(video_recorder.frames) if video_recorder else 0,
    }


async def main_async(args):
    config = None
    if args.config:
        from src.ScriptedVLA.utils import load_config

        config = load_config(args.config)

    save_video = args.save_video or (args.mode == "replay" and not args.no_save_video)
    video_root = None
    if save_video:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_root = Path(args.video_dir) / f"replay_{run_ts}"
        video_root.mkdir(parents=True, exist_ok=True)
        print(f"[mock] replay videos will be saved under: {video_root}")

    async with LiberoWSClient(args.ws_url) as client:
        await client.ping()
        print(f"[mock] connected to {args.ws_url}")

        task_ids = resolve_benchmark_task_ids(
            dataset_path=args.dataset_path,
            dataset_task_id=args.task_id,
            dataset_task_ids=args.task_ids,
            benchmark_task_id=args.benchmark_task_id,
            benchmark_task_ids=args.benchmark_task_ids,
            config_task_index=(
                config.get("dataset", {}).get("task_index") if config else None
            ),
        )
        dataset_indices = None
        if args.benchmark_task_id is not None or (
            args.benchmark_task_ids is not None and len(args.benchmark_task_ids) > 0
        ):
            dataset_indices = [None] * len(task_ids)
        elif args.task_ids is not None and len(args.task_ids) > 0:
            dataset_indices = [int(t) for t in args.task_ids]
        elif args.task_id is not None:
            dataset_indices = [int(args.task_id)]
        elif config is not None and config.get("dataset", {}).get("task_index") is not None:
            dataset_indices = [int(config["dataset"]["task_index"])]
        else:
            dataset_indices = [0]

        results = []
        for rollout_idx, (tid, ds_idx) in enumerate(zip(task_ids, dataset_indices), start=1):
            recorder = None
            if save_video:
                recorder = RolloutVideoRecorder(fps=args.fps, camera=args.camera)

            result = await run_episode(
                client=client,
                benchmark_task_id=tid,
                init_id=args.init_id,
                max_steps=args.max_steps,
                mode=args.mode,
                dataset_path=args.dataset_path,
                episode_index=args.episode_index,
                chunk_steps=args.chunk_steps,
                skip_obs_decode=args.skip_obs_decode,
                video_recorder=recorder,
                dataset_task_index=ds_idx,
            )

            video_path = None
            if recorder is not None and video_root is not None:
                out_path = rollout_video_path(
                    video_dir=video_root,
                    task_id=tid,
                    rollout_index=rollout_idx,
                    init_id=args.init_id,
                    success=result["success"],
                    task_name=result.get("task_name", ""),
                )
                if result.get("replay_episode_index") is not None:
                    out_path = out_path.with_name(
                        out_path.stem.replace(
                            f"_rollout{rollout_idx:03d}_",
                            f"_ep{result['replay_episode_index']:06d}_rollout{rollout_idx:03d}_",
                            1,
                        )
                        + out_path.suffix
                    )
                saved = recorder.save(out_path)
                video_path = str(saved) if saved else None
                result["video_path"] = video_path

            results.append(result)
            msg = (
                f"[mock] benchmark_task_id={tid} steps={result['steps']} "
                f"success={result['success']} mode={result['mode']}"
            )
            if video_path:
                msg += f" video={video_path} frames={result['num_video_frames']}"
            print(msg)

        if results:
            rate = sum(r["success"] for r in results) / len(results)
            print(f"[mock] summary: {len(results)} episodes, success_rate={rate:.2%}")
            if video_root is not None:
                summary_path = video_root / "summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mode": args.mode,
                            "success_rate": rate,
                            "results": results,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"[mock] summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="LIBERO WebSocket mock client")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--mode", choices=["zero", "random", "replay"], default="zero")
    parser.add_argument("--config", default=None, help="Optional config for dataset.task_index default")
    add_task_id_cli_arguments(parser)
    parser.add_argument("--init-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--dataset-path", type=str, default="./dada/libero-object")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument(
        "--skip-obs-decode",
        action="store_true",
        help="Skip image transfer when not saving video (faster action-only test)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout MP4 (enabled by default in replay mode)",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable video saving in replay mode",
    )
    parser.add_argument(
        "--video-dir",
        default="./results/replay_videos",
        help="Root directory for replay videos (timestamped subfolder per run)",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument(
        "--camera",
        choices=["agentview", "wrist", "both"],
        default="agentview",
        help="Camera view(s) to record",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
