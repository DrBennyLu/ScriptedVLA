#!/usr/bin/env python3
"""Full LIBERO-Object benchmark via WebSocket + baseline comparison report."""

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
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .libero_benchmark_baselines import (
    BC_TRANSFORMER_MULTITASK,
    LIBERO_OBJECT_TASK_NAMES,
    OPENVLA_LIBERO_OBJECT,
    RANDOM_POLICY,
    confidence_interval,
    mean_rate,
)
from .libero_task_mapping import map_dataset_task_indices
from .libero_ws_eval_core import run_eval_episode
from .libero_ws_client import LiberoWSClient
from inference import find_latest_checkpoint, load_model_from_checkpoint
from src.ScriptedVLA.utils import get_data_config, load_config


async def run_benchmark(
    ws_url: str,
    model,
    normalizer,
    image_keys,
    image_size: int,
    normalize_action: bool,
    normalize_state: bool,
    task_ids: List[int],
    n_eval: int,
    max_steps: int,
    chunk_steps: int,
    use_model: bool,
    mock_mode: str,
    dataset_path: Optional[str],
    align_joint_angles: bool = True,
    clip_normalized_state: bool = True,
) -> List[dict]:
    from .libero_ws_mock_client import run_episode as run_mock_episode

    results = []
    async with LiberoWSClient(ws_url) as client:
        await client.ping()
        for task_id in task_ids:
            successes = []
            steps_list = []
            for init_id in range(n_eval):
                if use_model:
                    r = await run_eval_episode(
                        client=client,
                        model=model,
                        normalizer=normalizer,
                        image_keys=image_keys,
                        image_size=image_size,
                        normalize_action=normalize_action,
                        normalize_state=normalize_state,
                        align_joint_angles=align_joint_angles,
                        clip_normalized_state=clip_normalized_state,
                        task_id=task_id,
                        init_id=init_id,
                        max_steps=max_steps,
                        chunk_steps=chunk_steps,
                        debug_ranges=False,
                    )
                else:
                    r = await run_mock_episode(
                        client=client,
                        task_id=task_id,
                        init_id=init_id,
                        max_steps=max_steps,
                        mode=mock_mode,
                        dataset_path=dataset_path,
                        episode_index=None,
                        chunk_steps=chunk_steps,
                        skip_obs_decode=True,
                    )
                successes.append(int(r["success"]))
                steps_list.append(r["steps"])
                print(
                    f"[benchmark] task_id={task_id} init_id={init_id} "
                    f"success={r['success']} steps={r['steps']}"
                )

            rate = sum(successes) / n_eval
            ci = confidence_interval(rate, n_eval)
            results.append(
                {
                    "task_id": task_id,
                    "task_name": LIBERO_OBJECT_TASK_NAMES[task_id],
                    "n_eval": n_eval,
                    "success_rate": rate,
                    "ci_95": ci,
                    "successes": sum(successes),
                    "avg_steps": sum(steps_list) / len(steps_list),
                }
            )
            print(
                f"[benchmark] task_id={task_id} rate={rate:.2%} ± {ci:.2%} "
                f"avg_steps={results[-1]['avg_steps']:.1f}"
            )
    return results


def write_comparison_report(
    results: List[dict],
    output_dir: Path,
    prefix: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{prefix}_{ts}.json"
    csv_path = output_dir / f"{prefix}_{ts}.csv"
    report_path = output_dir / f"{prefix}_{ts}_comparison.txt"

    overall = sum(r["success_rate"] for r in results) / len(results) if results else 0.0
    payload = {
        "timestamp": ts,
        "overall_success_rate": overall,
        "per_task": results,
        "baselines": {
            "bc_transformer_multitask_mean": mean_rate(BC_TRANSFORMER_MULTITASK),
            "openvla_libero_object_mean": mean_rate(OPENVLA_LIBERO_OBJECT),
            "random_policy_mean": 0.0,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "task_name",
                "success_rate",
                "ci_95",
                "avg_steps",
                "bc_transformer_ref",
                "openvla_ref",
                "random_ref",
            ],
        )
        writer.writeheader()
        for r in results:
            tid = r["task_id"]
            writer.writerow(
                {
                    "task_id": tid,
                    "task_name": r["task_name"],
                    "success_rate": f"{r['success_rate']:.4f}",
                    "ci_95": f"{r['ci_95']:.4f}",
                    "avg_steps": f"{r['avg_steps']:.1f}",
                    "bc_transformer_ref": f"{BC_TRANSFORMER_MULTITASK.get(tid, 0):.4f}",
                    "openvla_ref": f"{OPENVLA_LIBERO_OBJECT.get(tid, 0):.4f}",
                    "random_ref": "0.0000",
                }
            )

    lines = [
        "LIBERO-Object WebSocket Benchmark Comparison",
        f"Timestamp: {ts}",
        f"Overall success rate: {overall:.2%}",
        "",
        f"{'task_id':>7}  {'ours':>8}  {'±CI':>8}  {'bc_trans':>10}  {'openvla':>10}  task_name",
        "-" * 90,
    ]
    for r in results:
        tid = r["task_id"]
        lines.append(
            f"{tid:7d}  {r['success_rate']:7.2%}  {r['ci_95']:7.2%}  "
            f"{BC_TRANSFORMER_MULTITASK.get(tid, 0):9.2%}  "
            f"{OPENVLA_LIBERO_OBJECT.get(tid, 0):9.2%}  {r['task_name']}"
        )
    lines.extend(
        [
            "-" * 90,
            f"{'MEAN':>7}  {overall:7.2%}  {'':>8}  "
            f"{mean_rate(BC_TRANSFORMER_MULTITASK):9.2%}  "
            f"{mean_rate(OPENVLA_LIBERO_OBJECT):9.2%}",
            "",
            f"JSON: {json_path}",
            f"CSV: {csv_path}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report_path


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

    task_ids = list(range(10)) if args.task_ids is None else args.task_ids
    if args.dataset_task_ids is not None and len(args.dataset_task_ids) > 0:
        task_ids = map_dataset_task_indices(args.dataset_path, args.dataset_task_ids)
        print(
            f"[benchmark] mapped dataset task_index {args.dataset_task_ids} "
            f"-> benchmark task_id {task_ids}"
        )

    model = None
    normalizer = None
    use_model = not args.mock_mode

    if use_model:
        ckpt = args.checkpoint
        if ckpt is None:
            latest = find_latest_checkpoint(Path(args.checkpoint_dir))
            if latest is None:
                raise FileNotFoundError(
                    f"No checkpoint in {args.checkpoint_dir}. Use --mock-mode for pipeline test."
                )
            ckpt = str(latest)
        print(f"[benchmark] loading checkpoint: {ckpt}")
        model, normalizer = load_model_from_checkpoint(ckpt, args.config, device=args.device)

    results = await run_benchmark(
        ws_url=args.ws_url,
        model=model,
        normalizer=normalizer,
        image_keys=image_keys,
        image_size=image_size,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        align_joint_angles=align_joint_angles,
        clip_normalized_state=clip_normalized_state,
        task_ids=task_ids,
        n_eval=args.n_eval,
        max_steps=args.max_steps,
        chunk_steps=args.chunk_steps,
        use_model=use_model,
        mock_mode=args.mock_mode or "zero",
        dataset_path=args.dataset_path,
    )

    prefix = "libero_object_ws_eval" if use_model else f"libero_object_ws_mock_{args.mock_mode}"
    write_comparison_report(results, Path(args.output_dir), prefix)


def main():
    parser = argparse.ArgumentParser(description="LIBERO-Object WebSocket benchmark")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--config", default="libero/config_libero_object.yaml")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/libero_object")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="LIBERO benchmark task_ids (default: all 10 tasks, for baseline comparison)",
    )
    parser.add_argument(
        "--dataset-task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Dataset task_index list (mapped to benchmark task_id; overrides --task-ids)",
    )
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument(
        "--mock-mode",
        choices=["zero", "random", "replay"],
        default=None,
        help="If set, run mock client instead of model (for pipeline validation)",
    )
    parser.add_argument("--dataset-path", default="./dada/libero-object")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
