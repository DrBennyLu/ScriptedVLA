#!/usr/bin/env python3
"""Audit state/action normalizer stats vs episodes_stats and optional checkpoint."""

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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .libero_action_utils import action_dim_labels
from .libero_dataset_replay import resolve_training_episodes
from .libero_state_utils import state_dim_labels, state_normalization_diagnostics
from src.ScriptedVLA.utils import (
    Normalizer,
    clamp_action_stats_to_unit_bounds,
    compute_normalization_stats_from_episodes_stats_items,
    create_normalizer_from_dataset,
    create_normalizer_from_lerobot_meta,
    iter_episodes_stats_from_jsonl,
)


def _load_checkpoint_normalizer(checkpoint_path: Path) -> Optional[Normalizer]:
    if not checkpoint_path.exists():
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "normalizer" not in ckpt:
        return None
    return Normalizer.from_dict(ckpt["normalizer"])


def _stats_to_normalizer(
    action_min: np.ndarray,
    action_max: np.ndarray,
    state_min: Optional[np.ndarray],
    state_max: Optional[np.ndarray],
) -> Normalizer:
    return Normalizer(
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
    )


def _compare_action_stats(
    name_a: str,
    norm_a: Normalizer,
    name_b: str,
    norm_b: Normalizer,
) -> Dict[str, Any]:
    if norm_a.action_min is None or norm_b.action_min is None:
        return {"error": "missing action stats"}
    diff_min = np.abs(norm_a.action_min - norm_b.action_min)
    diff_max = np.abs(norm_a.action_max - norm_b.action_max)
    return {
        "min_max_abs_diff": float(max(diff_min.max(), diff_max.max())),
        "min_mean_abs_diff": float(diff_min.mean()),
        "max_mean_abs_diff": float(diff_max.mean()),
        "labels": action_dim_labels(),
        name_a: {
            "action_min": norm_a.action_min.tolist(),
            "action_max": norm_a.action_max.tolist(),
            "action_range": norm_a.action_range.tolist(),
        },
        name_b: {
            "action_min": norm_b.action_min.tolist(),
            "action_max": norm_b.action_max.tolist(),
            "action_range": norm_b.action_range.tolist(),
        },
        "per_dim_min_diff": diff_min.tolist(),
        "per_dim_max_diff": diff_max.tolist(),
    }


def verify_action_convention(dataset_path: Path, episode_id: int = 0) -> Dict[str, Any]:
    """Verify LeRobot parquet action layout and global min/max vs [-1, 1]."""
    import pandas as pd

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    rel = info["data_path"].format(episode_chunk=0, episode_index=episode_id)
    parquet_path = dataset_path / rel
    df = pd.read_parquet(parquet_path)
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"].values])
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    global_min = actions.min(axis=0)
    global_max = actions.max(axis=0)
    return {
        "layout": "7-dim OSC_POSE, dataset values in [-1, 1]",
        "action_dim_labels": action_dim_labels(),
        "episode_id": episode_id,
        "frame_count": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "global_min": global_min.tolist(),
        "global_max": global_max.tolist(),
        "within_unit_bounds": bool(global_min.min() >= -1.0 - 1e-5 and global_max.max() <= 1.0 + 1e-5),
        "frame0": actions[0].tolist(),
    }


def _action_roundtrip_case(
    normalizer: Normalizer,
    raw: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    norm = np.asarray(normalizer.normalize_action(raw), dtype=np.float64).reshape(-1)
    back = np.asarray(normalizer.denormalize_action(norm), dtype=np.float64).reshape(-1)
    err = np.abs(back - raw)
    norm_oob = [i for i, v in enumerate(norm) if v < -1.0 - 1e-5 or v > 1.0 + 1e-5]
    return {
        "label": label,
        "raw": raw.tolist(),
        "normalized": norm.tolist(),
        "denormalized": back.tolist(),
        "roundtrip_error": err.tolist(),
        "roundtrip_l2": float(np.linalg.norm(err)),
        "roundtrip_max_abs": float(err.max()),
        "norm_outside_unit_bounds_dims": norm_oob,
    }


def _action_roundtrip_suite(normalizer: Normalizer, sample_raw: np.ndarray) -> Dict[str, Any]:
    dim = len(sample_raw)
    cases = [
        _action_roundtrip_case(normalizer, sample_raw, "raw0"),
        _action_roundtrip_case(normalizer, np.full(dim, -1.0), "all(-1)"),
        _action_roundtrip_case(normalizer, np.full(dim, 1.0), "all(+1)"),
    ]
    rng = np.random.default_rng(42)
    rand = rng.uniform(-1.0, 1.0, size=dim)
    cases.append(_action_roundtrip_case(normalizer, rand, "random_in_unit"))
    return {"cases": cases}


def _compare_state_stats(
    name_a: str,
    norm_a: Normalizer,
    name_b: str,
    norm_b: Normalizer,
) -> Dict[str, Any]:
    if norm_a.state_min is None or norm_b.state_min is None:
        return {"error": "missing state stats"}
    diff_min = np.abs(norm_a.state_min - norm_b.state_min)
    diff_max = np.abs(norm_a.state_max - norm_b.state_max)
    return {
        "min_max_abs_diff": float(max(diff_min.max(), diff_max.max())),
        "min_mean_abs_diff": float(diff_min.mean()),
        "max_mean_abs_diff": float(diff_max.mean()),
        "labels": state_dim_labels(),
        name_a: {
            "state_min": norm_a.state_min.tolist(),
            "state_max": norm_a.state_max.tolist(),
            "state_range": norm_a.state_range.tolist(),
        },
        name_b: {
            "state_min": norm_b.state_min.tolist(),
            "state_max": norm_b.state_max.tolist(),
            "state_range": norm_b.state_range.tolist(),
        },
        "per_dim_min_diff": diff_min.tolist(),
        "per_dim_max_diff": diff_max.tolist(),
    }


def _normalize_preview(normalizer: Normalizer, raw: np.ndarray, align: bool = False) -> Dict[str, Any]:
    diag = state_normalization_diagnostics(
        normalizer, raw, align_joint_angles=align, clip=True
    )
    in_range = []
    raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    for i, x in enumerate(raw_arr):
        lo, hi = normalizer.state_min[i], normalizer.state_max[i]
        inside = bool(lo <= x <= hi) if lo <= hi else bool(hi <= x <= lo)
        in_range.append(inside)
    return {
        **diag,
        "in_training_range": in_range,
        "out_of_range_dims": [i for i, ok in enumerate(in_range) if not ok],
    }


def verify_state_convention(dataset_path: Path, episode_id: int = 0) -> Dict[str, Any]:
    """
    Verify LeRobot parquet observation.state layout: 6 joints + 2 gripper fingers.
    """
    import pandas as pd

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    rel = info["data_path"].format(episode_chunk=0, episode_index=episode_id)
    df = pd.read_parquet(dataset_path / rel)
    states = np.stack([np.asarray(s, dtype=np.float32) for s in df["observation.state"].values])

    joint_like = states[:, :6]
    grip_like = states[:, 6:8]
    return {
        "layout": "joint_pos[:6] + gripper_qpos[:2]",
        "episode_id": episode_id,
        "frame_count": int(states.shape[0]),
        "joint_dims_0_5_range": {
            "min": joint_like.min(axis=0).tolist(),
            "max": joint_like.max(axis=0).tolist(),
        },
        "gripper_dims_6_7_range": {
            "min": grip_like.min(axis=0).tolist(),
            "max": grip_like.max(axis=0).tolist(),
        },
        "gripper_sum_mean": float((grip_like[:, 0] + grip_like[:, 1]).mean()),
        "gripper_dims_narrow_band": bool(
            grip_like.max(axis=0).max() < 0.5 and grip_like.min(axis=0).min() > -0.5
        ),
        "frame0": states[0].tolist(),
    }


def run_audit(
    dataset_path: Path,
    checkpoint: Optional[Path],
    task_index: Optional[int],
    state_key: str = "observation.state",
    sample_raw_state: Optional[np.ndarray] = None,
    sample_raw_action: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    task_episodes = resolve_training_episodes(str(dataset_path), task_index, None)

    full_items = list(iter_episodes_stats_from_jsonl(episodes_stats_path))
    full_a_min, full_a_max, full_s_min, full_s_max = compute_normalization_stats_from_episodes_stats_items(
        full_items, state_key=state_key, action_key="action"
    )
    norm_full = _stats_to_normalizer(full_a_min, full_a_max, full_s_min, full_s_max)
    clamp_a_min, clamp_a_max = clamp_action_stats_to_unit_bounds(full_a_min, full_a_max)
    norm_full_clamped = _stats_to_normalizer(clamp_a_min, clamp_a_max, full_s_min, full_s_max)

    norm_task0 = None
    task0_count = 0
    if task_episodes is not None:
        task_set = set(task_episodes)
        task_items = [(i, s) for i, s in full_items if i in task_set]
        task0_count = len(task_items)
        ta_min, ta_max, ts_min, ts_max = compute_normalization_stats_from_episodes_stats_items(
            task_items, state_key=state_key, action_key="action"
        )
        norm_task0 = _stats_to_normalizer(ta_min, ta_max, ts_min, ts_max)

    norm_ckpt = _load_checkpoint_normalizer(checkpoint) if checkpoint else None

    report: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "task_index": task_index,
        "task_episode_count": task0_count,
        "task_episode_ids_preview": task_episodes[:10] if task_episodes else None,
        "state_dim_labels": state_dim_labels(),
        "action_dim_labels": action_dim_labels(),
        "state_convention": verify_state_convention(dataset_path, episode_id=0),
        "action_convention": verify_action_convention(dataset_path, episode_id=0),
        "full_dataset_stats": {
            "episode_count": len(full_items),
            "state_min": full_s_min.tolist() if full_s_min is not None else None,
            "state_max": full_s_max.tolist() if full_s_max is not None else None,
            "state_range": norm_full.state_range.tolist() if norm_full.state_range is not None else None,
            "action_min": full_a_min.tolist() if full_a_min is not None else None,
            "action_max": full_a_max.tolist() if full_a_max is not None else None,
            "action_range": norm_full.action_range.tolist() if norm_full.action_range is not None else None,
            "action_min_clamped": clamp_a_min.tolist() if clamp_a_min is not None else None,
            "action_max_clamped": clamp_a_max.tolist() if clamp_a_max is not None else None,
        },
    }

    if norm_task0 is not None:
        report["task_subset_stats"] = {
            "state_min": norm_task0.state_min.tolist(),
            "state_max": norm_task0.state_max.tolist(),
            "state_range": norm_task0.state_range.tolist(),
            "action_min": norm_task0.action_min.tolist(),
            "action_max": norm_task0.action_max.tolist(),
            "action_range": norm_task0.action_range.tolist(),
        }
        report["full_vs_task0"] = _compare_state_stats("full", norm_full, "task0", norm_task0)
        report["full_vs_task0_action"] = _compare_action_stats("full", norm_full, "task0", norm_task0)

    if norm_ckpt is not None:
        report["checkpoint_stats"] = {
            "state_min": norm_ckpt.state_min.tolist() if norm_ckpt.state_min is not None else None,
            "state_max": norm_ckpt.state_max.tolist() if norm_ckpt.state_max is not None else None,
            "state_range": norm_ckpt.state_range.tolist() if norm_ckpt.state_range is not None else None,
            "action_min": norm_ckpt.action_min.tolist() if norm_ckpt.action_min is not None else None,
            "action_max": norm_ckpt.action_max.tolist() if norm_ckpt.action_max is not None else None,
            "action_range": norm_ckpt.action_range.tolist() if norm_ckpt.action_range is not None else None,
        }
        report["checkpoint_vs_full"] = _compare_state_stats("checkpoint", norm_ckpt, "full", norm_full)
        report["checkpoint_vs_full_action"] = _compare_action_stats("checkpoint", norm_ckpt, "full", norm_full)
        if norm_task0 is not None:
            report["checkpoint_vs_task0"] = _compare_state_stats(
                "checkpoint", norm_ckpt, "task0", norm_task0
            )
            report["checkpoint_vs_task0_action"] = _compare_action_stats(
                "checkpoint", norm_ckpt, "task0", norm_task0
            )

    if sample_raw_action is not None:
        report["action_roundtrip"] = {
            "checkpoint": (
                _action_roundtrip_suite(norm_ckpt, sample_raw_action)
                if norm_ckpt is not None
                else None
            ),
            "full_stats_unclamped": _action_roundtrip_suite(norm_full, sample_raw_action),
            "full_stats_clamped": _action_roundtrip_suite(norm_full_clamped, sample_raw_action),
        }

    if sample_raw_state is not None and norm_ckpt is not None:
        report["sample_state_vs_checkpoint"] = _normalize_preview(norm_ckpt, sample_raw_state)
        # Representative sim OOD state (compare report task0 frame0 / init0)
        sim_example = np.array(
            [
                -0.014220706187188625,
                -0.16911499202251434,
                -0.017590703442692757,
                -2.4538979530334473,
                -0.010858085937798023,
                2.221376657485962,
                0.020674293860793114,
                -0.020668666809797287,
            ],
            dtype=np.float32,
        )
        report["sim_state_alignment_preview"] = state_normalization_diagnostics(
            norm_ckpt, sim_example, align_joint_angles=True, clip=True
        )

    # Posttrain recommendation
    if norm_ckpt is not None and norm_task0 is not None:
        ckpt_full_diff = report.get("checkpoint_vs_full", {}).get("min_max_abs_diff", 0.0)
        ckpt_task_diff = report.get("checkpoint_vs_task0", {}).get("min_max_abs_diff", 0.0)
        if ckpt_full_diff < 1e-5:
            report["posttrain_normalizer_recommendation"] = (
                "Keep init_checkpoint normalizer: checkpoint matches full-dataset episodes_stats "
                "(pretrain scope). With init_checkpoint_normalizer=true this is expected."
            )
        elif ckpt_task_diff < ckpt_full_diff:
            report["posttrain_normalizer_recommendation"] = (
                "Consider task0-only normalizer for posttrain-only eval; checkpoint closer to task0 "
                "subset than full dataset."
            )
        else:
            report["posttrain_normalizer_recommendation"] = (
                "Keep checkpoint normalizer for weight consistency unless retraining with new stats."
            )

    return report


def _print_report(report: Dict[str, Any]) -> None:
    labels = report.get("state_dim_labels", [])
    print("=" * 72)
    print("State normalizer audit")
    print("=" * 72)
    print(f"Dataset: {report['dataset_path']}")
    if report.get("checkpoint"):
        print(f"Checkpoint: {report['checkpoint']}")
    if report.get("task_index") is not None:
        print(f"Task index: {report['task_index']} ({report.get('task_episode_count')} episodes)")

    full = report["full_dataset_stats"]
    print("\n--- Full dataset state min/max/range ---")
    for i, label in enumerate(labels):
        smin = full["state_min"][i]
        smax = full["state_max"][i]
        srng = full["state_range"][i]
        print(f"  [{i}] {label:16s} min={smin:+.6f} max={smax:+.6f} range={srng:.6f}")

    if "task_subset_stats" in report:
        print("\n--- Task subset state min/max (first diff vs full) ---")
        task = report["task_subset_stats"]
        for i, label in enumerate(labels):
            dmin = abs(task["state_min"][i] - full["state_min"][i])
            dmax = abs(task["state_max"][i] - full["state_max"][i])
            flag = " *" if dmin > 1e-4 or dmax > 1e-4 else ""
            print(
                f"  [{i}] {label:16s} min={task['state_min'][i]:+.6f} "
                f"max={task['state_max'][i]:+.6f}  Δmin={dmin:.6f} Δmax={dmax:.6f}{flag}"
            )

    if "checkpoint_stats" in report:
        print("\n--- Checkpoint state min/max ---")
        ck = report["checkpoint_stats"]
        cmp_full = report.get("checkpoint_vs_full", {})
        print(
            f"  vs full: min_max_abs_diff={cmp_full.get('min_max_abs_diff', 'n/a')}"
        )
        if "checkpoint_vs_task0" in report:
            cmp_t0 = report["checkpoint_vs_task0"]
            print(
                f"  vs task0: min_max_abs_diff={cmp_t0.get('min_max_abs_diff', 'n/a')}"
            )

    if "sample_state_vs_checkpoint" in report:
        s = report["sample_state_vs_checkpoint"]
        print("\n--- Sample raw state normalization preview ---")
        for i, label in enumerate(labels):
            print(
                f"  [{i}] {label:16s} raw={s['raw'][i]:+.6f} "
                f"norm={s['normalized'][i]:+.4f} in_range={s['in_training_range'][i]}"
            )
        if s["out_of_range_dims"]:
            print(f"  OUT OF RANGE dims: {s['out_of_range_dims']}")

    if report.get("posttrain_normalizer_recommendation"):
        print("\n--- Recommendation ---")
        print(f"  {report['posttrain_normalizer_recommendation']}")

    action_labels = report.get("action_dim_labels", action_dim_labels())
    action_conv = report.get("action_convention", {})
    print("\n" + "=" * 72)
    print("Action normalizer audit")
    print("=" * 72)
    print(f"  Convention: {action_conv.get('layout', 'n/a')}")
    if action_conv.get("within_unit_bounds") is not None:
        print(f"  Parquet within [-1,1]: {action_conv['within_unit_bounds']}")

    full = report["full_dataset_stats"]
    if full.get("action_min"):
        print("\n--- Full dataset action min/max/range (episodes_stats, unclamped) ---")
        for i, label in enumerate(action_labels):
            amin = full["action_min"][i]
            amax = full["action_max"][i]
            arng = full["action_range"][i]
            print(f"  [{i}] {label:8s} min={amin:+.6f} max={amax:+.6f} range={arng:.6f}")
        if full.get("action_min_clamped"):
            print("\n--- Full dataset action min/max (clamped to [-1,1]) ---")
            for i, label in enumerate(action_labels):
                print(
                    f"  [{i}] {label:8s} min={full['action_min_clamped'][i]:+.6f} "
                    f"max={full['action_max_clamped'][i]:+.6f}"
                )

    if "checkpoint_stats" in report and report["checkpoint_stats"].get("action_min"):
        ck = report["checkpoint_stats"]
        cmp_a = report.get("checkpoint_vs_full_action", {})
        print("\n--- Checkpoint action min/max ---")
        print(f"  vs full (action): min_max_abs_diff={cmp_a.get('min_max_abs_diff', 'n/a')}")

    rt = report.get("action_roundtrip")
    if rt:
        for suite_name in ("checkpoint", "full_stats_unclamped", "full_stats_clamped"):
            suite = rt.get(suite_name)
            if not suite:
                continue
            print(f"\n--- Action roundtrip ({suite_name}) ---")
            for case in suite["cases"]:
                oob = case.get("norm_outside_unit_bounds_dims", [])
                oob_str = f" norm_oob_dims={oob}" if oob else ""
                print(
                    f"  {case['label']:16s} roundtrip_l2={case['roundtrip_l2']:.6f} "
                    f"max_err={case['roundtrip_max_abs']:.6f}{oob_str}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LIBERO state/action normalizer statistics")
    parser.add_argument("--dataset-path", type=Path, default=Path("./dada/libero-object"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sample_state = None
    sample_action = None
    try:
        import pandas as pd

        info_path = args.dataset_path / "meta" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        rel = info["data_path"].format(episode_chunk=0, episode_index=args.episode_id)
        df = pd.read_parquet(args.dataset_path / rel)
        sample_state = np.asarray(df.iloc[args.frame_index]["observation.state"], dtype=np.float32)
        raw_act = df.iloc[args.frame_index]["action"]
        sample_action = np.asarray(raw_act, dtype=np.float32).reshape(-1)
        if sample_action.ndim > 1:
            sample_action = sample_action[0]
    except Exception as exc:
        print(f"Warning: could not load sample state/action from parquet: {exc}")

    report = run_audit(
        dataset_path=args.dataset_path.resolve(),
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        task_index=args.task_index,
        sample_raw_state=sample_state,
        sample_raw_action=sample_action,
    )
    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
