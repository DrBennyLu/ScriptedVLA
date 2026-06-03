#!/usr/bin/env python3
"""
Compare offline replay_cache vs online step buffers (ref/reward/state distributions).

Example::

  python -m libero.debug_td3_replay_stats \\
    --replay-cache ./data/replay_buffers/libero_object_task6/replay_cache.pt \\
    --online-buffer-dir ./data/replay_buffers/online_ws_task6 \\
    --output ./results/replay_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.rl_td3_replay import (
    ReplayBuffer,
    ReplayTransition,
    load_replay_buffer,
    summarize_replay_buffer,
    transition_action_ref_mse,
)


def _load_online_samples(buffer_dir: Path, max_files: int) -> ReplayBuffer:
    files = sorted(buffer_dir.glob("step_buffer_*.pt"))
    if not files:
        raise FileNotFoundError(f"no step_buffer_*.pt under {buffer_dir}")
    files = files[-max_files:]
    replay = ReplayBuffer(capacity=len(files) + 1)
    import torch

    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        tr = data.get("transition")
        if tr is None:
            continue
        if isinstance(tr, dict):
            tr = ReplayTransition(**tr)
        replay.add(tr)
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit offline vs online TD3 replay stats")
    parser.add_argument("--replay-cache", type=str, required=True)
    parser.add_argument("--online-buffer-dir", type=str, default=None)
    parser.add_argument("--offline-capacity", type=int, default=40000)
    parser.add_argument("--max-online-files", type=int, default=500)
    parser.add_argument("--output", type=str, default="./results/replay_audit.json")
    args = parser.parse_args()

    cache_path = Path(args.replay_cache).expanduser().resolve()
    offline_replay, meta = load_replay_buffer(cache_path, capacity=args.offline_capacity)

    report = {
        "replay_cache_path": str(cache_path),
        "cache_meta": {k: meta.get(k) for k in ("chunk_len", "stride", "gamma", "rl_token_dim", "state_dim")},
        "offline": summarize_replay_buffer(offline_replay, label="offline_cache"),
    }

    if args.online_buffer_dir:
        online_dir = Path(args.online_buffer_dir).expanduser().resolve()
        online_replay = _load_online_samples(online_dir, args.max_online_files)
        report["online"] = summarize_replay_buffer(online_replay, label="online_buffer")
        # Expert vs VLA ref: offline ref==action (expert); online ref is VLA
        off = report["offline"]
        on = report["online"]
        report["comparison_notes"] = {
            "offline_ref_equals_expert_action": True,
            "online_ref_is_vla_predict": True,
            "action_ref_mse_gap": float(on.get("action_ref_mse_mean", 0) - off.get("action_ref_mse_mean", 0)),
            "online_mid_episode_success_leak": int(on.get("mid_episode_success_reward_count", 0)),
        }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[replay_stats] saved {out_path}")


if __name__ == "__main__":
    main()
