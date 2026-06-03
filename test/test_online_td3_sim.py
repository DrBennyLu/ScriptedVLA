"""
本地 LeRobot 数据上的 TD3 模拟在线训练脚本。

分步调试（任选其一）::

    python -m test.test_online_td3_sim --step dataset --dataset ./dataset/libero_object
    python -m test.test_online_td3_sim --step models
    python -m test.test_online_td3_sim --step replay
    python -m test.test_online_td3_sim --step warmup --save-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step online --load-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step eval --load-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step full

默认 --step full 为完整流程。

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from libero.rl_td3_replay import (
    LeRobotDatasetSubset,
    OnlineFeatureProvider,
    ReplayBuffer,
    build_episode_index,
    build_replay_from_dataset,
    collect_task_episode_ids,
    eval_with_vla_reference,
    export_q_logs,
    load_lerobot_dataset,
    load_td3_agent,
    load_trained_modules,
    online_train_loop,
    save_td3_agent_from_args,
    set_seed,
    warmup_train,
)
from src.ScriptedVLA.model import TD3ChunkAgent, TD3ChunkConfig
from src.ScriptedVLA.utils import (
    create_normalizer_from_dataset,
    create_normalizer_from_lerobot_meta,
    load_script_config,
)
from train_rl_token import create_delta_timestamps

# 固定测试子集（用于 dataset 步骤之外的快速迭代）
FIXED_EPISODE_SLICE = [
    0, 22, 25, 28, 30, 41, 47, 59, 63, 73, 91, 116, 119, 172, 206, 234, 236,
    237, 238, 239, 240, 242, 243, 266, 277, 286, 287, 307, 314, 315, 332, 339,
    348, 350, 352, 353, 365, 366, 368, 370, 390, 393, 400, 411, 420,
]


def build_argparser(td3_defaults: Dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LeRobot 上 TD3 模拟在线训练")
    p.add_argument(
        "--step",
        type=str,
        default="full",
        choices=("dataset", "models", "replay", "warmup", "online", "eval", "full"),
        help="分步调试或 full 全流程",
    )
    p.add_argument("--save-agent", type=str, default=None, help="保存 TD3 权重路径（.pt）")
    p.add_argument("--load-agent", type=str, default=None, help="加载已有 TD3 权重")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--dataset", type=str, default="./dataset/libero_object")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--chunk-len", type=int, default=td3_defaults["chunk_len"])
    p.add_argument("--stride", type=int, default=td3_defaults["stride"])
    p.add_argument("--batch-size", type=int, default=td3_defaults["batch_size"])
    p.add_argument("--replay-capacity", type=int, default=td3_defaults["replay_capacity"])
    p.add_argument("--warmup-updates", type=int, default=td3_defaults["warmup_updates"])
    p.add_argument("--warmup-update-ratio", type=int, default=td3_defaults["warmup_update_ratio"])
    p.add_argument("--online-train-episodes", type=int, default=td3_defaults["online_train_episodes"])
    p.add_argument("--max-eval-samples", type=int, default=td3_defaults["max_eval_samples"])
    p.add_argument("--vla-checkpoint", type=str, default=td3_defaults["vla_checkpoint"])
    p.add_argument("--rl-checkpoint", type=str, default=td3_defaults["rl_checkpoint"])
    p.add_argument("--output-dir", type=str, default=td3_defaults["output_dir"])
    p.add_argument("--gamma", type=float, default=td3_defaults["gamma"])
    p.add_argument("--tau", type=float, default=td3_defaults["tau"])
    p.add_argument("--actor-lr", type=float, default=td3_defaults["actor_lr"])
    p.add_argument("--critic-lr", type=float, default=td3_defaults["critic_lr"])
    p.add_argument("--policy-noise", type=float, default=td3_defaults["policy_noise"])
    p.add_argument("--noise-clip", type=float, default=td3_defaults["noise_clip"])
    p.add_argument("--policy-delay", type=int, default=td3_defaults["policy_delay"])
    p.add_argument("--actor-std", type=float, default=td3_defaults["actor_std"])
    p.add_argument("--max-action", type=float, default=td3_defaults["max_action"])
    p.add_argument("--ref-mask-prob", type=float, default=td3_defaults["ref_mask_prob"])
    p.add_argument("--policy-constraint-beta", type=float, default=td3_defaults["policy_constraint_beta"])
    p.add_argument(
        "--use-chunk-return-target",
        action=argparse.BooleanOptionalAction,
        default=td3_defaults["use_chunk_return_target"],
    )
    p.add_argument("--hidden-dim", type=int, default=td3_defaults["hidden_dim"])
    return p


def _load_fixed_subset_dataset(cfg, dataset_path_obj: Path):
    info_file = dataset_path_obj / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"本地数据集路径不存在或无效: {info_file}")

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    delta_timestamps = create_delta_timestamps(cfg.action_horizon, fps)
    dataset_name = dataset_path_obj.name
    root_path_str = str(dataset_path_obj)

    sub_ds = LeRobotDatasetSubset(
        repo_id=dataset_name,
        root=root_path_str,
        delta_timestamps=delta_timestamps,
        episodes=FIXED_EPISODE_SLICE,
    )
    print(f"  ✓ 固定测试集创建成功: repo_id={dataset_name}, root={root_path_str}")
    print(f"  ✓ 固定 episode 数量: {len(FIXED_EPISODE_SLICE)}")
    return sub_ds


def _assert_replay_chunk_spacing(
    replay: ReplayBuffer,
    episode_to_indices: dict,
    chunk_len: int,
) -> None:
    tr0 = next(iter(replay.buf))
    ep_indices = episode_to_indices[tr0.episode_id]
    local_curr = ep_indices.index(tr0.sample_index)
    local_next = ep_indices.index(tr0.next_sample_index)
    if local_next - local_curr != chunk_len:
        raise RuntimeError(
            f"expected next frame offset {chunk_len}, got {local_next - local_curr}"
        )


def _execute(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_once = load_script_config(args.config, dataset_path=args.dataset, seed=args.seed)

    out_dir = Path(args.output_dir)
    warmup_ckpt_path = out_dir / "warmup_agent.pt"
    print("=" * 80)
    print(f"[main] 启动脚本，step={args.step}, device={device}")
    print("=" * 80)

    if args.step == "dataset":
        print("[main] 当前任务：仅检查数据集与 task 对应 episode 列表")
        set_seed(args.seed)
        cfg = cfg_once
        full_ds = load_lerobot_dataset(Path(args.dataset), cfg.action_horizon)
        selected = collect_task_episode_ids(full_ds, args.task_index)
        sub_ds = load_lerobot_dataset(Path(args.dataset), cfg.action_horizon, episodes=selected)
        print(
            f"[dataset] task_index={args.task_index}, episodes={len(selected)}, len(sub_ds)={len(sub_ds)}"
        )
        if len(sub_ds) > 0:
            keys = sorted(sub_ds[0].keys())
            print(f"[dataset] 首帧字段示例: {keys[:24]}{'...' if len(keys) > 24 else ''}")
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_path = out_dir / "selected_episodes.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump({"task_index": args.task_index, "episodes": selected}, f, indent=2)
        print(f"[dataset] 已写入 {ep_path}")
        print("[dataset] ✅ 测试成功：数据集与 episode 切片检查完成")
        return

    if args.step == "models":
        print("[main] 当前任务：仅加载 VLA 与 RL token 模型")
        set_seed(args.seed)
        cfg, _, _ = load_trained_modules(
            args.config, args.dataset, args.vla_checkpoint, args.rl_checkpoint, device
        )
        print(
            f"[models] device={device}, state_dim={cfg.state_dim}, action_dim={cfg.action_dim}, "
            f"action_horizon={cfg.action_horizon}"
        )
        print("[models] ✅ 测试成功：模型与权重加载完成")
        return

    if args.step == "replay":
        print("[main] 当前任务：仅构建 ReplayBuffer")
        set_seed(args.seed)
        cfg, vla_model, rl_encoder = load_trained_modules(
            args.config, args.dataset, args.vla_checkpoint, args.rl_checkpoint, device
        )
        dataset_path_obj = Path(args.dataset).resolve()
        sub_ds = _load_fixed_subset_dataset(cfg, dataset_path_obj)

        try:
            _normalizer = create_normalizer_from_lerobot_meta(
                sub_ds,
                state_key=cfg.state_key,
                action_key="action",
            )
            print("  ✓ 归一化器已从 meta.episodes_stats 创建")
        except Exception as e:
            print(f"  从 meta.episodes_stats 创建归一化器失败: {e}，尝试 episodes_stats.jsonl")
            try:
                create_normalizer_from_dataset(dataset_path_obj)
                print("  ✓ 归一化器已从 episodes_stats.jsonl 创建")
            except Exception as e2:
                print(f"  ✗ 归一化器创建失败: {e2}")

        ep_map = build_episode_index(sub_ds)
        replay = build_replay_from_dataset(
            sub_ds,
            ep_map,
            cfg.state_key,
            args.chunk_len,
            args.stride,
            args.gamma,
            device,
            args.replay_capacity,
            vla_model,
            rl_encoder,
            cfg.image_keys,
            feature_batch_size=8,
        )
        print(f"[replay] len(replay)={len(replay)}, state_key={cfg.state_key}")
        _assert_replay_chunk_spacing(replay, ep_map, args.chunk_len)
        tr0 = next(iter(replay.buf))
        local_next = ep_map[tr0.episode_id].index(tr0.next_sample_index)
        local_curr = ep_map[tr0.episode_id].index(tr0.sample_index)
        print(
            f"[replay] 首条 transition: idx={tr0.sample_index}->{tr0.next_sample_index} "
            f"(+{local_next - local_curr} frames), z_rl.shape={tuple(tr0.z_rl.shape)}, "
            f"action.shape={tuple(tr0.action.shape)}"
        )
        print(f"[replay] ✅ 测试成功：ReplayBuffer 构建完成，样本数={len(replay)}")
        return

    set_seed(args.seed)
    print("[main] 当前任务：准备训练/评估所需模型与数据...")
    cfg, vla_model, rl_encoder = load_trained_modules(
        args.config, args.dataset, args.vla_checkpoint, args.rl_checkpoint, device
    )
    dataset_path_obj = Path(args.dataset).resolve()
    sub_ds = _load_fixed_subset_dataset(cfg, dataset_path_obj)

    try:
        _normalizer = create_normalizer_from_lerobot_meta(
            sub_ds,
            state_key=cfg.state_key,
            action_key="action",
        )
        print("  ✓ 归一化器已从 meta.episodes_stats 创建")
    except Exception as e:
        print(f"  从 meta.episodes_stats 创建归一化器失败: {e}")
        try:
            create_normalizer_from_dataset(dataset_path_obj)
            print("  ✓ 归一化器已从 episodes_stats.jsonl 创建")
        except Exception as e2:
            print(f"  ✗ 归一化器创建失败: {e2}")

    ep_map = build_episode_index(sub_ds)
    replay = build_replay_from_dataset(
        sub_ds,
        ep_map,
        cfg.state_key,
        args.chunk_len,
        args.stride,
        args.gamma,
        device,
        args.replay_capacity,
        vla_model,
        rl_encoder,
        cfg.image_keys,
        feature_batch_size=8,
    )
    rl_token_dim = int(rl_encoder.rl_token_dim)
    td3_cfg = TD3ChunkConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        fixed_std=args.actor_std,
        max_action=args.max_action,
        ref_mask_prob=args.ref_mask_prob,
        policy_constraint_beta=args.policy_constraint_beta,
        use_chunk_return_target=args.use_chunk_return_target,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    )
    provider = OnlineFeatureProvider(
        sub_ds,
        cfg.image_keys,
        cfg.state_key,
        vla_model,
        rl_encoder,
        device,
        args.chunk_len,
    )
    _assert_replay_chunk_spacing(replay, ep_map, args.chunk_len)

    default_warmup = max(1, int(np.ceil(len(replay) / max(args.batch_size, 1))))
    warmup_n = args.warmup_updates if args.warmup_updates > 0 else default_warmup

    if args.step == "warmup":
        print("[main] 当前任务：执行 warmup 训练")
        agent = TD3ChunkAgent(
            rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
        )
        logs = warmup_train(agent, replay, args.batch_size, warmup_n, args.warmup_update_ratio)
        export_q_logs(logs, out_dir, base_name="q_curve_warmup")
        save_td3_agent_from_args(warmup_ckpt_path, agent, args, cfg)
        if args.save_agent:
            save_td3_agent_from_args(Path(args.save_agent), agent, args, cfg)
        print(f"[warmup] ✅ 完成，更新步数={len(logs)}")
        return

    if args.step == "online":
        print("[main] 当前任务：执行 online 训练")
        if args.load_agent:
            agent = load_td3_agent(Path(args.load_agent), device)
        elif warmup_ckpt_path.exists():
            agent = load_td3_agent(warmup_ckpt_path, device)
        else:
            agent = TD3ChunkAgent(
                rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
            )
            warmup_train(agent, replay, args.batch_size, warmup_n, args.warmup_update_ratio)
            save_td3_agent_from_args(warmup_ckpt_path, agent, args, cfg)
        logs = online_train_loop(agent, replay, args.batch_size, args.online_train_episodes)
        export_q_logs(logs, out_dir, base_name="q_curve_online")
        if args.save_agent:
            save_td3_agent_from_args(Path(args.save_agent), agent, args, cfg)
        print(f"[online] ✅ 完成，更新步数={len(logs)}")
        return

    if args.step == "eval":
        print("[main] 当前任务：执行评估")
        if args.load_agent:
            agent = load_td3_agent(Path(args.load_agent), device)
        elif warmup_ckpt_path.exists():
            agent = load_td3_agent(warmup_ckpt_path, device)
        else:
            raise FileNotFoundError(
                f"[eval] 未找到可用权重：{warmup_ckpt_path}。"
                "请先执行 --step warmup，或通过 --load-agent 指定权重文件。"
            )
        metrics = eval_with_vla_reference(agent, replay, provider, args.max_eval_samples)
        print(
            f"[eval] mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
            f"l2={metrics['mean_l2_per_step']:.6f} ref_mse={metrics['ref_mse']:.6f} "
            f"q_mean={metrics['eval_q_mean']:.6f}"
        )
        print("[eval] ✅ 测试成功：评估完成")
        return

    if args.step == "full":
        print("[main] 当前任务：执行 full 全流程（warmup -> online -> eval）")
        agent = TD3ChunkAgent(
            rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
        )
        all_logs: List[Dict[str, Any]] = []
        all_logs.extend(
            warmup_train(agent, replay, args.batch_size, warmup_n, args.warmup_update_ratio)
        )
        all_logs.extend(
            online_train_loop(agent, replay, args.batch_size, args.online_train_episodes)
        )
        export_q_logs(all_logs, out_dir, base_name="q_curve")
        metrics = eval_with_vla_reference(agent, replay, provider, args.max_eval_samples)
        print(
            f"[full] mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
            f"l2={metrics['mean_l2_per_step']:.6f} ref_mse={metrics['ref_mse']:.6f} "
            f"q_mean={metrics['eval_q_mean']:.6f}"
        )
        if args.save_agent:
            save_td3_agent_from_args(Path(args.save_agent), agent, args, cfg)
        print("[full] ✅ 全流程完成")
        return

    raise RuntimeError(f"未知 step: {args.step}")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    cfg_for_defaults = load_script_config(pre_args.config)
    td3_defaults = cfg_for_defaults.raw_config["training"]["online_rl"]["td3"]

    _execute(build_argparser(td3_defaults).parse_args())


def run_online_td3_sim(args: argparse.Namespace) -> None:
    args = argparse.Namespace(**vars(args))
    args.step = "full"
    _execute(args)


if __name__ == "__main__":
    main()
