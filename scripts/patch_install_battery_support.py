#!/usr/bin/env python3
"""Minimal patches for install_battery (LeRobot v3 + pyav + meta.stats normalizer)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/data/share/1919650160032350208/users/lxg/ScriptedVLA")


def patch_normalization() -> None:
    path = ROOT / "src/ScriptedVLA/utils/normalization.py"
    text = path.read_text(encoding="utf-8")
    marker = "def create_normalizer_from_lerobot_meta("
    helper = '''
def create_normalizer_from_meta_stats(
    meta: Any,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> Normalizer:
    """Build Normalizer from LeRobot v3 global meta.stats (no episodes_stats)."""
    if not hasattr(meta, "stats") or meta.stats is None:
        raise AttributeError("meta.stats is missing; cannot build normalizer from global stats")
    stats = meta.stats
    if action_key not in stats:
        raise KeyError(f"action key {action_key!r} not in meta.stats")
    a = stats[action_key]
    if "min" not in a or "max" not in a:
        raise KeyError(f"meta.stats[{action_key}] missing min/max")
    action_min = np.asarray(a["min"], dtype=np.float64).flatten()
    action_max = np.asarray(a["max"], dtype=np.float64).flatten()
    state_min = state_max = None
    if state_key in stats and "min" in stats[state_key] and "max" in stats[state_key]:
        state_min = np.asarray(stats[state_key]["min"], dtype=np.float64).flatten()
        state_max = np.asarray(stats[state_key]["max"], dtype=np.float64).flatten()
    # Do NOT clamp to [-1,1] for real-robot eef/joint stats.
    return Normalizer(
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
    )


'''
    if "def create_normalizer_from_meta_stats(" not in text:
        if marker not in text:
            raise RuntimeError("create_normalizer_from_lerobot_meta not found")
        text = text.replace(marker, helper + marker, 1)
        path.write_text(text, encoding="utf-8")
        print("patched normalization.py: +create_normalizer_from_meta_stats")
    else:
        print("normalization.py already patched")


def patch_train() -> None:
    path = ROOT / "train.py"
    text = path.read_text(encoding="utf-8")

    # import helper
    old_imp = "create_normalizer_from_lerobot_meta,"
    new_imp = "create_normalizer_from_lerobot_meta,\n    create_normalizer_from_meta_stats,"
    if "create_normalizer_from_meta_stats" not in text:
        if old_imp not in text:
            raise RuntimeError("import create_normalizer_from_lerobot_meta missing")
        text = text.replace(old_imp, new_imp, 1)

    # pass video_backend
    old_ds = """        lerobot_dataset = LeRobotDatasetSubset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps,
            **episodes_kw
        )"""
    new_ds = """        video_backend = dataset_config.get("video_backend", "pyav")
        print(f"  video_backend: {video_backend}")
        lerobot_dataset = LeRobotDatasetSubset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            **episodes_kw
        )"""
    if "video_backend=video_backend" not in text:
        if old_ds not in text:
            raise RuntimeError("LeRobotDatasetSubset construction block not found")
        text = text.replace(old_ds, new_ds, 1)

    # normalizer fallback to meta.stats
    old_norm = """    try:
        normalizer = create_normalizer_from_lerobot_meta(
            lerobot_dataset,
            state_key=state_key,
            action_key="action",
        )
        print(f" ✓ 归一化器已从 meta.episodes_stats 创建")
        if normalizer.action_min is not None:
            print(f"  Action 范围（用于反归一化）: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
        if normalizer.state_min is not None:
            print(f"  State 范围（用于传入动作头前归一化）: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
    except Exception as e:
        print(f" 从 meta.episodes_stats 创建归一化器失败: {e}，尝试从 meta/episodes_stats.jsonl 创建")
        try:
            normalizer = create_normalizer_from_dataset(dataset_path_obj)
            print(f" ✓ 归一化器已从 episodes_stats.jsonl 创建")
        except Exception as e2:
            print(f" ✗ 归一化器创建失败: {e2}")
            print(f" 警告: 将不使用归一化，训练可能不稳定")
            normalizer = None"""
    new_norm = """    try:
        normalizer = create_normalizer_from_lerobot_meta(
            lerobot_dataset,
            state_key=state_key,
            action_key="action",
        )
        print(f" ✓ 归一化器已从 meta.episodes_stats 创建")
        if normalizer.action_min is not None:
            print(f"  Action 范围（用于反归一化）: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
        if normalizer.state_min is not None:
            print(f"  State 范围（用于传入动作头前归一化）: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
    except Exception as e:
        print(f" 从 meta.episodes_stats 创建归一化器失败: {e}，尝试 meta.stats / episodes_stats.jsonl")
        try:
            normalizer = create_normalizer_from_meta_stats(
                lerobot_dataset.meta,
                state_key=state_key,
                action_key="action",
            )
            print(f" ✓ 归一化器已从 meta.stats 创建")
            if normalizer.action_min is not None:
                print(f"  Action 范围: [{normalizer.action_min.min():.4f}, {normalizer.action_max.max():.4f}]")
            if normalizer.state_min is not None:
                print(f"  State 范围: [{normalizer.state_min.min():.4f}, {normalizer.state_max.max():.4f}]")
        except Exception as e_stats:
            print(f" 从 meta.stats 创建失败: {e_stats}，尝试 episodes_stats.jsonl")
            try:
                normalizer = create_normalizer_from_dataset(dataset_path_obj)
                print(f" ✓ 归一化器已从 episodes_stats.jsonl 创建")
            except Exception as e2:
                print(f" ✗ 归一化器创建失败: {e2}")
                print(f" 警告: 将不使用归一化，训练可能不稳定")
                normalizer = None"""
    if "create_normalizer_from_meta_stats(" not in text.split("步骤2.6")[-1][:2500]:
        if old_norm not in text:
            raise RuntimeError("normalizer block not found for patch")
        text = text.replace(old_norm, new_norm, 1)

    path.write_text(text, encoding="utf-8")
    print("patched train.py")


def main() -> None:
    patch_normalization()
    patch_train()
    print("PATCH_OK")


if __name__ == "__main__":
    main()
