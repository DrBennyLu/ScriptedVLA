#!/usr/bin/env python3
"""Patch ScriptedVLA/train.py for install_battery (pyav + meta.stats normalizer)."""

from __future__ import annotations

import re
from pathlib import Path

PATH = Path("/data/share/1919650160032350208/users/lxg/ScriptedVLA/train.py")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    if "create_normalizer_from_meta_stats," not in text:
        text2, n = re.subn(
            r"(create_normalizer_from_lerobot_meta,\n)",
            r"\1    create_normalizer_from_meta_stats,\n",
            text,
            count=1,
        )
        if n != 1:
            raise RuntimeError(f"import patch failed n={n}")
        text = text2
        print("import: applied")
    else:
        print("import: already")

    if "video_backend=video_backend" not in text:
        pat = re.compile(
            r"([ \t]*)lerobot_dataset = LeRobotDatasetSubset\(\n"
            r"([ \t]*)repo_id=dataset_name,\n"
            r"([ \t]*)root=root_path_str,\n"
            r"([ \t]*)delta_timestamps=delta_timestamps,\n"
            r"([ \t]*)\*\*episodes_kw\n"
            r"([ \t]*)\)",
            re.M,
        )

        def repl(m: re.Match[str]) -> str:
            ind = m.group(1)
            ind2 = m.group(2)
            return (
                f'{ind}video_backend = dataset_config.get("video_backend", "pyav")\n'
                f'{ind}print(f"  video_backend: {{video_backend}}")\n'
                f"{ind}lerobot_dataset = LeRobotDatasetSubset(\n"
                f"{ind2}repo_id=dataset_name,\n"
                f"{ind2}root=root_path_str,\n"
                f"{ind2}delta_timestamps=delta_timestamps,\n"
                f"{ind2}video_backend=video_backend,\n"
                f"{ind2}**episodes_kw\n"
                f"{m.group(6)})"
            )

        text2, n = pat.subn(repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"video_backend patch failed n={n}")
        text = text2
        print("video_backend: applied")
    else:
        print("video_backend: already")

    if "create_normalizer_from_meta_stats(" not in text:
        pat = re.compile(
            r'([ \t]*)print\(f"\s*从 meta\.episodes_stats 创建归一化器失败: \{e\}，尝试从 meta/episodes_stats\.jsonl 创建"\)\n'
            r"([ \t]*)try:\n"
            r"([ \t]*)normalizer = create_normalizer_from_dataset\(dataset_path_obj\)\n"
            r'([ \t]*)print\(f"\s*✓ 归一化器已从 episodes_stats\.jsonl 创建"\)\n'
            r"([ \t]*)except Exception as e2:\n"
            r'([ \t]*)print\(f"\s*✗ 归一化器创建失败: \{e2\}"\)\n'
            r'([ \t]*)print\(f"\s*警告: 将不使用归一化，训练可能不稳定"\)\n'
            r"([ \t]*)normalizer = None",
            re.M,
        )

        def repl_fb(m: re.Match[str]) -> str:
            ind = m.group(1)
            ind_try = m.group(2)
            ind_body = m.group(3)
            return (
                f'{ind}print(f" 从 meta.episodes_stats 创建归一化器失败: {{e}}，尝试 meta.stats / episodes_stats.jsonl")\n'
                f"{ind_try}try:\n"
                f"{ind_body}normalizer = create_normalizer_from_meta_stats(\n"
                f"{ind_body}    lerobot_dataset.meta,\n"
                f"{ind_body}    state_key=state_key,\n"
                f'{ind_body}    action_key="action",\n'
                f"{ind_body})\n"
                f'{ind_body}print(f" ✓ 归一化器已从 meta.stats 创建")\n'
                f"{ind_body}if normalizer.action_min is not None:\n"
                f'{ind_body}    print(f"  Action 范围: [{{normalizer.action_min.min():.4f}}, {{normalizer.action_max.max():.4f}}]")\n'
                f"{ind_body}if normalizer.state_min is not None:\n"
                f'{ind_body}    print(f"  State 范围: [{{normalizer.state_min.min():.4f}}, {{normalizer.state_max.max():.4f}}]")\n'
                f"{ind_try}except Exception as e_stats:\n"
                f'{ind_body}print(f" 从 meta.stats 创建失败: {{e_stats}}，尝试 episodes_stats.jsonl")\n'
                f"{ind_body}try:\n"
                f"{ind_body}    normalizer = create_normalizer_from_dataset(dataset_path_obj)\n"
                f'{ind_body}    print(f" ✓ 归一化器已从 episodes_stats.jsonl 创建")\n'
                f"{ind_body}except Exception as e2:\n"
                f'{ind_body}    print(f" ✗ 归一化器创建失败: {{e2}}")\n'
                f'{ind_body}    print(f" 警告: 将不使用归一化，训练可能不稳定")\n'
                f"{ind_body}    normalizer = None"
            )

        text2, n = pat.subn(repl_fb, text, count=1)
        if n != 1:
            raise RuntimeError(f"normalizer fallback patch failed n={n}")
        text = text2
        print("normalizer_fallback: applied")
    else:
        print("normalizer_fallback: already")

    PATH.write_text(text, encoding="utf-8")
    print("TRAIN_PATCH_OK")


if __name__ == "__main__":
    main()
