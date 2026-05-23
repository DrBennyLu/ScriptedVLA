"""Load GT action sequences from LeRobot libero-object parquet files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def load_episode_actions(
    dataset_path: str,
    episode_index: int,
) -> Tuple[np.ndarray, str, int]:
    """
    Returns:
        actions: [T, 7]
        instruction: task text
        task_index: dataset task index
    """
    root = Path(dataset_path)
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    data_path_tpl = info["data_path"]
    rel = data_path_tpl.format(episode_chunk=0, episode_index=episode_index)
    parquet_path = root / rel
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"].values], axis=0)
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else -1

    instruction = ""
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists() and task_index >= 0:
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row["task_index"]) == task_index:
                    instruction = row["task"]
                    break
    if not instruction and "tasks" in df.columns:
        val = df["tasks"].iloc[0]
        if isinstance(val, (list, tuple)) and val:
            instruction = str(val[0])
        else:
            instruction = str(val)
    return actions, instruction, task_index


def normalize_instruction(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def find_episode_by_instruction(dataset_path: str, instruction: str) -> int:
    target = normalize_instruction(instruction)
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks = row.get("tasks", [])
            if not tasks:
                continue
            if normalize_instruction(tasks[0]) == target:
                return int(row["episode_index"])
    raise ValueError(
        f"No dataset episode found for instruction: {instruction!r} in {dataset_path}"
    )


def get_task_description(dataset_path: str, task_index: int) -> str:
    """从 meta/tasks.jsonl 读取指定 task_index 的任务描述。"""
    root = Path(dataset_path)
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_path}")

    total_tasks = 0
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            idx = int(row["task_index"])
            total_tasks = max(total_tasks, idx + 1)
            if idx == task_index:
                return row["task"]

    raise ValueError(
        f"task_index={task_index} out of range; valid range is 0..{total_tasks - 1}"
    )


def get_episode_ids_for_task_index(dataset_path: str, task_index: int) -> List[int]:
    """扫描 meta/episodes.jsonl，返回属于指定 task_index 的全部 episode 列表。"""
    task_text = get_task_description(dataset_path, task_index)
    target = normalize_instruction(task_text)

    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")

    episode_ids: List[int] = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks = row.get("tasks", [])
            if not tasks:
                continue
            if normalize_instruction(tasks[0]) == target:
                episode_ids.append(int(row["episode_index"]))

    if not episode_ids:
        raise ValueError(
            f"No episodes found for task_index={task_index} ({task_text!r}) in {dataset_path}"
        )
    return sorted(episode_ids)


def resolve_training_episodes(
    dataset_path: str,
    task_index: Optional[int],
    episode_slice: Optional[List[int]],
) -> Optional[List[int]]:
    """
    解析训练用 episode 列表。
    优先级: episode_slice > task_index > None（全量）。
    """
    if episode_slice is not None and len(episode_slice) > 0:
        return list(episode_slice)
    if task_index is not None:
        return get_episode_ids_for_task_index(dataset_path, int(task_index))
    return None


def resolve_eval_episode_id(
    dataset_path: str,
    task_index: Optional[int],
    eval_episode_id: Optional[int],
    default: int = 0,
) -> int:
    """解析评估默认 episode_id。"""
    if eval_episode_id is not None:
        return int(eval_episode_id)
    if task_index is not None:
        return get_episode_ids_for_task_index(dataset_path, int(task_index))[0]
    return default


def find_first_episode_for_task(dataset_path: str, task_index: int) -> int:
    return get_episode_ids_for_task_index(dataset_path, task_index)[0]
