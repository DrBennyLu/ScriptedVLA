"""Map LeRobot dataset task_index to LIBERO benchmark task_id."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from libero_benchmark_baselines import LIBERO_OBJECT_TASK_NAMES
from libero_dataset_replay import normalize_instruction


def benchmark_task_language(task_id: int) -> str:
    """Natural-language instruction for a LIBERO benchmark task_id."""
    if task_id < 0 or task_id >= len(LIBERO_OBJECT_TASK_NAMES):
        raise ValueError(
            f"benchmark task_id={task_id} out of range; valid range is 0..{len(LIBERO_OBJECT_TASK_NAMES) - 1}"
        )
    name = LIBERO_OBJECT_TASK_NAMES[task_id]
    return normalize_instruction(name.replace("_", " "))


@lru_cache(maxsize=8)
def build_dataset_to_benchmark_map(dataset_path: str) -> Dict[int, int]:
    """
    Build dataset task_index -> LIBERO benchmark task_id by matching task text.
    """
    tasks_file = Path(dataset_path) / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_file}")

    benchmark_by_text = {
        benchmark_task_language(bid): bid for bid in range(len(LIBERO_OBJECT_TASK_NAMES))
    }
    mapping: Dict[int, int] = {}

    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ds_idx = int(row["task_index"])
            ds_text = normalize_instruction(row["task"])
            if ds_text not in benchmark_by_text:
                raise ValueError(
                    f"Dataset task_index={ds_idx} ({row['task']!r}) has no matching "
                    f"LIBERO benchmark task in {dataset_path}"
                )
            mapping[ds_idx] = benchmark_by_text[ds_text]

    return mapping


def dataset_task_index_to_benchmark_task_id(dataset_path: str, task_index: int) -> int:
    mapping = build_dataset_to_benchmark_map(dataset_path)
    if task_index not in mapping:
        raise ValueError(
            f"dataset task_index={task_index} not found in {dataset_path}/meta/tasks.jsonl"
        )
    return mapping[task_index]


def benchmark_task_id_to_dataset_task_index(dataset_path: str, task_id: int) -> int:
    mapping = build_dataset_to_benchmark_map(dataset_path)
    reverse = {bid: ds_idx for ds_idx, bid in mapping.items()}
    if task_id not in reverse:
        raise ValueError(f"benchmark task_id={task_id} has no dataset task_index mapping")
    return reverse[task_id]


def map_dataset_task_indices(dataset_path: str, dataset_task_indices: Sequence[int]) -> List[int]:
    return [
        dataset_task_index_to_benchmark_task_id(dataset_path, int(idx))
        for idx in dataset_task_indices
    ]


def format_task_mapping_line(dataset_path: str, dataset_task_index: int) -> str:
    benchmark_id = dataset_task_index_to_benchmark_task_id(dataset_path, dataset_task_index)
    from libero_dataset_replay import get_task_description

    text = get_task_description(dataset_path, dataset_task_index)
    return (
        f"dataset task_index={dataset_task_index} ({text!r}) "
        f"-> benchmark task_id={benchmark_id}"
    )


def resolve_benchmark_task_ids(
    *,
    dataset_path: str,
    dataset_task_id: Optional[int] = None,
    dataset_task_ids: Optional[Sequence[int]] = None,
    benchmark_task_id: Optional[int] = None,
    benchmark_task_ids: Optional[Sequence[int]] = None,
    config_task_index: Optional[int] = None,
) -> List[int]:
    """
    Resolve CLI/config task selectors to LIBERO benchmark task_id list.

    Priority:
      1. benchmark_task_ids / benchmark_task_id (no conversion)
      2. dataset_task_ids / dataset_task_id (mapped via tasks.jsonl)
      3. config dataset.task_index
      4. default dataset task_index 0
    """
    if benchmark_task_ids is not None and len(benchmark_task_ids) > 0:
        return [int(t) for t in benchmark_task_ids]
    if benchmark_task_id is not None:
        return [int(benchmark_task_id)]

    if dataset_task_ids is not None and len(dataset_task_ids) > 0:
        indices = [int(t) for t in dataset_task_ids]
    elif dataset_task_id is not None:
        indices = [int(dataset_task_id)]
    elif config_task_index is not None:
        indices = [int(config_task_index)]
    else:
        indices = [0]

    benchmark_ids = map_dataset_task_indices(dataset_path, indices)
    for ds_idx, bid in zip(indices, benchmark_ids):
        print(f"[task_map] {format_task_mapping_line(dataset_path, ds_idx)}")
    return benchmark_ids


def add_task_id_cli_arguments(parser) -> None:
    """Register task selection args shared by WebSocket eval scripts."""
    group = parser.add_argument_group(
        "Task selection",
        "By default uses config dataset.task_index (or 0). "
        "Values passed to --task-id / --task-ids are LeRobot dataset task_index. "
        "Use --benchmark-task-id* to pass LIBERO simulator task_id directly.",
    )
    group.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Dataset task_index (mapped to LIBERO benchmark task_id before eval)",
    )
    group.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Multiple dataset task_index values (overrides --task-id)",
    )
    group.add_argument(
        "--benchmark-task-id",
        type=int,
        default=None,
        help="LIBERO benchmark task_id (skip dataset mapping; overrides --task-id)",
    )
    group.add_argument(
        "--benchmark-task-ids",
        type=int,
        nargs="*",
        default=None,
        help="Multiple LIBERO benchmark task_ids (skip dataset mapping)",
    )


def resolve_task_ids_from_args(args, *, dataset_path: str, config: Optional[dict] = None) -> List[int]:
    config_task_index = None
    if config is not None:
        config_task_index = config.get("dataset", {}).get("task_index")

    return resolve_benchmark_task_ids(
        dataset_path=dataset_path,
        dataset_task_id=args.task_id,
        dataset_task_ids=args.task_ids,
        benchmark_task_id=args.benchmark_task_id,
        benchmark_task_ids=args.benchmark_task_ids,
        config_task_index=config_task_index,
    )
