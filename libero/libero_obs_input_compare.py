#!/usr/bin/env python3
"""
Compare LIBERO dataset frame vs WebSocket sim obs at VLA model input boundary.

Training path: LeRobotDatasetSubset -> create_collate_fn
Inference path: LiberoWSClient.create_episode -> ws_obs_to_model_inputs -> prepare_images_input

Usage:
  # Terminal 1 (LIBERO env): python scripts/libero_ws_server.py --suite libero_object
  # Terminal 2 (ScriptedVLA env):
  python -m libero.libero_obs_input_compare \\
    --config libero/config_libero_object.yaml \\
    --dataset-path ./dada/libero-object \\
    --ws-url ws://127.0.0.1:8765 \\
    --task-index 0 \\
    --checkpoint ./checkpoints/libero_object_task0_posttrain/checkpoint_step_100000.pt \\
    --deep
"""

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
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from inference import find_latest_checkpoint, load_model_from_checkpoint, prepare_images_input, run_inference
from .libero_action_utils import action_dim_labels
from .libero_dataset_replay import get_task_description, resolve_eval_episode_id
from .libero_state_utils import state_normalization_diagnostics
from .libero_obs_adapter import decode_b64_image, ws_obs_to_model_inputs
from .libero_task_mapping import dataset_task_index_to_benchmark_task_id
from .libero_ws_client import LiberoWSClient
from src.ScriptedVLA.utils import get_data_config, load_config
from src.ScriptedVLA.utils.normalization import (
    create_normalizer_from_dataset,
    create_normalizer_from_lerobot_meta,
)
from train import (
    LeRobotDatasetSubset,
    create_collate_fn,
    create_delta_timestamps,
    load_dataset_info,
)

CAMERA_ROWS = [
    ("image", "agentview", "agentview_b64"),
    ("wrist_image", "wrist", "wrist_b64"),
]

ALIGNMENT_WARNINGS = [
    "Dataset frame is episode frame_index=0; sim obs is after WARMUP_STEPS=5 zero actions.",
    "Sim init_id may not match the initial state used when the dataset episode was recorded.",
    "Large raw image/state differences may reflect different physical scenes, not only preprocessing bugs.",
]


def _tensor_to_uint8_hwc(img_tensor: torch.Tensor) -> np.ndarray:
    """LeRobot [C,H,W] float tensor -> uint8 [H,W,C] without resize."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    arr = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _pil_to_uint8_hwc(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _state_to_numpy(state: Any) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _compare_vectors(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
        }
    diff = a - b
    return {
        "shape_match": True,
        "l2": float(np.linalg.norm(diff)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "per_dim_diff": diff.tolist(),
        "a": a.tolist(),
        "b": b.tolist(),
    }


def _image_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    diff = np.abs(a_f - b_f)
    mse = float(np.mean((a_f - b_f) ** 2))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return {
        "mae": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
        "psnr": psnr,
    }


def _format_state_values(state: Optional[np.ndarray], precision: int = 4) -> str:
    if state is None:
        return "N/A"
    return ", ".join(f"{v:.{precision}f}" for v in state.reshape(-1))


def _state_overlay_text(raw_state: Optional[np.ndarray], norm_state: Optional[np.ndarray]) -> str:
    return (
        f"raw: [{_format_state_values(raw_state)}]\n"
        f"norm: [{_format_state_values(norm_state)}]"
    )


def _draw_image_panel(
    ax,
    img: np.ndarray,
    title: str,
    raw_state: Optional[np.ndarray] = None,
    norm_state: Optional[np.ndarray] = None,
) -> None:
    ax.imshow(img)
    h, w = img.shape[:2]
    ax.set_title(f"{title} ({w}x{h})", fontsize=9)
    ax.axis("off")
    if raw_state is not None or norm_state is not None:
        ax.text(
            0.02,
            0.02,
            _state_overlay_text(raw_state, norm_state),
            transform=ax.transAxes,
            fontsize=6,
            va="bottom",
            ha="left",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.72),
            family="monospace",
        )


def _save_compare_grid(
    rows: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
    left_label: str = "Dataset",
    right_label: str = "Sim",
    left_raw_state: Optional[np.ndarray] = None,
    left_norm_state: Optional[np.ndarray] = None,
    right_raw_state: Optional[np.ndarray] = None,
    right_norm_state: Optional[np.ndarray] = None,
) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5.0 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=11)

    for row_idx, (camera_label, ds_img, sim_img) in enumerate(rows):
        _draw_image_panel(
            axes[row_idx, 0],
            ds_img,
            f"{left_label} {camera_label}",
            raw_state=left_raw_state,
            norm_state=left_norm_state,
        )
        _draw_image_panel(
            axes[row_idx, 1],
            sim_img,
            f"{right_label} {camera_label}",
            raw_state=right_raw_state,
            norm_state=right_norm_state,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_diff_grid(
    rows: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
    left_raw_state: Optional[np.ndarray] = None,
    left_norm_state: Optional[np.ndarray] = None,
    right_raw_state: Optional[np.ndarray] = None,
    right_norm_state: Optional[np.ndarray] = None,
) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 5.0 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle(title, fontsize=11)

    for row_idx, (camera_label, ds_img, sim_img) in enumerate(rows):
        diff = np.abs(ds_img.astype(np.float32) - sim_img.astype(np.float32))
        diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
        panels = [
            (ds_img, f"Dataset {camera_label}", left_raw_state, left_norm_state),
            (sim_img, f"Sim {camera_label}", right_raw_state, right_norm_state),
            (diff_u8, f"|diff| {camera_label}", None, None),
        ]
        for col_idx, (img, subtitle, raw_state, norm_state) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if col_idx == 2:
                im = ax.imshow(diff, cmap="hot")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(subtitle, fontsize=9)
                ax.axis("off")
            else:
                _draw_image_panel(ax, img, subtitle, raw_state=raw_state, norm_state=norm_state)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _camera_names_from_image_keys(image_keys: List[str]) -> List[str]:
    return [key.replace("observation.images.", "") for key in image_keys]


def _images_list_to_dict(images_list: List, camera_names: List[str]) -> Dict[str, Image.Image]:
    if not images_list:
        return {}
    first = images_list[0]
    if isinstance(first, list):
        cams = first
    else:
        cams = images_list
    out: Dict[str, Image.Image] = {}
    for name, img in zip(camera_names, cams):
        out[name] = img
    return out


def _build_infer_model_inputs(
    obs_msg: Dict[str, Any],
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    normalizer,
    normalize_state: bool,
    align_joint_angles: bool = True,
    clip_normalized_state: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], np.ndarray, np.ndarray, str, Dict[str, Any]]:
    images_dict, state_raw, instruction = ws_obs_to_model_inputs(
        obs_msg, image_size=image_size, skip_image_decode=False
    )
    images_pil_dict = prepare_images_input(images_dict, device, image_size=image_size)
    camera_names = _camera_names_from_image_keys(image_keys)
    images_for_model = [[images_pil_dict[name] for name in camera_names]]

    state_diag: Dict[str, Any] = {}
    state_aligned = state_raw
    state_raw_np = np.array([], dtype=np.float32)
    states = None
    if state_raw is not None:
        state_raw_np = _state_to_numpy(state_raw)
        if normalizer is not None:
            state_diag = state_normalization_diagnostics(
                normalizer,
                state_raw_np,
                align_joint_angles=align_joint_angles,
                clip=clip_normalized_state,
            )
            state_aligned = np.asarray(state_diag["aligned_raw"], dtype=np.float32)
        states = torch.tensor(state_aligned, dtype=torch.float32, device=device).unsqueeze(0)
        if normalize_state and normalizer is not None:
            states = normalizer.normalize_state(states, clip=clip_normalized_state)

    model_inputs = {
        "images": images_for_model,
        "instructions": [instruction],
        "states": states,
    }
    return (
        model_inputs,
        images_dict,
        state_raw_np if state_raw is not None else np.array([]),
        state_aligned if state_raw is not None else np.array([]),
        instruction,
        state_diag,
    )


def _extract_dataset_raw_action(sample: Dict[str, Any]) -> np.ndarray:
    """First action step from LeRobot sample (matches collate first timestep)."""
    action = sample.get("action")
    if action is None:
        raise KeyError("sample missing 'action'")
    if isinstance(action, torch.Tensor):
        arr = action.detach().cpu().numpy()
    else:
        arr = np.asarray(action, dtype=np.float32)
    if arr.ndim >= 2:
        arr = arr[0]
    return arr.reshape(-1).astype(np.float32)


def build_train_action_views(
    sample: Dict[str, Any],
    config: dict,
    normalizer,
    normalize_action: Optional[bool] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return raw action, collate-normalized first step, and full normalized chunk."""
    data_config = get_data_config(config)
    if normalize_action is None:
        normalize_action = data_config.get("normalize_action", True) if normalizer else False
    raw = _extract_dataset_raw_action(sample)

    dataset_config = config.get("dataset", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    state_key = dataset_config.get("state_key", "observation.state")
    model_config = config.get("model", {})
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    task_description_config = dataset_config.get("task_description", {})
    use_batch_task = task_description_config.get("use_batch_task", True)

    collate = create_collate_fn(
        image_keys=image_keys,
        state_key=state_key,
        image_size=image_size,
        use_batch_task=use_batch_task,
        normalizer=normalizer,
        normalize_action=normalize_action,
        normalize_state=False,
        augmentation_config=None,
    )
    batch = collate([sample])
    norm_chunk = batch["action"]
    if isinstance(norm_chunk, torch.Tensor):
        norm_first = norm_chunk[0, 0].detach().cpu().numpy().reshape(-1)
        norm_chunk_np = norm_chunk[0].detach().cpu().numpy()
    else:
        norm_chunk_np = np.asarray(norm_chunk[0])
        norm_first = norm_chunk_np[0].reshape(-1)

    normalized_first = norm_first if normalize_action else None
    return raw, normalized_first, norm_chunk_np if normalize_action else None


def load_dataset_sample(
    dataset_path: Path,
    config: dict,
    episode_id: int,
    frame_index: int,
) -> Tuple[Dict[str, Any], Any]:
    dataset_config = config.get("dataset", {})
    dataset_info = load_dataset_info(dataset_path)
    fps = dataset_info.get("fps", 10)
    action_horizon = dataset_config.get("action_horizon", 50)
    delta_timestamps = create_delta_timestamps(action_horizon, fps)

    dataset = LeRobotDatasetSubset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=delta_timestamps,
        episodes=[episode_id],
    )
    if frame_index >= len(dataset):
        raise IndexError(
            f"frame_index={frame_index} out of range for episode {episode_id} (len={len(dataset)})"
        )
    return dataset[frame_index], dataset


def build_train_model_inputs(
    sample: Dict[str, Any],
    config: dict,
    normalizer,
    normalize_state: bool,
    normalize_action: Optional[bool] = None,
) -> Tuple[Dict[str, Any], np.ndarray, str]:
    dataset_config = config.get("dataset", {})
    data_config = get_data_config(config)
    model_config = config.get("model", {})
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    state_key = dataset_config.get("state_key", "observation.state")
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    task_description_config = dataset_config.get("task_description", {})
    use_batch_task = task_description_config.get("use_batch_task", True)
    if normalize_action is None:
        normalize_action = data_config.get("normalize_action", True) if normalizer else False
    clip_normalized_state = data_config.get("clip_normalized_state", True) if normalizer else False

    collate = create_collate_fn(
        image_keys=image_keys,
        state_key=state_key,
        image_size=image_size,
        use_batch_task=use_batch_task,
        normalizer=normalizer,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        clip_normalized_state=clip_normalized_state,
        augmentation_config=None,
    )
    batch = collate([sample])

    instruction = batch["text"][0] if batch.get("text") else ""
    raw_state = _state_to_numpy(sample.get(state_key, sample.get("observation.state")))

    model_inputs = {
        "images": batch["images"],
        "instructions": batch["text"],
        "states": batch.get("state"),
    }
    return model_inputs, raw_state, instruction


def extract_dataset_raw_images(sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam_key, _label, _b64_key in CAMERA_ROWS:
        full_key = f"observation.images.{cam_key}"
        if full_key in sample:
            out[cam_key] = _tensor_to_uint8_hwc(sample[full_key])
    return out


def extract_sim_raw_images(obs_msg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cam_key, _label, b64_key in CAMERA_ROWS:
        if b64_key in obs_msg:
            out[cam_key] = decode_b64_image(obs_msg[b64_key], flip_vertical=True)
    return out


async def fetch_sim_obs(
    ws_url: str,
    benchmark_task_id: int,
    init_id: int,
    suite: str,
) -> Dict[str, Any]:
    async with LiberoWSClient(ws_url) as client:
        await client.ping()
        obs_msg = await client.create_episode(
            task_id=benchmark_task_id,
            init_id=init_id,
            suite=suite,
            max_steps=600,
        )
        episode_id = obs_msg.get("episode_id")
        try:
            return obs_msg
        finally:
            if episode_id:
                try:
                    await client.close_episode(episode_id)
                except Exception:
                    pass


def _compare_tensor_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        if key not in a or key not in b:
            result[key] = {"present": False, "in_a": key in a, "in_b": key in b}
            continue
        ta = a[key].detach().cpu().float()
        tb = b[key].detach().cpu().float()
        if ta.shape != tb.shape:
            result[key] = {"shape_match": False, "a_shape": list(ta.shape), "b_shape": list(tb.shape)}
            continue
        diff = (ta - tb).abs()
        result[key] = {
            "shape_match": True,
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose": bool(torch.allclose(ta, tb, atol=1e-5, rtol=1e-4)),
        }
    return result


async def run_compare(args) -> Dict[str, Any]:
    config = load_config(args.config)
    data_config = get_data_config(config)
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})

    dataset_path = Path(args.dataset_path).resolve()
    image_keys = dataset_config.get("image_keys", ["observation.images.image"])
    state_key = dataset_config.get("state_key", "observation.state")
    image_size = model_config.get("vlm", {}).get("image_size", 224)
    camera_names = _camera_names_from_image_keys(image_keys)
    use_normalizer = data_config.get("use_normalizer", True)
    normalize_state = (
        data_config.get("normalize_state", True) if use_normalizer and not args.no_normalize_state else False
    )
    normalize_action = data_config.get("normalize_action", True) if use_normalizer else False
    if args.normalize_action is not None:
        normalize_action = args.normalize_action
    align_joint_angles = data_config.get("align_joint_angles", True) if use_normalizer else False
    clip_normalized_state = data_config.get("clip_normalized_state", True) if use_normalizer else False

    ckpt_path = args.checkpoint
    if ckpt_path is None and args.checkpoint_dir:
        latest = find_latest_checkpoint(Path(args.checkpoint_dir))
        ckpt_path = str(latest) if latest else None

    normalizer = None
    if use_normalizer:
        if ckpt_path:
            _, normalizer = load_model_from_checkpoint(ckpt_path, args.config, device="cpu")
        else:
            dataset_info = load_dataset_info(dataset_path)
            fps = dataset_info.get("fps", 10)
            action_horizon = dataset_config.get("action_horizon", 50)
            delta_timestamps = create_delta_timestamps(action_horizon, fps)
            tmp_ds = LeRobotDatasetSubset(
                repo_id=dataset_path.name,
                root=str(dataset_path),
                delta_timestamps=delta_timestamps,
                episodes=[0],
            )
            try:
                normalizer = create_normalizer_from_lerobot_meta(
                    tmp_ds, state_key=state_key, action_key="action"
                )
            except Exception:
                normalizer = create_normalizer_from_dataset(dataset_path)

    episode_id = resolve_eval_episode_id(
        str(dataset_path),
        task_index=args.task_index,
        eval_episode_id=args.episode_id,
    )
    benchmark_task_id = dataset_task_index_to_benchmark_task_id(str(dataset_path), args.task_index)
    task_text = get_task_description(str(dataset_path), args.task_index)

    sample, _dataset = load_dataset_sample(dataset_path, config, episode_id, args.frame_index)
    train_inputs, train_raw_state, train_instruction = build_train_model_inputs(
        sample, config, normalizer, normalize_state, normalize_action=normalize_action
    )
    gt_raw_action, gt_norm_action, gt_norm_chunk = build_train_action_views(
        sample, config, normalizer, normalize_action=normalize_action
    )

    obs_msg = await fetch_sim_obs(
        args.ws_url, benchmark_task_id, args.init_id, args.suite
    )
    device = torch.device("cpu")
    (
        infer_inputs,
        infer_tensors,
        infer_raw_state,
        infer_aligned_state,
        infer_instruction,
        infer_state_diag,
    ) = _build_infer_model_inputs(
        obs_msg,
        image_keys=image_keys,
        image_size=image_size,
        device=device,
        normalizer=normalizer,
        normalize_state=normalize_state,
        align_joint_angles=align_joint_angles,
        clip_normalized_state=clip_normalized_state,
    )

    ds_raw = extract_dataset_raw_images(sample)
    sim_raw = extract_sim_raw_images(obs_msg)

    train_pil = _images_list_to_dict(train_inputs["images"], camera_names)
    infer_pil = _images_list_to_dict(infer_inputs["images"], camera_names)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []
    model_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []
    image_metrics: Dict[str, Any] = {}

    for cam_key, _label, _b64 in CAMERA_ROWS:
        if cam_key not in ds_raw or cam_key not in sim_raw:
            continue
        raw_rows.append((cam_key, ds_raw[cam_key], sim_raw[cam_key]))
        if cam_key in train_pil and cam_key in infer_pil:
            ds_224 = _pil_to_uint8_hwc(train_pil[cam_key])
            sim_224 = _pil_to_uint8_hwc(infer_pil[cam_key])
            if ds_224.shape != sim_224.shape:
                sim_224 = np.array(
                    infer_pil[cam_key].resize((ds_224.shape[1], ds_224.shape[0]), Image.Resampling.LANCZOS)
                )
            model_rows.append((cam_key, ds_224, sim_224))
            image_metrics[cam_key] = _image_metrics(ds_224, sim_224)

    title_base = (
        f"task_index={args.task_index} episode={episode_id} frame={args.frame_index} "
        f"benchmark_task_id={benchmark_task_id} init_id={args.init_id}\n"
        f"{task_text[:80]}"
    )

    train_norm_state = (
        _state_to_numpy(train_inputs["states"][0])
        if train_inputs.get("states") is not None
        else None
    )
    infer_norm_state = (
        _state_to_numpy(infer_inputs["states"][0])
        if infer_inputs.get("states") is not None
        else None
    )

    raw_png = out_dir / "compare_raw_images.png"
    model_png = out_dir / "compare_model_input_images.png"
    diff_png = out_dir / "compare_model_input_diff.png"

    state_kwargs = dict(
        left_raw_state=train_raw_state,
        left_norm_state=train_norm_state,
        right_raw_state=infer_raw_state,
        right_norm_state=infer_norm_state,
    )

    if raw_rows:
        _save_compare_grid(raw_rows, raw_png, f"Raw images\n{title_base}", **state_kwargs)
        print(f"[compare] saved raw image grid: {raw_png.resolve()}")
    if model_rows:
        _save_compare_grid(
            model_rows, model_png, f"Model input (224) images\n{title_base}", **state_kwargs
        )
        _save_diff_grid(model_rows, diff_png, f"Model input abs diff\n{title_base}", **state_kwargs)
        print(f"[compare] saved model input grid: {model_png.resolve()}")
        print(f"[compare] saved model input diff: {diff_png.resolve()}")

    report: Dict[str, Any] = {
        "timestamp": run_ts,
        "config": args.config,
        "dataset_path": str(dataset_path),
        "ws_url": args.ws_url,
        "checkpoint": ckpt_path,
        "alignment_warnings": ALIGNMENT_WARNINGS,
        "meta": {
            "task_index": args.task_index,
            "task_text": task_text,
            "episode_id": episode_id,
            "frame_index": args.frame_index,
            "benchmark_task_id": benchmark_task_id,
            "init_id": args.init_id,
            "suite": args.suite,
            "image_size": image_size,
            "image_keys": image_keys,
        },
        "instruction": {
            "dataset": train_instruction,
            "sim": infer_instruction,
            "exact_match": train_instruction == infer_instruction,
        },
        "state": {
            "align_joint_angles": align_joint_angles,
            "clip_normalized_state": clip_normalized_state,
            "dataset": {
                "raw": train_raw_state.tolist(),
                "normalized": train_norm_state.tolist() if train_norm_state is not None else None,
            },
            "sim": {
                "raw": infer_raw_state.tolist(),
                "aligned_raw": infer_aligned_state.tolist() if infer_aligned_state.size else None,
                "normalized_unclipped": infer_state_diag.get("normalized_unclipped"),
                "normalized": infer_norm_state.tolist() if infer_norm_state is not None else None,
                "out_of_range_dims_before_clip": infer_state_diag.get(
                    "out_of_range_dims_before_clip", []
                ),
                "clipped_dims": infer_state_diag.get("clipped_dims", []),
            },
            "raw": _compare_vectors(train_raw_state, infer_raw_state),
            "aligned_raw": (
                _compare_vectors(train_raw_state, infer_aligned_state)
                if infer_aligned_state.size
                else None
            ),
            "normalized": (
                _compare_vectors(train_norm_state, infer_norm_state)
                if train_norm_state is not None and infer_norm_state is not None
                else None
            ),
        },
        "action": {
            "dim_labels": action_dim_labels(),
            "normalize_action": normalize_action,
            "dataset": {
                "raw_first_step": gt_raw_action.tolist(),
                "normalized_first_step": (
                    gt_norm_action.tolist() if gt_norm_action is not None else None
                ),
                "normalized_chunk_shape": (
                    list(gt_norm_chunk.shape) if gt_norm_chunk is not None else None
                ),
            },
        },
        "images": image_metrics,
        "model_inputs_summary": {
            "images_structure_match": (
                isinstance(train_inputs["images"], list)
                and isinstance(infer_inputs["images"], list)
                and len(train_inputs["images"]) == len(infer_inputs["images"])
            ),
            "camera_order": camera_names,
        },
        "artifacts": {
            "compare_raw_images": str(raw_png.resolve()) if raw_rows else None,
            "compare_model_input_images": str(model_png.resolve()) if model_rows else None,
            "compare_model_input_diff": str(diff_png.resolve()) if model_rows else None,
        },
    }

    if args.deep and ckpt_path:
        model, _ = load_model_from_checkpoint(ckpt_path, args.config, device="cpu")
        vlm = model.qwen_vl_interface
        use_state_vlm = model.use_state_vlm

        train_qwen = vlm.build_qwenvl_inputs(
            images=train_inputs["images"],
            instructions=train_inputs["instructions"],
            states=train_inputs["states"] if use_state_vlm else None,
        )
        infer_qwen = vlm.build_qwenvl_inputs(
            images=infer_inputs["images"],
            instructions=infer_inputs["instructions"],
            states=infer_inputs["states"] if use_state_vlm else None,
        )
        deep_keys = ["input_ids", "attention_mask", "pixel_values"]
        report["deep_vlm_inputs"] = {}
        for key in deep_keys:
            if key in train_qwen and key in infer_qwen:
                report["deep_vlm_inputs"][key] = _compare_tensor_dict(
                    {key: train_qwen[key]}, {key: infer_qwen[key]}
                )[key]

        train_actions_norm = model.predict_action(
            {
                "images": train_inputs["images"],
                "instructions": train_inputs["instructions"],
                "states": train_inputs["states"],
            }
        )["normalized_actions"]
        infer_actions_norm = model.predict_action(
            {
                "images": infer_inputs["images"],
                "instructions": infer_inputs["instructions"],
                "states": infer_inputs["states"],
            }
        )["normalized_actions"]
        report["predict_action_check"] = _compare_vectors(
            train_actions_norm.reshape(-1), infer_actions_norm.reshape(-1)
        )
        report["predict_action_check"]["train_actions_first3"] = train_actions_norm[0, :3].tolist()
        report["predict_action_check"]["infer_actions_first3"] = infer_actions_norm[0, :3].tolist()

        def _to_numpy1d(x) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().reshape(-1)
            return np.asarray(x, dtype=np.float32).reshape(-1)

        predict_norm_first = _to_numpy1d(train_actions_norm[0, 0])
        camera_tensors = {}
        for key in image_keys:
            if key in sample:
                cam = key.replace("observation.images.", "")
                camera_tensors[cam] = sample[key]
        states_tensor = None
        if train_raw_state is not None:
            states_tensor = torch.tensor(train_raw_state, dtype=torch.float32).unsqueeze(0)
        predict_denorm_chunk = run_inference(
            model=model,
            images=camera_tensors,
            instruction=train_instruction,
            image_keys=image_keys,
            states=states_tensor,
            normalizer=normalizer,
            image_size=image_size,
            normalize_action=normalize_action,
            normalize_state=normalize_state,
            align_joint_angles=align_joint_angles,
            clip_normalized_state=clip_normalized_state,
        )
        predict_denorm_first = np.asarray(predict_denorm_chunk[0]).reshape(-1)

        gt_vs_predict: Dict[str, Any] = {
            "raw_l2": float(np.linalg.norm(predict_denorm_first - gt_raw_action)),
            "denorm_l2": float(np.linalg.norm(predict_denorm_first - gt_raw_action)),
            "per_dim_diff": (predict_denorm_first - gt_raw_action).tolist(),
        }
        if gt_norm_action is not None:
            gt_vs_predict["normalized_l2"] = float(
                np.linalg.norm(predict_norm_first - gt_norm_action)
            )
            gt_vs_predict["normalized_per_dim_diff"] = (
                predict_norm_first - gt_norm_action
            ).tolist()

        report["action"]["model"] = {
            "predict_normalized_first_step": predict_norm_first.tolist(),
            "predict_denormalized_first_step": predict_denorm_first.tolist(),
            "predict_denormalized_chunk_shape": list(predict_denorm_chunk.shape),
        }
        report["action"]["gt_vs_predict"] = gt_vs_predict

    summary_path = out_dir / "report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[compare] report saved: {summary_path.resolve()}")
    print(f"[compare] instruction match={report['instruction']['exact_match']}")
    if report["state"]["raw"].get("shape_match"):
        print(
            f"[compare] raw state L2={report['state']['raw']['l2']:.6f} "
            f"max_abs={report['state']['raw']['max_abs']:.6f}"
        )
    sim_state = report["state"].get("sim", {})
    if sim_state.get("out_of_range_dims_before_clip") is not None:
        print(
            f"[compare] sim state OOD dims before clip: "
            f"{sim_state.get('out_of_range_dims_before_clip')} "
            f"clipped_dims={sim_state.get('clipped_dims')}"
        )
    if sim_state.get("normalized"):
        norm_vals = sim_state["normalized"]
        oob = [i for i, v in enumerate(norm_vals) if v < -1.0 - 1e-6 or v > 1.0 + 1e-6]
        print(f"[compare] sim normalized in [-1,1]: {len(oob) == 0} (oob_dims={oob})")
    for cam, metrics in image_metrics.items():
        print(
            f"[compare] {cam}@224 MAE={metrics['mae']:.4f} "
            f"max={metrics['max_abs']:.4f} PSNR={metrics['psnr']:.2f}"
        )
    action_report = report.get("action", {})
    if action_report.get("gt_vs_predict"):
        gvp = action_report["gt_vs_predict"]
        print(
            f"[compare] action GT vs predict denorm L2={gvp['denorm_l2']:.6f} "
            f"raw_l2={gvp['raw_l2']:.6f}"
        )
        if "normalized_l2" in gvp:
            print(f"[compare] action GT vs predict normalized L2={gvp['normalized_l2']:.6f}")
    for warn in ALIGNMENT_WARNINGS:
        print(f"[compare][WARN] {warn}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare dataset vs sim obs at VLA model input boundary"
    )
    parser.add_argument("--config", default="libero/config_libero_object.yaml")
    parser.add_argument("--dataset-path", default="./dada/libero-object")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--init-id", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-dir", default="./results/obs_compare")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--no-normalize-state", action="store_true")
    parser.add_argument(
        "--normalize-action",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=None,
        help="Override config data.normalize_action (e.g. true for legacy checkpoints)",
    )
    args = parser.parse_args()

    asyncio.run(run_compare(args))


if __name__ == "__main__":
    main()
