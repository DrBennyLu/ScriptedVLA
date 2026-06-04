"""VLA WebSocket rollout with per-step frame recording for replay buffer construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from inference import run_inference

from .libero_action_adapter import model_action_to_libero
from .libero_obs_adapter import ws_obs_to_model_inputs
from .libero_rollout_video import RolloutVideoRecorder
from .libero_ws_client import LiberoWSClient
from .rl_td3_replay import WSRolloutFrame


@dataclass(frozen=True)
class VlaWsInferenceSettings:
    """与 libero_ws_eval 一致的 WebSocket VLA 推理预处理。"""

    normalizer: Optional[object]
    normalize_action: bool
    normalize_state: bool
    align_joint_angles: bool
    clip_normalized_state: bool


def vla_ws_inference_settings_from_config(
    config: dict, normalizer: Optional[object]
) -> VlaWsInferenceSettings:
    from src.ScriptedVLA.utils import get_data_config

    data_config = get_data_config(config)
    use_normalizer = bool(data_config.get("use_normalizer", True))
    norm = normalizer if use_normalizer else None
    return VlaWsInferenceSettings(
        normalizer=norm,
        normalize_action=bool(data_config.get("normalize_action", True)) if use_normalizer else False,
        normalize_state=bool(data_config.get("normalize_state", True)) if use_normalizer else False,
        align_joint_angles=bool(data_config.get("align_joint_angles", True)) if use_normalizer else False,
        clip_normalized_state=bool(data_config.get("clip_normalized_state", True)) if use_normalizer else False,
    )


def ws_obs_to_normalized_vla_inputs(
    obs_msg: dict,
    *,
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    instruction: str,
    ws_infer: VlaWsInferenceSettings,
) -> dict:
    """构建 extract_vla_tokens / predict_action 用的输入（state 已按 eval 路径归一化）。"""
    from .libero_obs_input_compare import _build_infer_model_inputs

    model_inputs, *_rest = _build_infer_model_inputs(
        obs_msg,
        image_keys=image_keys,
        image_size=image_size,
        device=device,
        normalizer=ws_infer.normalizer,
        normalize_state=ws_infer.normalize_state,
        align_joint_angles=ws_infer.align_joint_angles,
        clip_normalized_state=ws_infer.clip_normalized_state,
    )
    if instruction:
        model_inputs["instructions"] = [instruction]
    return model_inputs


@torch.no_grad()
def _predict_vla_action_chunk(
    vla_model,
    obs_msg: dict,
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    instruction: str,
    chunk_size: int,
    ws_infer: VlaWsInferenceSettings,
) -> np.ndarray:
    images, state, instr = ws_obs_to_model_inputs(
        obs_msg, image_size=image_size, skip_image_decode=False
    )
    action_chunk = run_inference(
        model=vla_model,
        images=images,
        instruction=instruction or instr,
        image_keys=image_keys,
        states=state,
        normalizer=ws_infer.normalizer,
        image_size=image_size,
        normalize_action=ws_infer.normalize_action,
        normalize_state=ws_infer.normalize_state,
        align_joint_angles=ws_infer.align_joint_angles,
        clip_normalized_state=ws_infer.clip_normalized_state,
    )
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    return chunk[:chunk_size].copy()


async def run_vla_collect_episode(
    client: LiberoWSClient,
    vla_model,
    image_keys: List[str],
    image_size: int,
    chunk_size: int,
    task_id: int,
    init_id: int,
    max_steps: int,
    chunk_steps: int,
    device: torch.device,
    ws_infer: VlaWsInferenceSettings,
    video_recorder: Optional[RolloutVideoRecorder] = None,
    rollout_label: str = "",
) -> Tuple[dict, List[WSRolloutFrame]]:
    """
    Run pure VLA closed-loop episode and record per-step frames for replay conversion.

    Returns:
        (result dict, frames) — frames empty when episode failed.
    """
    label = rollout_label or f"task_id={task_id} init_id={init_id}"
    created = await client.create_episode(task_id=task_id, init_id=init_id, max_steps=max_steps)
    episode_id = created["episode_id"]
    instruction = created.get("instruction", "")
    task_name = created.get("task_name", "")
    print(f"[vla_collect] {label} episode={episode_id} instruction={instruction!r}")

    step_count = 0
    done = False
    success = False
    action_buffer: List[list] = []
    buffer_idx = 0
    current_ref_chunk: Optional[np.ndarray] = None
    frames: List[WSRolloutFrame] = []

    if video_recorder is not None:
        video_recorder.reset()
        video_recorder.append_obs(created)

    try:
        obs_msg = created
        while not done and step_count < max_steps:
            if buffer_idx >= len(action_buffer):
                ref_chunk = _predict_vla_action_chunk(
                    vla_model,
                    obs_msg,
                    image_keys,
                    image_size,
                    device,
                    instruction,
                    chunk_size,
                    ws_infer,
                )
                if ref_chunk.ndim == 1:
                    ref_chunk = ref_chunk.reshape(1, -1)
                current_ref_chunk = ref_chunk.copy()
                action_buffer = [
                    model_action_to_libero(ref_chunk[i])
                    for i in range(len(ref_chunk))
                ]
                buffer_idx = 0

            steps_this_round = min(chunk_steps, len(action_buffer) - buffer_idx)
            for _ in range(steps_this_round):
                action_list = action_buffer[buffer_idx]
                action_arr = np.asarray(action_list, dtype=np.float32)
                assert current_ref_chunk is not None
                frames.append(
                    WSRolloutFrame(
                        local_index=step_count,
                        obs_msg=copy.deepcopy(obs_msg),
                        action=action_arr,
                        ref_chunk=current_ref_chunk.copy(),
                    )
                )
                buffer_idx += 1
                obs_msg = await client.step(episode_id, action_list, include_images=True)
                if video_recorder is not None:
                    video_recorder.append_obs(obs_msg)
                step_count += 1
                done = bool(obs_msg.get("done"))
                success = bool(obs_msg.get("success"))
                if done or step_count >= max_steps:
                    break
    finally:
        closed = await client.close_episode(episode_id)

    final_success = success or bool(closed.get("success", False))
    result = {
        "task_id": task_id,
        "init_id": init_id,
        "task_name": task_name,
        "instruction": instruction,
        "steps": step_count,
        "success": final_success,
        "num_video_frames": len(video_recorder.frames) if video_recorder else 0,
    }
    if not final_success:
        frames = []
    return result, frames
