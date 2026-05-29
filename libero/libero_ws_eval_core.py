"""Shared LIBERO WebSocket eval episode loop."""

from __future__ import annotations

from typing import Optional

import numpy as np

from inference import run_inference
from .libero_action_adapter import model_action_to_libero
from .libero_obs_adapter import ws_obs_to_model_inputs
from .libero_rollout_video import RolloutVideoRecorder
from .libero_ws_client import LiberoWSClient


async def run_eval_episode(
    client: LiberoWSClient,
    model,
    normalizer,
    image_keys,
    image_size: int,
    normalize_action: bool,
    normalize_state: bool,
    task_id: int,
    init_id: int,
    max_steps: int,
    chunk_steps: int,
    debug_ranges: bool,
    video_recorder: Optional[RolloutVideoRecorder] = None,
    rollout_label: str = "",
    align_joint_angles: bool = True,
    clip_normalized_state: bool = True,
) -> dict:
    label = rollout_label or f"task_id={task_id} init_id={init_id}"
    created = await client.create_episode(
        task_id=task_id, init_id=init_id, max_steps=max_steps
    )
    episode_id = created["episode_id"]
    instruction = created.get("instruction", "")
    task_name = created.get("task_name", "")
    print(f"[eval] {label} episode={episode_id} instruction={instruction!r}")

    step_count = 0
    done = False
    success = False
    action_buffer = []
    buffer_idx = 0

    if video_recorder is not None:
        video_recorder.reset()
        video_recorder.append_obs(created)

    try:
        obs_msg = created
        while not done and step_count < max_steps:
            if buffer_idx >= len(action_buffer):
                images, state, instr = ws_obs_to_model_inputs(
                    obs_msg, image_size=image_size, skip_image_decode=False
                )
                if debug_ranges and state is not None:
                    print(
                        f"  [debug] state min={state.min():.4f} max={state.max():.4f}"
                    )

                action_chunk = run_inference(
                    model=model,
                    images=images,
                    instruction=instruction or instr,
                    image_keys=image_keys,
                    states=state,
                    normalizer=normalizer,
                    image_size=image_size,
                    normalize_action=normalize_action,
                    normalize_state=normalize_state,
                    align_joint_angles=align_joint_angles,
                    clip_normalized_state=clip_normalized_state,
                )
                action_chunk = np.asarray(action_chunk, dtype=np.float32)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk.reshape(1, -1)

                if debug_ranges:
                    print(
                        f"  [debug] action_chunk min={action_chunk.min():.4f} "
                        f"max={action_chunk.max():.4f} shape={action_chunk.shape}"
                    )

                action_buffer = [
                    model_action_to_libero(action_chunk[i])
                    for i in range(len(action_chunk))
                ]
                buffer_idx = 0

            steps_this_round = min(chunk_steps, len(action_buffer) - buffer_idx)
            for _ in range(steps_this_round):
                action = action_buffer[buffer_idx]
                buffer_idx += 1
                obs_msg = await client.step(episode_id, action, include_images=True)
                if video_recorder is not None:
                    video_recorder.append_obs(obs_msg)
                step_count += 1
                done = bool(obs_msg.get("done"))
                success = bool(obs_msg.get("success"))
                if done or step_count >= max_steps:
                    break
    finally:
        closed = await client.close_episode(episode_id)

    final_success = success or closed.get("success", False)
    return {
        "task_id": task_id,
        "init_id": init_id,
        "task_name": task_name,
        "instruction": instruction,
        "steps": step_count,
        "success": final_success,
        "num_video_frames": len(video_recorder.frames) if video_recorder else 0,
    }
