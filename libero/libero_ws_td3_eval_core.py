"""LIBERO WebSocket eval episode loop with VLA + RL token + TD3 actor."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inference import prepare_images_input
from .libero_action_adapter import model_action_to_libero
from .libero_obs_adapter import ws_obs_to_model_inputs
from .libero_rollout_video import RolloutVideoRecorder
from .libero_ws_client import LiberoWSClient


def ws_obs_to_vla_model_inputs(
    obs_msg: dict,
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    instruction: str = "",
) -> Tuple[dict, Optional[np.ndarray]]:
    """
    Build VLA inputs from a WebSocket obs payload (aligned with rl_td3_replay.build_single_input).

    Returns:
        (inputs dict for extract_vla_tokens / predict_action, raw state ndarray or None)
    """
    images_dict, state_raw, instr = ws_obs_to_model_inputs(
        obs_msg, image_size=image_size, skip_image_decode=False
    )
    images_pil = prepare_images_input(images_dict, device, image_size=image_size)

    if len(image_keys) == 1:
        images_input = [list(images_pil.values())[0]]
    else:
        camera_names = []
        for key in image_keys:
            camera_name = key.replace("observation.images.", "")
            if camera_name in images_pil:
                camera_names.append(camera_name)
        images_input = [[images_pil[name] for name in camera_names]]

    states_tensor = None
    if state_raw is not None:
        states_tensor = torch.tensor(
            np.asarray(state_raw, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        if states_tensor.dim() == 1:
            states_tensor = states_tensor.unsqueeze(0)

    inputs = {
        "images": images_input,
        "instructions": [instruction or instr],
        "states": states_tensor,
    }
    return inputs, state_raw


@torch.no_grad()
def predict_td3_action_chunk(
    vla_model,
    rl_encoder,
    td3_agent,
    inputs: dict,
    state_raw: Optional[np.ndarray],
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run VLA + RL token + TD3 to produce an action chunk [T, action_dim]."""
    z_tokens = vla_model.extract_vla_tokens(inputs)
    z_rl = rl_encoder.encode(z_tokens).float()

    pred = vla_model.predict_action(inputs)
    ref_actions = pred["normalized_actions"]
    if isinstance(ref_actions, np.ndarray):
        ref_tensor = torch.as_tensor(ref_actions, dtype=torch.float32, device=device)
    else:
        ref_tensor = ref_actions.to(device=device, dtype=torch.float32)
    if ref_tensor.dim() == 2:
        ref_tensor = ref_tensor.unsqueeze(0)
    ref_chunk = ref_tensor[:, :chunk_size, :]

    if state_raw is None:
        state_tensor = torch.zeros(
            (1, td3_agent.actor.state_dim), dtype=torch.float32, device=device
        )
    else:
        state_tensor = torch.as_tensor(
            np.asarray(state_raw, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=device,
        )

    action_chunk = td3_agent.act(
        z_rl,
        state_tensor,
        ref_chunk,
        deterministic=True,
        apply_ref_mask=False,
    )
    actions = action_chunk.squeeze(0).detach().cpu().numpy()
    return np.asarray(actions, dtype=np.float32)


async def run_td3_eval_episode(
    client: LiberoWSClient,
    vla_model,
    rl_encoder,
    td3_agent,
    image_keys: List[str],
    image_size: int,
    chunk_size: int,
    task_id: int,
    init_id: int,
    max_steps: int,
    chunk_steps: int,
    debug_ranges: bool,
    video_recorder: Optional[RolloutVideoRecorder] = None,
    rollout_label: str = "",
) -> dict:
    device = td3_agent.device
    label = rollout_label or f"task_id={task_id} init_id={init_id}"
    created = await client.create_episode(
        task_id=task_id, init_id=init_id, max_steps=max_steps
    )
    episode_id = created["episode_id"]
    instruction = created.get("instruction", "")
    task_name = created.get("task_name", "")
    print(f"[eval_td3] {label} episode={episode_id} instruction={instruction!r}")

    step_count = 0
    done = False
    success = False
    action_buffer: List[list] = []
    buffer_idx = 0

    if video_recorder is not None:
        video_recorder.reset()
        video_recorder.append_obs(created)

    try:
        obs_msg = created
        while not done and step_count < max_steps:
            if buffer_idx >= len(action_buffer):
                inputs, state_raw = ws_obs_to_vla_model_inputs(
                    obs_msg,
                    image_keys=image_keys,
                    image_size=image_size,
                    device=device,
                    instruction=instruction,
                )
                if debug_ranges and state_raw is not None:
                    print(
                        f"  [debug] state min={state_raw.min():.4f} max={state_raw.max():.4f}"
                    )

                action_chunk = predict_td3_action_chunk(
                    vla_model=vla_model,
                    rl_encoder=rl_encoder,
                    td3_agent=td3_agent,
                    inputs=inputs,
                    state_raw=state_raw,
                    chunk_size=chunk_size,
                    device=device,
                )
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
