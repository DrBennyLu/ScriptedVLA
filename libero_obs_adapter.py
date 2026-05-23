"""Convert LIBERO WebSocket observations to ScriptedVLA model inputs."""

from __future__ import annotations

import base64
import io
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def decode_b64_image(b64_str: str, flip_vertical: bool = True) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if flip_vertical:
        arr = arr[::-1].copy()
    return arr


def b64_to_tensor(b64_str: str, image_size: Optional[int] = None) -> torch.Tensor:
    arr = decode_b64_image(b64_str)
    img = Image.fromarray(arr)
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return tensor


def ws_obs_to_model_inputs(
    obs_msg: Dict,
    image_size: Optional[int] = None,
    skip_image_decode: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[np.ndarray], str]:
    """
    Map server obs payload to model image dict, state array, instruction.

    Server keys: agentview_b64 -> image, wrist_b64 -> wrist_image
    """
    instruction = obs_msg.get("instruction", "")
    state = None
    if "state" in obs_msg and obs_msg["state"]:
        state = np.asarray(obs_msg["state"], dtype=np.float32)

    images: Dict[str, torch.Tensor] = {}
    if not skip_image_decode:
        if "agentview_b64" in obs_msg:
            images["image"] = b64_to_tensor(obs_msg["agentview_b64"], image_size=image_size)
        if "wrist_b64" in obs_msg:
            images["wrist_image"] = b64_to_tensor(
                obs_msg["wrist_b64"], image_size=image_size
            )
    return images, state, instruction
