"""Save LIBERO WebSocket rollout frames as MP4 videos."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import imageio
import numpy as np

from .libero_obs_adapter import decode_b64_image

CameraMode = Literal["agentview", "wrist", "both"]


class RolloutVideoRecorder:
    """Accumulate decoded RGB frames from server obs messages and write MP4."""

    def __init__(self, fps: int = 20, camera: CameraMode = "agentview"):
        self.fps = fps
        self.camera = camera
        self.frames: List[np.ndarray] = []

    def append_obs(self, obs_msg: dict) -> None:
        if self.camera == "both":
            left = right = None
            if "agentview_b64" in obs_msg:
                left = decode_b64_image(obs_msg["agentview_b64"], flip_vertical=True)
            if "wrist_b64" in obs_msg:
                right = decode_b64_image(obs_msg["wrist_b64"], flip_vertical=True)
            if left is None and right is None:
                return
            if left is None:
                self.frames.append(right)
                return
            if right is None:
                self.frames.append(left)
                return
            if left.shape[0] != right.shape[0]:
                target_h = max(left.shape[0], right.shape[0])
                left = _resize_height(left, target_h)
                right = _resize_height(right, target_h)
            self.frames.append(np.concatenate([left, right], axis=1))
            return

        key = "agentview_b64" if self.camera == "agentview" else "wrist_b64"
        if key not in obs_msg:
            return
        self.frames.append(decode_b64_image(obs_msg[key], flip_vertical=True))

    def save(self, output_path: str | Path) -> Optional[Path]:
        if not self.frames:
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8)
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return path

    def reset(self) -> None:
        self.frames = []


def _resize_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / h
    new_w = max(1, int(w * scale))
    from PIL import Image

    pil = Image.fromarray(image)
    pil = pil.resize((new_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(pil)


def rollout_video_path(
    video_dir: Path,
    task_id: int,
    rollout_index: int,
    init_id: int,
    success: bool,
    task_name: str = "",
) -> Path:
    status = "success" if success else "fail"
    safe_name = task_name.replace(" ", "_")[:40] if task_name else "task"
    filename = (
        f"task{task_id:02d}_{safe_name}_init{init_id:03d}_"
        f"rollout{rollout_index:03d}_{status}.mp4"
    )
    return video_dir / filename
