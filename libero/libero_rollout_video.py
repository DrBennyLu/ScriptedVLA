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


def rollout_q_curve_paths(video_path: Path) -> tuple[Path, Path]:
    """PNG/CSV paths alongside rollout MP4 (same stem + _q_curve)."""
    stem = video_path.with_suffix("")
    return stem.with_name(stem.name + "_q_curve.png"), stem.with_name(stem.name + "_q_curve.csv")


def save_rollout_q_curve(
    video_path: Path,
    q_history: List[dict],
    *,
    success: bool,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Plot per-env-step critic Q (q1, q2, min) and write CSV next to rollout video.

    Each entry in q_history: {"step", "q1", "q2", "q_min"}.
    """
    if not q_history:
        return None, None

    png_path, csv_path = rollout_q_curve_paths(video_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(row["step"]) for row in q_history]
    q1_vals = [float(row["q1"]) for row in q_history]
    q2_vals = [float(row["q2"]) for row in q_history]
    q_min_vals = [float(row["q_min"]) for row in q_history]

    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "q1", "q2", "q_min"])
        writer.writeheader()
        for row in q_history:
            writer.writerow(
                {
                    "step": int(row["step"]),
                    "q1": float(row["q1"]),
                    "q2": float(row["q2"]),
                    "q_min": float(row["q_min"]),
                }
            )

    try:
        import matplotlib.pyplot as plt

        status = "success" if success else "fail"
        plt.figure(figsize=(10, 5))
        plt.plot(steps, q1_vals, label="Q1", alpha=0.9)
        plt.plot(steps, q2_vals, label="Q2", alpha=0.9)
        plt.plot(steps, q_min_vals, label="min(Q1,Q2)", linewidth=2.0)
        plt.xlabel("Environment step")
        plt.ylabel("Critic Q (chunk action)")
        plt.title(f"TD3 critic Q per step ({status})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[warn] failed to save Q curve PNG {png_path}: {exc}")
        png_path = None

    return png_path, csv_path
