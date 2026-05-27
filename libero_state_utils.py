"""LIBERO observation.state layout helpers (LeRobot libero-object convention)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

# LeRobot libero-object: observation.state is 8-D =
#   robot0_joint_pos[:6] (first 6 arm joints) + robot0_gripper_qpos[:2] (both fingers).
# The 7th arm joint is omitted in the dataset; do not use mean(gripper) here.
STATE_DIM = 8
JOINT_COUNT = 6
GRIPPER_COUNT = 2
TWO_PI = 2.0 * np.pi


def extract_state_from_robosuite_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Build observation.state matching LeRobot libero-object parquet / episodes_stats.

    Args:
        obs: robosuite/LIBERO observation dict.

    Returns:
        float32 vector of shape [8].
    """
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if joint.size < JOINT_COUNT:
        raise ValueError(f"Expected at least {JOINT_COUNT} joint values, got {joint.size}")
    if gripper.size < GRIPPER_COUNT:
        raise ValueError(f"Expected at least {GRIPPER_COUNT} gripper qpos values, got {gripper.size}")
    state = np.concatenate([joint[:JOINT_COUNT], gripper[:GRIPPER_COUNT]], axis=0)
    return state.astype(np.float32, copy=False)


def align_joint_angles_to_stats(
    state: Union[np.ndarray, Iterable[float]],
    state_min: Union[np.ndarray, Iterable[float]],
    state_max: Union[np.ndarray, Iterable[float]],
    joint_dims: Iterable[int] | None = None,
) -> np.ndarray:
    """
    Shift joint angles by k*2pi so they best match training stats per dimension.

    For each joint dim, prefer a candidate in [state_min, state_max]; otherwise pick
    the candidate closest to the interval midpoint. Gripper dims are unchanged.
    """
    out = np.asarray(state, dtype=np.float64).reshape(-1).copy()
    smin = np.asarray(state_min, dtype=np.float64).reshape(-1)
    smax = np.asarray(state_max, dtype=np.float64).reshape(-1)
    dims = list(joint_dims) if joint_dims is not None else list(range(JOINT_COUNT))

    for i in dims:
        lo, hi = float(smin[i]), float(smax[i])
        mid = 0.5 * (lo + hi)
        in_range_val: Optional[float] = None
        best_out = float(out[i])
        best_dist = float("inf")
        for k in range(-2, 3):
            cand = float(out[i]) + k * TWO_PI
            if lo <= cand <= hi:
                in_range_val = cand
                break
            dist = abs(cand - mid)
            if dist < best_dist:
                best_dist = dist
                best_out = cand
        out[i] = in_range_val if in_range_val is not None else best_out

    return out.astype(np.float32, copy=False)


def prepare_raw_state_for_inference(
    state: Union[np.ndarray, Iterable[float]],
    normalizer: Any,
    align_joint_angles: bool = True,
) -> np.ndarray:
    """Align sim raw state before normalize_state (inference / eval only)."""
    raw = np.asarray(state, dtype=np.float32).reshape(-1)
    if (
        not align_joint_angles
        or normalizer is None
        or normalizer.state_min is None
        or normalizer.state_max is None
    ):
        return raw
    return align_joint_angles_to_stats(raw, normalizer.state_min, normalizer.state_max)


def state_normalization_diagnostics(
    normalizer: Any,
    raw_state: Union[np.ndarray, Iterable[float]],
    *,
    align_joint_angles: bool = True,
    clip: bool = True,
) -> Dict[str, Any]:
    """Report raw -> aligned -> normalized (unclipped/clipped) for one state vector."""
    raw = np.asarray(raw_state, dtype=np.float64).reshape(-1)
    aligned = prepare_raw_state_for_inference(raw, normalizer, align_joint_angles=align_joint_angles)
    aligned64 = np.asarray(aligned, dtype=np.float64)

    if normalizer is None or normalizer.state_min is None:
        return {
            "raw": raw.tolist(),
            "aligned_raw": aligned.tolist(),
            "normalized_unclipped": aligned.tolist(),
            "normalized": aligned.tolist(),
            "out_of_range_dims_before_clip": [],
            "clipped_dims": [],
        }

    smin = np.asarray(normalizer.state_min, dtype=np.float64).reshape(-1)
    srange = np.asarray(normalizer.state_range, dtype=np.float64).reshape(-1)
    unclipped = 2.0 * (aligned64 - smin) / srange - 1.0
    clipped = np.clip(unclipped, -1.0, 1.0) if clip else unclipped

    oob = [int(i) for i, v in enumerate(unclipped) if v < -1.0 - 1e-6 or v > 1.0 + 1e-6]
    clipped_dims = [int(i) for i in oob if clip and abs(clipped[i] - unclipped[i]) > 1e-6]

    return {
        "raw": raw.tolist(),
        "aligned_raw": aligned.tolist(),
        "normalized_unclipped": unclipped.tolist(),
        "normalized": clipped.tolist(),
        "out_of_range_dims_before_clip": oob,
        "clipped_dims": clipped_dims,
    }


def state_dim_labels() -> list[str]:
    return [f"joint_{i}" for i in range(JOINT_COUNT)] + [
        "gripper_finger_0",
        "gripper_finger_1",
    ]
