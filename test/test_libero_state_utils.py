"""Tests for LIBERO state alignment and normalization diagnostics."""

import numpy as np

from libero.libero_state_utils import align_joint_angles_to_stats, state_normalization_diagnostics
from src.ScriptedVLA.utils import create_normalizer_from_dataset
from pathlib import Path


def _full_stats_vectors():
    smin = np.zeros(8, dtype=np.float64)
    smax = np.ones(8, dtype=np.float64)
    smin[3] = 2.289
    smax[3] = 3.4
    smin[2] = 0.008
    smax[2] = 0.386
    smin[5] = -1.06
    smax[5] = 0.664
    return smin, smax


def test_align_joint_prefers_2pi_shift_toward_stats():
    smin, smax = _full_stats_vectors()
    raw = np.zeros(8, dtype=np.float32)
    raw[3] = -2.45
    aligned = align_joint_angles_to_stats(raw, smin, smax)
    mid = 0.5 * (smin[3] + smax[3])
    assert abs(float(aligned[3]) - mid) < abs(float(raw[3]) - mid)
    assert float(aligned[3]) > 3.0


def test_align_joint2_stays_when_no_better_2pi_candidate():
    smin = np.zeros(8, dtype=np.float64)
    smax = np.ones(8, dtype=np.float64)
    smin[2] = 0.008
    smax[2] = 0.386
    raw = np.zeros(8, dtype=np.float32)
    raw[2] = -0.018
    aligned = align_joint_angles_to_stats(raw, smin, smax)
    assert float(aligned[2]) == float(raw[2])


def test_gripper_dims_unchanged():
    smin = np.zeros(8)
    smax = np.ones(8)
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02, -0.02], dtype=np.float32)
    aligned = align_joint_angles_to_stats(raw, smin, smax)
    assert aligned[6] == raw[6]
    assert aligned[7] == raw[7]


def test_sim_example_normalized_within_unit_after_clip():
    dataset = Path("./dada/libero-object")
    if not dataset.exists():
        return
    norm = create_normalizer_from_dataset(dataset)
    sim = np.array(
        [
            -0.014220706187188625,
            -0.16911499202251434,
            -0.017590703442692757,
            -2.4538979530334473,
            -0.010858085937798023,
            2.221376657485962,
            0.020674293860793114,
            -0.020668666809797287,
        ],
        dtype=np.float32,
    )
    diag = state_normalization_diagnostics(norm, sim, align_joint_angles=True, clip=True)
    clipped = np.asarray(diag["normalized"], dtype=np.float64)
    assert clipped.min() >= -1.0 - 1e-6
    assert clipped.max() <= 1.0 + 1e-6
