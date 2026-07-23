# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Generate a SYNTHETIC expert motion clip in X1 joint space for Gate A smoke validation.

This is NOT the real retargeted expert (that arrives at Gate B from
``data/0008_normal_walk4_stageii.npz`` retargeting). It only exists so the AMP
discriminator plumbing + expert/policy schema match can be exercised without the
retargeted dataset. It is produced entirely in X1 joint order and unit system so
that ``MotionLib`` can load it and the loader-vs-builder stats stay consistent.

Output: data/smoke_expert/walk_smoke.npz
  root_translation (N,3) world, meters
  root_rotation    (N,4) xyzw quaternion
  joint_positions  (N,12) X1 joint order (URDF dof_names)
  foot_contact     (N,2) {0,1}
  fps              scalar Hz (120 -> MotionLib resamples to the 100 Hz policy rate)

Run:  python humanoid/scripts/gen_smoke_expert.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

# Load motion_lib by file path so this script stays headless (no isaacgym/torch).
# NOTE: the module must be registered in sys.modules *before* exec, because
# motion_lib uses @dataclass and Python's dataclass machinery resolves
# cls.__module__ through sys.modules.
_ML = os.path.join(os.path.dirname(__file__), "..", "algo", "amp", "motion_lib.py")
_spec = importlib.util.spec_from_file_location("_smoke_motion_lib", os.path.abspath(_ML))
ml = importlib.util.module_from_spec(_spec)
sys.modules["_smoke_motion_lib"] = ml
_spec.loader.exec_module(ml)

DEFAULT = np.asarray(ml.X1_DEFAULT_DOF_POS, dtype=np.float64)   # (12,)
LIMITS = ml.X1_JOINT_LIMITS                              # ((lo,hi),)*12
JOINT_NAMES = ml.X1_JOINT_NAMES

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "smoke_expert"))
OUT_PATH = os.path.join(OUT_DIR, "walk_smoke.npz")


def _gait_joint_trajectory(n: int, fps: float) -> np.ndarray:
    """Procedural sagittal walk: left/right legs in antiphase, within X1 limits.

    Indices (X1 order):
      0,6 hip_pitch | 1,7 hip_roll | 2,8 hip_yaw | 3,9 knee_pitch
      4,10 ankle_pitch | 5,11 ankle_roll   (left block 0..5, right block 6..11)
    """
    t = np.arange(n) / fps                      # seconds
    cycle = 1.0                                  # 1.0 s gait cycle (== cfg rewards.cycle_time)
    phase = 2.0 * np.pi * t / cycle
    left, right = np.cos(phase), np.cos(phase + np.pi)   # antiphase

    j = np.tile(DEFAULT, (n, 1)).astype(np.float64)
    # swing amplitudes chosen to stay strictly inside manual_joint_clip windows
    j[:, 0] += 0.25 * left                       # left hip_pitch
    j[:, 6] += 0.25 * right                      # right hip_pitch
    j[:, 3] += 0.35 * np.clip(-left, 0, None)    # left knee flexes on swing
    j[:, 9] += 0.35 * np.clip(-right, 0, None)   # right knee flexes on swing
    j[:, 4] += 0.12 * np.clip(left, 0, None)     # left ankle_pitch
    j[:, 10] += 0.12 * np.clip(right, 0, None)   # right ankle_pitch
    j[:, 1] += 0.06 * left                       # left hip_roll
    j[:, 7] += 0.06 * right                      # right hip_roll
    j[:, 5] += 0.04 * left                       # left ankle_roll
    j[:, 11] += 0.04 * right                     # right ankle_roll
    # clamp to the conservative locomotion windows (defensive; should already be inside)
    for i, (lo, hi) in enumerate(LIMITS):
        j[:, i] = np.clip(j[:, i], lo, hi)
    return j


def _root_trajectory(n: int, fps: float, forward_v: float = 0.4, height: float = 0.61):
    t = np.arange(n) / fps
    root_t = np.zeros((n, 3), dtype=np.float64)
    root_t[:, 0] = forward_v * t                 # +X forward
    root_t[:, 2] = height + 0.01 * np.sin(2 * np.pi * t / 1.0)   # small vertical bob
    # identity orientation with a tiny forward-facing yaw jitter (xyzw)
    root_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)).astype(np.float64)
    return root_t, root_q


def _foot_contact(n: int, fps: float) -> np.ndarray:
    """Alternating stance: left in contact when cos(phase)>0, right antiphase."""
    t = np.arange(n) / fps
    phase = 2.0 * np.pi * t / 1.0
    left = (np.cos(phase) > -0.2).astype(np.float64)
    right = (np.cos(phase + np.pi) > -0.2).astype(np.float64)
    return np.stack([left, right], axis=1)


def main(out_path: str = OUT_PATH, fps: float = 120.0, seconds: float = 3.0) -> str:
    n = int(round(fps * seconds))
    root_t, root_q = _root_trajectory(n, fps)
    joints = _gait_joint_trajectory(n, fps)
    contact = _foot_contact(n, fps)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             root_translation=root_t,
             root_rotation=root_q,
             joint_positions=joints,
             foot_contact=contact,
             fps=np.float64(fps),
             note=("SYNTHETIC X1-space walk for Gate A smoke validation only; "
                   "NOT the retargeted expert (Gate B)."))

    # in-file range check (must match MotionLib's expectation)
    ok = True
    for i, (lo, hi) in enumerate(LIMITS):
        cmin, cmax = float(joints[:, i].min()), float(joints[:, i].max())
        if cmin < lo - 0.05 or cmax > hi + 0.05:
            ok = False
            print(f"  WARN joint {JOINT_NAMES[i]} out of [{lo},{hi}]: [{cmin:.3f},{cmax:.3f}]")
    print(f"[gen_smoke_expert] wrote {out_path}")
    print(f"  frames={n} fps={fps} duration={seconds:.3f}s joints_range_ok={ok}")
    print(f"  root_x: {root_t[0,0]:.3f} -> {root_t[-1,0]:.3f}  "
          f"forward_v~{(root_t[-1,0]-root_t[0,0])/(n/fps):.3f} m/s")
    return out_path


if __name__ == "__main__":
    main()
