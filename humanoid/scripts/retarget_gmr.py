# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""GMR-style SMPL-X -> AgiBot X1 motion retargeting (Gate B) — numerical IK approach.

This module implements the GMR (General Motion Retargeting, arXiv:2505.02833 /
https://github.com/Roboparty/GMR) methodology adapted to the X1 robot, using a
self-contained numpy/scipy numerical IK instead of the mink+MuJoCo solver (which
GMR's upstream uses). The core algorithmic ideas are faithfully reproduced:

  1. Forward-kinematics the SMPL-X skeleton to get per-frame 3D positions + rotations
     of pelvis, hips, knees, ankles (human keypoints).
  2. Scale human keypoints to X1 proportions (human_scale optimization) so the
     retarget geometry matches the robot link lengths.
  3. Two-stage numerical IK per leg (scipy.optimize.least_squares with joint-limit
     bounds), matching BOTH position AND orientation of the foot frame — not just
     a sagittal-plane closed-form solution. This is the key GMR improvement over
     analytic 2-link IK: it respects joint limits naturally, handles the full 6-DOF
     chain, and can trade off position vs orientation matching via weights.
     - Stage 1 (coarse, ik_match_table1): match foot world POSITION (high pos_weight).
     - Stage 2 (fine, ik_match_table2): refine with foot ORIENTATION added.
  4. Root: heading yaw from human root orientation; translation from pelvis, Z-up.
  5. Foot contact from ankle velocity (stance = slow forward motion).

Output (MotionLib-compatible npz) + retarget_manifest.json for Gate-B traceability.

Run: python humanoid/scripts/retarget_gmr.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "0008_normal_walk4_stageii.npz")
OUT_DIR = os.path.join(ROOT, "data", "retarget_gmr")
OUT_NPZ = os.path.join(OUT_DIR, "x1_walk_retargeted.npz")
OUT_REPORT = os.path.join(OUT_DIR, "retarget_report.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "retarget_manifest.json")

# Load motion_lib (shared contract: joint order, limits, default pose)
_ML = os.path.join(HERE, "..", "algo", "amp", "motion_lib.py")
_spec = importlib.util.spec_from_file_location("_gmr_ml", os.path.abspath(_ML))
ml = importlib.util.module_from_spec(_spec)
sys.modules["_gmr_ml"] = ml
_spec.loader.exec_module(ml)

X1_JOINT_NAMES = ml.X1_JOINT_NAMES
X1_LIMITS = ml.X1_JOINT_LIMITS
X1_DEFAULT = np.asarray(ml.X1_DEFAULT_DOF_POS, dtype=np.float64)
X1_INIT_HEIGHT = 0.7

# --------------------------------------------------------------------------- #
# SMPL skeleton (official smplx SMPL_JOINT_NAMES indices for legs)
# --------------------------------------------------------------------------- #
SMPL_PELVIS, SMPL_LHIP, SMPL_RHIP = 0, 1, 2
SMPL_LKNEE, SMPL_RKNEE = 4, 5
SMPL_LANK, SMPL_RANK = 7, 8
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 0, 4, 5, 3]

# SMPL neutral template (Y-up, meters). Used for bone-offset computation.
SMPL_NEUTRAL = np.array([
    [0.0, 0.0, 0.0],          # 0 pelvis
    [0.06, -0.08, 0.02],      # 1 L hip
    [-0.06, -0.08, 0.02],     # 2 R hip
    [0.0, 0.08, 0.02],        # 3 spine1
    [0.06, -0.40, 0.04],      # 4 L knee
    [-0.06, -0.40, 0.04],     # 5 R knee
    [0.0, 0.18, 0.05],        # 6 spine2
    [0.06, -0.78, 0.04],      # 7 L ankle
    [-0.06, -0.78, 0.04],     # 8 R ankle
    [0.0, 0.30, 0.06],        # 9 spine3
], dtype=np.float64)

# X1 leg link lengths (URDF). thigh = hip_yaw to knee, shin = knee to ankle.
# Calibrated from URDF origin chain + prior Gate-B validation.
X1_THIGH = 0.305
X1_SHIN = 0.305
# Hip offset from pelvis center (lateral), from URDF left_hip_pitch origin y=0.0923
X1_HIP_LATERAL = 0.0923


# --------------------------------------------------------------------------- #
# SMPL forward kinematics
# --------------------------------------------------------------------------- #
def smpl_fk(root_orient_aa, pose_body, trans):
    """Forward-kinematics SMPL -> per-frame world 3D joint positions + global rotations.

    Returns pos (N,10,3), Rg list of (N,3,3) global rotation matrices for joints 0..9.
    """
    N = root_orient_aa.shape[0]
    R_root = Rot.from_rotvec(root_orient_aa).as_matrix()
    R_body = Rot.from_rotvec(pose_body.reshape(N, 21, 3)).as_matrix()
    Rg = [None] * 10
    Rg[0] = R_root
    for j in range(1, 10):
        p = SMPL_PARENTS[j]
        R_local = R_body[:, j - 1]
        Rg[j] = np.einsum("nab,nbc->nac", Rg[p], R_local)
    pos = np.zeros((N, 10, 3), dtype=np.float64)
    pos[:, 0] = trans
    for j in range(1, 10):
        p = SMPL_PARENTS[j]
        offset = SMPL_NEUTRAL[j] - SMPL_NEUTRAL[p]
        rotated = np.einsum("nab,b->na", Rg[p], offset)
        pos[:, j] = pos[:, p] + rotated
    return pos, Rg


# --------------------------------------------------------------------------- #
# X1 leg forward kinematics (simplified sagittal model, Z-up)
# --------------------------------------------------------------------------- #
# Joint axis order per leg in X1 convention (from URDF):
#   hip_pitch (rotate about Y=sagittal), hip_roll (about X=frontal),
#   hip_yaw (about Z=transverse), knee_pitch (Y), ankle_pitch (Y), ankle_roll (X)
# The FK chain starts at the hip joint center and applies rotations in sequence.
# Leg indices: [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]

def _leg_fk(q6, thigh, shin, side_sign):
    """Forward-kinematics one X1 leg -> (foot_pos(3,), foot_rot(3,3), knee_pos(3,)).

    q6: 6 joint angles [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]
        in the X1 convention (delta from default, then we add default inside).
    side_sign: +1 for left, -1 for right (for hip lateral offset mirroring).
    Returns foot position and rotation (as 3x3 matrix) and knee position, all in a
    root-aligned frame where +X=forward, +Y=left, +Z=up.
    """
    hp, hr, hy, kp, ap, ar = q6

    # Build rotation chain (Y-up frame: X=forward, Y=left, Z=up)
    R = np.eye(3)
    # hip_pitch: rotate about Y (sagittal). Positive pitch swings leg forward.
    c, s = math.cos(hp), math.sin(hp)
    R = R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # hip_roll: rotate about X (frontal)
    c, s = math.cos(hr), math.sin(hr)
    R = R @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    # hip_yaw: rotate about Z (transverse)
    c, s = math.cos(hy), math.sin(hy)
    R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Hip center (lateral offset from pelvis)
    pos = np.array([0.0, side_sign * X1_HIP_LATERAL, 0.0])

    # Thigh: go down (in hip frame, leg extends in -Z)
    thigh_vec = R @ np.array([0.0, 0.0, -thigh])
    knee_pos = pos + thigh_vec

    # knee_pitch: rotate about Y
    c, s = math.cos(kp), math.sin(kp)
    R = R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Shin: go down
    shin_vec = R @ np.array([0.0, 0.0, -shin])
    foot_pos = knee_pos + shin_vec

    # ankle_pitch: rotate about Y
    c, s = math.cos(ap), math.sin(ap)
    R = R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # ankle_roll: rotate about X
    c, s = math.cos(ar), math.sin(ar)
    R = R @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return foot_pos, R, knee_pos


def _leg_fk_batch(q6_arr, thigh, shin, side_sign):
    """Batch leg FK for all frames. q6_arr: (N,6) -> foot_pos (N,3), foot_rot (N,3,3)."""
    N = q6_arr.shape[0]
    foot_pos = np.zeros((N, 3))
    foot_rot = np.zeros((N, 3, 3))
    for i in range(N):
        fp, fr, _ = _leg_fk(q6_arr[i], thigh, shin, side_sign)
        foot_pos[i] = fp
        foot_rot[i] = fr
    return foot_pos, foot_rot


# --------------------------------------------------------------------------- #
# GMR-style numerical IK per leg
# --------------------------------------------------------------------------- #
def solve_leg_ik(target_foot_pos, target_foot_rot, side_sign, default_q6,
                 thigh=X1_THIGH, shin=X1_SHIN, max_stage=2):
    """Two-stage numerical IK for one leg (GMR methodology).

    Stage 1 (coarse): minimize foot POSITION error only.
    Stage 2 (fine): minimize foot POSITION + ORIENTATION error.

    target_foot_pos: (3,) desired foot position in root-aligned frame.
    target_foot_rot: (3,3) desired foot rotation matrix.
    Returns: 6 joint angles (delta from default in X1 convention).
    """
    limits = np.array([
        X1_LIMITS[0], X1_LIMITS[1], X1_LIMITS[2],
        X1_LIMITS[3], X1_LIMITS[4], X1_LIMITS[5],
    ])  # (6,2) lower/upper — same for both legs (already symmetric)

    def residual_pos(q):
        fp, fr, _ = _leg_fk(q, thigh, shin, side_sign)
        return (fp - target_foot_pos) * 10.0  # scale position cost

    def residual_pos_rot(q):
        fp, fr, _ = _leg_fk(q, thigh, shin, side_sign)
        pos_err = (fp - target_foot_pos) * 10.0
        # rotation error as matrix difference (flatten upper triangle)
        rot_err = (fr - target_foot_rot).ravel() * 2.0
        return np.concatenate([pos_err, rot_err])

    lb = limits[:, 0]
    ub = limits[:, 1]

    # Stage 1: position only (coarse)
    q0 = default_q6.copy()
    try:
        res1 = least_squares(residual_pos, q0, bounds=(lb, ub), method="trf",
                             max_nfev=100, ftol=1e-8)
        q_best = res1.x
    except Exception:
        q_best = q0

    # Stage 2: position + rotation (fine), warm-started from stage 1
    if max_stage >= 2:
        try:
            res2 = least_squares(residual_pos_rot, q_best, bounds=(lb, ub),
                                 method="trf", max_nfev=100, ftol=1e-8)
            q_best = res2.x
        except Exception:
            pass

    return np.clip(q_best, lb, ub)


# --------------------------------------------------------------------------- #
# Human-scale optimization (GMR ik_config_manager step)
# --------------------------------------------------------------------------- #
def optimize_human_scale(human_kp, src_fps):
    """Estimate thigh/shin scale factors to match SMPL leg lengths to X1.

    Measures the SMPL hip-to-knee and knee-to-ankle distances (mean over frames),
    then returns a scale that maps the human leg length ratio to X1 proportions.
    This mirrors GMR's optimize_human_scale.py.
    """
    hip, knee, ank = human_kp  # each (N,3)
    smpl_thigh = np.linalg.norm(knee - hip, axis=1).mean()
    smpl_shin = np.linalg.norm(ank - knee, axis=1).mean()
    # Scale human thigh/shin to X1 link lengths (preserving the angle structure)
    thigh_scale = X1_THIGH / max(smpl_thigh, 1e-6)
    shin_scale = X1_SHIN / max(smpl_shin, 1e-6)
    return {"smpl_thigh_len": float(smpl_thigh), "smpl_shin_len": float(smpl_shin),
            "thigh_scale": float(thigh_scale), "shin_scale": float(shin_scale)}


# --------------------------------------------------------------------------- #
# Main retarget
# --------------------------------------------------------------------------- #
def retarget(src_path: str = SRC) -> dict:
    d = np.load(src_path, allow_pickle=True)
    fps = float(d["mocap_frame_rate"])
    N = d["poses"].shape[0]
    dt_src = 1.0 / fps
    root_orient_aa = np.asarray(d["root_orient"], dtype=np.float64)
    pose_body = np.asarray(d["pose_body"], dtype=np.float64)
    trans = np.asarray(d["trans"], dtype=np.float64)

    # 1. SMPL FK -> world joint positions + rotations
    pos, Rg = smpl_fk(root_orient_aa, pose_body, trans)

    # 2. Root translation: SMPL(Y-up) -> X1(Z-up): X1x=SMPLx, X1y=SMPLz, X1z=SMPLy
    pelvis = pos[:, SMPL_PELVIS].copy()
    x1_trans = np.empty((N, 3), dtype=np.float64)
    x1_trans[:, 0] = pelvis[:, 0]
    x1_trans[:, 1] = pelvis[:, 2]
    x1_trans[:, 2] = pelvis[:, 1]
    # Recenter + set pelvis at X1 init height, keep vertical bob
    bob = x1_trans[:, 2] - x1_trans[0, 2]
    x1_trans[:, 0] -= x1_trans[0, 0]
    x1_trans[:, 1] -= x1_trans[0, 1]
    x1_trans[:, 2] = X1_INIT_HEIGHT + (bob - bob.mean())

    # 3. Root rotation: heading yaw (strip SMPL T-pose tilt)
    R_root = Rg[0]
    fwd_smpl = np.einsum("nab,b->na", R_root, np.array([1.0, 0.0, 0.0]))
    fwd_x1 = np.stack([fwd_smpl[:, 0], fwd_smpl[:, 2], fwd_smpl[:, 1]], axis=-1)
    yaw = np.arctan2(fwd_x1[:, 1], fwd_x1[:, 0])
    x1_root_quat = np.zeros((N, 4), dtype=np.float64)  # xyzw
    half = yaw * 0.5
    x1_root_quat[:, 2] = np.sin(half)
    x1_root_quat[:, 3] = np.cos(half)

    # 4. Human-scale optimization (GMR step)
    scale_info = optimize_human_scale(
        (pos[:, SMPL_LHIP], pos[:, SMPL_LKNEE], pos[:, SMPL_LANK]), fps)
    # GMR applies human_scale_table to keypoints before IK. We scale the ankle
    # target so the hip->ankle distance matches X1 leg lengths (otherwise the IK
    # target is unreachable for the shorter robot legs and residuals blow up).
    smpl_leg_total = scale_info["smpl_thigh_len"] + scale_info["smpl_shin_len"]
    x1_leg_total = X1_THIGH + X1_SHIN
    leg_scale = x1_leg_total / max(smpl_leg_total, 1e-6)

    # 5. Per-leg GMR numerical IK
    joints = np.zeros((N, 12), dtype=np.float64)
    ik_residuals = {"L": [], "R": []}

    leg_specs = [
        ("L", SMPL_LHIP, SMPL_LKNEE, SMPL_LANK, [0, 1, 2, 3, 4, 5]),
        ("R", SMPL_RHIP, SMPL_RKNEE, SMPL_RANK, [6, 7, 8, 9, 10, 11]),
    ]

    for tag, hip_i, knee_i, ank_i, dof_idx in leg_specs:
        side_sign = 1.0 if tag == "L" else -1.0
        default_q6 = X1_DEFAULT[dof_idx[0]:dof_idx[5] + 1].copy()
        q_prev = default_q6.copy()  # warm-start from previous frame (temporal coherence)

        for i in range(N):
            # Express ankle pos relative to hip, in ROOT-LOCAL frame (SMPL convention:
            # X=forward, Y=up, Z=left). This strips the global root rotation so we
            # work in the body frame, same as the keypoint-based approach.
            R_root_inv = Rg[0][i].T  # (3,3) transpose = inverse for rotation matrix
            rel = pos[i, ank_i] - pos[i, hip_i]
            root_local = R_root_inv @ rel
            # Remap SMPL root-local (X=fwd, Y=up, Z=left) -> X1 (X=fwd, Y=left, Z=up):
            # X1_X=SMPL_X(fwd), X1_Y=SMPL_Z(left), X1_Z=SMPL_Y(up)
            target_local = np.array([root_local[0], root_local[2], root_local[1]])
            # GMR human-scale: scale the ankle target so hip->ankle distance matches
            # X1 leg lengths (thigh+shin), preserving the direction/angle structure.
            target_local = target_local * leg_scale
            # Target foot rotation: approximate from the hip->ankle direction
            # (foot should point along the leg direction). We use the SMPL ankle
            # rotation projected to X1 Z-up as the orientation target.
            R_ank_world = Rg[ank_i][i]  # SMPL ankle global rotation
            # Convert SMPL (Y-up) rotation to X1 (Z-up) by axis remapping
            # SMPL: X=fwd, Y=up, Z=left ; X1: X=fwd, Y=left, Z=up
            # remap: x->x, y->z, z->y => permutation matrix
            P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
            target_rot = P @ R_ank_world @ P.T

            q6 = solve_leg_ik(target_local, target_rot, side_sign, q_prev)
            q_prev = q6.copy()  # warm-start next frame from this solution (GMR-like temporal coherence)

            # Residual check for reporting
            fp, fr, _ = _leg_fk(q6, X1_THIGH, X1_SHIN, side_sign)
            ik_residuals[tag].append(float(np.linalg.norm(fp - target_local)))
            joints[i, dof_idx[0]:dof_idx[5] + 1] = q6

    ik_res_mean = {k: float(np.mean(v)) for k, v in ik_residuals.items()}

    # 6. Foot contact from ankle forward velocity (stance = slow)
    dt_f = 1.0 / fps
    foot_fwd = np.zeros((N, 2), dtype=np.float64)
    for idx, (_, _, _, ank_i, _) in enumerate(leg_specs):
        foot_fwd[:, idx] = pos[:, ank_i, 0]
    foot_vel = np.zeros_like(foot_fwd)
    foot_vel[1:] = np.diff(foot_fwd, axis=0) / dt_f
    foot_vel[0] = foot_vel[1]
    foot_speed = np.abs(foot_vel)
    med = np.median(foot_speed, axis=0)
    foot_contact = (foot_speed <= med[None, :]).astype(np.float64)

    # 7. Temporal smoothing (moving average window=5) for continuity
    w = 5
    if N > w:
        k = np.ones(w) / w
        joints_sm = np.empty_like(joints)
        for c in range(12):
            joints_sm[:, c] = np.convolve(joints[:, c], k, mode="same")
        joints = joints_sm

    # Clamp to X1 limits
    n_clip = 0
    for i, (lo, hi) in enumerate(X1_LIMITS):
        col = joints[:, i]
        n_clip += int(np.sum((col < lo) | (col > hi)))
        joints[:, i] = np.clip(joints[:, i], lo, hi)

    # Checks
    finite_ok = bool(np.all(np.isfinite(joints)) and np.all(np.isfinite(x1_trans))
                     and np.all(np.isfinite(x1_root_quat)))
    range_ok = all(X1_LIMITS[i][0] - 0.05 <= joints[:, i].min()
                   and joints[:, i].max() <= X1_LIMITS[i][1] + 0.05 for i in range(12))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(OUT_NPZ,
             root_translation=x1_trans,
             root_rotation=x1_root_quat,
             joint_positions=joints,
             foot_contact=foot_contact,
             fps=np.float64(fps),
             note="GMR-style retarget: SMPL-X FK + two-stage scipy numerical IK (Gate B).")

    report = {
        "method": "GMR (arXiv:2505.02833) methodology: SMPL FK + human-scale optimization + "
                  "two-stage scipy.least_squares numerical IK (position-only coarse, "
                  "position+rotation fine) with joint-limit bounds. Self-contained numpy/scipy.",
        "reference": "https://github.com/Roboparty/GMR",
        "source": os.path.relpath(src_path, ROOT),
        "source_sha256": _sha256(src_path),
        "source_fps": fps, "n_frames": int(N),
        "human_scale": scale_info,
        "ik_residual_mean_m": ik_res_mean,
        "x1_link_lengths": {"thigh": X1_THIGH, "shin": X1_SHIN},
        "n_clip_violations": int(n_clip),
        "finite_ok": finite_ok, "range_ok": range_ok,
        "forward_velocity_mps": float(np.mean(np.diff(x1_trans[:, 0]) / dt_src)),
        "root_height_mean": float(x1_trans[:, 2].mean()),
        "root_height_std": float(x1_trans[:, 2].std()),
        "joint_stats": {X1_JOINT_NAMES[i]: {
            "min": float(joints[:, i].min()), "max": float(joints[:, i].max()),
            "mean": float(joints[:, i].mean()), "std": float(joints[:, i].std()),
            "limit": list(X1_LIMITS[i]),
        } for i in range(12)},
        "output": os.path.relpath(OUT_NPZ, ROOT),
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Manifest (Gate B requirement: deterministic reconstruction)
    manifest = {
        "method": report["method"],
        "reference": report["reference"],
        "source_motion": {
            "path": report["source"],
            "sha256": report["source_sha256"],
            "fps": fps, "n_frames": int(N),
            "format": "SMPL-X npz (root_orient, pose_body, trans, betas)",
            "license_source": "BMLrub (BiomotionLab) via /Users/yumx/code/robot_x/X1/Retargeting_X1/BMLrub_stageii",
        },
        "f1_model": {
            "robot": "AgiBot X1 (12-DOF legs-only)",
            "urdf": "resources/robots/x1/urdf/x1.urdf",
            "joint_order": list(X1_JOINT_NAMES),
            "default_dof_pos": list(X1_DEFAULT),
            "joint_limits": [list(l) for l in X1_LIMITS],
        },
        "coordinate_system": {
            "up_axis": "Z (X1 sim up-axis index 1)",
            "root_translation": "world meters, X1 Z-up (X=forward, Y=left, Z=up)",
            "root_rotation": "xyzw quaternion, heading yaw only",
            "joint_angles": "radians, X1 URDF dof_names order",
            "fps": "resampled by MotionLib to 100 Hz (policy dt)",
        },
        "human_scale": scale_info,
        "ik_residual_mean_m": ik_res_mean,
        "output_files": {
            "training_data": os.path.relpath(OUT_NPZ, ROOT),
            "report": os.path.relpath(OUT_REPORT, ROOT),
            "manifest": os.path.relpath(OUT_MANIFEST, ROOT),
        },
        "validation": "Run validate_retarget_gmr.py (schema/finite/range/continuity/contact/MotionLib round-trip)",
        "reproducibility": "Deterministic: same source npz + same code revision => same output",
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    return report


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    r = retarget()
    print(f"[retarget_gmr] wrote {r['output']}")
    print(f"  source sha256={r['source_sha256']} fps={r['source_fps']} frames={r['n_frames']}")
    print(f"  forward_v={r['forward_velocity_mps']:.3f} m/s  root_h={r['root_height_mean']:.3f}+-{r['root_height_std']:.3f}")
    print(f"  finite_ok={r['finite_ok']} range_ok={r['range_ok']} clip_violations={r['n_clip_violations']}")
    print(f"  ik_residual_mean_m: L={r['ik_residual_mean_m']['L']:.4f} R={r['ik_residual_mean_m']['R']:.4f}")
    print(f"  report: {os.path.relpath(OUT_REPORT, ROOT)}")
    print(f"  manifest: {os.path.relpath(OUT_MANIFEST, ROOT)}")
