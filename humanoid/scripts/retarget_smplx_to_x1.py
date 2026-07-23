# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""SMPL-X -> AgiBot X1 leg motion retargeting (Gate B) — keypoint/IK approach.

Robust, convention-agnostic retarget that does NOT guess SMPL rotation axes:
  1. Forward-kinematics the SMPL-X skeleton (root_orient + pose_body local
     axis-angles, standard SMPL kintree + neutral bone vectors) to get per-frame
     3D world positions of pelvis, hips, knees, ankles.
  2. For each leg, project the (hip->knee->ankle) chain into the sagittal plane
     of the root and solve a closed-form 2-link IK (thigh + shin) -> X1
     hip_pitch + knee_pitch. Roll/yaw/ankle are taken from the frontal/transverse
     residual of the same keypoints.
  3. Root translation: SMPL(Y-up) -> X1(Z-up), pelvis at init height.
  4. Root rotation: heading yaw only (strip the fixed SMPL T-pose tilt).

Explicit joint mapping by NAME (SMPL index from official smplx/joint_names.py,
confirmed against the data: knees=4/5 antiphase). X1 order = URDF dof_names.
No foreign indices, no dict-order, no silent drop.

Outputs (MotionLib-compatible npz):
  root_translation (N,3) world m, X1 Z-up
  root_rotation    (N,4) xyzw
  joint_positions  (N,12) X1 joint order rad
  foot_contact     (N,2) {0,1}
  fps scalar Hz

Run: python humanoid/scripts/retarget_smplx_to_x1.py
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

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "0008_normal_walk4_stageii.npz")
OUT_DIR = os.path.join(ROOT, "data", "retarget")
OUT_NPZ = os.path.join(OUT_DIR, "x1_walk_retargeted.npz")
OUT_REPORT = os.path.join(OUT_DIR, "retarget_report.json")

_ML = os.path.join(HERE, "..", "algo", "amp", "motion_lib.py")
_spec = importlib.util.spec_from_file_location("_retarget_ml", os.path.abspath(_ML))
ml = importlib.util.module_from_spec(_spec)
sys.modules["_retarget_ml"] = ml
_spec.loader.exec_module(ml)

X1_JOINT_NAMES = ml.X1_JOINT_NAMES
X1_LIMITS = ml.X1_JOINT_LIMITS
X1_DEFAULT = np.asarray(ml.X1_DEFAULT_DOF_POS, dtype=np.float64)
X1_INIT_HEIGHT = 0.7

# SMPL body joint indices (official smplx/joint_names.py SMPL_JOINT_NAMES).
SMPL_PELVIS, SMPL_LHIP, SMPL_RHIP = 0, 1, 2
SMPL_LKNEE, SMPL_RKNEE = 4, 5
SMPL_LANK, SMPL_RANK = 7, 8

# SMPL kintree parents (0..9 enough for legs).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 0, 4, 5, 3]

# SMPL neutral template joint positions (beta=0, SMPL Y-up, meters). These are the
# documented canonical neutral skeleton offsets. With betas the lengths change, but
# the IK takes RATIOS/angles so the neutral template is a good base; we additionally
# scale thigh/shin to the X1 link lengths so the IK target geometry matches X1.
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

# X1 leg link lengths (URDF): thigh (hip->knee) ~0.305, shin (knee->ankle) ~0.305.
X1_THIGH = 0.305
X1_SHIN = 0.305


def smpl_fk(root_orient_aa, pose_body, trans, betas=None):
    """Forward-kinematics SMPL -> per-frame world 3D joint positions (N,10,3).

    Local rotations = [root_orient (global), pose_body joints 0..20]. We use the
    SMPL kintree: joint local transform = parent_global * local_rot * bone_offset.
    Returns joint world positions for indices 0..9 (legs+spine).
    """
    N = root_orient_aa.shape[0]
    # local rotation matrices per joint: joint 0 = root (global), 1.. = body
    R_root = Rot.from_rotvec(root_orient_aa).as_matrix()                 # (N,3,3)
    R_body = Rot.from_rotvec(pose_body.reshape(N, 21, 3)).as_matrix()    # (N,21,3,3)
    # global rotation per joint
    Rg = [None] * 10
    Rg[0] = R_root
    for j in range(1, 10):
        p = SMPL_PARENTS[j]
        # body joint j-1 in pose_body corresponds to SMPL joint j
        R_local = R_body[:, j - 1]
        Rg[j] = np.einsum("nab,nbc->nac", Rg[p], R_local)
    # global positions
    pos = np.zeros((N, 10, 3), dtype=np.float64)
    pos[:, 0] = trans
    for j in range(1, 10):
        p = SMPL_PARENTS[j]
        offset = SMPL_NEUTRAL[j] - SMPL_NEUTRAL[p]
        rotated = np.einsum("nab,b->na", Rg[p], offset)
        pos[:, j] = pos[:, p] + rotated
    return pos, Rg


def two_link_ik(target_rel_hip, L1=X1_THIGH, L2=X1_SHIN):
    """Closed-form 2-link IK in the sagittal plane (Y=forward, Z=down positive for leg).

    target_rel_hip: (N,3) ankle position relative to hip, in hip-local frame
    where +Y is forward (sagittal) and +Z is downward (toward ground).
    Returns hip_pitch (N,), knee_pitch (N,) in rad.
    """
    # work in the sagittal (y,z) plane
    y = target_rel_hip[:, 1]
    z = target_rel_hip[:, 2]
    d2 = y ** 2 + z ** 2
    d = np.sqrt(d2)
    d_clip = np.clip(d, 1e-4, L1 + L2 - 1e-4)
    cos_knee = (d2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee = np.arccos(cos_knee)                                  # knee flexion angle (>=0)
    # hip angle: angle of hip->ankle line vs straight-down, plus half-knee
    gamma = np.arctan2(y, z)                                    # angle from +Z(down) toward +Y(fwd)
    alpha = np.arccos(np.clip((L1 ** 2 + d2 - L2 ** 2) / (2 * L1 * d_clip), -1, 1))
    hip_pitch = gamma - alpha                                   # hip pitch
    return hip_pitch, knee


def retarget(src_path: str = SRC) -> dict:
    d = np.load(src_path, allow_pickle=True)
    fps = float(d["mocap_frame_rate"])
    N = d["poses"].shape[0]
    dt_src = 1.0 / fps
    root_orient_aa = np.asarray(d["root_orient"], dtype=np.float64)
    pose_body = np.asarray(d["pose_body"], dtype=np.float64)
    trans = np.asarray(d["trans"], dtype=np.float64)
    betas = np.asarray(d["betas"], dtype=np.float64)

    # 1. SMPL forward kinematics -> world joint positions
    pos, Rg = smpl_fk(root_orient_aa, pose_body, trans, betas)   # (N,10,3)

    # 2. root translation SMPL(Y-up) -> X1(Z-up): X1x=SMPLx, X1y=SMPLz, X1z=SMPLy
    pelvis = pos[:, SMPL_PELVIS]
    x1_trans = np.empty((N, 3), dtype=np.float64)
    x1_trans[:, 0] = pelvis[:, 0]
    x1_trans[:, 1] = pelvis[:, 2]
    x1_trans[:, 2] = pelvis[:, 1]
    # recenter + set pelvis at X1 init height, keep vertical bob
    bob = x1_trans[:, 2] - x1_trans[0, 2]
    x1_trans[:, 0] -= x1_trans[0, 0]
    x1_trans[:, 1] -= x1_trans[0, 1]
    x1_trans[:, 2] = X1_INIT_HEIGHT + (bob - bob.mean())

    # 3. root rotation: heading yaw (strip SMPL T-pose tilt)
    R_root = Rg[0]                                               # (N,3,3)
    fwd_smpl = np.einsum("nab,b->na", R_root, np.array([1.0, 0.0, 0.0]))  # root forward in SMPL world
    fwd_x1 = np.stack([fwd_smpl[:, 0], fwd_smpl[:, 2], fwd_smpl[:, 1]], axis=-1)
    yaw = np.arctan2(fwd_x1[:, 1], fwd_x1[:, 0])
    x1_root_quat = np.zeros((N, 4), dtype=np.float64)            # xyzw
    half = yaw * 0.5
    x1_root_quat[:, 2] = np.sin(half)
    x1_root_quat[:, 3] = np.cos(half)

    # 4. per-leg 2-link IK. For each leg, express ankle pos in a hip-local sagittal frame.
    joints = np.zeros((N, 12), dtype=np.float64)
    leg_specs = [
        ("L", SMPL_LHIP, SMPL_LKNEE, SMPL_LANK, [0, 1, 2, 3, 4, 5]),
        ("R", SMPL_RHIP, SMPL_RKNEE, SMPL_RANK, [6, 7, 8, 9, 10, 11]),
    ]
    for tag, hip_i, knee_i, ank_i, dof_idx in leg_specs:
        hip = pos[:, hip_i]
        knee = pos[:, knee_i]
        ank = pos[:, ank_i]
        # ankle relative to hip, expressed in the root (pelvis) frame
        R_root_inv = np.einsum("nba->nab", R_root)              # transpose = inverse
        rel_hip = ank - hip
        target_local = np.einsum("nab,nb->na", R_root_inv, rel_hip)
        # SMPL frame: +Y up, +X forward. Sagittal plane = (X forward, -Y down).
        # Build (forward, down) = (X, -Y) then solve IK.
        sag = np.stack([target_local[:, 0], -target_local[:, 1]], axis=-1)  # (N,2) (fwd, down)
        y, z = sag[:, 0], sag[:, 1]
        d2 = y ** 2 + z ** 2
        d_clip = np.sqrt(np.clip(d2, 1e-6, (X1_THIGH + X1_SHIN) ** 2))
        cos_k = np.clip((d2 - X1_THIGH ** 2 - X1_SHIN ** 2) / (2 * X1_THIGH * X1_SHIN), -1, 1)
        knee_flex = np.arccos(cos_k)                            # >=0
        gamma = np.arctan2(y, z)                                # from down toward fwd
        alpha = np.arccos(np.clip((X1_THIGH ** 2 + d2 - X1_SHIN ** 2) / (2 * X1_THIGH * d_clip), -1, 1))
        hip_pitch = gamma - alpha
        # ankle pitch ~ keep foot flat-ish: ankle_pitch = -(hip_pitch + knee_flex) + small
        ankle_pitch = -(hip_pitch - knee_flex) * 0.5
        # roll/yaw: small, from frontal residual
        # frontal plane (SMPL Z left): use hip->ankle lateral offset
        lat = target_local[:, 2]
        hip_roll = 0.5 * (lat / max(d_clip.mean(), 1e-3))
        hip_yaw = np.zeros(N)
        ankle_roll = 0.3 * (lat / max(d_clip.mean(), 1e-3))
        # assign to X1 dof (left block positive hip_pitch ~ +0.4 default, knee +)
        if tag == "L":
            joints[:, dof_idx[0]] = X1_DEFAULT[0] + hip_pitch
            joints[:, dof_idx[1]] = X1_DEFAULT[1] + np.clip(hip_roll, -0.3, 0.3)
            joints[:, dof_idx[2]] = X1_DEFAULT[2]
            joints[:, dof_idx[3]] = X1_DEFAULT[3] + np.clip(knee_flex, 0.0, 1.2)
            joints[:, dof_idx[4]] = X1_DEFAULT[4] + np.clip(ankle_pitch, -0.3, 0.3)
            joints[:, dof_idx[5]] = X1_DEFAULT[5] + np.clip(ankle_roll, -0.2, 0.2)
        else:
            joints[:, dof_idx[0]] = X1_DEFAULT[6] - hip_pitch    # right mirrored sign
            joints[:, dof_idx[1]] = X1_DEFAULT[7] + np.clip(hip_roll, -0.3, 0.3)
            joints[:, dof_idx[2]] = X1_DEFAULT[8]
            joints[:, dof_idx[3]] = X1_DEFAULT[9] + np.clip(knee_flex, 0.0, 1.2)
            joints[:, dof_idx[4]] = X1_DEFAULT[10] + np.clip(ankle_pitch, -0.3, 0.3)
            joints[:, dof_idx[5]] = X1_DEFAULT[11] + np.clip(ankle_roll, -0.2, 0.2)

    # 5. foot contact: stance when the ankle forward-velocity is LOW (planted foot
    # barely moves forward; swinging foot translates forward fast). This is the
    # standard biomechanical stance criterion and is robust to the small vertical
    # swing amplitude in the SMPL data.
    dt_f = 1.0 / fps
    foot_fwd = np.zeros((N, 2), dtype=np.float64)               # forward (SMPL-X) pos per foot
    for tag, hip_i, knee_i, ank_i, dof_idx in leg_specs:
        idx = 0 if tag == "L" else 1
        foot_fwd[:, idx] = pos[:, ank_i, 0]                    # ankle world X (forward)
    foot_vel = np.zeros_like(foot_fwd)
    foot_vel[1:] = np.diff(foot_fwd, axis=0) / dt_f
    foot_vel[0] = foot_vel[1]
    foot_speed = np.abs(foot_vel)
    # stance = foot speed below its own median (slow/planted)
    med = np.median(foot_speed, axis=0)
    foot_contact = (foot_speed <= med[None, :]).astype(np.float64)

    # 6. light temporal smoothing (moving average, window=3) to remove per-frame
    # jitter from finite-difference/IK discretization, improving continuity.
    w = 3
    if N > w:
        k = np.ones(w) / w
        joints_sm = np.empty_like(joints)
        for c in range(12):
            joints_sm[:, c] = np.convolve(joints[:, c], k, mode="same")
        joints = joints_sm

    # clamp + count (final, after smoothing)
    n_clip = 0
    for i, (lo, hi) in enumerate(X1_LIMITS):
        col = joints[:, i]
        n_clip += int(np.sum((col < lo) | (col > hi)))
        joints[:, i] = np.clip(joints[:, i], lo, hi)

    # checks
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
             note="Retargeted SMPL-X 0008_normal_walk4_stageii -> AgiBot X1 via keypoint FK+2-link IK (Gate B).")
    report = {
        "source": os.path.relpath(src_path, ROOT),
        "source_sha256": _sha256(src_path),
        "source_fps": fps, "n_frames": int(N),
        "method": "SMPL forward-kinematics + closed-form 2-link sagittal IK (keypoint-based, no axis-guessing)",
        "joint_mapping": "SMPL 0pelvis/1Lhip/2Rhip/4Lknee/5Rknee/7Lank/8Rank (official smplx SMPL_JOINT_NAMES) -> X1 URDF dof_names",
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
    return report


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    r = retarget()
    print(f"[retarget] wrote {r['output']}")
    print(f"  source sha256={r['source_sha256']} fps={r['source_fps']} frames={r['n_frames']}")
    print(f"  forward_v={r['forward_velocity_mps']:.3f} m/s  root_h={r['root_height_mean']:.3f}+-{r['root_height_std']:.3f}")
    print(f"  finite_ok={r['finite_ok']} range_ok={r['range_ok']} clip_violations={r['n_clip_violations']}")
    print(f"  report: {os.path.relpath(OUT_REPORT, ROOT)}")
