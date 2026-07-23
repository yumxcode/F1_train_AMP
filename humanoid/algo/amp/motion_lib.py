# SPDX-License-Identifier: BSD-3-Clause
"""Expert motion library + shared AMP feature builder for AgiBot X1.

This module owns the *single source of truth* for the AMP discriminator feature layout so
that expert (retargeted mocap) and policy (env rollout) produce bit-identical schemas.

Contract: see ``data/amp_contract.md``. Joint order is the URDF ``dof_names`` order
(= action order), never dict-insertion order or a foreign robot's indices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:  # torch is only needed at train time; numpy path lets the loader be unit-tested headless.
    import torch
except Exception:  # pragma: no cover - headless audit/testing without torch
    torch = None  # type: ignore

# --------------------------------------------------------------------------- #
# Authoritative X1 joint order & default pose (== URDF dof_names, == action order)
# Mirrors humanoid/envs/x1/x1_dh_stand_config.py init_state.default_joint_angles.
# --------------------------------------------------------------------------- #
X1_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
# Conservative locomotion limits (== cfg.safety.manual_joint_clip). Used for expert range check.
X1_JOINT_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-0.90, 0.70), (-0.35, 0.35), (-0.45, 0.45), (0.00, 1.45), (-0.55, 0.45), (-0.30, 0.30),
    (-0.90, 0.70), (-0.35, 0.35), (-0.45, 0.45), (0.00, 1.45), (-0.55, 0.45), (-0.30, 0.30),
)
# default_dof_pos in X1 joint order (rad). Must equal the env default so dof_pos_rel matches.
X1_DEFAULT_DOF_POS: Tuple[float, ...] = (
    0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
    -0.4, -0.05, 0.31, 0.49, -0.21, 0.0,
)
X1_NUM_JOINTS = len(X1_JOINT_NAMES)

# --------------------------------------------------------------------------- #
# AMP feature layout. dim F = 35. See data/amp_contract.md §2.
# Purely motion-derived (NO gait-phase reference) so the discriminator learns style
# from the actual motion, matching standard AMP. Expert == policy, exactly.
# --------------------------------------------------------------------------- #
_REF_SWING_IDX = (0, 3, 4, 6, 9, 10)  # kept for retarget reporting only (not a disc feature)
_AMP_LAYOUT = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("dof_pos_rel", X1_NUM_JOINTS),
    ("dof_vel", X1_NUM_JOINTS),
    ("foot_contact", 2),
]
AMP_BLOCKS: Dict[str, Tuple[int, int]] = {}
_off = 0
for _name, _dim in _AMP_LAYOUT:
    AMP_BLOCKS[_name] = (_off, _off + _dim)
    _off += _dim
AMP_OBS_DIM: int = _off  # 3+3+3+12+12+2 = 35


# --------------------------------------------------------------------------- #
# Quaternion helpers (x,y,z,w) - Isaac Gym tensor order.
# --------------------------------------------------------------------------- #
def quat_from_axis_angle_np(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (...,3) axis-angle (rotation vector) to (...,4) quat (x,y,z,w)."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, angle, out=np.zeros_like(axis_angle), where=angle != 0)
    s = np.sin(angle * 0.5)
    c = np.cos(angle * 0.5)
    q = np.concatenate([axis * s, c], axis=-1)
    # zero-rotation guard
    q = np.where(angle == 0, np.array([0.0, 0.0, 0.0, 1.0]), q)
    return q


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (...,3) by the inverse of quat q (...,4 xyzw)."""
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    # q* v q, with v as pure quaternion
    t = 2.0 * np.cross(np.stack([x, y, z], axis=-1), v)
    res = v - np.stack([w, w, w], axis=-1) * t + np.cross(np.stack([x, y, z], axis=-1), t)
    return res


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v (...,3) by quat q (...,4 xyzw)."""
    qc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return quat_rotate_inverse_np(np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1), v)


# --------------------------------------------------------------------------- #
# Feature builders. The expert path uses numpy; the policy path mirrors this in torch.
# --------------------------------------------------------------------------- #
def _default_dof_vec() -> np.ndarray:
    return np.asarray(X1_DEFAULT_DOF_POS, dtype=np.float64)


def build_amp_obs_numpy(
    root_pos_t: np.ndarray,       # (3,) world root translation
    root_pos_prev: np.ndarray,    # (3,) previous-frame world root translation
    root_quat_t: np.ndarray,      # (4,) xyzw base orientation THIS frame
    root_quat_prev: np.ndarray,   # (4,) previous frame
    joint_pos_t: np.ndarray,      # (12,) X1-order joint positions (rad)
    joint_pos_prev: np.ndarray,   # (12,)
    foot_contact_t: np.ndarray,   # (2,) {0,1}
    foot_contact_prev: np.ndarray, # (2,)
    dt: float,                    # seconds (must be the 100 Hz policy dt)
    gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> np.ndarray:
    """Build one AMP feature vector (AMP_OBS_DIM,) from adjacent retargeted frames.

    All quantities are expressed in the base (pelvis) frame and use ``dt`` for finite-diff
    velocities. Expert and policy MUST call the same builder (mirrored in torch in the env).
    """
    assert dt > 0, "dt must be positive (100 Hz policy dt)"
    # base linear velocity in base frame
    world_lin_vel = (root_pos_t - root_pos_prev) / dt
    base_lin_vel = quat_rotate_inverse_np(root_quat_t, world_lin_vel)
    # base angular velocity (world) from quaternion difference, then to base frame
    qrel = _quat_mul_np(_quat_conj_np(root_quat_prev), root_quat_t)
    omega_world = 2.0 * qrel[..., :3] / dt
    base_ang_vel = quat_rotate_inverse_np(root_quat_t, omega_world)
    # projected gravity in base frame
    proj_grav = quat_rotate_inverse_np(root_quat_t, gravity)
    # joint pos rel default + joint vel
    default = _default_dof_vec()
    dof_pos_rel = joint_pos_t - default
    dof_vel = (joint_pos_t - joint_pos_prev) / dt

    out = np.concatenate([
        base_lin_vel.reshape(-1),
        base_ang_vel.reshape(-1),
        proj_grav.reshape(-1),
        dof_pos_rel.reshape(-1),
        dof_vel.reshape(-1),
        np.asarray(foot_contact_t, dtype=np.float64).reshape(-1),
    ]).astype(np.float64)
    assert out.shape[0] == AMP_OBS_DIM, f"AMP feature dim {out.shape[0]} != {AMP_OBS_DIM}"
    return out


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1)


# --------------------------------------------------------------------------- #
# Motion library
# --------------------------------------------------------------------------- #
@dataclass
class ClipStats:
    n_frames: int
    fps: float
    dt: float
    amp_mean: np.ndarray
    amp_std: np.ndarray
    amp_min: np.ndarray
    amp_max: np.ndarray
    joint_range_ok: bool


class MotionLib:
    """Loads retargeted expert motion and serves expert AMP minibatches.

    Expected clip file (npz) keys:
        root_translation : (N,3)  world, meters
        root_rotation    : (N,4)  xyzw quaternion  (retarget converts SMPL axis-angle)
        joint_positions  : (N,12) X1 joint order
        ref_dof_pos_rel  : (N,6)  swing pitch joint delta vs default (or None -> zeros)
        foot_contact     : (N,2)  {0,1} (or None -> height-based)
        fps              : scalar Hz of the stored clip
    """

    def __init__(self, clip_paths: List[str], target_fps: float = 100.0,
                 loopable: bool = False, device: str = "cpu"):
        self.target_fps = float(target_fps)
        self.target_dt = 1.0 / self.target_fps
        self.loopable = loopable
        self.device = device
        self.clips_amp: List[np.ndarray] = []
        self.stats: List[ClipStats] = []
        for p in clip_paths:
            amp, st = self._load_one(p)
            self.clips_amp.append(amp)
            self.stats.append(st)
        self._all = np.concatenate(self.clips_amp, axis=0)
        if self.loopable:
            # only loop if first/last amp obs are close (continuity guard)
            if np.linalg.norm(self._all[0] - self._all[-1]) < 5.0:
                self._all = np.concatenate([self._all, self._all[:1]], axis=0)
        self._n = self._all.shape[0]

    def _load_one(self, path: str) -> Tuple[np.ndarray, ClipStats]:
        d = np.load(path, allow_pickle=True)
        fps = float(d["fps"]) if "fps" in d.files else self.target_fps
        root_t = np.asarray(d["root_translation"], dtype=np.float64)
        root_q = np.asarray(d["root_rotation"], dtype=np.float64)
        joints = np.asarray(d["joint_positions"], dtype=np.float64)
        contact = (np.asarray(d["foot_contact"], dtype=np.float64)
                   if "foot_contact" in d.files else None)
        if root_q.shape[-1] == 3:  # still axis-angle -> convert
            root_q = quat_from_axis_angle_np(root_q)
        root_q = root_q / (np.linalg.norm(root_q, axis=-1, keepdims=True) + 1e-12)
        # resample to target fps
        root_t, root_q, joints, contact, new_fps = _resample_clip(
            root_t, root_q, joints, contact, fps, self.target_fps)
        dt = 1.0 / new_fps
        N = root_t.shape[0]
        if contact is None:
            # height-based contact placeholder: foot below threshold -> contact
            contact = np.zeros((N, 2), dtype=np.float64)
        amp = np.zeros((max(N - 1, 0), AMP_OBS_DIM), dtype=np.float64)
        for i in range(N - 1):
            amp[i] = build_amp_obs_numpy(
                root_t[i + 1], root_t[i], root_q[i + 1], root_q[i],
                joints[i + 1], joints[i],
                contact[i + 1], contact[i], dt)
        joint_ok = _check_joint_ranges(joints)
        stats = ClipStats(
            n_frames=N, fps=new_fps, dt=dt,
            amp_mean=amp.mean(axis=0) if len(amp) else np.zeros(AMP_OBS_DIM),
            amp_std=amp.std(axis=0) if len(amp) else np.zeros(AMP_OBS_DIM),
            amp_min=amp.min(axis=0) if len(amp) else np.zeros(AMP_OBS_DIM),
            amp_max=amp.max(axis=0) if len(amp) else np.zeros(AMP_OBS_DIM),
            joint_range_ok=joint_ok,
        )
        _validate_finite(amp, path)
        return amp, stats

    @property
    def num_samples(self) -> int:
        return self._n

    def sample_expert(self, batch_size: int):
        """Return (batch_size, AMP_OBS_DIM) expert minibatch as torch tensor."""
        if torch is None:
            raise RuntimeError("torch required to sample expert minibatches")
        idx = np.random.randint(0, max(self._n, 1), size=batch_size)
        out = self._all[idx]
        return torch.from_numpy(out).float().to(self.device)

    def expert_logit_ema_init(self):
        """Initial estimate of mean expert discriminator logit (for reward transform)."""
        return 0.0


def _resample_clip(root_t, root_q, joints, contact, src_fps, dst_fps):
    if abs(src_fps - dst_fps) < 1e-3:
        return root_t, root_q, joints, contact, dst_fps
    N = root_t.shape[0]
    t_src = np.arange(N) / src_fps
    N_dst = int(math.floor(t_src[-1] * dst_fps)) + 1
    t_dst = np.arange(N_dst) / dst_fps
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])

    def interp(arr):
        out = np.empty((N_dst,) + arr.shape[1:], dtype=np.float64)
        for c in range(arr.shape[1]):
            out[:, c] = np.interp(t_dst, t_src, arr[:, c])
        return out

    rt = interp(root_t)
    j = interp(joints)
    ct = interp(contact) if contact is not None else None
    # quaternion resample: normalize of linear interp (fine for dense mocap)
    qi = interp(root_q)
    qi = qi / (np.linalg.norm(qi, axis=-1, keepdims=True) + 1e-12)
    # enforce continuity (flip sign if dot with previous < 0)
    for i in range(1, qi.shape[0]):
        if np.dot(qi[i], qi[i - 1]) < 0:
            qi[i] *= -1.0
    return rt, qi, j, ct, dst_fps


def _check_joint_ranges(joints: np.ndarray, tol: float = 0.05) -> bool:
    ok = True
    for i, (lo, hi) in enumerate(X1_JOINT_LIMITS):
        col = joints[:, i]
        if np.nanmin(col) < lo - tol or np.nanmax(col) > hi + tol:
            ok = False
            print(f"[MotionLib] joint {X1_JOINT_NAMES[i]} out of range "
                  f"[{lo},{hi}]: min={np.nanmin(col):.3f} max={np.nanmax(col):.3f}")
    return ok


def _validate_finite(arr: np.ndarray, path: str):
    if not np.all(np.isfinite(arr)):
        bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(f"[MotionLib] {path}: {bad} non-finite AMP feature values")
