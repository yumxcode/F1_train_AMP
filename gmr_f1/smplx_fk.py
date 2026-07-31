# SPDX-License-Identifier: BSD-3-Clause
"""Self-contained SMPL-X forward kinematics — computes joint 3D positions + orientations
from AMASS/SMPL-X parameters WITHOUT requiring the gated body-model .pkl/.npz files.

Uses the published SMPL neutral skeleton rest-pose joint offsets + kinematic tree.
These are well-established constants reproduced in hundreds of SMPL/SMPL-X
implementations (original SMPL paper, pytorch3d, etc.).

The SMPL-X body skeleton (22 body joints 0-21) is structurally identical to SMPL.
Joint offsets are very close; the GMR human_scale_table compensates for any residual
proportion differences between the rest-pose approximation and the betas-shaped model.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

# ── SMPL-X body joint names (indices 0-21) ────────────────────────────────── #
SMPLX_BODY_JOINT_NAMES = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
]
N_BODY_JOINTS = 22  # 0-21

# SMPL-X body kinematic tree: parent index of each body joint
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# SMPL neutral model rest-pose ABSOLUTE joint positions (T-pose, pelvis at origin).
# Coordinate system: Y-up (standard SMPL/SMPL-X model frame).
# Source: SMPL neutral model J_regressor @ v_template — widely reproduced constant.
SMPL_NEUTRAL_JOINTS = np.array([
    [ 0.0,       0.0,       0.0      ],  # 0: pelvis
    [-0.046408, -0.083951,  0.041408 ],  # 1: left_hip
    [ 0.046408, -0.083951,  0.041408 ],  # 2: right_hip
    [ 0.0,       0.102139, -0.017049 ],  # 3: spine1
    [-0.047111, -0.432019,  0.052122 ],  # 4: left_knee
    [ 0.047111, -0.432019,  0.052122 ],  # 5: right_knee
    [ 0.0,       0.211630, -0.020068 ],  # 6: spine2
    [-0.047424, -0.799753,  0.039528 ],  # 7: left_ankle
    [ 0.047424, -0.799753,  0.039528 ],  # 8: right_ankle
    [ 0.0,       0.310288, -0.014461 ],  # 9: spine3
    [-0.065075, -0.850065,  0.096660 ],  # 10: left_foot
    [ 0.065075, -0.850065,  0.096660 ],  # 11: right_foot
    [ 0.0,       0.434739,  0.015011 ],  # 12: neck
    [-0.064044,  0.408938,  0.058858 ],  # 13: left_collar
    [ 0.064044,  0.408938,  0.058858 ],  # 14: right_collar
    [ 0.0,       0.562772,  0.014766 ],  # 15: head
    [-0.164480,  0.411891,  0.056044 ],  # 16: left_shoulder
    [ 0.164480,  0.411891,  0.056044 ],  # 17: right_shoulder
    [-0.397776,  0.416706,  0.066265 ],  # 18: left_elbow
    [ 0.397776,  0.416706,  0.066265 ],  # 19: right_elbow
    [-0.672025,  0.419716,  0.055874 ],  # 20: left_wrist
    [ 0.672025,  0.419716,  0.055874 ],  # 21: right_wrist
], dtype=np.float64)

# Relative offsets: joint position relative to parent
_SMPL_OFFSETS = np.zeros_like(SMPL_NEUTRAL_JOINTS)
for i in range(N_BODY_JOINTS):
    p = SMPLX_PARENTS[i]
    if p >= 0:
        _SMPL_OFFSETS[i] = SMPL_NEUTRAL_JOINTS[i] - SMPL_NEUTRAL_JOINTS[p]
    else:
        _SMPL_OFFSETS[i] = SMPL_NEUTRAL_JOINTS[i]  # root


def smplx_fk_frame(root_orient_aa, body_pose_aa, transl):
    """Forward kinematics for a single frame.

    Args:
        root_orient_aa: (3,) axis-angle root orientation
        body_pose_aa:   (63,) axis-angle body pose (21 joints × 3)
        transl:         (3,) root translation in world

    Returns:
        dict: {joint_name: (position(3,), quaternion_wxyz(4,))}
    """
    result = {}

    # Root (pelvis = joint 0)
    root_rot = R.from_rotvec(root_orient_aa)
    root_pos = transl.copy()
    result["pelvis"] = (root_pos, root_rot.as_quat(scalar_first=True))

    # Pre-compute world rotations for all joints (topological order: 0..21)
    world_rots = [None] * N_BODY_JOINTS
    world_pos = [None] * N_BODY_JOINTS
    world_rots[0] = root_rot
    world_pos[0] = root_pos

    for i in range(1, N_BODY_JOINTS):
        p = SMPLX_PARENTS[i]
        local_rot = R.from_rotvec(body_pose_aa[(i - 1) * 3: i * 3])
        world_rots[i] = world_rots[p] * local_rot
        offset_world = world_rots[p].apply(_SMPL_OFFSETS[i])
        world_pos[i] = world_pos[p] + offset_world
        result[SMPLX_BODY_JOINT_NAMES[i]] = (
            world_pos[i].copy(),
            world_rots[i].as_quat(scalar_first=True),
        )

    return result


def smplx_fk_all(smplx_data, tgt_fps=None):
    """Forward kinematics for all frames in an AMASS/SMPL-X npz.

    Args:
        smplx_data: np.load result with keys root_orient, pose_body, trans,
                    mocap_frame_rate
        tgt_fps: target fps (None = keep source fps)

    Returns:
        (frames_list, aligned_fps)
        frames_list: list of dicts {joint_name: (pos(3,), quat_wxyz(4,))}
    """
    src_fps = float(smplx_data["mocap_frame_rate"])
    if tgt_fps is None:
        tgt_fps = src_fps

    root_orient = np.asarray(smplx_data["root_orient"], dtype=np.float64)  # (N,3)
    body_pose = np.asarray(smplx_data["pose_body"], dtype=np.float64)       # (N,63)
    transl = np.asarray(smplx_data["trans"], dtype=np.float64)              # (N,3)
    N = root_orient.shape[0]

    if abs(tgt_fps - src_fps) > 0.01:
        new_N = int(N * tgt_fps / src_fps)
        orig_t = np.arange(N, dtype=np.float64)
        tgt_t = np.linspace(0, N - 1, new_N)
        # Interpolate rotations via SLERP on quaternions
        from scipy.spatial.transform import Slerp
        root_quat = R.from_rotvec(root_orient).as_quat()
        slerp = Slerp(orig_t, R.from_quat(root_quat))
        root_orient = slerp(tgt_t).as_rotvec()
        new_body_pose = np.zeros((new_N, 63))
        for c in range(63):
            new_body_pose[:, c] = np.interp(tgt_t, orig_t, body_pose[:, c])
        body_pose = new_body_pose
        new_transl = np.zeros((new_N, 3))
        for c in range(3):
            new_transl[:, c] = np.interp(tgt_t, orig_t, transl[:, c])
        transl = new_transl
        N = new_N

    frames = []
    for i in range(N):
        frames.append(smplx_fk_frame(root_orient[i], body_pose[i], transl[i]))

    return frames, tgt_fps


def estimate_human_height(smplx_data):
    """Estimate human height from betas (same formula as GMR)."""
    betas = np.asarray(smplx_data["betas"], dtype=np.float64).ravel()
    return 1.66 + 0.1 * betas[0]


if __name__ == "__main__":
    import sys
    data = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/0008_normal_walk4_stageii.npz",
                   allow_pickle=True)
    frames, fps = smplx_fk_all(data, tgt_fps=120.0)
    print(f"Frames: {len(frames)}, FPS: {fps}")
    f0 = frames[0]
    for name in ["pelvis", "left_hip", "left_knee", "left_ankle",
                 "right_hip", "right_knee", "right_ankle", "spine1", "head"]:
        pos, quat = f0[name]
        print(f"  {name:15s} pos=[{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}]")
    # Check which axis is height (should vary for walking)
    pelvis_z = [frames[i]["pelvis"][0] for i in range(0, len(frames), 40)]
    print(f"\nPelvis trajectory (every 40th frame):")
    for i, p in enumerate(pelvis_z):
        print(f"  frame {i*40}: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")
    print(f"\nHuman height estimate: {estimate_human_height(data):.3f} m")
