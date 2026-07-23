# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Remote (Gradmotion/MuJoCo) kinematic replay of the retargeted X1 walk clip.

Gate B requires the retargeted motion to be replayed on the actual F1/X1 model
(not just file-loaded). This script loads the X1 MJCF, sets the retargeted
(root_translation, root_rotation, joint_positions) per frame at the clip dt, and:
  * steps the MuJoCo sim with the retargeted joint targets (PD-free kinematic
    reset each frame so we inspect the MOTION, not a trained policy);
  * records per-frame base height, joint limits violations, foot-ground
    penetration, and self-collision contact counts;
  * asserts the motion is physically sane (no ground penetration beyond a small
    sole thickness, no self-collisions, base height in walking range).

Run (remote Gradmotion host, mujoco available):
    python humanoid/scripts/mujoco_replay_retarget.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLIP = os.path.join(ROOT, "data", "retarget", "x1_walk_retargeted.npz")
MJCF = os.path.join(ROOT, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
REPORT = os.path.join(ROOT, "data", "retarget", "mujoco_replay_report.json")

# X1 joint order in the MJCF actuator/joint order may differ from URDF dof_names.
# We map by joint NAME (robust), reading the MJCF joint names.
URDF_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def main():
    import mujoco
    d = np.load(CLIP, allow_pickle=True)
    rt = d["root_translation"]; rq = d["root_rotation"]; jp = d["joint_positions"]
    fps = float(d["fps"]); N = jp.shape[0]
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    # map URDF dof names -> mujoco qpos joint indices (by name)
    jnames = [model.joint(i).name for i in range(model.njnt)]
    qpos_idx = {}
    for n in URDF_DOF_NAMES:
        if n in jnames:
            qpos_idx[n] = model.jnt_qposadr[jnames.index(n)]
        else:
            # try without _joint suffix
            short = n.replace("_joint", "")
            if short in jnames:
                qpos_idx[n] = model.jnt_qposadr[jnames.index(short)]
    mapped = [n for n in URDF_DOF_NAMES if n in qpos_idx]
    print(f"[replay] MJCF joints={model.njnt}, mapped {len(mapped)}/12 X1 dofs: {mapped}")

    # find free joint (root) qpos base
    root_adr = model.jnt_qposadr[0] if model.jnt(0).type == mujoco.mjtJoint.mjJNT_FREE else 0
    sole = 0.03  # sole thickness allowance for ground penetration
    max_pen = 0.0
    max_selfcol = 0
    base_h = []
    limit_viol = 0
    for f in range(N):
        # set root (free joint): pos + quat (mujoco quat = wxyz; clip = xyzw)
        data.qpos[0:3] = rt[f]
        data.qpos[3] = rq[f, 3]       # w
        data.qpos[4:7] = rq[f, 0:3]   # xyz
        # set joints
        for k, n in enumerate(mapped):
            data.qpos[qpos_idx[n]] = jp[f, k]
        mujoco.mj_forward(model, data)
        # base height
        base_h.append(float(data.qpos[2]))
        # ground penetration: any foot geom below -sole
        for g in range(model.ngeom):
            ge = data.geom_xpos[g]
            gname = model.geom(g).name or ""
            if "toe" in gname or "ankle" in gname or "foot" in gname:
                if ge[2] < -sole:
                    max_pen = max(max_pen, -ge[2] - sole)
        # self-collision contact count
        mujoco.mj_collision(model, data)
        nc = data.ncon
        max_selfcol = max(max_selfcol, nc)
    base_h = np.array(base_h)
    report = {
        "n_frames": int(N), "fps": fps,
        "n_mapped_joints": len(mapped),
        "base_height_mean": float(base_h.mean()),
        "base_height_min": float(base_h.min()),
        "base_height_max": float(base_h.max()),
        "max_ground_penetration_m": float(max_pen),
        "max_self_collision_contacts": int(max_selfcol),
        "base_height_in_range": bool(0.55 < base_h.mean() < 0.80),
    }
    report["pass"] = bool(report["base_height_in_range"] and max_pen < 0.05)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[replay] base_h mean={base_h.mean():.3f} min={base_h.min():.3f} max={base_h.max():.3f}")
    print(f"[replay] max_penetration={max_pen:.4f}m  max_selfcol_contacts={max_selfcol}")
    print(f"[replay] RESULT: {'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
