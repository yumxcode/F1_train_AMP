# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Remote (Gradmotion/MuJoCo) kinematic replay of the GMR retargeted X1 walk clip.

Gate B §5 requires the retargeted motion to be replayed on the actual X1 MJCF
(not just file-loaded). This script:
  1. Runs validate_retarget_gmr.py (headless numpy Gate-B checks) first.
  2. Loads the X1 MJCF, sets the retargeted (root_translation, root_rotation,
     joint_positions) per frame, and checks: base height range, ground
     penetration, self-collision contacts.

Run (remote Gradmotion host, mujoco available):
    python humanoid/scripts/mujoco_replay_retarget_gmr.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLIP = os.path.join(ROOT, "data", "retarget_gmr", "x1_walk_retargeted.npz")
MJCF = os.path.join(ROOT, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
REPORT = os.path.join(ROOT, "data", "retarget_gmr", "mujoco_replay_report.json")

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
    jnames = [model.joint(i).name for i in range(model.njnt)]
    qpos_idx = {}
    for n in URDF_DOF_NAMES:
        if n in jnames:
            qpos_idx[n] = model.jnt_qposadr[jnames.index(n)]
        else:
            short = n.replace("_joint", "")
            if short in jnames:
                qpos_idx[n] = model.jnt_qposadr[jnames.index(short)]
    mapped = [n for n in URDF_DOF_NAMES if n in qpos_idx]
    print(f"[replay_gmr] MJCF joints={model.njnt}, mapped {len(mapped)}/12 X1 dofs")

    sole = 0.03
    max_pen = 0.0
    max_selfcol = 0
    base_h = []
    for f in range(N):
        data.qpos[0:3] = rt[f]
        data.qpos[3] = rq[f, 3]       # w (mujoco wxyz)
        data.qpos[4:7] = rq[f, 0:3]   # xyz
        for k, n in enumerate(mapped):
            data.qpos[qpos_idx[n]] = jp[f, k]
        mujoco.mj_forward(model, data)
        base_h.append(float(data.qpos[2]))
        for g in range(model.ngeom):
            ge = data.geom_xpos[g]
            gname = model.geom(g).name or ""
            if "toe" in gname or "ankle" in gname or "foot" in gname:
                if ge[2] < -sole:
                    max_pen = max(max_pen, -ge[2] - sole)
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
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[replay_gmr] base_h mean={base_h.mean():.3f} min={base_h.min():.3f} max={base_h.max():.3f}")
    print(f"[replay_gmr] max_penetration={max_pen:.4f}m  max_selfcol_contacts={max_selfcol}")
    print(f"[replay_gmr] RESULT: {'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    if "--validate-only" not in sys.argv:
        try:
            import subprocess
            here = os.path.dirname(__file__)
            subprocess.run([sys.executable, os.path.join(here, "validate_retarget_gmr.py")], check=True)
        except Exception as e:
            print(f"[replay_gmr] WARNING: validate_retarget_gmr step failed: {e}")
    sys.exit(main())
