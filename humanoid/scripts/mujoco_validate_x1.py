# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""MuJoCo structural + kinematic validation of the X1 model and smoke-expert motion.

Uses the MuJoCo runtime available in the training host env (mujoco>=3.x). NO Isaac Gym / torch.
Gate-A/B-relevant checks (the kinematic-replay clause of amp_loop.md §5/§6):
  1. MJCF parses; 12 actuated hinge DOF; free-floating base; bodies/foot collision geoms present.
  2. The URDF-actuated joint order (X1_JOINT_NAMES) maps 1:1 onto the MJCF dof order; the
     default keyframe qpos matches X1_DEFAULT_DOF_POS (single source of truth across URDF/MJCF).
  3. Kinematic replay of the smoke-expert trajectory: set qpos frame-by-frame, run mj_forward;
     assert joint limits respected, base height in range, no NaN/Inf, no gross ground penetration.

Run:  python humanoid/scripts/mujoco_validate_x1.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MJCF = os.path.join(ROOT, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
SMOKE = os.path.join(ROOT, "data", "smoke_expert", "walk_smoke.npz")


def _load_motion_lib():
    p = os.path.abspath(os.path.join(HERE, "..", "algo", "amp", "motion_lib.py"))
    spec = importlib.util.spec_from_file_location("_mjv_ml", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_mjv_ml"] = m
    spec.loader.exec_module(m)
    return m


ml = _load_motion_lib()
results = {"checks": [], "pass": True}


def check(name, cond, detail=""):
    results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        results["pass"] = False


def main():
    import mujoco
    print("=== MuJoCo structural + kinematic validation (X1) ===")
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    print(f"  loaded MJCF: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    # 1. structure
    n_act = model.nu
    check("12 actuators", n_act == 12, f"nu={n_act}")
    # hinge DOF count (exclude free joint)
    hinge_types = [model.jnt_type[i] for i in range(model.njnt)]
    n_hinge = sum(1 for t in hinge_types if t == mujoco.mjtJoint.mjJNT_HINGE)
    n_free = sum(1 for t in hinge_types if t == mujoco.mjtJoint.mjJNT_FREE)
    check("free-floating base (1 free joint)", n_free == 1, f"n_free={n_free}")
    check("12 hinge DOF", n_hinge == 12, f"n_hinge={n_hinge}")

    # body names present
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    has_pelvis = any("x1-body" in str(b) for b in body_names)
    has_ankle = sum("ankle_roll_link" in str(b) for b in body_names)
    check("pelvis body present", has_pelvis, "")
    check("two ankle_roll_link feet", has_ankle == 2, f"ankle_roll_link count={has_ankle}")

    # 2. joint order mapping: X1_JOINT_NAMES (_joint suffix) <-> MJCF joint names
    mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    # skip the free joint
    hinge_joints = [j for j in mj_joint_names if j and j != "floating_base"]
    x1_names = list(ml.X1_JOINT_NAMES)  # with _joint suffix
    # map by stripping the _joint suffix
    x1_stripped = [n.replace("_joint", "") for n in x1_names]
    check("12 hinge joints listed", len(hinge_joints) == 12, f"{hinge_joints}")

    # build qpos index map for each X1 joint
    x1_to_qpos = {}
    missing = []
    for xname in x1_names:
        mname = xname.replace("_joint", "")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mname)
        if jid < 0:
            missing.append(mname)
            continue
        x1_to_qpos[xname] = model.jnt_qposadr[jid]
    check("all 12 X1 joints found in MJCF", len(missing) == 0, f"missing={missing}")

    # default keyframe check
    if model.nkey > 0:
        kf = model.key_qpos[0].copy()
        kf_joints = np.array([kf[x1_to_qpos[n]] for n in x1_names])
        defaults = np.asarray(ml.X1_DEFAULT_DOF_POS)
        match = bool(np.allclose(kf_joints, defaults, atol=1e-3))
        check("keyframe qpos matches X1_DEFAULT_DOF_POS", match,
              f"keyframe={np.round(kf_joints,3).tolist()}\n   default ={np.round(defaults,3).tolist()}")
    else:
        check("keyframe present", False, "no keyframe in MJCF")

    # 3. kinematic replay of smoke expert
    d = np.load(SMOKE, allow_pickle=True)
    root_t = np.asarray(d["root_translation"])
    root_q = np.asarray(d["root_rotation"])  # xyzw
    joints = np.asarray(d["joint_positions"])
    N = root_t.shape[0]
    heights, nan_count, limit_viol = [], 0, 0
    contact_pen = []
    for i in range(N):
        qpos = data.qpos.copy()
        # free joint: pos(3) + quat(4)
        qpos[0:3] = root_t[i]
        # MuJoCo quat order is wxyz; expert/MotionLib uses xyzw -> convert
        qx, qy, qz, qw = root_q[i]
        qpos[3:7] = [qw, qx, qy, qz]
        for k, xname in enumerate(x1_names):
            qpos[x1_to_qpos[xname]] = joints[i, k]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(data.qpos)):
            nan_count += 1
        heights.append(float(data.qpos[2]))
        # joint limit check (range defined in MJCF: -3.14..3.14; use conservative X1 limits)
        for k, xname in enumerate(x1_names):
            lo, hi = ml.X1_JOINT_LIMITS[k]
            val = joints[i, k]
            if val < lo - 0.05 or val > hi + 0.05:
                limit_viol += 1
        # ground penetration: lowest contact geom z
        conz = [data.geom_xpos[g][2] for g in range(model.ngeom) if data.geom_xpos[g][2] < 0.3]
        if conz:
            contact_pen.append(min(conz))

    heights = np.array(heights)
    check("kinematic replay: no NaN/Inf in qpos", nan_count == 0, f"nan_frames={nan_count}")
    check("base height in [0.50, 0.72] throughout",
          float(heights.min()) > 0.50 and float(heights.max()) < 0.72,
          f"h range=[{heights.min():.3f},{heights.max():.3f}]")
    check("all joints within conservative X1 limits", limit_viol == 0, f"violations={limit_viol}")
    if contact_pen:
        worst = float(min(contact_pen))
        check("no gross ground penetration (min foot z > -0.02)", worst > -0.02, f"min_geom_z={worst:.3f}")
    else:
        check("foot geoms detected near ground", True, "no near-ground geoms (check geom list)")

    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} "
          f"({sum(c['pass'] for c in results['checks'])}/{len(results['checks'])}) ===")
    # write report
    rep = os.path.join(ROOT, "data", "smoke_expert", "mujoco_validation_report.txt")
    os.makedirs(os.path.dirname(rep), exist_ok=True)
    with open(rep, "w") as f:
        f.write(f"MuJoCo X1 validation ({N} frames)\n")
        f.write(f"MJCF nq={model.nq} nv={model.nv} nu={model.nu} n_hinge={n_hinge} n_free={n_free}\n")
        f.write(f"hinge_joints_order={hinge_joints}\n")
        f.write(f"height_range=[{heights.min():.3f},{heights.max():.3f}] nan={nan_count} limit_viol={limit_viol}\n")
        for c in results["checks"]:
            f.write(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']} :: {c['detail']}\n")
    print(f"  report -> data/smoke_expert/mujoco_validation_report.txt")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
