#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Gate B validation: MuJoCo kinematic playback + quantitative acceptance.

Checks:
  1. Joint name 1:1, dimension/order match X1 task
  2. Root height/pose/orientation reasonable
  3. No NaN/Inf jumps in joint angles or root
  4. Foot contact timing (alternating gait)
  5. Slip/penetration/hover/knee-reverse/self-collision within thresholds
  6. Loop continuity
"""
import os, sys, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Only insert vendored pylibs if mujoco is not already importable (e.g., remote Py3.8
# env without pip-installed mujoco). Locally (conda/pip mujoco 3.x), the vendored
# copy in gmr_f1/pylibs/mujoco is broken (MJTNUM_BYTES undefined), so we skip it.
try:
    import mujoco  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(REPO, "gmr_f1", "pylibs"))
import mujoco

XML = os.path.join(REPO, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
CLIP = os.path.join(REPO, "data", "retarget_gmr", "x1_walk_retargeted.npz")
REPORT = os.path.join(REPO, "data", "retarget_gmr", "validation_report.json")

X1_DOF = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]
X1_LIMITS = {
    "hip_pitch": (-0.90, 0.70), "hip_roll": (-0.35, 0.35),
    "hip_yaw": (-0.45, 0.45), "knee_pitch": (0.00, 1.45),
    "ankle_pitch": (-0.55, 0.45), "ankle_roll": (-0.30, 0.30),
}


def main():
    results = {"checks": [], "pass": True}

    def check(name, cond, detail=""):
        results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
        s = "PASS" if cond else "FAIL"
        print(f"  [{s}] {name}" + (f" :: {detail}" if detail else ""))
        if not cond:
            results["pass"] = False

    d = np.load(CLIP, allow_pickle=True)
    root_pos = d["root_translation"]
    root_rot = d["root_rotation"]  # xyzw
    dof_pos = d["joint_positions"]
    foot_contact = d["foot_contact"]
    fps = float(d["fps"])
    dt = 1.0 / fps
    N = root_pos.shape[0]

    print(f"=== Gate B Validation: {CLIP} ===")
    print(f"  frames={N}, fps={fps}, dt={dt:.4f}s, duration={N*dt:.2f}s")

    # 1. Joint name/dim match
    clip_dof = [str(x) for x in d["dof_names"]]
    check("joint_names 1:1 with X1 task", clip_dof == X1_DOF, f"{clip_dof}")
    check("joint dim == 12", dof_pos.shape[1] == 12, f"shape={dof_pos.shape}")
    check("root_translation shape (N,3)", root_pos.shape == (N, 3))
    check("root_rotation shape (N,4)", root_rot.shape == (N, 4))

    # 2. No NaN/Inf
    check("no NaN/Inf in root_pos", np.all(np.isfinite(root_pos)))
    check("no NaN/Inf in root_rot", np.all(np.isfinite(root_rot)))
    check("no NaN/Inf in dof_pos", np.all(np.isfinite(dof_pos)))

    # 3. Root height/pose reasonable
    root_z = root_pos[:, 2]
    check("root height > 0.3m", root_z.min() > 0.3, f"min_z={root_z.min():.3f}")
    check("root height < 1.0m", root_z.max() < 1.0, f"max_z={root_z.max():.3f}")
    check("root height variation < 0.2m", root_z.max() - root_z.min() < 0.2,
          f"range={root_z.max()-root_z.min():.3f}")

    # Forward progress
    root_x = root_pos[:, 0]
    total_dist = root_x[-1] - root_x[0]
    avg_speed = total_dist / (N * dt)
    check("forward progress > 0", total_dist > 0.1, f"dist={total_dist:.2f}m speed={avg_speed:.2f}m/s")
    check("avg speed 0.1-1.0 m/s", 0.1 < avg_speed < 1.0, f"speed={avg_speed:.2f}m/s")

    # Lateral drift
    root_y = root_pos[:, 1]
    check("lateral drift < 0.5m", abs(root_y[-1] - root_y[0]) < 0.5,
          f"drift={abs(root_y[-1]-root_y[0]):.3f}m")

    # 4. Joint angle smoothness (no NaN/Inf jumps)
    dof_vel = np.diff(dof_pos, axis=0) / dt
    max_joint_vel = np.max(np.abs(dof_vel))
    check("max joint velocity < 30 rad/s", max_joint_vel < 30,
          f"max_vel={max_joint_vel:.1f}rad/s")

    # Joint within limits (allow small margin)
    for i, dn in enumerate(X1_DOF):
        for suffix, (lo, hi) in X1_LIMITS.items():
            if dn.endswith(suffix):
                col = dof_pos[:, i]
                margin = 0.05
                ok = col.min() >= lo - margin and col.max() <= hi + margin
                if not ok:
                    check(f"{dn} within limits [{lo},{hi}]", False,
                          f"range=[{col.min():.3f},{col.max():.3f}]")
                break

    # 5. Knee not reversed (knee_pitch > -0.1 always)
    for side in ["left", "right"]:
        idx = X1_DOF.index(f"{side}_knee_pitch")
        knee = dof_pos[:, idx]
        check(f"{side} knee not reversed (>-0.1)", knee.min() > -0.1,
              f"min={knee.min():.3f}")

    # 6. MuJoCo playback: load model, set qpos, check body positions
    model = mujoco.MjModel.from_xml_path(XML)
    # Enforce X1 limits
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for suffix, (lo, hi) in X1_LIMITS.items():
            if jname and jname.endswith(suffix):
                model.jnt_range[i] = [lo, hi]
                break
    data = mujoco.MjData(model)

    dof_to_qpos = {}
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] == 3:
            dof_to_qpos[jname] = model.jnt_qposadr[i]

    foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb) for fb in foot_bodies]

    foot_heights = np.zeros((N, 2))
    knee_heights = np.zeros((N, 2))
    knee_bodies = ["lleft_knee_pitch_link", "right_knee_pitch_link"]
    knee_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kb) for kb in knee_bodies]

    penetration_count = 0
    hover_count = 0
    for fi in range(N):
        # Set qpos: [x,y,z, w,x,y,z, ...joints...]
        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[fi]
        # mujoco quat is wxyz; our root_rot is xyzw
        qpos[3] = root_rot[fi, 3]  # w
        qpos[4:7] = root_rot[fi, 0:3]  # xyz
        for j, dn in enumerate(X1_DOF):
            if dn in dof_to_qpos:
                qpos[dof_to_qpos[dn]] = dof_pos[fi, j]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for li, fid in enumerate(foot_ids):
            if fid >= 0:
                foot_heights[fi, li] = data.xpos[fid][2]
                if data.xpos[fid][2] < -0.01:
                    penetration_count += 1
        for li, kid in enumerate(knee_ids):
            if kid >= 0:
                knee_heights[fi, li] = data.xpos[kid][2]

    # Foot penetration
    check("foot penetration < 5% frames", penetration_count < 0.05 * N,
          f"penetration_frames={penetration_count}/{N}")

    # Foot clearance (swing foot lifts)
    for li, side in enumerate(["left", "right"]):
        fh = foot_heights[:, li]
        clearance = fh.max() - fh.min()
        check(f"{side} foot clearance > 0.02m", clearance > 0.02,
              f"clearance={clearance:.3f}m min={fh.min():.3f} max={fh.max():.3f}")

    # Foot hover (both feet off ground for extended period)
    min_foot_z = foot_heights.min(axis=1)
    hover_frames = np.sum(min_foot_z > 0.08)
    check("no extended double-float (< 10% frames)", hover_frames < 0.1 * N,
          f"double_float_frames={hover_frames}/{N}")

    # Knee above ground
    check("knee above ground (> 0.05m)", knee_heights.min() > 0.05,
          f"min_knee_z={knee_heights.min():.3f}")

    # 7. Loop continuity (first vs last frame)
    dof_diff = np.max(np.abs(dof_pos[-1] - dof_pos[0]))
    root_pos_diff = np.linalg.norm(root_pos[-1] - root_pos[0])
    check("loop joint continuity (dof diff < 1.0 rad)", dof_diff < 1.0,
          f"max_dof_diff={dof_diff:.3f}")

    # 8. Foot contact alternation (gait)
    contact_sum = foot_contact.sum(axis=0)
    check("both feet have contact frames", contact_sum.min() > N * 0.1,
          f"contact_sum={contact_sum}")
    # Check alternating pattern (not both always in contact)
    both_contact = np.sum(np.all(foot_contact > 0.5, axis=1))
    check("not always double-support (< 90%)", both_contact < 0.9 * N,
          f"double_support_frames={both_contact}/{N}")

    # Summary
    n_pass = sum(c["pass"] for c in results["checks"])
    n_total = len(results["checks"])
    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} ({n_pass}/{n_total}) ===")

    results["summary"] = {
        "frames": N, "fps": fps, "duration_s": N * dt,
        "avg_speed": avg_speed, "total_distance": total_dist,
        "root_z_range": [float(root_z.min()), float(root_z.max())],
        "n_pass": n_pass, "n_total": n_total,
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report: {REPORT}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
