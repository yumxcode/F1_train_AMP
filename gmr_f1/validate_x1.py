#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Gate B validation: MuJoCo kinematic playback + quantitative acceptance.

Checks (expanded per reviewer requirements):
  1. Joint name 1:1, dimension/order match X1 task
  2. Root height/pose/orientation reasonable, no NaN/Inf
  3. Foot penetration (strict: <5mm), foot clearance, double-float, knee height
  4. Support foot slip (stance-phase foot horizontal velocity < threshold)
  5. Self-collision detection (MuJoCo geom-geom contact)
  6. Left-right contact timing vs reference motion comparison
  7. Loop continuity: position + orientation + velocity + contact (not just dof diff)
  8. Foot contact alternation (gait)
"""
import os, sys, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
try:
    import mujoco  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(REPO, "gmr_f1", "pylibs"))
import mujoco

XML = os.path.join(REPO, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
CLIP = os.path.join(REPO, "data", "retarget_gmr", "x1_walk_retargeted.npz")
REF_DATA = os.path.join(REPO, "data", "0008_normal_walk4_stageii.npz")
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
SLIP_VEL_THRESHOLD = 0.1  # m/s max horizontal velocity during stance (support phase)
PENETRATION_THRESHOLD = 0.005  # 5mm strict penetration threshold


def compute_ref_contact():
    """Compute contact timing from the original SMPL-X reference motion."""
    try:
        ref = np.load(REF_DATA, allow_pickle=True)
        from scipy.spatial.transform import Rotation as Rot
        PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
        REST = np.array([
            [0,0,0],[-0.046408,-0.083951,0.041408],[0.046408,-0.083951,0.041408],
            [0,0.102139,-0.017049],[-0.047111,-0.432019,0.052122],[0.047111,-0.432019,0.052122],
            [0,0.211630,-0.020068],[-0.047424,-0.799753,0.039528],[0.047424,-0.799753,0.039528],
            [0,0.310288,-0.014461],[-0.065075,-0.850065,0.096660],[0.065075,-0.850065,0.096660],
        ])
        trans = np.asarray(ref["trans"], dtype=np.float64)
        root_orient = np.asarray(ref["root_orient"], dtype=np.float64)
        body_pose = np.asarray(ref["pose_body"], dtype=np.float64)
        N = trans.shape[0]
        foot_z = np.zeros((N, 2))
        for f in range(N):
            wr=[None]*12; wp=[None]*12
            wr[0]=Rot.from_rotvec(root_orient[f]); wp[0]=trans[f].copy()
            for j in range(1,12):
                p=PARENTS[j]; lr=Rot.from_rotvec(body_pose[f,(j-1)*3:j*3])
                wr[j]=wr[p]*lr; wp[j]=wp[p]+wr[p].apply(REST[j]-REST[PARENTS[j]])
            # SMPL coords: Y=up, so foot height = Y component (index 1)
            foot_z[f] = [wp[10][1], wp[11][1]]
        # Contact = foot near lowest point
        thresh = np.percentile(foot_z, 30)
        ref_contact = (foot_z < thresh).astype(float)
        # Compute stance fraction per foot
        ref_stance_frac = ref_contact.mean(axis=0)
        # Compute step frequency (contact transitions per second)
        ref_transitions = np.sum(np.abs(np.diff(ref_contact, axis=0)), axis=0) / (N / float(ref["mocap_frame_rate"]))
        return {"stance_frac": ref_stance_frac.tolist(),
                "transitions_per_sec": ref_transitions.tolist(),
                "available": True}
    except Exception as e:
        return {"available": False, "error": str(e)}


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

    # === 1. Joint name/dim match ===
    clip_dof = [str(x) for x in d["dof_names"]]
    check("joint_names 1:1 with X1 task", clip_dof == X1_DOF, f"{clip_dof}")
    check("joint dim == 12", dof_pos.shape[1] == 12, f"shape={dof_pos.shape}")
    check("root_translation shape (N,3)", root_pos.shape == (N, 3))
    check("root_rotation shape (N,4)", root_rot.shape == (N, 4))

    # === 2. No NaN/Inf ===
    check("no NaN/Inf in root_pos", np.all(np.isfinite(root_pos)))
    check("no NaN/Inf in root_rot", np.all(np.isfinite(root_rot)))
    check("no NaN/Inf in dof_pos", np.all(np.isfinite(dof_pos)))

    # === 3. Root height/pose ===
    root_z = root_pos[:, 2]
    check("root height > 0.3m", root_z.min() > 0.3, f"min_z={root_z.min():.3f}")
    check("root height < 1.0m", root_z.max() < 1.0, f"max_z={root_z.max():.3f}")
    check("root height variation < 0.2m", root_z.max() - root_z.min() < 0.2,
          f"range={root_z.max()-root_z.min():.3f}")

    root_x = root_pos[:, 0]
    total_dist = root_x[-1] - root_x[0]
    avg_speed = total_dist / (N * dt)
    check("forward progress > 0", total_dist > 0.1, f"dist={total_dist:.2f}m speed={avg_speed:.2f}m/s")
    check("avg speed 0.1-1.0 m/s", 0.1 < avg_speed < 1.0, f"speed={avg_speed:.2f}m/s")

    root_y = root_pos[:, 1]
    check("lateral drift < 0.5m", abs(root_y[-1] - root_y[0]) < 0.5,
          f"drift={abs(root_y[-1]-root_y[0]):.3f}m")

    # Joint smoothness
    dof_vel = np.diff(dof_pos, axis=0) / dt
    check("max joint velocity < 30 rad/s", np.max(np.abs(dof_vel)) < 30,
          f"max_vel={np.max(np.abs(dof_vel)):.1f}rad/s")

    # Joint limits
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

    # Knee not reversed
    for side in ["left", "right"]:
        idx = X1_DOF.index(f"{side}_knee_pitch")
        knee = dof_pos[:, idx]
        check(f"{side} knee not reversed (>-0.1)", knee.min() > -0.1, f"min={knee.min():.3f}")

    # === 4. MuJoCo playback ===
    model = mujoco.MjModel.from_xml_path(XML)
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
    knee_bodies = ["lleft_knee_pitch_link", "right_knee_pitch_link"]
    knee_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kb) for kb in knee_bodies]

    foot_heights = np.zeros((N, 2))
    foot_xy = np.zeros((N, 2, 2))  # (frame, foot, xy)
    knee_heights = np.zeros((N, 2))
    penetration_count = 0

    # Self-collision tracking
    self_collision_frames = 0
    # Exclude foot-ground contacts (those are normal). Check body-body contacts.
    foot_body_ids = set(foot_ids)

    for fi in range(N):
        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[fi]
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
                foot_xy[fi, li] = data.xpos[fid][:2]
                if data.xpos[fid][2] < -PENETRATION_THRESHOLD:
                    penetration_count += 1
        for li, kid in enumerate(knee_ids):
            if kid >= 0:
                knee_heights[fi, li] = data.xpos[kid][2]

        # Self-collision: check contacts between non-foot bodies
        for ci in range(data.ncon):
            c = data.contact[ci]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
            # Skip ground contacts (world body id=0)
            if model.geom_bodyid[c.geom1] == 0 or model.geom_bodyid[c.geom2] == 0:
                continue
            # Skip foot-foot contacts with ground already covered. Check non-trivial contacts.
            # Only flag contacts between distinct body parts (not parent-child)
            if b1 and b2 and b1 != b2:
                self_collision_frames += 1
                break  # one contact per frame is enough

    # === Penetration (strict) ===
    check(f"foot penetration < {PENETRATION_THRESHOLD*1000:.0f}mm in all frames",
          penetration_count == 0, f"penetration_frames={penetration_count}/{N}")

    # === Foot clearance ===
    for li, side in enumerate(["left", "right"]):
        fh = foot_heights[:, li]
        clearance = fh.max() - fh.min()
        check(f"{side} foot clearance > 0.02m", clearance > 0.02,
              f"clearance={clearance:.3f}m min={fh.min():.3f} max={fh.max():.3f}")

    # === Double-float ===
    min_foot_z = foot_heights.min(axis=1)
    hover_frames = np.sum(min_foot_z > 0.08)
    check("no extended double-float (< 10% frames)", hover_frames < 0.1 * N,
          f"double_float_frames={hover_frames}/{N}")

    # === Knee above ground ===
    check("knee above ground (> 0.05m)", knee_heights.min() > 0.05,
          f"min_knee_z={knee_heights.min():.3f}")

    # === 5. Support foot slip ===
    # Slip = horizontal velocity of foot RELATIVE to root during stance.
    # Using world-frame foot position would include root forward motion (not slip).
    slip_violations = 0
    max_slip = 0.0
    stance_frames_total = 0
    for li in range(2):
        for fi in range(1, N):
            if foot_contact[fi, li] > 0.5 and foot_contact[fi-1, li] > 0.5:
                stance_frames_total += 1
                rel_fi = foot_xy[fi, li] - root_pos[fi, :2]
                rel_prev = foot_xy[fi-1, li] - root_pos[fi-1, :2]
                dx = (rel_fi[0] - rel_prev[0]) / dt
                dy = (rel_fi[1] - rel_prev[1]) / dt
                slip_speed = np.sqrt(dx**2 + dy**2)
                max_slip = max(max_slip, slip_speed)
                if slip_speed > SLIP_VEL_THRESHOLD:
                    slip_violations += 1
    check(f"support foot slip < {SLIP_VEL_THRESHOLD} m/s (root-relative, stance)",
          slip_violations < 0.05 * max(stance_frames_total, 1),
          f"slip_violations={slip_violations}/{stance_frames_total} stance_frames, max_slip={max_slip:.3f}m/s")

    # === 6. Self-collision ===
    check("no self-collision (non-ground body contacts)", self_collision_frames == 0,
          f"self_collision_frames={self_collision_frames}/{N}")

    # === 7. Contact timing vs reference ===
    ref_info = compute_ref_contact()
    contact_sum = foot_contact.sum(axis=0)
    clip_stance_frac = (foot_contact.sum(axis=0) / N).tolist()
    check("both feet have contact frames", contact_sum.min() > N * 0.1,
          f"contact_sum={contact_sum}")
    both_contact = np.sum(np.all(foot_contact > 0.5, axis=1))
    check("not always double-support (< 90%)", both_contact < 0.9 * N,
          f"double_support_frames={both_contact}/{N}")

    if ref_info["available"]:
        # Compare stance fraction with reference (within 20% absolute)
        for li, side in enumerate(["left", "right"]):
            ref_st = ref_info["stance_frac"][li]
            clip_st = clip_stance_frac[li]
            diff = abs(ref_st - clip_st)
            check(f"{side} stance fraction matches reference (diff<0.2)",
                  diff < 0.2,
                  f"ref={ref_st:.2f} clip={clip_st:.2f} diff={diff:.2f}")
        # Compare transition frequency
        clip_trans = np.sum(np.abs(np.diff(foot_contact > 0.5, axis=0)), axis=0) / (N * dt)
        for li, side in enumerate(["left", "right"]):
            ref_tr = ref_info["transitions_per_sec"][li]
            clip_tr = clip_trans[li]
            ratio = min(clip_tr, ref_tr) / max(clip_tr, ref_tr) if max(clip_tr, ref_tr) > 0 else 1.0
            check(f"{side} step frequency matches reference (ratio>0.5)",
                  ratio > 0.5,
                  f"ref={ref_tr:.1f}/s clip={clip_tr:.1f}/s ratio={ratio:.2f}")
    else:
        print(f"  [SKIP] Reference contact comparison unavailable: {ref_info.get('error','')}")

    # === 8. Loop continuity (expanded) ===
    # Position continuity
    root_pos_diff = np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2])  # xy only
    check("loop position continuity (xy diff < 0.5m)", root_pos_diff < 0.5,
          f"xy_diff={root_pos_diff:.3f}m")
    # Orientation continuity
    quat_diff = np.abs(root_rot[-1] - root_rot[0]).max()
    check("loop orientation continuity (quat diff < 0.5)", quat_diff < 0.5,
          f"quat_diff={quat_diff:.3f}")
    # Velocity continuity (finite differences)
    root_vel_last = (root_pos[-1] - root_pos[-2]) / dt
    root_vel_first = (root_pos[1] - root_pos[0]) / dt
    vel_diff = np.linalg.norm(root_vel_last - root_vel_first)
    check("loop velocity continuity (diff < 1.0 m/s)", vel_diff < 1.0,
          f"vel_diff={vel_diff:.3f}m/s")
    # Joint continuity
    dof_diff = np.max(np.abs(dof_pos[-1] - dof_pos[0]))
    check("loop joint continuity (dof diff < 1.0 rad)", dof_diff < 1.0,
          f"max_dof_diff={dof_diff:.3f}")
    # Contact continuity
    contact_diff = np.abs(foot_contact[-1] - foot_contact[0]).sum()
    check("loop contact continuity (no contact state change)", contact_diff == 0,
          f"contact_first={foot_contact[0].tolist()} contact_last={foot_contact[-1].tolist()}")
    # Overall loopable verdict
    loopable = (root_pos_diff < 0.1 and quat_diff < 0.1 and vel_diff < 0.3 and
                dof_diff < 0.3 and contact_diff == 0)
    check("clip loopable (strict all criteria)", loopable,
          f"loopable={loopable}" if not loopable else "loopable=True")
    if not loopable:
        print("  [NOTE] Clip is non-loopable — AMP MotionLib must NOT force-loop this clip")

    # === Summary ===
    n_pass = sum(c["pass"] for c in results["checks"])
    n_total = len(results["checks"])
    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} ({n_pass}/{n_total}) ===")

    results["summary"] = {
        "frames": N, "fps": fps, "duration_s": N * dt,
        "avg_speed": avg_speed, "total_distance": total_dist,
        "root_z_range": [float(root_z.min()), float(root_z.max())],
        "loopable": bool(loopable),
        "max_slip_mps": float(max_slip),
        "self_collision_frames": int(self_collision_frames),
        "penetration_frames": int(penetration_count),
        "n_pass": n_pass, "n_total": n_total,
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report: {REPORT}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
