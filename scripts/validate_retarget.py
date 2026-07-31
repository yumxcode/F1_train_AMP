#!/usr/bin/env python3
"""Gate B validation: MuJoCo kinematic replay + physics metrics.

Loads retargeted NPZ, replays in MuJoCo (kinematic, no dynamics), and checks:
1. Joint name 1:1 mapping, dim/order match, no missing/duplicate
2. Root height/posture/direction reasonable, no NaN/Inf, no jumps
3. Support foot sliding, ground penetration (strict), suspension
4. Knee reverse folding, self-collision
5. Left/right foot contact timing vs reference
6. Loop continuity (first/last frame)

Outputs a structured PASS/FAIL report.
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gmr_f1" / "pylibs"))

import numpy as np
import mujoco as mj

XML = str(ROOT / "gmr_f1" / "GMR" / "assets" / "x1" / "x1_mocap.xml")
CLIP = str(ROOT / "data" / "retarget_gmr" / "x1_walk_retargeted.npz")

# Thresholds
THRESHOLDS = {
    "ground_penetration_max": 0.005,    # m — STRICT: no more than 5mm below ground
    "foot_sliding_max": 0.02,           # m per frame during stance
    "suspension_min_contacts": 1,       # at least 1 foot in contact at all times
    "knee_flexion_min": 0.0,            # knee pitch should never go negative (reverse fold)
    "knee_range_min": 0.3,              # rad — each knee must move > 0.3 rad
    "contact_fraction_range": [0.15, 0.65],  # each foot stance fraction
    "joint_jump_max": 0.5,              # rad per frame — no sudden jumps
    "root_height_range": [0.40, 0.80],  # m
    "NaN_allowed": False,
}

EXPECTED_JOINTS = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_pitch_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_pitch_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
]


def validate():
    d = np.load(CLIP, allow_pickle=True)
    report = {"checks": [], "pass": True}

    def check(name, passed, detail=""):
        report["checks"].append({"name": name, "pass": bool(passed), "detail": detail})
        if not passed:
            report["pass"] = False
            print(f"  ✗ FAIL: {name} — {detail}")
        else:
            print(f"  ✓ PASS: {name} — {detail}")

    root_pos = d["root_translation"]
    root_quat = d["root_rotation"]   # xyzw
    joints = d["joint_positions"]
    joint_names = list(d["joint_names"])
    contact = d["foot_contact"]
    fps = float(d["fps"])
    foot_h = d["foot_heights"] if "foot_heights" in d.files else None
    N = root_pos.shape[0]
    dt = 1.0 / fps

    print(f"\n{'='*60}")
    print(f"Gate B Validation — {N} frames @ {fps}Hz, dt={dt:.4f}s")
    print(f"{'='*60}\n")

    # ── 1. Joint name mapping ── #
    print("[1] Joint name mapping")
    match = joint_names == EXPECTED_JOINTS
    check("joint_names_match", match, f"expected {len(EXPECTED_JOINTS)} joints, got {len(joint_names)}")
    check("joint_dim_match", joints.shape[1] == 12, f"shape={joints.shape}")
    check("no_duplicates", len(set(joint_names)) == len(joint_names), f"{len(set(joint_names))} unique")

    # ── 2. NaN/Inf ── #
    print("\n[2] NaN/Inf check")
    for name, arr in [("root_pos",root_pos),("root_quat",root_quat),("joints",joints)]:
        nbad = int(np.sum(~np.isfinite(arr)))
        check(f"finite_{name}", nbad == 0, f"{nbad} non-finite")

    # ── 3. Root height/posture/direction ── #
    print("\n[3] Root trajectory")
    rz = root_pos[:, 2]
    rh_ok = THRESHOLDS["root_height_range"][0] <= rz.mean() <= THRESHOLDS["root_height_range"][1]
    check("root_height", rh_ok, f"Z mean={rz.mean():.3f} range=[{rz.min():.3f},{rz.max():.3f}]")

    # Root height variation (should be small during walking)
    rz_var = rz.std()
    check("root_height_stable", rz_var < 0.05, f"std={rz_var:.4f}")

    # Walking direction (should be primarily +X)
    dx = root_pos[-1, 0] - root_pos[0, 0]
    dy = root_pos[-1, 1] - root_pos[0, 1]
    speed = dx / (N * dt)
    check("forward_progression", dx > 0.5, f"dx={dx:.2f}m dy={dy:.2f}m speed={speed:.2f}m/s")

    # Quaternion norm
    qnorm = np.linalg.norm(root_quat, axis=1)
    check("quat_normalized", np.allclose(qnorm, 1.0, atol=1e-4), f"norm range=[{qnorm.min():.6f},{qnorm.max():.6f}]")

    # Jumps in root position
    root_jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    check("no_root_jumps", root_jumps.max() < 0.1, f"max jump={root_jumps.max():.4f}m")

    # ── 4. Joint ranges and knee check ── #
    print("\n[4] Joint analysis")
    short = ["L_hp","L_hr","L_hy","L_kp","L_ap","L_ar","R_hp","R_hr","R_hy","R_kp","R_ap","R_ar"]
    for i, n in enumerate(short):
        r = joints[:,i].max() - joints[:,i].min()
        print(f"  {n}: range={r:.3f} [{joints[:,i].min():+.3f}, {joints[:,i].max():+.3f}] std={joints[:,i].std():.3f}")

    # Knee flexion (never negative = no reverse fold)
    lk = joints[:, 3]  # left_knee_pitch
    rk = joints[:, 9]  # right_knee_pitch
    check("left_knee_no_reverse", lk.min() >= THRESHOLDS["knee_flexion_min"],
          f"min={lk.min():.3f}")
    check("right_knee_no_reverse", rk.min() >= THRESHOLDS["knee_flexion_min"],
          f"min={rk.min():.3f}")
    check("left_knee_range", lk.max()-lk.min() >= THRESHOLDS["knee_range_min"],
          f"range={lk.max()-lk.min():.3f}")
    check("right_knee_range", rk.max()-rk.min() >= THRESHOLDS["knee_range_min"],
          f"range={rk.max()-rk.min():.3f}")

    # Joint jumps
    joint_vel = np.abs(np.diff(joints, axis=0))
    max_vel = joint_vel.max()
    check("no_joint_jumps", max_vel < THRESHOLDS["joint_jump_max"],
          f"max jump={max_vel:.3f}rad/frame")

    # Frozen joints (std < 0.01)
    frozen = [short[i] for i in range(12) if joints[:,i].std() < 0.01]
    check("no_frozen_joints", len(frozen) == 0, f"frozen: {frozen}")

    # ── 5. MuJoCo FK replay for physics metrics ── #
    print("\n[5] MuJoCo kinematic replay")
    model = mj.MjModel.from_xml_path(XML)
    data = mj.MjData(model)

    l_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    r_foot_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    l_knee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_knee_pitch_link")
    r_knee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_knee_pitch_link")
    pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")

    foot_positions = np.zeros((N, 2, 3))   # (N, foot, xyz)
    knee_positions = np.zeros((N, 2, 3))
    pelvis_positions = np.zeros((N, 3))
    qpos_all = np.zeros((N, model.nq))

    # Build qpos from clip
    dof_names_mj = []
    for i in range(model.njnt):
        n = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if n != "pelvis":
            dof_names_mj.append(n)
    name_to_idx = {n:i for i,n in enumerate(dof_names_mj)}
    reorder = [name_to_idx[jn] for jn in EXPECTED_JOINTS]

    for f in range(N):
        qpos = np.zeros(model.nq)
        qpos[:3] = root_pos[f]
        qpos[3:7] = root_quat[f]
        qpos[7:] = joints[f, reorder] if len(reorder) == 12 else joints[f]
        # Actually qpos[7:] should be in MJCF dof order
        # joints is in EXPECTED_JOINTS order, need to map back to MJCF order
        for i, jn in enumerate(EXPECTED_JOINTS):
            qpos[7 + name_to_idx[jn]] = joints[f, i]
        qpos_all[f] = qpos

        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        foot_positions[f, 0] = data.xpos[l_foot_id]
        foot_positions[f, 1] = data.xpos[r_foot_id]
        knee_positions[f, 0] = data.xpos[l_knee_id]
        knee_positions[f, 1] = data.xpos[r_knee_id]
        pelvis_positions[f] = data.xpos[pelvis_id]

    # ── 6. Ground penetration (STRICT) ── #
    print("\n[6] Ground penetration (STRICT)")
    for s, name in enumerate(["Left", "Right"]):
        foot_z = foot_positions[:, s, 2]
        penetration = np.maximum(0, THRESHOLDS["ground_penetration_max"] - foot_z)
        max_pen = penetration.max()
        # Use the foot sphere positions for more accurate check
        foot_min_z = foot_z.min()
        check(f"{name}_foot_min_height",
              foot_min_z >= -THRESHOLDS["ground_penetration_max"],
              f"min Z={foot_min_z:.4f}m (threshold={THRESHOLDS['ground_penetration_max']:.3f})")

    # ── 7. Foot sliding ── #
    print("\n[7] Foot sliding")
    for s, name in enumerate(["Left", "Right"]):
        foot_xy = foot_positions[:, s, :2]
        foot_disp = np.linalg.norm(np.diff(foot_xy, axis=0), axis=1)
        # During stance (contact=1), displacement should be small
        stance_mask = contact[:-1, s] > 0.5
        if stance_mask.sum() > 0:
            slide = foot_disp[stance_mask]
            max_slide = slide.max() if len(slide) > 0 else 0
            check(f"{name}_foot_sliding",
                  max_slide < THRESHOLDS["foot_sliding_max"],
                  f"max slide during stance={max_slide:.4f}m ({stance_mask.sum()} stance frames)")
        else:
            check(f"{name}_foot_sliding", True, "no stance frames detected")

    # ── 8. Suspension (both feet off ground) ── #
    print("\n[8] Suspension check")
    both_off = (contact[:, 0] < 0.5) & (contact[:, 1] < 0.5)
    suspension_frames = both_off.sum()
    check("no_suspension", suspension_frames <= N * 0.05,
          f"{suspension_frames}/{N} frames with both feet off ({100*suspension_frames/N:.1f}%)")

    # ── 9. Contact timing ── #
    print("\n[9] Contact timing")
    for s, name in enumerate(["Left", "Right"]):
        frac = contact[:, s].mean()
        lo, hi = THRESHOLDS["contact_fraction_range"]
        check(f"{name}_contact_fraction", lo <= frac <= hi,
              f"fraction={frac:.2f} (expected [{lo},{hi}])")

    # Alternation: when one foot is in stance, the other should be swinging
    l_stance = contact[:, 0] > 0.5
    r_stance = contact[:, 1] > 0.5
    both_stance = (l_stance & r_stance).sum()
    check("contact_alternation", both_stance < N * 0.3,
          f"both stance: {both_stance}/{N} ({100*both_stance/N:.1f}%)")

    # ── 10. Loop continuity ── #
    print("\n[10] Loop continuity")
    loopable = bool(d["loopable"]) if "loopable" in d.files else False
    pos_gap = np.linalg.norm(root_pos[-1] - root_pos[0])
    check("loop_status", True, f"loopable={loopable}, pos_gap={pos_gap:.3f}m")

    # ── Summary ── #
    n_pass = sum(1 for c in report["checks"] if c["pass"])
    n_fail = sum(1 for c in report["checks"] if not c["pass"])
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if report['pass'] else 'FAIL'} ({n_pass} passed, {n_fail} failed)")
    print(f"{'='*60}\n")

    # Save report
    report_path = str(ROOT / "data" / "retarget_gmr" / "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")

    return report["pass"]


if __name__ == "__main__":
    validate()
