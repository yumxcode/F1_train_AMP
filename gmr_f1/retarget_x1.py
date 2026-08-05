#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Gate B retargeting driver: SMPL-X walk → X1 robot motion via GMR differential IK.

Pipeline:
  1. Load AMASS/SMPL-X reference walk (data/0008_normal_walk4_stageii.npz)
  2. SMPL-X FK (self-contained, no gated body-model files) → human joint frames
  3. GMR differential IK (mujoco + mink) → X1 joint angles per frame
  4. Post-process: height adjust, origin offset, extract 12 leg DOF in X1 order
  5. Save retargeted expert clip for AMP MotionLib

Output: data/retarget_gmr/x1_walk_retargeted.npz
"""
import os, sys, json, time
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────── #
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GMR = os.path.join(REPO, "gmr_f1")
sys.path.insert(0, GMR)           # for smplx_fk
sys.path.insert(0, os.path.join(GMR, "GMR"))  # for general_motion_retargeting
sys.path.insert(0, os.path.join(GMR, "pylibs"))

from smplx_fk import smplx_fk_all, estimate_human_height, SMPLX_BODY_JOINT_NAMES
import mujoco
import mink
from scipy.spatial.transform import Rotation as Rot

# ── X1 joint order (from motion_lib / URDF dof_names) ─────────────────────── #
X1_DOF_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
]

XML_FILE = os.path.join(REPO, "resources", "robots", "x1", "mjcf", "xyber_x1_flat.xml")
IK_CONFIG = os.path.join(GMR, "smplx_to_x1.json")
REF_DATA = os.path.join(REPO, "data", "0008_normal_walk4_stageii.npz")
OUT_DIR = os.path.join(REPO, "data", "retarget_gmr")
OUT_PATH = os.path.join(OUT_DIR, "x1_walk_retargeted.npz")


def load_ik_config(path, actual_human_height):
    with open(path) as f:
        cfg = json.load(f)
    ratio = actual_human_height / cfg["human_height_assumption"]
    for key in cfg["human_scale_table"]:
        cfg["human_scale_table"][key] *= ratio
    return cfg


def setup_retargeter(xml_file, ik_config):
    """Build the GMR-style differential IK retargeter for X1."""
    model = mujoco.MjModel.from_xml_path(xml_file)

    # Enforce actual X1 joint limits (from env config manual_joint_clip)
    # The MJCF has ±π placeholders; tighten to real locomotion ranges so the
    # mink ConfigurationLimit prevents extreme over-rotation.
    X1_LIMITS = {
        "hip_pitch": (-0.90, 0.70), "hip_roll": (-0.35, 0.35),
        "hip_yaw": (-0.45, 0.45), "knee_pitch": (0.00, 1.45),
        "ankle_pitch": (-0.55, 0.45), "ankle_roll": (-0.30, 0.30),
    }
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for suffix, (lo, hi) in X1_LIMITS.items():
            if jname and jname.endswith(suffix):
                model.jnt_range[i] = [lo, hi]
                break

    # Collect DOF names
    dof_names = []
    for i in range(model.nv):
        jid = model.dof_jntid[i]
        dof_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))

    configuration = mink.Configuration(model)

    # Build IK tasks from config
    tasks1, tasks2 = [], []
    human_body_map1, human_body_map2 = {}, {}

    for table, tasks, hmap in [(ik_config["ik_match_table1"], tasks1, human_body_map1),
                                (ik_config["ik_match_table2"], tasks2, human_body_map2)]:
        for frame_name, entry in table.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name, frame_type="body",
                    position_cost=pos_weight, orientation_cost=rot_weight,
                    lm_damping=1,
                )
                hmap[body_name] = task
                tasks.append(task)

    ik_limits = [mink.ConfigurationLimit(model)]
    return model, configuration, tasks1, tasks2, human_body_map1, human_body_map2, ik_limits, dof_names


def scale_and_offset_human(human_data, ik_config):
    """Scale human data and apply pos/rot offsets (mirrors GMR update_targets)."""
    root_name = ik_config["human_root_name"]
    scale_table = ik_config["human_scale_table"]
    ground = np.array(ik_config["ground_height"]) * np.array([0, 0, 1])

    # Get pos/rot offsets from table1 (same for both tables)
    pos_offsets, rot_offsets = {}, {}
    for frame_name, entry in ik_config["ik_match_table1"].items():
        body_name, _, _, po, ro = entry
        pos_offsets[body_name] = np.array(po) - ground
        rot_offsets[body_name] = Rot.from_quat(ro, scalar_first=True)

    root_pos, root_quat = human_data[root_name]
    root_pos = np.asarray(root_pos)
    root_quat = np.asarray(root_quat)

    scaled_root_pos = scale_table[root_name] * root_pos

    result = {}
    for body_name in human_data:
        pos, quat = human_data[body_name]
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if body_name in scale_table and body_name != root_name:
            pos = (pos - root_pos) * scale_table[body_name] + scaled_root_pos

        # apply rot offset
        if body_name in rot_offsets:
            quat = (Rot.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            # apply pos offset in rotated frame
            local_off = pos_offsets.get(body_name, np.zeros(3))
            global_off = Rot.from_quat(quat, scalar_first=True).apply(local_off)
            pos = pos + global_off

        result[body_name] = (pos, quat)

    return result


def retarget_frame(human_data, configuration, tasks1, tasks2,
                   hmap1, hmap2, ik_limits, solver="daqp", damping=0.5):
    """Retarget one frame via differential IK."""
    # Set targets
    for hmap, tasks in [(hmap1, tasks1), (hmap2, tasks2)]:
        for body_name, task in hmap.items():
            if body_name in human_data:
                pos, quat = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(
                    mink.SO3(np.asarray(quat)), np.asarray(pos)))

    dt = configuration.model.opt.timestep
    max_iter = 10

    for tasks in [tasks1, tasks2]:
        if not tasks:
            continue
        def error():
            return np.linalg.norm(np.concatenate(
                [t.compute_error(configuration) for t in tasks]))
        curr_err = error()
        vel = mink.solve_ik(configuration, tasks, dt, solver, damping, ik_limits)
        configuration.integrate_inplace(vel, dt)
        next_err = error()
        n = 0
        while curr_err - next_err > 0.001 and n < max_iter:
            curr_err = next_err
            vel = mink.solve_ik(configuration, tasks, dt, solver, damping, ik_limits)
            configuration.integrate_inplace(vel, dt)
            next_err = error()
            n += 1

    return configuration.data.qpos.copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load reference SMPL-X data
    smplx_data = np.load(REF_DATA, allow_pickle=True)
    src_fps = float(smplx_data["mocap_frame_rate"])
    human_height = estimate_human_height(smplx_data)
    print(f"Reference: {REF_DATA}")
    print(f"  src_fps={src_fps}, frames={smplx_data['pose_body'].shape[0]}, human_height={human_height:.3f}m")

    # 2. SMPL-X FK (self-contained)
    tgt_fps = 100.0  # match AMP policy rate
    human_frames, aligned_fps = smplx_fk_all(smplx_data, tgt_fps=tgt_fps)
    print(f"  SMPL-X FK: {len(human_frames)} frames @ {aligned_fps} Hz")

    # 3. Setup retargeter
    ik_config = load_ik_config(IK_CONFIG, human_height)
    model, configuration, tasks1, tasks2, hmap1, hmap2, ik_limits, dof_names = setup_retargeter(XML_FILE, ik_config)
    print(f"  X1 model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"  DOF names: {dof_names}")

    # Build dof name → qpos index map (skip floating base: qpos[0:7])
    dof_to_qpos = {}
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] == 3:  # hinge
            dof_to_qpos[jname] = model.jnt_qposadr[i]

    # 4. Retarget each frame
    qpos_list = []
    t0 = time.time()
    for fi, human_data in enumerate(human_frames):
        scaled = scale_and_offset_human(human_data, ik_config)
        qpos = retarget_frame(scaled, configuration, tasks1, tasks2,
                              hmap1, hmap2, ik_limits)
        qpos_list.append(qpos.copy())
        if fi % 100 == 0:
            print(f"  frame {fi}/{len(human_frames)}...", flush=True)

    elapsed = time.time() - t0
    print(f"  Retargeting done: {len(qpos_list)} frames in {elapsed:.1f}s ({len(qpos_list)/elapsed:.1f} fps)")

    qpos_arr = np.array(qpos_list)  # (N, nq)
    N = qpos_arr.shape[0]

    # 5. Extract root pos/rot + leg DOF
    root_pos = qpos_arr[:, :3].copy()
    root_rot_xyzw = qpos_arr[:, 3:7].copy()  # mujoco quat is wxyz in qpos!

    # MuJoCo qpos quaternion order: [w, x, y, z]. Convert to xyzw for MotionLib.
    root_rot = root_rot_xyzw.copy()
    root_rot[:, [0,1,2,3]] = root_rot[:, [1,2,3,0]]  # wxyz → xyzw

    # Extract 12 leg DOF in X1 order
    dof_pos = np.zeros((N, 12))
    for i, dn in enumerate(X1_DOF_NAMES):
        if dn in dof_to_qpos:
            dof_pos[:, i] = qpos_arr[:, dof_to_qpos[dn]]
        else:
            print(f"  WARNING: DOF '{dn}' not found in model!")

    # 6. Height adjustment: ensure lowest foot on ground
    # Use mujoco FK to get body positions — compute for ALL frames (also needed for contact)
    data = mujoco.MjData(model)
    foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_ids_fk = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fb) for fb in foot_bodies]
    foot_z_all = np.zeros((N, 2))
    for fi in range(N):
        data.qpos[:] = qpos_arr[fi]
        mujoco.mj_forward(model, data)
        for li, fid in enumerate(foot_ids_fk):
            if fid >= 0:
                foot_z_all[fi, li] = data.xpos[fid][2]

    ground_offset = 0.02
    lowest_z = foot_z_all.min()
    height_shift = -lowest_z + ground_offset
    root_pos[:, 2] += height_shift
    print(f"  Height adjust: lowest_foot_z={lowest_z:.4f}, shift=+{height_shift:.4f}")

    # Do NOT scale root xy — it breaks foot world positions (support foot slip).
    # The IK already produced consistent root translations matching X1 kinematics.

    # Origin offset: zero first frame xy
    root_pos[:, :2] -= root_pos[0, :2]

    # Note: leg_scale root xy scaling was removed — it caused massive world-frame
    # foot slip (17+ m/s) because scaling root translation without scaling joint
    # angles disconnects foot positions from root. The IK already produces correct
    # root translations matching X1 kinematics.

    # 7. Compute foot contact (height-based heuristic, per-frame)
    # Recompute foot z with the final root_pos (after height + origin adjustments)
    for fi in range(N):
        qpos = np.zeros(model.nq)
        qpos[0:3] = root_pos[fi]
        qpos[3:7] = qpos_arr[fi, 3:7]
        for j, dn in enumerate(X1_DOF_NAMES):
            if dn in dof_to_qpos:
                qpos[dof_to_qpos[dn]] = dof_pos[fi, j]
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for li, fid in enumerate(foot_ids_fk):
            if fid >= 0:
                foot_z_all[fi, li] = data.xpos[fid][2]
    contact_threshold = ground_offset + 0.05  # foot within 5cm of ground = contact
    foot_contact = (foot_z_all < contact_threshold).astype(np.float64)
    # Smooth: a single-frame spike shouldn't toggle contact
    for c in range(2):
        col = foot_contact[:, c].copy()
        for i in range(1, N - 1):
            if col[i] != col[i - 1] and col[i] != col[i + 1]:
                col[i] = col[i - 1]  # remove single-frame spikes
        foot_contact[:, c] = col
    contact_pct = foot_contact.mean(axis=0) * 100
    print(f"  Foot contact: left={contact_pct[0]:.0f}% right={contact_pct[1]:.0f}%")

    # 8. Stats
    print(f"\n=== Retargeted motion stats ===")
    print(f"  frames={N}, fps={aligned_fps}")
    print(f"  root_pos z range: {root_pos[:,2].min():.3f} - {root_pos[:,2].max():.3f}")
    print(f"  root_pos x range: {root_pos[:,0].min():.3f} - {root_pos[:,0].max():.3f}")
    print(f"  dof_pos range per joint:")
    for i, dn in enumerate(X1_DOF_NAMES):
        print(f"    {dn:25s}: [{dof_pos[:,i].min():.3f}, {dof_pos[:,i].max():.3f}]")
    finite = np.all(np.isfinite(dof_pos)) and np.all(np.isfinite(root_pos))
    print(f"  all_finite={finite}")

    # 9. Save
    np.savez_compressed(OUT_PATH,
        root_translation=root_pos,
        root_rotation=root_rot,
        joint_positions=dof_pos,
        foot_contact=foot_contact,
        fps=aligned_fps,
        dof_names=np.array(X1_DOF_NAMES),
    )
    print(f"\nSaved to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
