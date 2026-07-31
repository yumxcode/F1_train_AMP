#!/usr/bin/env python3
"""SMPLX→X1 retargeting via gait-parameter extraction + clean gait synthesis.

Extracts gait cycle, duty factor, step length, and clearance from SMPLX data,
then synthesizes a physically valid walking gait for the X1 robot.
Foot placement: planted during stance (zero sliding), bezier during swing.
"""
import sys, os, math, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gmr_f1" / "pylibs"))

import numpy as np
import mujoco as mj
import mink
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import find_peaks

XML = str(ROOT / "gmr_f1" / "GMR" / "assets" / "x1" / "x1_mocap.xml")
X1_JOINT_ORDER = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_pitch_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_pitch_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
]

# SMPLX skeleton (for gait parameter extraction only)
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,12,12,12,13,14,16,17,18,19]
NAMES = ["pelvis","left_hip","right_hip","spine1","left_knee","right_knee",
         "spine2","left_ankle","right_ankle","spine3","left_foot","right_foot"]
REST = {"left_hip":[-0.05,-0.03,0.09],"right_hip":[-0.05,-0.03,-0.09],
        "left_knee":[0,-0.42,0],"right_knee":[0,-0.42,0],
        "left_ankle":[0,-0.40,0],"right_ankle":[0,-0.40,0],
        "left_foot":[0.06,-0.04,0],"right_foot":[0.06,-0.04,0],
        "spine1":[0,0.12,0],"spine2":[0,0.13,0],"spine3":[0,0.13,0]}


def extract_gait_params(d):
    """Extract walking gait parameters from SMPLX data."""
    trans = np.asarray(d["trans"], dtype=np.float64)
    root_orient = np.asarray(d["root_orient"], dtype=np.float64)
    body_pose = np.asarray(d["pose_body"], dtype=np.float64)
    src_fps = float(d["mocap_frame_rate"])
    hr = (1.66 + 0.1*d["betas"][0]) / 1.70
    N = trans.shape[0]

    # FK for foot Z (height) and relative X (step pattern)
    foot_z = np.zeros((N, 2))
    foot_xrel = np.zeros((N, 2))
    for f in range(N):
        Rg = [None]*12
        Rg[0] = Rot.from_rotvec(root_orient[f]).as_matrix()
        for j in range(1, 12):
            Rg[j] = Rg[PARENTS[j]] @ Rot.from_rotvec(body_pose[f,(j-1)*3:j*3]).as_matrix()
        pos = [None]*12
        pos[0] = trans[f].copy()
        for j in range(1, 12):
            pos[j] = pos[PARENTS[j]] + Rg[PARENTS[j]] @ (np.array(REST[NAMES[j]])*hr)
        foot_z[f] = [pos[10][2], pos[11][2]]
        foot_xrel[f] = [pos[10][0]-pos[0][0], pos[11][0]-pos[0][0]]

    # Detect swing peaks
    lp, _ = find_peaks(foot_z[:,0], height=np.mean(foot_z[:,0]), distance=int(src_fps*0.3))
    rp, _ = find_peaks(foot_z[:,1], height=np.mean(foot_z[:,1]), distance=int(src_fps*0.3))

    cycle_s = np.mean(np.diff(lp))/src_fps if len(lp)>1 else 0.84
    cadence_hz = 1.0 / cycle_s

    # Step length from relative X oscillation
    stride = np.mean([foot_xrel[:,0].max()-foot_xrel[:,0].min(),
                      foot_xrel[:,1].max()-foot_xrel[:,1].min()])
    step_len = stride / 2.0
    clearance = np.mean([foot_z[:,0].max()-foot_z[:,0].min(),
                         foot_z[:,1].max()-foot_z[:,1].min()])
    walk_speed = (trans[-1,0]-trans[0,0]) / (N/src_fps)

    # Duty factor from stance time
    thresh = np.percentile(foot_z, 40)
    duty = 1.0 - np.mean(foot_z < thresh, axis=0).mean()

    return {
        "cycle_s": cycle_s, "cadence_hz": cadence_hz,
        "step_len_human": step_len, "clearance_human": clearance,
        "walk_speed_human": walk_speed, "duty_factor": duty,
        "human_height": 1.66+0.1*d["betas"][0],
        "src_fps": src_fps, "N_src": N,
    }


def generate_walking_gait(params, target_fps=100.0, duration_s=None):
    """Generate clean walking gait for X1 robot."""
    pos_scale = 0.61 / (params["human_height"] * 0.54)  # pelvis_h/height ratio
    pos_scale = min(pos_scale, 0.7)

    step_len = params["step_len_human"] * pos_scale
    clearance = params["clearance_human"] * pos_scale * 0.8
    cycle_s = params["cycle_s"]
    duty = min(max(params["duty_factor"], 0.55), 0.70)
    speed = step_len / (cycle_s * (1 - duty/2))  # approximate walking speed

    if duration_s is None:
        duration_s = params["N_src"] / params["src_fps"]

    N = int(duration_s * target_fps)
    dt = 1.0 / target_fps
    cycle_frames = int(cycle_s * target_fps)
    stance_frames = int(cycle_frames * duty)
    swing_frames = cycle_frames - stance_frames
    half_cycle = cycle_frames // 2

    ROBOT_PELVIS_Z = 0.61
    HIP_WIDTH = 0.092
    SOLE_H = 0.06   # ankle_roll body origin Z when flat (foot geom offset ~4cm)
    CONTACT_THRESH = 0.09

    print(f"[gait] cycle={cycle_s:.3f}s duty={duty:.2f} step={step_len:.3f}m "
          f"clearance={clearance:.3f}m speed={speed:.2f}m/s scale={pos_scale:.3f}")
    print(f"[gait] frames: cycle={cycle_frames} stance={stance_frames} swing={swing_frames} "
          f"half={half_cycle} total={N}")

    # Time array
    t = np.arange(N) * dt

    # Pelvis trajectory
    pelvis_x = speed * t
    pelvis_y = np.zeros(N)
    pelvis_z = ROBOT_PELVIS_Z + 0.008 * np.sin(2*np.pi*t/cycle_s)  # small vertical oscillation

    def gen_foot(phase_offset):
        """Generate foot world trajectory with proper stance/swing."""
        fx = np.zeros(N)
        fz = np.full(N, SOLE_H)
        prev_plant_x = 0.0
        swing_updated = False

        for f in range(N):
            phase = ((f / target_fps + phase_offset) % cycle_s) / cycle_s

            if phase < duty:
                # STANCE: foot planted at fixed world X
                fx[f] = prev_plant_x
                fz[f] = SOLE_H
                swing_updated = False
            else:
                # SWING: bezier from current to next plant
                s = (phase - duty) / (1 - duty)
                next_plant = prev_plant_x + step_len
                # Cubic bezier for smooth step
                fx[f] = (prev_plant_x*(1-s)**3 + 3*prev_plant_x*(1-s)**2*s +
                         3*next_plant*(1-s)*s**2 + next_plant*s**3)
                fz[f] = SOLE_H + clearance * 4*s*(1-s)
                # Update plant ONCE at end of swing
                if s > 0.98 and not swing_updated:
                    prev_plant_x = next_plant
                    swing_updated = True

        return fx, fz

    # Left foot: starts in stance
    l_fx, l_fz = gen_foot(0.0)
    # Right foot: half cycle offset
    r_fx, r_fz = gen_foot(cycle_s / 2)

    l_fy = np.full(N, +HIP_WIDTH)
    r_fy = np.full(N, -HIP_WIDTH)

    return {
        "pelvis_x": pelvis_x, "pelvis_y": pelvis_y, "pelvis_z": pelvis_z,
        "l_foot": np.stack([l_fx, l_fy, l_fz], axis=1),
        "r_foot": np.stack([r_fx, r_fy, r_fz], axis=1),
        "N": N, "fps": target_fps, "dt": dt,
        "sole_h": SOLE_H,
    }


def solve_ik(gait, out_path):
    """Solve IK for the generated gait."""
    N = gait["N"]
    model = mj.MjModel.from_xml_path(XML)
    model.opt.timestep = 0.01
    configuration = mink.Configuration(model)

    tasks = [
        mink.FrameTask("pelvis", "body", position_cost=500, orientation_cost=50, lm_damping=1),
        mink.FrameTask("left_ankle_roll_link", "body", position_cost=300, orientation_cost=20, lm_damping=1),
        mink.FrameTask("right_ankle_roll_link", "body", position_cost=300, orientation_cost=20, lm_damping=1),
    ]
    limits = [mink.ConfigurationLimit(model)]
    UPRIGHT = np.array([1.,0.,0.,0.])

    kf_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home_default")
    warm = model.key_qpos[kf_id].copy() if kf_id >= 0 else model.qpos0.copy()

    all_qpos = []
    MAX_ITER = 40

    for f in range(N):
        configuration.update(warm)

        targets = [
            np.array([gait["pelvis_x"][f], gait["pelvis_y"][f], gait["pelvis_z"][f]]),
            gait["l_foot"][f],
            gait["r_foot"][f],
        ]

        for task, pos in zip(tasks, targets):
            se3 = mink.SE3.from_rotation_and_translation(mink.SO3(UPRIGHT), pos)
            task.set_target(se3)

        dt = configuration.model.opt.timestep
        prev_err = float('inf')
        for it in range(MAX_ITER):
            vel = mink.solve_ik(configuration, tasks, dt, "daqp", 0.5, limits)
            configuration.integrate_inplace(vel, dt)
            err = np.linalg.norm(np.concatenate(
                [t.compute_error(configuration) for t in tasks]))
            if abs(prev_err - err) < 1e-5:
                break
            prev_err = err

        all_qpos.append(configuration.data.qpos.copy())
        warm = configuration.data.qpos.copy()
        if f % 100 == 0:
            print(f"  frame {f}/{N}, err={err:.4f}")

    all_qpos = np.array(all_qpos)

    # Extract and reorder
    root_pos = all_qpos[:, :3].copy()
    root_quat = all_qpos[:, 3:7].copy()
    dof_pos = all_qpos[:, 7:].copy()

    dof_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                 for i in range(model.njnt) if mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) != "pelvis"]
    name_to_idx = {n:i for i,n in enumerate(dof_names)}

    # Height adjust
    data = mj.MjData(model)
    foot_h = np.zeros((N, 2))
    for f in range(N):
        data.qpos[:3] = root_pos[f]
        data.qpos[3:7] = root_quat[f]
        for i, jn in enumerate(X1_JOINT_ORDER):
            data.qpos[7+name_to_idx[jn]] = dof_pos[f, name_to_idx[jn]]
        mj.mj_forward(model, data)
        for s, bn in enumerate(["left_ankle_roll_link","right_ankle_roll_link"]):
            foot_h[f,s] = data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, bn), 2]
    adjust = gait["sole_h"] - foot_h.min()
    root_pos[:, 2] += adjust
    foot_h += adjust

    # Reorder dof to X1 order
    reorder = [name_to_idx[jn] for jn in X1_JOINT_ORDER]
    dof_ordered = dof_pos[:, reorder]

    # Clip
    lo = np.array([-0.90,-0.35,-0.45, 0.00,-0.55,-0.30, -0.90,-0.35,-0.45, 0.00,-0.55,-0.30])
    hi = np.array([ 0.70, 0.35, 0.45, 1.45, 0.45, 0.30,  0.70, 0.35, 0.45, 1.45, 0.45, 0.30])
    dof_clipped = np.clip(dof_ordered, lo, hi)

    # Contact
    contact = (foot_h < 0.04).astype(np.float32)

    # Loop check
    loop_ok = (np.linalg.norm(root_pos[-1,:2]-root_pos[0,:2]) < 0.1 and
               np.linalg.norm(dof_clipped[-1]-dof_clipped[0]) < 0.3)

    # Validation summary
    short = ["L_hp","L_hr","L_hy","L_kp","L_ap","L_ar","R_hp","R_hr","R_hy","R_kp","R_ap","R_ar"]
    print("\n[validation] joint ranges:")
    for i,n in enumerate(short):
        r = dof_clipped[:,i].max()-dof_clipped[:,i].min()
        print(f"  {n}: range={r:.3f} std={dof_clipped[:,i].std():.3f}")
    print(f"[validation] foot_h: L[{foot_h[:,0].min():.3f},{foot_h[:,0].max():.3f}] "
          f"R[{foot_h[:,1].min():.3f},{foot_h[:,1].max():.3f}]")
    print(f"[validation] contact: L={contact[:,0].mean():.2f} R={contact[:,1].mean():.2f}")
    sw = sum(1 for i in range(N-1) if (contact[i]!=contact[i+1]).any())
    print(f"[validation] switches: {sw}, speed: {(root_pos[-1,0]-root_pos[0,0])/(N/gait['fps']):.2f}m/s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
        root_translation=root_pos.astype(np.float32),
        root_rotation=root_quat.astype(np.float32),
        joint_positions=dof_clipped.astype(np.float32),
        joint_names=np.array(X1_JOINT_ORDER),
        foot_contact=contact.astype(np.float32),
        fps=np.float32(gait["fps"]),
        loopable=np.bool_(loop_ok),
        foot_heights=foot_h.astype(np.float32),
    )
    print(f"\n[retarget] saved → {out_path}")


if __name__ == "__main__":
    d = np.load(str(ROOT/"data"/"0008_normal_walk4_stageii.npz"), allow_pickle=True)
    params = extract_gait_params(d)
    print(f"[params] {json.dumps(params, indent=2)}" if 'json' in dir() else f"[params] {params}")
    gait = generate_walking_gait(params, target_fps=100.0)
    solve_ik(gait, str(ROOT/"data"/"retarget_gmr"/"x1_walk_retargeted.npz"))
