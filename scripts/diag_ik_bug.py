"""Root-cause the left-knee-freeze IK bug in retarget_gmr.solve_leg_ik.
Reproduce the exact per-frame IK targets for the LEFT leg, then test variants:
  (a) full pipeline (stage2 pos+rot, warm-start)  -> reproduces freeze?
  (b) stage1 only (position), warm-start           -> does knee unfreeze?
  (c) stage2 but fresh-start (no warm-start) every frame -> warm-start the cause?
  (d) stage1 only, fresh-start
Reports the knee_pitch (leg index 3) trajectory range for each variant.
"""
import importlib.util, os, sys
import numpy as np

RG = os.path.join(os.getcwd(), "humanoid", "scripts", "retarget_gmr.py")
spec = importlib.util.spec_from_file_location("_rg", RG)
rg = importlib.util.module_from_spec(spec); sys.modules["_rg"] = rg; spec.loader.exec_module(rg)

d = np.load("data/0008_normal_walk4_stageii.npz", allow_pickle=True)
root_orient = np.asarray(d["root_orient"], dtype=np.float64)
pose_body = np.asarray(d["pose_body"], dtype=np.float64)
trans = np.asarray(d["trans"], dtype=np.float64)
N = pose_body.shape[0]
pos, Rg = rg.smpl_fk(root_orient, pose_body, trans)

# replicate retarget() target construction for LEFT leg
scale_info = rg.optimize_human_scale((pos[:, rg.SMPL_LHIP], pos[:, rg.SMPL_LKNEE], pos[:, rg.SMPL_LANK]), float(d["mocap_frame_rate"]))
smpl_leg_total = scale_info["smpl_thigh_len"] + scale_info["smpl_shin_len"]
leg_scale = (rg.X1_THIGH + rg.X1_SHIN) / max(smpl_leg_total, 1e-6)
P = np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=np.float64)
default_q6_L = rg.X1_DEFAULT[0:6].copy()  # left leg default [0.4,0.05,-0.31,0.49,-0.21,0.0]

print(f"leg_scale={leg_scale:.4f}  left default_q6={default_q6_L}  (knee idx3 default={default_q6_L[3]})")

# Build all left-leg targets
targets_local = np.zeros((N,3)); targets_rot = np.zeros((N,3,3))
for i in range(N):
    R_root_inv = Rg[0][i].T
    rel = pos[i, rg.SMPL_LANK] - pos[i, rg.SMPL_LHIP]
    root_local = R_root_inv @ rel
    tl = np.array([root_local[0], root_local[2], root_local[1]]) * leg_scale
    targets_local[i] = tl
    targets_rot[i] = P @ Rg[rg.SMPL_LANK][i] @ P.T

# reachability: distance from hip to target vs max leg reach (thigh+shin)
dist = np.linalg.norm(targets_local, axis=1)
print(f"\nLEFT target hip->ankle distance: mean={dist.mean():.4f} min={dist.min():.4f} max={dist.max():.4f}  (max reach thigh+shin={rg.X1_THIGH+rg.X1_SHIN:.4f})")

def run_variant(stage, warm):
    """stage in {1,2}; warm in {True,False}. Returns knee (idx3) trajectory."""
    knee = np.zeros(N)
    q_prev = default_q6_L.copy()
    for i in range(N):
        q0 = q_prev.copy() if warm else default_q6_L.copy()
        q6 = rg.solve_leg_ik(targets_local[i], targets_rot[i], 1.0, q0, max_stage=stage)
        if warm: q_prev = q6.copy()
        knee[i] = q6[3]
    return knee

variants = [("stage2 warm (full pipeline)", 2, True),
            ("stage1-only warm", 1, True),
            ("stage2 fresh-start", 2, False),
            ("stage1-only fresh-start", 1, False)]
print("\n=== LEFT knee_pitch (idx3) trajectory per variant ===")
print(f"{'variant':32s} {'mean':>7s} {'min':>7s} {'max':>7s} {'range':>7s}  frozen?")
for name, stage, warm in variants:
    k = run_variant(stage, warm)
    rng = k.max()-k.min()
    print(f"{name:32s} {k.mean():7.3f} {k.min():7.3f} {k.max():7.3f} {rng:7.3f}  {'FROZEN' if rng<0.05 else 'moves'}")

# Compare RIGHT leg (same, to confirm right is healthy)
targets_local_R = np.zeros((N,3)); targets_rot_R = np.zeros((N,3,3))
for i in range(N):
    R_root_inv = Rg[0][i].T
    rel = pos[i, rg.SMPL_RANK] - pos[i, rg.SMPL_RHIP]
    root_local = R_root_inv @ rel
    targets_local_R[i] = np.array([root_local[0], root_local[2], root_local[1]]) * leg_scale
    targets_rot_R[i] = P @ Rg[rg.SMPL_RANK][i] @ P.T
distR = np.linalg.norm(targets_local_R, axis=1)
print(f"\nRIGHT target hip->ankle distance: mean={distR.mean():.4f} min={distR.min():.4f} max={distR.max():.4f}")
def run_R(stage, warm):
    knee=np.zeros(N); q_prev=rg.X1_DEFAULT[6:12].copy()
    for i in range(N):
        q0=q_prev.copy() if warm else rg.X1_DEFAULT[6:12].copy()
        q6=rg.solve_leg_ik(targets_local_R[i], targets_rot_R[i], -1.0, q0, max_stage=stage)
        if warm: q_prev=q6.copy()
        knee[i]=q6[3]
    return knee
kR=run_R(2,True)
print(f"RIGHT stage2 warm: knee mean={kR.mean():.3f} range={kR.max()-kR.min():.3f} {'FROZEN' if kR.max()-kR.min()<0.05 else 'moves'}")
