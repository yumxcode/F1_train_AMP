"""Diagnose whether the frozen left knee is in the SMPL-X source or introduced by
the retarget IK. Compute the true knee FLEXION angle (hip->knee->ankle) for both
legs from SMPL forward kinematics, for all frames. A normal walk flexes BOTH knees
alternately. If L flexion is constant in the SOURCE, the mocap is the issue; if the
source flexes both but the retarget froze one, it's an IK bug.
"""
import importlib.util, os, sys
import numpy as np

HERE = os.path.join(os.getcwd(), "humanoid", "scripts")
ML = os.path.join(os.getcwd(), "humanoid", "algo", "amp", "motion_lib.py")
spec = importlib.util.spec_from_file_location("_ml", ML)
ml = importlib.util.module_from_spec(spec); sys.modules["_ml"] = ml; spec.loader.exec_module(ml)

# reuse retarget_gmr's smpl_fk
RG = os.path.join(os.getcwd(), "humanoid", "scripts", "retarget_gmr.py")
spec2 = importlib.util.spec_from_file_location("_rg", RG)
rg = importlib.util.module_from_spec(spec2); sys.modules["_rg"] = rg; spec2.loader.exec_module(rg)

d = np.load("data/0008_normal_walk4_stageii.npz", allow_pickle=True)
root_orient = np.asarray(d["root_orient"], dtype=np.float64)
pose_body = np.asarray(d["pose_body"], dtype=np.float64)
trans = np.asarray(d["trans"], dtype=np.float64)
N = pose_body.shape[0]
print(f"source frames={N}, fps={float(d['mocap_frame_rate'])}")

pos, Rg = rg.smpl_fk(root_orient, pose_body, trans)   # (N,10,3)

def knee_flexion_deg(hip, knee, ank):
    """Angle at the knee between hip->knee and knee->ankle vectors (degrees).
    ~180 = straight leg, smaller = more flexed."""
    v1 = hip - knee
    v2 = ank - knee
    v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(v1n * v2n, axis=1)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

L_flex = knee_flexion_deg(pos[:, rg.SMPL_LHIP], pos[:, rg.SMPL_LKNEE], pos[:, rg.SMPL_LANK])
R_flex = knee_flexion_deg(pos[:, rg.SMPL_RHIP], pos[:, rg.SMPL_RKNEE], pos[:, rg.SMPL_RANK])

print("\n=== SOURCE knee flexion angle (deg); 180=straight, <180=flexed ===")
print(f"LEFT  knee flexion: mean={L_flex.mean():.1f} std={L_flex.std():.1f} min={L_flex.min():.1f} max={L_flex.max():.1f} range={L_flex.max()-L_flex.min():.1f}")
print(f"RIGHT knee flexion: mean={R_flex.mean():.1f} std={R_flex.std():.1f} min={R_flex.min():.1f} max={R_flex.max():.1f} range={R_flex.max()-R_flex.min():.1f}")

# ankle height (Z, SMPL Y-up so index 1) relative to hip — does each foot lift during swing?
print("\n=== SOURCE ankle vertical motion (foot lift) ===")
L_lift = pos[:, rg.SMPL_LANK, 1] - pos[:, rg.SMPL_LHIP, 1]
R_lift = pos[:, rg.SMPL_RANK, 1] - pos[:, rg.SMPL_RHIP, 1]
print(f"LEFT  ankle-vs-hip height: range={L_lift.max()-L_lift.min():.3f}m (mean {L_lift.mean():.3f})")
print(f"RIGHT ankle-vs-hip height: range={R_lift.max()-R_lift.min():.3f}m (mean {R_lift.mean():.3f})")

print("\n=== correlation L vs R knee flexion (walk: strongly negative, ~-0.7..-0.95) ===")
print(f"corr(L_flex, R_flex) = {np.corrcoef(L_flex, R_flex)[0,1]:.3f}")

print("\n=== VERDICT ===")
l_moves = (L_flex.max() - L_flex.min()) > 8.0
r_moves = (R_flex.max() - R_flex.min()) > 8.0
print(f"source LEFT knee flexes: {'YES' if l_moves else 'NO (static)'} (range {L_flex.max()-L_flex.min():.1f} deg)")
print(f"source RIGHT knee flexes: {'YES' if r_moves else 'NO (static)'} (range {R_flex.max()-R_flex.min():.1f} deg)")
if l_moves and r_moves:
    print("=> SOURCE has bilateral knee flexion. The frozen L_knee in the retarget is an IK BUG (introduced by retarget_gmr.py).")
elif not l_moves:
    print("=> SOURCE left knee is static. The mocap clip itself lacks left-knee motion (asymmetric source).")
else:
    print("=> mixed; see above.")

# Also re-run the retarget IK on a few frames with verbose to see the IK behavior
print("\n=== spot-check: IK on source frame by frame, left leg, where does knee go? ===")
# build the same target as retarget() for left leg, frame 0 and a mid frame
for i in [0, 50, 100, 150, 200, 250]:
    R_root_inv = Rg[0][i].T
    rel = pos[i, rg.SMPL_LANK] - pos[i, rg.SMPL_LHIP]
    root_local = R_root_inv @ rel
    target_local = np.array([root_local[0], root_local[2], root_local[1]]) * (rg.X1_THIGH + rg.X1_SHIN) / (rg.scale_info_get if hasattr(rg,'scale_info_get') else 1.0)
