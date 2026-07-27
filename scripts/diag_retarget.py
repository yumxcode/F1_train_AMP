import numpy as np
d = np.load("data/retarget_gmr/x1_walk_retargeted.npz")
jp = d["joint_positions"]; rt = d["root_translation"]; fc = d["foot_contact"]
N = jp.shape[0]

print("=== X1 joint order (from motion_lib): 6/leg = hip-pitch/roll/yaw, knee-pitch, ankle-pitch/roll ===")
names = ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee_pitch","L_ankle_pitch","L_ankle_roll",
         "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee_pitch","R_ankle_pitch","R_ankle_roll"]

print("\n=== per-joint motion stats (range = max-min over 405 frames) ===")
print(f"{'joint':16s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'range':>8s}  status")
for i in range(12):
    r = jp[:, i]
    rng = r.max() - r.min()
    status = "FROZEN" if rng < 0.02 else ("near-static" if rng < 0.1 else "active")
    print(f"{names[i]:16s} {r.mean():8.3f} {r.std():8.3f} {r.min():8.3f} {r.max():8.3f} {rng:8.3f}  {status}")

print("\n=== left vs right knee direct comparison (should alternate in a walk) ===")
print(f"L_knee(j3): range={jp[:,3].max()-jp[:,3].min():.4f}  mean={jp[:,3].mean():.4f}")
print(f"R_knee(j9): range={jp[:,9].max()-jp[:,9].min():.4f}  mean={jp[:,9].mean():.4f}")

print("\n=== foot_contact signal: does LEFT foot ever lift? bilateral alternation? ===")
print(f"foot_contact shape={fc.shape}")
print(f"L foot: mean={fc[:,0].mean():.3f} min={fc[:,0].min():.3f} max={fc[:,0].max():.3f} (range {fc[:,0].max()-fc[:,0].min():.3f})")
print(f"R foot: mean={fc[:,1].mean():.3f} min={fc[:,1].min():.3f} max={fc[:,1].max():.3f} (range {fc[:,1].max()-fc[:,1].min():.3f})")
both = np.mean((fc[:,0]>0.5)&(fc[:,1]>0.5))
neither = np.mean((fc[:,0]<0.5)&(fc[:,1]<0.5))
single = np.mean(((fc[:,0]>0.5)&(fc[:,1]<0.5)) | ((fc[:,0]<0.5)&(fc[:,1]>0.5)))
print(f"double-support(both)={both:.3f}  flight(neither)={neither:.3f}  single-support={single:.3f}")

print("\n=== bilateral symmetry check ===")
print(f"L_hip_pitch(j0) range={jp[:,0].max()-jp[:,0].min():.4f}")
print(f"R_hip_pitch(j6) range={jp[:,6].max()-jp[:,6].min():.4f}")
print(f"corr(L_hip_pitch j0, R_hip_pitch j6) = {np.corrcoef(jp[:,0],jp[:,6])[0,1]:.3f} (walk: antisymmetric ~-0.5..-0.9)")
print(f"corr(L_knee j3, R_knee j9) = {np.corrcoef(jp[:,3],jp[:,9])[0,1]:.3f}")

print("\n=== root motion ===")
print(f"root X: {rt[0,0]:.3f} -> {rt[-1,0]:.3f} (delta {rt[-1,0]-rt[0,0]:.3f}m over {N/100:.2f}s = {(rt[-1,0]-rt[0,0])/(N/100):.2f} m/s)")
print(f"root Z(height): mean={rt[:,2].mean():.3f} min={rt[:,2].min():.3f} max={rt[:,2].max():.3f}")

# Check the SMPL-X source too, to see if the frozen knee is in the source or introduced by retarget
try:
    src = np.load("data/0008_normal_walk4_stageii.npz")
    print("\n=== SMPL-X SOURCE pose_body check (is the source knee moving?) ===")
    print(f"source keys: {src.files}")
    pb = src["pose_body"]  # (N,63) SMPL-X body pose (21 joints x 3 axis-angles)
    print(f"pose_body shape={pb.shape}")
    # SMPL-X body joints: 0=pelvis(spine chain), ... knees are joints 1(L_knee) and 6? need SMPL ordering
    # SMPL joint order: 0=pelvis,1=l_hip,2=r_hip,3=spine1,4=l_knee,5=r_knee,6=spine2,7=l_ankle,8=r_ankle,...
    # Actually SMPL 24-joint: 0=pelvis,1=l_hip,2=r_hip,3=spine1,4=l_knee,5=r_knee,6=spine2,7=l_ankle,8=r_ankle,
    #   9=spine3,10=l_collar,11=r_collar,12=neck,13=head,14=l_shoulder... L_knee=4, R_knee=5
    print(f"L_knee(SMPL j4) axis-angle range per-comp: {pb[:,4*3:4*3+3].max(0)-pb[:,4*3:4*3+3].min(0)}")
    print(f"R_knee(SMPL j5) axis-angle range per-comp: {pb[:,5*3:5*3+3].max(0)-pb[:,5*3:5*3+3].min(0)}")
except Exception as e:
    print(f"\n(source check skipped: {e})")
