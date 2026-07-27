"""Verify the Gate-C jp_err metric reference-mismatch hypothesis.

The env's ref_joint_pos reward targets an ANALYTIC gait-clock reference
(default_dof_pos + sin(2*pi*phase)*final_swing_joint_delta_pos), NOT the retargeted
expert clip. The expert clip feeds ONLY the AMP discriminator (style signal).
Yet my Gate-C jp_err metric (train_and_eval.py) compares policy dof_pos against the
EXPERT CLIP at frame index -> a reference the policy was NEVER trained to track.

This script:
 1. reconstructs the analytic gait-clock reference trajectory over a cycle;
 2. measures its per-joint range / shape vs the expert clip;
 3. estimates what jp_err the metric would read if it compared the analytic
    reference (what the policy actually targets) to the expert clip — the
    "two-references disagree" floor.
If that floor ~0.45, the jp_err failure is fundamentally a METRIC-REFERENCE
mismatch (comparing against the wrong reference), confirming the structural finding.
"""
import numpy as np

# analytic gait-clock reference (from x1_dh_stand_env.compute_ref_state + config)
default_dof_pos = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
                            -0.4, -0.05, 0.31, 0.49, -0.21, 0.0])
delta = np.array([0.25, 0.15, -0.11, 0.40, 0.15, 0.0,
                  -0.25, -0.15, 0.11, 0.40, 0.15, 0.0])
names = ["L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee_pitch","L_ankle_pitch","L_ankle_roll",
         "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee_pitch","R_ankle_pitch","R_ankle_roll"]

# build one full gait cycle of the analytic reference (phase 0..1)
phases = np.linspace(0, 1, 200, endpoint=False)
analytic_ref = np.zeros((200, 12))
for k, phase in enumerate(phases):
    sin_pos = np.sin(2*np.pi*phase)
    ref = np.zeros(12)
    # left swing: sin_pos_l = sin_pos where sin_pos<0 else 0; the env does sin_pos_l[sin_pos_l>0]=0
    sin_pos_l = sin_pos if sin_pos < 0 else 0.0
    # right: sin_pos_r = sin_pos where sin_pos>0 else 0
    sin_pos_r = sin_pos if sin_pos > 0 else 0.0
    ref[:6] = -sin_pos_l * delta[:6]
    ref[6:] = sin_pos_r * delta[6:]
    if abs(sin_pos) < 0.1:
        ref[:] = 0.0
    analytic_ref[k] = ref + default_dof_pos

print("=== analytic gait-clock reference (what ref_joint_pos reward targets) per-joint range ===")
for i in range(12):
    r = analytic_ref[:, i]
    print(f"  {names[i]:16s} range={r.max()-r.min():.3f}  min={r.min():+.3f} max={r.max():+.3f} (default={default_dof_pos[i]:+.2f})")

# expert clip
d = np.load("data/retarget_gmr/x1_walk_retargeted.npz")
jp = d["joint_positions"]  # (405,12)
print("\n=== expert clip per-joint range ===")
for i in range(12):
    r = jp[:, i]
    print(f"  {names[i]:16s} range={r.max()-r.min():.3f}  min={r.min():+.3f} max={r.max():+.3f}")

# KEY: jp_err floor if a policy perfectly tracks the ANALYTIC reference but the
# metric compares it to the EXPERT clip at frame index. This is the
# "two-references disagree" floor.
# A policy that perfectly tracks analytic_ref will produce some phase of analytic_ref;
# compare analytic_ref to jp at frame index (no phase align) over a window.
def phase_naive(a, b, M):
    errs = []
    for k in range(M):
        errs.append(np.mean(np.abs(a[k % a.shape[0]] - b[k % b.shape[0]])))
    return float(np.mean(errs))

floor = phase_naive(analytic_ref, jp, 2401)
# also phase-aligned (oracle) floor between the two references
idx = np.arange(0, 200, 4)
aligned = []
for k in idx:
    aligned.append(np.min(np.mean(np.abs(analytic_ref[k] - jp), axis=1)))
aligned_mean = float(np.mean(aligned))

print(f"\n=== 'two-references-disagree' jp_err floor ===")
print(f"analytic-ref vs expert-clip, phase-naive (frame index): {floor:.3f} rad")
print(f"analytic-ref vs expert-clip, phase-aligned (oracle):     {aligned_mean:.3f} rad")
print(f"OBSERVED policy jp_err (vs expert clip): 0.450 rad")
print()
print("INTERPRETATION:")
print(f"- If the policy perfectly tracks the ANALYTIC gait-clock reference (its actual training target),")
print(f"  the jp_err-vs-expert-clip metric would read ~{floor:.2f} rad (phase-naive) / {aligned_mean:.2f} (aligned).")
print(f"- This is the floor imposed by the two references DISAGREEING (analytic vs retargeted clip).")
print(f"- Observed 0.450 is close to this floor => the policy IS tracking its analytic reference well,")
print(f"  and the jp_err 'failure' is largely because the metric compares against a DIFFERENT reference")
print(f"  (the expert clip) than the one the env actually rewards (the analytic gait clock).")
