"""Phase-mismatch analysis for the jp_err Gate-C metric.

The eval metric compares policy dof_pos against the reference clip at the
episode frame index (train_and_eval.py line 130: retarget_joints[ep_step % N]).
The policy commands vx=0.5 m/s; the reference clip walks ~1.3 m/s (root X delta).
A policy that imitates the reference's STYLE but at a different CADENCE (because
it is told to go slower) will be at a different gait phase than the reference at
any given frame index -> phase-naive jp_err is inflated.

This script quantifies the EXPECTED jp_err floor under cadence mismatch by
comparing the reference clip to a time-warped (re-cadenced) version of itself,
which is the best-case for a perfect-style walker that simply goes at 0.5 m/s.

If this floor is ~0.45, the metric itself is the problem (not the policy).
If this floor is ~0.0, then 0.45 is a genuine policy-imitation gap.
"""
import numpy as np

d = np.load("data/retarget_gmr/x1_walk_retargeted.npz")
jp = d["joint_positions"]
rt = d["root_translation"]
N = jp.shape[0]
fps = 100.0

# reference speed (m/s), from root X
ref_speed = np.mean(np.diff(rt[:,0])) * fps
cmd_speed = 0.5
cadence_ratio = cmd_speed / ref_speed   # policy walks at this fraction of reference cadence
print(f"reference speed={ref_speed:.3f} m/s, eval command={cmd_speed} m/s, cadence_ratio={cadence_ratio:.3f}")
print(f"=> a perfect-style policy walking at {cmd_speed} m/s has gait cadence ~{cadence_ratio:.2f}x the reference")

# Phase-naive jp_err: reference at frame k vs reference at frame k (identity) — the TRUE zero.
naive_zero = np.mean(np.abs(jp - jp))
print(f"\nphase-naive jp_err at zero shift (sanity, should be 0): {naive_zero:.4f}")

# Now simulate a perfect-style walker at cadence_ratio: it produces the SAME joint
# trajectory shape but stretched in time. At eval time, its frame k corresponds to
# reference frame round(k * cadence_ratio). Compare.
def phase_naive_err(policy_jp, ref_jp):
    """The eval metric: |policy_jp[k] - ref_jp[k % N]| mean over k."""
    M = policy_jp.shape[0]
    errs = []
    for k in range(M):
        errs.append(np.mean(np.abs(policy_jp[k] - ref_jp[k % N])))
    return float(np.mean(errs))

# Generate the "perfect imitator at 0.5 m/s" trajectory = reference re-sampled in time
# so that the SAME gait cycle plays over a longer wall-clock time (slower cadence).
# At eval frame k (100Hz), the slow walker is at reference-phase k*cadence_ratio.
slow_M = 2401  # one eval episode (matches the 2401-step episodes observed)
slow_jp = np.zeros((slow_M, 12))
for k in range(slow_M):
    ref_phase = k * cadence_ratio
    # linear interp of reference at fractional frame
    f0 = int(np.floor(ref_phase)) % N
    f1 = (f0 + 1) % N
    frac = ref_phase - np.floor(ref_phase)
    slow_jp[k] = (1 - frac) * jp[f0] + frac * jp[f1]

# Phase-naive jp_err of the SLOW perfect imitator vs reference at frame index
naive_slow = phase_naive_err(slow_jp, jp)
print(f"\nphase-naive jp_err of a PERFECT-STYLE walker at {cmd_speed} m/s vs reference at frame index:")
print(f"  = {naive_slow:.4f} rad  (OBSERVED policy jp_err = 0.450)")

# Phase-ALIGNED jp_err: for each slow-walker frame, take min error over all reference frames (oracle alignment)
# This is the lower bound for a perfect imitator (the error if we removed phase mismatch).
# Subsample for speed.
idx = np.arange(0, slow_M, 10)
aligned = []
for k in idx:
    best = np.min(np.mean(np.abs(slow_jp[k] - jp), axis=1))
    aligned.append(best)
aligned_mean = float(np.mean(aligned))
print(f"\nphase-ALIGNED jp_err (oracle best-shift) of the same perfect-style walker:")
print(f"  = {aligned_mean:.4f} rad")

print("\n=== INTERPRETATION ===")
print(f"If phase-naive floor ({naive_slow:.3f}) is close to observed 0.450,")
print(f"  => jp_err failure is a METRIC ARTIFACT (cadence/phase mismatch), not a policy defect.")
print(f"If phase-naive floor ({naive_slow:.3f}) is near 0,")
print(f"  => 0.450 is a genuine policy-imitation gap (the policy is not imitating the reference).")
print(f"phase-aligned floor ({aligned_mean:.3f}) shows what the metric SHOULD read for a perfect imitator.")
