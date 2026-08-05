#!/usr/bin/env python3
"""Diagnostic: expert vs random-policy AMP feature separability analysis.

Computes expert AMP features and random-init policy features, then measures
how trivially separable they are using a simple logistic regression probe.

If separability > 0.99 at iter-0, the discriminator will trivially saturate
regardless of regularization. This confirms the distribution problem.
"""
import numpy as np
import sys, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Load expert clip
    d = np.load(os.path.join(REPO, "data", "retarget_gmr", "x1_walk_retargeted.npz"), allow_pickle=True)
    root_pos = d["root_translation"].astype(np.float64)
    joints = d["joint_positions"].astype(np.float64)
    contact = d["foot_contact"].astype(np.float64)
    fps = float(d["fps"])
    dt = 1.0 / fps
    N = root_pos.shape[0]
    default_dof = np.array([0.4,0.05,-0.31,0.49,-0.21,0.0,-0.4,-0.05,0.31,0.49,-0.21,0.0])

    # Build single-step AMP features for expert (35-dim, matching AMP_OBS_DIM)
    expert_feats = []
    for i in range(N-1):
        wlv = (root_pos[i+1] - root_pos[i]) / dt  # world lin vel
        dpr = joints[i+1] - default_dof            # dof pos rel
        dv = (joints[i+1] - joints[i]) / dt        # dof vel
        expert_feats.append(np.concatenate([wlv[:3], dpr, dv, contact[i+1]]))
    expert_feats = np.array(expert_feats)

    # Simulate random-init policy features (standing at default pose)
    # At init: base_lin_vel≈0, dof_pos≈default, dof_vel≈0, contact=[1,1]
    n_policy = 1000
    policy_feats = np.zeros((n_policy, expert_feats.shape[1]))
    policy_feats[:, :3] = np.random.randn(n_policy, 3) * 0.01   # near-zero velocity
    policy_feats[:, 3:15] = np.random.randn(n_policy, 12) * 0.02  # near-default dof
    policy_feats[:, 15:27] = np.random.randn(n_policy, 12) * 0.01  # near-zero dof vel
    policy_feats[:, 27:29] = 1.0  # standing: both feet in contact

    print("=== Expert vs Random-Policy Feature Separability ===")
    print(f"Expert samples: {expert_feats.shape[0]}")
    print(f"Policy samples: {n_policy}")
    print(f"Feature dim: {expert_feats.shape[1]}")

    # Per-dimension statistics
    print("\nExpert feature statistics (key dimensions):")
    names = ["vx", "vy", "vz"] + [f"dof_{i}" for i in range(12)] + [f"dvel_{i}" for i in range(12)] + ["contact_L", "contact_R"]
    for i in range(min(5, len(names))):
        e_m, e_s = expert_feats[:, i].mean(), expert_feats[:, i].std()
        p_m, p_s = policy_feats[:, i].mean(), policy_feats[:, i].std()
        sep = abs(e_m - p_m) / max(e_s + p_s, 0.001)
        print(f"  {names[i]:10s}: expert={e_m:+.3f}±{e_s:.3f}  policy={p_m:+.3f}±{p_s:.3f}  separation={sep:.1f}σ")

    # Simple separability: logistic regression accuracy
    X = np.vstack([expert_feats, policy_feats])
    y = np.concatenate([np.ones(len(expert_feats)), np.zeros(n_policy)])
    
    # Compute Fisher's Linear Discriminant (no sklearn needed)
    mu_e = expert_feats.mean(axis=0)
    mu_p = policy_feats.mean(axis=0)
    Sigma_e = np.cov(expert_feats.T) + np.eye(expert_feats.shape[1]) * 1e-6
    Sigma_p = np.cov(policy_feats.T) + np.eye(expert_feats.shape[1]) * 1e-6
    Sw = 0.5 * (Sigma_e + Sigma_p)  # within-class scatter
    w = np.linalg.solve(Sw, mu_e - mu_p)  # Fisher discriminant direction
    
    # Project and compute separation
    proj_e = expert_feats @ w
    proj_p = policy_feats @ w
    proj_all = X @ w
    threshold = (proj_e.mean() + proj_p.mean()) / 2
    
    acc_e = np.mean(proj_e > threshold)
    acc_p = np.mean(proj_p < threshold)
    fisher_acc = 0.5 * (acc_e + acc_p)
    
    print(f"\nFisher Linear Discriminant separability:")
    print(f"  Expert projection: mean={proj_e.mean():.3f} std={proj_e.std():.3f}")
    print(f"  Policy projection: mean={proj_p.mean():.3f} std={proj_p.std():.3f}")
    print(f"  Gap: {abs(proj_e.mean() - proj_p.mean()):.3f} ({abs(proj_e.mean() - proj_p.mean()) / max(proj_e.std() + proj_p.std(), 0.001):.1f}σ)")
    print(f"  Classification accuracy: {fisher_acc:.4f}")
    
    if fisher_acc > 0.99:
        print(f"\n*** CONFIRMED: Features are TRIVIALLY SEPARABLE ({fisher_acc:.4f}) ***")
        print(f"    Discriminator WILL saturate regardless of regularization.")
        print(f"    Root cause: expert walks at ~0.9 m/s with alternating contacts,")
        print(f"    while random policy stands still with both feet planted.")
        print(f"    This is a DISTRIBUTION problem, not a regularization problem.")
        print(f"\nRecommended solutions:")
        print(f"  1. Add standing/slow-walking clips to diversify expert distribution")
        print(f"  2. Pretrain policy to basic walking BEFORE adding AMP (warm start)")
        print(f"  3. Use curriculum: start AMP loss weight at 0, ramp up after policy walks")
        print(f"  4. Add state-action features (not state-only) for temporal complexity")
    elif fisher_acc > 0.9:
        print(f"\n*** HIGH separability ({fisher_acc:.4f}) — saturation likely ***")
    else:
        print(f"\nSeparability {fisher_acc:.4f} — regularization may help")
    
    return fisher_acc

if __name__ == "__main__":
    main()
