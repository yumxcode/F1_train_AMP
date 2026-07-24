# F1 AMP — Task Specification (frozen Gate-C thresholds, iteration 3)

**Status:** FROZEN at first entry to amp_training (iteration 3). Thresholds derived
from X1 size, control frequency (100 Hz policy), and the GMR retargeted reference clip.
Subsequent iterations must NOT lower these to claim success.

## 1. Fixed evaluation protocol

- **Eval seeds (≥5):** [5, 17, 42, 123, 2024] — seeds NOT used for hyperparameter search.
- **Episodes per seed:** 4 (bounded for remote cost).
- **Episode length:** 24 s nominal (the env episode_length_s). Goal: ≥60 s stable in §6.3.
- **Nominal command:** forward velocity vx=0.5 m/s (within the reference clip's ~1.6 m/s
  scaled-walk range; 0.5 m/s is the sustainable tracking target for X1 size).
- **Terrain:** flat (curriculum locked at level 0 for eval; training uses curriculum).
- **Domain randomization:** eval runs WITH the cfg.domain_rand enabled (same as training)
  to measure robustness, plus a separate Sim2Sim check (§6.3.5).
- **Noise:** enabled (cfg.noise.add_noise=True).

## 2. Gate-C acceptance thresholds (frozen)

All measured with mean over the 5 seeds × 4 episodes (20 episodes), reporting mean/std/
worst. Per §6, the four metric groups:

### 2a. Imitation quality
| Metric | Threshold | Source |
|---|---|---|
| joint_pos_err_ref (rad, mean) | ≤ 0.30 | X1 has conservative locomotion joint windows (±0.3–1.45 rad); 0.30 rad is the smallest window margin |
| base_height (m, mean) | 0.55–0.80 | X1 standing height 0.70 m (retarget root_h_mean=0.700) |
| base_pitch (deg, mean |abs|) | ≤ 10.0 | stable walking keeps torso near upright |
| AMP style reward | reported (no hard gate; must be > 0 and non-decreasing vs baseline) | discriminator-driven |

### 2b. Walking task
| Metric | Threshold |
|---|---|
| vx_track_err (m/s, mean) | ≤ 0.25 (tracking 0.5 m/s nominal; half-velocity tolerance) |
| lateral_drift (m, mean over episode) | ≤ 0.5 |
| yaw_drift (deg/s, mean |abs|) | ≤ 15 |

### 2c. Stability
| Metric | Threshold |
|---|---|
| fall_rate (fraction, mean) | ≤ 0.05 (§6: episode success ≥95%, fall ≤5%) |
| episode_length_steps (mean) | ≥ 2000 steps (=20 s at 100 Hz; partial toward 60 s goal) |
| dof_pos_limit_viol_total | 0 (no hard joint-limit violations across all episodes) |
| foot_contact alternation | both feet show alternating stance (not stuck; reported) |

### 2d. Safety & deployability
| Metric | Threshold |
|---|---|
| NaN/Inf in any rollout | 0 (hard fail if any) |
| Sim2Sim (Isaac→MuJoCo) | checkpoint exportable (JIT/ONNX) + plays in sim2sim without immediate fall |

## 3. Training configuration (frozen for this run)
- Task: x1_amp (AMPOnPolicyRunner, Gate-A validated).
- Expert: data/retarget_gmr/x1_walk_retargeted.npz (Gate-B validated GMR retarget).
- Seeds: train seed=5 (single long run, then fixed 5-seed eval).
- max_iterations: start 2000 (bounded; extend only if improving, per §10).
- num_envs: 4096, rl_device cuda:0.

## 4. Sim2Sim
- Export checkpoint to JIT (export_policy_dh.py) and run sim2sim.py (MuJoCo).
- Must not immediately fall; counts as Gate-C item even if shorter than 60 s.

## 5. Reproducibility
- code_revision: git commit hash on branch amp-gateA-smoke.
- dataset_revision: GMR retarget sha256 (Gate-B validated).
- All eval metrics + logs committed to data/amp_eval/.
