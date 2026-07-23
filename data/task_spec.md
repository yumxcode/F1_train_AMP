# F1 (AgiBot X1) AMP Task Specification — FROZEN at amp_training entry

**Robot:** AgiBot X1 ("F1" code-name), 12-DOF legs, URDF dof_names order.
**Control:** P-control, 100 Hz policy (sim dt=1e-3, decimation=10).
**Reference clip:** SMPL-X 0008_normal_walk4 retargeted (data/retarget/x1_walk_retargeted.npz),
forward velocity ~1.6 m/s, root height 0.70m, 405 frames @ 120Hz.
**Frozen at:** amp_training iteration 3. **DO NOT lower these thresholds to declare success.**

## 1. Fixed evaluation set (frozen, used for ALL gate decisions)
- **Seeds:** `[5, 17, 42, 123, 2024]` (5 seeds, fixed, NOT used for tuning).
- **Eval episodes per seed:** 10 (nominal), report mean/std/worst.
- **Command (nominal):** forward velocity 0.5 m/s (vx), vy=0, yaw=0.
    - Command range explored during eval: vx ∈ {0.2, 0.5, 1.0} m/s, vy ∈ [-0.1,0.1], yaw ∈ [-0.2,0.2].
- **Terrain:** flat + trimesh (curriculum disabled at eval), heightfield off.
- **Episode length:** 60 s (must hold without fall for full eval).
- **Domain randomization:** ON at eval (the trained DR config; tests robustness).

## 2. Frozen acceptance thresholds (Gate C)
These are derived from F1/X1 size (0.7m hip height, 0.305m links), control (100Hz),
and the reference walk (1.6 m/s). Reporting mean over 5 seeds unless noted.

### 2a. Imitation quality
| metric | target | threshold (must meet) |
|---|---|---|
| AMP style reward | high | ≥ 0.6 (clipped reward scale [0,2]) |
| reference joint pos error (rad, mean) | low | ≤ 0.35 |
| reference joint vel error (rad/s) | low | ≤ 3.0 |
| root height error (m) | low | ≤ 0.05 |
| root orientation error (deg) | low | ≤ 15 |
| gait cycle period error (vs ref) | low | ≤ 25% |
| foot contact phase consistency (L/R antiphase) | high | cross-corr ≤ -0.3 |

### 2b. Walking task
| metric | threshold |
|---|---|
| forward velocity tracking error (m/s) | ≤ 0.25 at vx=0.5 |
| lateral drift (m over 60s) | ≤ 1.0 |
| yaw drift (rad over 60s) | ≤ 0.5 |
| distance completion rate (actual/target) | ≥ 0.7 |

### 2c. Stability
| metric | threshold |
|---|---|
| episode completion rate (60s no fall) | ≥ 0.95 |
| fall rate | ≤ 0.05 |
| continuous no-fall duration | ≥ 60s nominal |
| base roll/pitch (deg, mean) | ≤ 10 |
| foot slip (m/s stance, mean) | ≤ 0.15 |
| abnormal double-flight / double-stance ratio | ≤ 0.30 |

### 2d. Safety & deployability
| metric | threshold |
|---|---|
| joint limit violation rate | ≤ 0.01 |
| velocity/torque saturation rate | ≤ 0.10 |
| non-foot collision rate | 0 |
| self-collision rate | 0 |
| NaN/Inf in any rollout | 0 |
| Isaac-Gym -> MuJoCo Sim2Sim: stable walk ≥ 20s | PASS |

## 3. Minimum acceptance protocol (amp_loop.md §6)
- ≥5 fixed seeds; report mean, std, worst seed. No cherry-picking best video/checkpoint.
- Nominal env, reference command: continuous stable walk ≥ 60s, no fall, no NaN/Inf, no hard joint-limit violation.
- Fixed batch eval: episode success ≥ 95%, fall rate ≤ 5%.
- Repeat eval under controlled DR + MuJoCo Sim2Sim; must meet frozen Sim2Sim threshold.
- Final checkpoint: resumable, playable, exportable; exported policy obs/action contract == training.

## 4. Training config (locked for this run)
- Task: x1_amp (AMPOnPolicyRunner). Baseline: x1_dh_stand (plain PPO, regression control).
- Expert: data/retarget/x1_walk_retargeted.npz (Gate B validated).
- max_iterations: 15000; num_envs: 4096; seed: 5 (train); eval seeds fixed above.
- Early-stop: plateau ≥ ~3000 iter with no fixed-eval improvement -> terminate, extract findings.

## 5. Trend judgment
Training improvement judged ONLY by fixed eval (above), not training-window reward peaks,
single videos, or discriminator accuracy alone.
