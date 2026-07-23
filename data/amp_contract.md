# AMP (Adversarial Motion Priors) Contract — AgiBot X1 ("F1" code-name)

This is the authoritative expert/policy feature contract for the `x1_amp` task.
It is the human-readable source for `state/amp_contract.json`. Single source of
truth for feature layout: `humanoid/algo/amp/motion_lib.py`.

Verification commands (all exit 0 on PASS):
- `python humanoid/scripts/test_amp_contract.py`        — schema/loader (numpy-only, 21 checks)
- `python humanoid/scripts/amp_algo_smoke_numpy.py`      — disc train + style-reward sign (numpy-only, 10 checks)
- `python humanoid/scripts/check_amp_runner_config.py`   — runner path/log wiring (numpy-only, 4 checks)
- `python humanoid/scripts/smoke_amp.py --headless`      — FULL Isaac-Gym closed-loop smoke (GPU host)

## 1. Robot identity & joint order (F1 = AgiBot X1)
- URDF: `resources/robots/x1/urdf/x1.urdf`; MJCF: `resources/robots/x1/mjcf/xyber_x1_flat.xml`
- Actuated DOF = 12 (legs only; arms/waist fixed in the locomotion URDF).
- Authoritative joint order == URDF dof_names order == action order == expert clip order:

| idx | joint                       | idx | joint                        |
|-----|-----------------------------|-----|------------------------------|
| 0   | left_hip_pitch_joint        | 6   | right_hip_pitch_joint        |
| 1   | left_hip_roll_joint         | 7   | right_hip_roll_joint         |
| 2   | left_hip_yaw_joint          | 8   | right_hip_yaw_joint          |
| 3   | left_knee_pitch_joint       | 9   | right_knee_pitch_joint       |
| 4   | left_ankle_pitch_joint      | 10  | right_ankle_pitch_joint      |
| 5   | left_ankle_roll_joint       | 11  | right_ankle_roll_joint       |

Default dof pos (rad): `(0.4,0.05,-0.31,0.49,-0.21,0.0, -0.4,-0.05,0.31,0.49,-0.21,0.0)`
== `X1DHStandCfg.init_state.default_joint_angles` == `X1_DEFAULT_DOF_POS`.
Foot link = `ankle_roll` (2 feet). Control: P, 100 Hz policy (sim dt=1e-3, decimation=10).

## 2. AMP observation / transition feature (F = 35)
Expert and policy MUST produce the identical 35-dim feature via the SAME builder
(`motion_lib.build_amp_obs_numpy` expert-side; `X1AMPEnv.compute_amp_obs` policy-side).
All velocities are in the base (pelvis) frame; expert uses finite-diff over real `dt`.

| block            | offset | dim | source                                              |
|------------------|--------|-----|-----------------------------------------------------|
| base_lin_vel     | 0:3    | 3   | root world lin vel rotated into base frame          |
| base_ang_vel     | 3:6    | 3   | root world ang vel rotated into base frame          |
| projected_gravity| 6:9    | 3   | world gravity (0,0,-9.81) rotated into base frame   |
| dof_pos_rel      | 9:21   | 12  | joint pos - default (X1 order)                      |
| dof_vel          | 21:33  | 12  | joint velocity (X1 order)                           |
| foot_contact     | 33:35  | 2   | {0,1} per foot (policy: contact_force>5N)           |

State-only discriminator (no action in the feature), matching standard AMP.
Expert `dt` = 1/100 s (resampled to the policy rate); velocities use this `dt`.

## 3. Expert clip file format (npz, retargeted — Gate B output)
Keys: `root_translation (N,3)` world meters; `root_rotation (N,4)` xyzw quaternion;
`joint_positions (N,12)` X1 order rad; `foot_contact (N,2)` {0,1}; `fps` scalar Hz.
Optional `ref_dof_pos_rel (N,6)` (swing-pitch deltas, for env ref reward, not a disc feature).
Gate-A smoke uses the SYNTHETIC clip `data/smoke_expert/walk_smoke.npz` (NOT Gate-B data).

## 4. Discriminator
MLP `[1024,512]->1` (ELU), single logit `D(x)`. Loss = standard GAN + R1 gradient penalty
on policy samples (`disc_grad_penalty_coef=5.0`). Trained `disc_train_iters` (2) times per
PPO iteration on fresh expert minibatch (`disc_batch_size=4096`) vs policy replay buffer.

## 5. Policy transition replay buffer
Circular, capacity `policy_buffer_capacity=1e6`, dim 35. Filled each step from
`env.compute_amp_obs()`. Transient reservoir — NOT persisted in checkpoints (avoids
all-zero-sample corruption on resume; refills within one PPO iteration).

## 6. Style reward & task-reward combination
`r_style = clamp(1 - 0.25*(ema(D(expert)) - D(policy)), 0, 2)`. Sign: reward RISES as the
policy logit rises toward the expert (more expert-like). Added to the env task reward
BEFORE `process_env_step` so GAE observes it: `reward += style_weight * r_style`
(`style_weight=1.0`). `ema(D(expert))` tracked with `expert_logit_ema_decay=0.95`.

## 7. Normalization
AMP features are NOT running-normalized (raw base-frame quantities + binary contacts).
PPO observation uses the env's `num_single_obs=47` stacked history (unchanged from baseline).
AMP style reward is added in env reward units; task reward scales per `x1_amp_config`.

## 8. Checkpoint / resume / export
Checkpoint dict keys: `model_state_dict`, `optimizer_state_dict`, `es_optimizer_state_dict`,
`disc_state_dict`, `disc_optimizer_state_dict`, `expert_logit_ema`, `policy_amp_buffer_state`,
`amp_style_weight`, `iter`. Resume restores discriminator + its optimizer + expert-logit EMA
(buffers intentionally reset). Export: JIT/ONNX via the unchanged policy contract
(`ActorCriticDH`); AMP does not change the deployed observation/action schema.

## 9. Task registration & training entry
- `x1_amp`: `X1AMPEnv` + `X1AMPCfg`/`X1AMPCfgPPO`, runner `AMPOnPolicyRunner`.
- `x1_dh_stand`: plain PPO baseline, UNCHANGED (regression control).
- Train: `python humanoid/scripts/train_amp.py --task=x1_amp --headless`
- Play/export/Sim2Sim: same scripts as baseline (AMP only changes the runner/reward).
