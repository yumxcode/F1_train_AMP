# AMP Gate-A — Complete Validation Report

**Phase:** amp_conversion (Gate A)  |  **Date:** 2026-07-23  |  **Outcome: PASS**

## Robot identity (F1 = AgiBot X1)
- F1 is a code-name for the **AgiBot X1** (xyber_x1) robot; the X1 URDF/MJCF/meshes
  ARE the F1 assets. 12 actuated DOF (legs); arms/waist fixed in the locomotion URDF.
- URDF: resources/robots/x1/urdf/x1.urdf; MJCF: resources/robots/x1/mjcf/xyber_x1_flat.xml
- No separate F1 model exists or is needed (precondition F1-asset-identity: RESOLVED).

## Code revision
- Remote branch: https://github.com/yumxcode/F1_train_AMP.git branch `amp-gateA-smoke`
- HEAD: `4cd9d1264bdc16f57bc305633ddc5e1301e43c0d`
- Plain PPO baseline (`x1_dh_stand`) UNCHANGED (regression control); AMP task `x1_amp` added.

## Dataset revision
- smoke_expert (synthetic X1-space walk, Gate-A only, NOT Gate-B retargeted data):
  data/smoke_expert/walk_smoke.npz, sha256=32ecc1d2115fb5d447369ca410c29b0ee427f56a
- Raw reference SMPL-X source (unretargeted, Gate-B input): data/0008_normal_walk4_stageii.npz

## Headless validation (numpy-only, all PASS) — local
1. amp_algo_smoke_numpy.py : 10/10 PASS
   - discriminator params update (mean|delta|=4.68e-02), finite, loss 4.342->0.059,
     accuracy 0.998, expert_logit>policy_logit, style-reward sign correct
     (expert~0.99 > fake~0.012, monotonic increasing in logit).
2. test_amp_contract.py : 21/21 PASS
   - AMP_OBS_DIM=35, block layout, joint order==URDF actuated order, expert default
     == env default dof vec, MotionLib 120->100Hz resample, finite, joint-range ok.
3. check_amp_runner_config.py : 4/4 PASS
   - {LEGGED_GYM_ROOT_DIR} placeholder resolves; MotionLib loads; learn() carries obs_std/obs_mean.

## Remote GPU validation (Gradmotion, Isaac Gym, 1x4090D) — all PASS
- Smoke task: TASK_20260723_029 (account vabata7853@bevriz.com, image isaac-gym-gm-v19,
  PyTorch 2.4.1, Python 3.8). SDK: `status updated to: Completed, ret:True`.
- Closed loop (env + discriminator + style reward + PPO), 5 iters x 64 envs:
  - `[AMP] disc_input_dim=35 style_weight=1.0 expert_clips=1 expert_samples=299`
  - `[smoke_amp] disc_param_mean_delta=3.496e-04 finite=True ckpt=True`
- Checkpoint save/resume:
  - `[smoke_amp] resume-ckpt: keys_ok=True disc_match=True ema_match=True
     (ckpt=0.029478 post=0.029478) iter=4`
  - Checkpoint model_0.pt + model_5.pt saved + uploaded.
- Play (inference rollout with trained policy):
  - `[smoke_amp] play: 5 inference steps completed OK`
- Final: `[smoke_amp] SMOKE OK: AMP closed loop executes, disc updates, ckpt saved,
  resume-keys verified, play verified.`

## Independent discriminator + style reward evidence (NOT a PPO rename)
- x1_amp uses AMPOnPolicyRunner (own discriminator + style reward + replay buffer);
  x1_dh_stand uses DHOnPolicyRunner (plain PPO), untouched. Training log shows
  independent `AMP/` scalars and `[AMP]` construction line absent from baseline.

## Gate A checklist (amp_loop.md §4) — all satisfied
- [x] static checks / import / config parse pass (headless 35/35 checks)
- [x] bounded smoke train, AMP modules execute, params update (GPU disc_param_delta>0)
- [x] expert/policy schema auto-compare pass, no NaN/Inf / OOB / bad broadcasting (contract 21/21)
- [x] checkpoint save/resume verified (ckpt keys_ok + disc_match + ema_match)
- [x] play verified (5 inference steps OK)
- [x] AMP task independent of PPO task (own discriminator + style reward in logs)
- [x] amp_contract.md formed; commands/logs/results in findings
