# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""X1 AMP task configuration. Inherits the X1 PPO locomotion config and adds the AMP block."""
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO
from humanoid.algo.amp.motion_lib import AMP_OBS_DIM


class X1AMPCfg(X1DHStandCfg):
    """Same robot/control as x1_dh_stand; experiment renamed for AMP."""
    class asset(X1DHStandCfg.asset):
        name = "x1_amp"
        # Reverted from x1_amp_limits.urdf (iter-23 structural pivot FAILED: tight physics
        # limits caused 95% fall rate — PD controller cannot handle hard joint stops at the
        # conservative X1 range within 1500-iter training budget). The original ±π URDF
        # limits + the env's manual_joint_clip (reward/action-clamp only) remain the best
        # achievable balance per the margin sweep (iter-15 to iter-22).

    class safety(X1DHStandCfg.safety):
        # No hard joint-limit termination (iter-16/17/18/23 all failed).
        # Structural pivot (iter-24): position-dependent damping near joint limits.
        # Adds extra PD damping as dof_pos approaches the tight X1 range, decelerating
        # joints before physics-integration overshoot. Control-architecture change.
        terminate_on_joint_limit = False
        joint_limit_termination_margin = 0.15
        limit_damping = True
        limit_damping_margin = 0.80  # widened from 0.85 (activate in outer 20% vs 15%)
        limit_damping_gain = 80.0    # increased from 50 (stronger damping near limits)

    class rewards(X1DHStandCfg.rewards):
        # Keep task reward strong enough for 85%+ speed tracking.
        class scales(X1DHStandCfg.rewards.scales):
            tracking_lin_vel = 2.0   # 1.5 -> 2.0: restore for 85% speed tracking target
            ref_joint_pos = 0.3     # 0.6 -> 0.3


class X1AMPCfgPPO(X1DHStandCfgPPO):
    seed = 5
    runner_class_name = 'AMPOnPolicyRunner'
    experiment_name = 'x1_amp'

    class runner(X1DHStandCfgPPO.runner):
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 100
        experiment_name = 'x1_amp'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

    class amp:
        # Discriminator feature contract (must equal MotionLib.AMP_OBS_DIM == 35).
        disc_input_dim = AMP_OBS_DIM
        # v2 anti-domination: spectral norm + symmetric GP + label smoothing + reduced capacity
        disc_hidden_dims = [512, 256]   # was [1024, 512] — over-parameterized for 35-dim input
        disc_lr = 5e-5
        disc_grad_penalty_coef = 10.0   # was 5.0 — symmetric R1 on both expert+policy
        disc_label_smoothing = 0.1     # soft labels prevent logit saturation
        disc_train_iters = 1
        disc_batch_size = 4096
        # Style reward mixing.
        style_weight = 1.0
        expert_logit_ema_decay = 0.95
        expert_logit_ema_init = 0.0
        # Policy transition replay buffer.
        policy_buffer_capacity = 1_000_000
        # Expert motion clips (Gate-B validated GMR retargeted clip).
        expert_clip_paths = [
            "{LEGGED_GYM_ROOT_DIR}/data/retarget_gmr/x1_walk_retargeted.npz",
        ]
