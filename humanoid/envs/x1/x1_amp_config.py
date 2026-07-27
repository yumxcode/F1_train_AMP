# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""X1 AMP task configuration. Inherits the X1 PPO locomotion config and adds the AMP block."""
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO
from humanoid.algo.amp.motion_lib import AMP_OBS_DIM


class X1AMPCfg(X1DHStandCfg):
    """Same robot/control as x1_dh_stand; experiment renamed for AMP."""
    class asset(X1DHStandCfg.asset):
        name = "x1_amp"

    class safety(X1DHStandCfg.safety):
        # Hard joint-limit termination (Gate-C safety): terminate episode if any dof exceeds the
        # tight X1 limits by > 0.05 rad margin. Forces the policy to learn zero-violation behavior
        # to survive, without the soft-penalty safety-vs-performance tradeoff (iter-15 finding).
        # The plain PPO baseline (x1_dh_stand) inherits safety without this flag -> unaffected.
        terminate_on_joint_limit = True
        joint_limit_termination_margin = 0.15  # widened from 0.05 (iter-16: too tight, caused lateral drift 4m);
                                                # 0.15 allows the ~0.014 rad PD overshoot while catching severe violations

    class rewards(X1DHStandCfg.rewards):
        # Keep the task reward (gait/velocity) but down-weight tracking slightly so the
        # style reward (added by the AMP runner) has room to shape the motion.
        class scales(X1DHStandCfg.rewards.scales):
            tracking_lin_vel = 1.5   # 2.5 -> 1.5 (style reward now contributes)
            ref_joint_pos = 0.3     # 0.6 -> 0.3
            # dof_pos_limits scale stays at -10 (the -30 experiment, iter-15, degraded vx tracking
            # without sufficiently reducing violations; hard termination handles safety instead)


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
        disc_hidden_dims = [1024, 512]
        disc_lr = 1e-4
        disc_grad_penalty_coef = 5.0
        disc_train_iters = 2           # discriminator updates per PPO iteration
        disc_batch_size = 4096
        # Style reward mixing.
        style_weight = 1.0
        expert_logit_ema_decay = 0.95
        expert_logit_ema_init = 0.0
        # Policy transition replay buffer.
        policy_buffer_capacity = 1_000_000
        # Expert motion clips. Real training (amp_training) uses the Gate-B validated
        # GMR retargeted clip (https://github.com/Roboparty/GMR methodology).
        # The synthetic smoke clip remains available for Gate-A smoke.
        expert_clip_paths = [
            "{LEGGED_GYM_ROOT_DIR}/data/retarget_gmr/x1_walk_retargeted.npz",
        ]
