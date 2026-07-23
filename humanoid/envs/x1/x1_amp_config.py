# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""X1 AMP task configuration. Inherits the X1 PPO locomotion config and adds the AMP block."""
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg, X1DHStandCfgPPO
from humanoid.algo.amp.motion_lib import AMP_OBS_DIM


class X1AMPCfg(X1DHStandCfg):
    """Same robot/control as x1_dh_stand; experiment renamed for AMP."""
    class asset(X1DHStandCfg.asset):
        name = "x1_amp"

    class rewards(X1DHStandCfg.rewards):
        # Keep the task reward (gait/velocity) but down-weight tracking slightly so the
        # style reward (added by the AMP runner) has room to shape the motion.
        class scales(X1DHStandCfg.rewards.scales):
            tracking_lin_vel = 1.5   # 2.5 -> 1.5 (style reward now contributes)
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
        # retargeted clip. The synthetic smoke clip remains available for Gate-A smoke.
        expert_clip_paths = [
            "{LEGGED_GYM_ROOT_DIR}/data/retarget/x1_walk_retargeted.npz",
        ]
