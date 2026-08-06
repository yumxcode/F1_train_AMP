# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from robolab import ROBOLAB_ROOT_DIR
import torch
from robolab.tasks.manager_based.amp.mdp.symmetry import x1


@configclass
class RslRlAmpCfg:
    disc_obs_buffer_size: int = 1000
    grad_penalty_scale: float = 10.0
    disc_trunk_weight_decay: float = 1.0e-4
    disc_linear_weight_decay: float = 1.0e-2
    disc_learning_rate: float = 1.0e-5
    disc_max_grad_norm: float = 1.0

    @configclass
    class AMPDiscriminatorCfg:
        hidden_dims: list[int] = MISSING
        activation: str = "elu"
        style_reward_scale: float = 1.0
        task_style_lerp: float = 0.0

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOAmp"
    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()


@configclass
class X1AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "x1_amp"
    wandb_project = "x1_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=x1.compute_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=1.5,
                task_style_lerp=0.6,
            ),
            loss_type="LSGAN",
        ),
    )
