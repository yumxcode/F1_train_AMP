#!/usr/bin/env python3
"""
Diagnostic: verify that commands written into env.commands[:] propagate
through step() → compute_observations() → obs_buf.

Usage: python scripts/diag_obs.py --task=x1_dh_stand --headless --num_envs=1
"""
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humanoid.utils import get_args, task_registry
from humanoid.envs import *


def diag():
    args = get_args()
    # fixed seed for reproducibility
    args.seed = 42

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False

    # env init
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(f"[DIAG] env created, num_envs={env.num_envs} obs_dim={env.num_obs}")

    # reset (first step with zero actions)
    obs, privileged_obs = env.reset()
    print(f"[DIAG] after reset — obs[0, -5:] = {obs[0, -5:].tolist()}")
    print(f"[DIAG] after reset — commands[0] = {env.commands[0].tolist()}")

    # Set commands explicitly
    env.commands[0, 0] = 0.5   # forward velocity
    env.commands[0, 1] = 0.0
    env.commands[0, 2] = 0.0
    print(f"[DIAG] SET commands[0] = {env.commands[0].tolist()}")

    # Step with zero actions, inspect returned obs
    for i in range(5):
        obs, critic_obs, rews, dones, infos = env.step(torch.zeros((1, env.num_actions), device=env.device))
        
        # The current obs's last frame (latest single-obs frame)
        last_frame = obs[0, -47:] if env.num_single_obs == 47 else obs[0, -env.num_single_obs:]
        print(f"[DIAG] step {i}: obs[0, -47:-42]={last_frame[:5].tolist()}  cmd[0]={env.commands[0,:3].tolist()}")
        
        if dones[0]:
            print(f"[DIAG] step {i}: env 0 done, resetting...")
            # After reset the commands get resampled — re-set
            env.commands[0, 0] = 0.5
            env.commands[0, 1] = 0.0
            env.commands[0, 2] = 0.0

    print("[DIAG] done.")


if __name__ == '__main__':
    diag()
