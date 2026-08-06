# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry functions for X1 (29-DOF humanoid).

Runtime joint ordering (VERIFIED from gradmotion retarget run):
[
  'left_hip_pitch_joint',      #0   <->1
  'lumbar_yaw_joint',          #1   self  negate(yaw)
  'right_hip_pitch_joint',     #2   <->0
  'left_hip_roll_joint',       #3   <->4  negate
  'lumbar_roll_joint',         #4   self  negate(roll)
  'right_hip_roll_joint',      #5   <->3  negate
  'left_hip_yaw_joint',        #6   <->8  negate
  'lumbar_pitch_joint',        #7   self
  'right_hip_yaw_joint',       #8   <->6  negate
  'left_knee_pitch_joint',     #9   <->12
  'left_shoulder_pitch_joint', #10  <->11
  'right_shoulder_pitch_joint',#11  <->10
  'right_knee_pitch_joint',    #12  <->9
  'left_ankle_pitch_joint',    #13  <->16
  'left_shoulder_roll_joint',  #14  <->15 negate
  'right_shoulder_roll_joint', #15  <->14 negate
  'right_ankle_pitch_joint',   #16  <->13
  'left_ankle_roll_joint',     #17  <->20 negate
  'left_shoulder_yaw_joint',   #18  <->19 negate
  'right_shoulder_yaw_joint',  #19  <->18 negate
  'right_ankle_roll_joint',    #20  <->17 negate
  'left_elbow_pitch_joint',    #21  <->22
  'right_elbow_pitch_joint',   #22  <->21
  'left_elbow_yaw_joint',      #23  <->24 negate
  'right_elbow_yaw_joint',     #24  <->23 negate
  'left_wrist_pitch_joint',    #25  <->26
  'right_wrist_pitch_joint',   #26  <->25
  'left_wrist_roll_joint',     #27  <->28 negate
  'right_wrist_roll_joint',    #28  <->27 negate
]
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

NUM_DOFS = 29


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(env, obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(env, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _history_length(env: ManagerBasedRLEnv, group_name: str) -> int:
    cfg = getattr(env, "unwrapped", env).cfg
    history_length = getattr(getattr(cfg.observations, group_name), "history_length", 0)
    return history_length if history_length is not None and history_length > 0 else 1


def _transform_policy_obs_left_right(env, obs: torch.Tensor) -> torch.Tensor:
    obs_shape = obs.shape
    history_length = _history_length(env, "policy")
    expected_dim = history_length * (3 + 3 + 3 + NUM_DOFS + NUM_DOFS + NUM_DOFS)
    assert obs_shape[-1] == expected_dim, f"Expected policy obs dim {expected_dim}, got {obs_shape[-1]}."
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = NUM_DOFS * history_length
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    return obs


def _transform_critic_obs_left_right(env, obs: torch.Tensor) -> torch.Tensor:
    obs_shape = obs.shape
    history_length = _history_length(env, "critic")
    expected_dim = history_length * (3 + 3 + 3 + 3 + NUM_DOFS + NUM_DOFS + NUM_DOFS)
    assert obs_shape[-1] == expected_dim, f"Expected critic obs dim {expected_dim}, got {obs_shape[-1]}."
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = NUM_DOFS * history_length
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
    return obs


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, NUM_DOFS)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Left-right symmetry for 29-DOF X1 joint data (runtime order)."""
    joint_data_switched = joint_data.clone()

    # left indices <-> right indices (based on runtime BFS order)
    left_idx  = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 23, 25, 27]
    right_idx = [2, 5, 8, 12, 11, 16, 15, 20, 19, 22, 24, 26, 28]

    joint_data_switched[..., left_idx] = joint_data[..., right_idx]
    joint_data_switched[..., right_idx] = joint_data[..., left_idx]

    # negate yaw/roll joints (including self-symmetric lumbar yaw/roll)
    # Runtime indices: lumbar_yaw=1, lumbar_roll=4,
    # left_hip_roll=3, right_hip_roll=5, left_hip_yaw=6, right_hip_yaw=8,
    # left_shoulder_roll=14, right_shoulder_roll=15,
    # left_shoulder_yaw=18, right_shoulder_yaw=19,
    # left_ankle_roll=17, right_ankle_roll=20,
    # left_elbow_yaw=23, right_elbow_yaw=24,
    # left_wrist_roll=27, right_wrist_roll=28
    negate_idx = [1, 4, 3, 5, 6, 8, 14, 15, 17, 18, 19, 20, 23, 24, 27, 28]
    joint_data_switched[..., negate_idx] = -1 * joint_data_switched[..., negate_idx]

    return joint_data_switched
