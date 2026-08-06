# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry functions for X1 (29-DOF humanoid).

Isaac Lab joint ordering (BFS articulation traversal):
[
  'lumbar_yaw_joint',             #0   self  negate(yaw)
  'left_hip_pitch_joint',         #1   <->2
  'right_hip_pitch_joint',        #2   <->1
  'lumbar_roll_joint',            #3   self  negate(roll)
  'left_hip_roll_joint',          #4   <->5  negate
  'right_hip_roll_joint',         #5   <->4  negate
  'lumbar_pitch_joint',           #6   self
  'left_hip_yaw_joint',           #7   <->8  negate
  'right_hip_yaw_joint',          #8   <->7  negate
  'left_shoulder_pitch_joint',    #9   <->10
  'right_shoulder_pitch_joint',   #10  <->9
  'left_knee_pitch_joint',        #11  <->12
  'right_knee_pitch_joint',       #12  <->11
  'left_shoulder_roll_joint',     #13  <->14 negate
  'right_shoulder_roll_joint',    #14  <->13 negate
  'left_ankle_pitch_joint',       #15  <->16
  'right_ankle_pitch_joint',      #16  <->15
  'left_shoulder_yaw_joint',      #17  <->18 negate
  'right_shoulder_yaw_joint',     #18  <->17 negate
  'left_ankle_roll_joint',        #19  <->20 negate
  'right_ankle_roll_joint',       #20  <->19 negate
  'left_elbow_pitch_joint',       #21  <->22
  'right_elbow_pitch_joint',      #22  <->21
  'left_elbow_yaw_joint',         #23  <->24 negate
  'right_elbow_yaw_joint',        #24  <->23 negate
  'left_wrist_pitch_joint',       #25  <->26
  'right_wrist_pitch_joint',      #26  <->25
  'left_wrist_roll_joint',        #27  <->28 negate
  'right_wrist_roll_joint',       #28  <->27 negate
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
    """Left-right symmetry for 29-DOF X1 joint data."""
    joint_data_switched = joint_data.clone()

    # left indices <-> right indices
    left_idx  = [1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    right_idx = [2, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

    joint_data_switched[..., left_idx] = joint_data[..., right_idx]
    joint_data_switched[..., right_idx] = joint_data[..., left_idx]

    # negate yaw/roll joints (including self-symmetric lumbar yaw/roll)
    negate_idx = [0, 3, 4, 5, 7, 8, 13, 14, 17, 18, 19, 20, 23, 24, 27, 28]
    joint_data_switched[..., negate_idx] = -1 * joint_data_switched[..., negate_idx]

    return joint_data_switched
