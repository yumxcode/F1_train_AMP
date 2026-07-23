# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""X1 AMP environment. Extends X1DHStandEnv with the AMP feature builder.

``compute_amp_obs`` is the policy-side mirror of ``motion_lib.build_amp_obs_numpy``: both
produce the identical 35-dim feature (base lin/ang vel in base frame, projected gravity,
dof_pos rel default, dof_vel, foot contact). The AMP runner reads this to (a) fill the policy
replay buffer and (b) compute the style reward from the discriminator. Joint order is the
URDF dof_names order; the default pose is the X1 default used by the expert loader.
"""
from isaacgym.torch_utils import *
import torch

from humanoid.envs.x1.x1_dh_stand_env import X1DHStandEnv
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.algo.amp.motion_lib import AMP_OBS_DIM, X1_DEFAULT_DOF_POS

# world gravity vector identical to the expert builder (magnitude 9.81) for schema parity
_GRAVITY_WORLD = (0.0, 0.0, -9.81)


class X1AMPEnv(X1DHStandEnv):
    """X1 locomotion env + AMP discriminator feature builder."""

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # default dof vector broadcast over envs, in X1 joint order (== expert default)
        self._amp_default = torch.tensor(X1_DEFAULT_DOF_POS, dtype=torch.float, device=self.device).unsqueeze(0)
        self._gravity_world = torch.tensor(_GRAVITY_WORLD, dtype=torch.float, device=self.device).unsqueeze(0)
        # cached gravity over envs for quat_rotate_inverse
        self._gravity_world_n = self._gravity_world.expand(self.num_envs, -1)
        self.amp_obs_dim = AMP_OBS_DIM

    @torch.no_grad()
    def compute_amp_obs(self) -> torch.Tensor:
        """Policy AMP feature (N, AMP_OBS_DIM). Mirrors motion_lib.build_amp_obs_numpy exactly.

        Layout (see data/amp_contract.md §2):
          base_lin_vel(3) base_ang_vel(3) projected_gravity(3)
          dof_pos_rel(12) dof_vel(12) foot_contact(2)
        """
        base_quat = self.root_states[:, 3:7]
        # refresh from authoritative root_states so the feature is always current
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self._gravity_world_n)
        dof_pos_rel = self.dof_pos - self._amp_default
        foot_contact = (self.contact_forces[:, self.feet_indices, 2] > 5.0).float()
        amp_obs = torch.cat([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            dof_pos_rel,
            self.dof_vel,
            foot_contact,
        ], dim=-1)
        return amp_obs
