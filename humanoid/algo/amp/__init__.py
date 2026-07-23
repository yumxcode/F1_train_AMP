# SPDX-License-Identifier: BSD-3-Clause
# AMP (Adversarial Motion Priors) closed-loop for AgiBot X1 ("F1" code-name).
# See data/amp_contract.md for the authoritative expert/policy feature contract.
from .motion_lib import MotionLib, X1_JOINT_NAMES, X1_DEFAULT_DOF_POS, AMP_OBS_DIM, AMP_BLOCKS
from .discriminator import Discriminator, compute_disc_loss
from .replay_buffer import AMPReplayBuffer
from .amp_on_policy_runner import AMPOnPolicyRunner

__all__ = [
    "MotionLib",
    "Discriminator",
    "compute_disc_loss",
    "AMPReplayBuffer",
    "AMPOnPolicyRunner",
    "X1_JOINT_NAMES",
    "X1_DEFAULT_DOF_POS",
    "AMP_OBS_DIM",
    "AMP_BLOCKS",
]
