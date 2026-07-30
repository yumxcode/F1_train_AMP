# SPDX-License-Identifier: BSD-3-Clause
"""AMP discriminator. State-only (no action): expert/policy share the amp_obs feature."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Single-logit discriminator over AMP transition features.

    Input dim = AMP_OBS_DIM (see data/amp_contract.md). Produces a scalar logit D(x).
    """

    def __init__(self, input_dim: int, hidden_dims=(1024, 512), activation="elu"):
        super().__init__()
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.trunk = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logit D(x) of shape (batch, 1)
        return self.trunk(x)

    def compute_reward(self, policy_logit: torch.Tensor, expert_logit_ema: torch.Tensor,
                       reward_clamp: float = 2.0) -> torch.Tensor:
        """AMP style reward (Peng et al. 2021 ASE/AMP formulation).

        Uses exp(-0.25 * max(0, ema_e - p_logit)) which NEVER collapses to zero even
        when the discriminator dominates. This preserves a non-trivial gradient when
        the policy is far from expert (prevents the "dead style reward" failure mode
        observed in iter-1 training where the clamped linear reward hit exactly 0).

        r = exp(-0.25 * max(0, ema(D(e)) - D(p)))

        Checks:
          * policy == expert  -> D(p) ~ D(e)        -> r ~ 1.0
          * policy clearly fake -> D(p) -> -inf     -> r -> exp(-inf) -> 0 (but never EXACTLY 0)
          * policy fools D     -> D(p) > D(e)       -> r ~ 1.0 (capped)

        The previous linear-clamp formula r=clamp(1-0.25*(ema-p),0,2) clamped to exactly
        0 when the logit gap exceeded 4 (observed at iter ~100), killing the style
        gradient entirely. The exponential formulation preserves gradient everywhere.
        """
        logit_diff = torch.clamp(expert_logit_ema - policy_logit.squeeze(-1), min=0.0)
        r = torch.exp(-0.25 * logit_diff)
        return r.clamp(0.0, reward_clamp)


def compute_disc_loss(disc: Discriminator, expert: torch.Tensor, policy: torch.Tensor,
                      grad_penalty_coef: float = 5.0) -> Dict[str, torch.Tensor]:
    """AMP discriminator loss + gradient penalty on policy samples.

    Returns dict with loss, expert_loss, policy_loss, accuracy, expert_logit, policy_logit,
    grad_penalty.
    """
    d_e = disc(expert)
    d_p = disc(policy)
    expert_loss = nn.functional.softplus(-d_e).mean()
    policy_loss = nn.functional.softplus(d_p).mean()
    # gradient penalty on policy samples (R1-style on the fake distribution)
    grad_penalty = torch.tensor(0.0, device=expert.device)
    if grad_penalty_coef > 0:
        policy_g = policy.detach().requires_grad_(True)
        d_pg = disc(policy_g)
        grads = torch.autograd.grad(
            outputs=d_pg.sum(), inputs=policy_g,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = (grads.pow(2).reshape(grads.shape[0], -1).sum(-1)).mean()
    loss = expert_loss + policy_loss + grad_penalty_coef * grad_penalty
    with torch.no_grad():
        acc = ((d_e > 0).float().mean() + (d_p < 0).float().mean()) * 0.5
    return {
        "loss": loss,
        "expert_loss": expert_loss,
        "policy_loss": policy_loss,
        "grad_penalty": grad_penalty,
        "accuracy": acc,
        "expert_logit": d_e.mean(),
        "policy_logit": d_p.mean(),
    }
