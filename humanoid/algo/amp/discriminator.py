# SPDX-License-Identifier: BSD-3-Clause
"""AMP discriminator with anti-domination regularization.

v2 changes vs original:
  * Spectral normalization on all linear layers (constrains Lipschitz constant).
  * Symmetric R1 gradient penalty on BOTH expert and policy samples.
  * Reduced default capacity [512, 256] (was [1024, 512]) for 35-dim input.
  * Label smoothing (default 0.1) prevents logit saturation.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn


class Discriminator(nn.Module):
    """Single-logit discriminator over AMP transition features."""

    def __init__(self, input_dim: int, hidden_dims=(512, 256), activation="elu",
                 use_spectral_norm: bool = True):
        super().__init__()
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            if use_spectral_norm:
                lin = sn(lin)
            layers += [lin, act()]
            prev = h
        out_lin = nn.Linear(prev, 1)
        if use_spectral_norm:
            out_lin = sn(out_lin)
        layers += [out_lin]
        self.trunk = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def compute_reward(self, policy_logit: torch.Tensor, expert_logit_ema: torch.Tensor,
                       reward_clamp: float = 2.0) -> torch.Tensor:
        """AMP style reward: r = exp(-0.25 * max(0, ema(D(e)) - D(p)))."""
        logit_diff = torch.clamp(expert_logit_ema - policy_logit.squeeze(-1), min=0.0)
        r = torch.exp(-0.25 * logit_diff)
        return r.clamp(0.0, reward_clamp)


def compute_disc_loss(disc: Discriminator, expert: torch.Tensor, policy: torch.Tensor,
                      grad_penalty_coef: float = 10.0,
                      label_smooth: float = 0.1) -> Dict[str, torch.Tensor]:
    """AMP discriminator loss with symmetric R1 GP + label smoothing.

    Anti-domination: gradient penalty applied to BOTH expert and policy (symmetric).
    Label smoothing mixes softplus losses to prevent over-confident logits.
    """
    d_e = disc(expert)
    d_p = disc(policy)

    # Label-smoothed softplus losses
    if label_smooth > 0:
        # real (expert): (1-2s)*softplus(-d_e) + s*softplus(d_e)
        expert_loss = (1 - 2*label_smooth) * nn.functional.softplus(-d_e).mean() + \
                       label_smooth * nn.functional.softplus(d_e).mean()
        # fake (policy): (1-2s)*softplus(d_p) + s*softplus(-d_p)
        policy_loss = (1 - 2*label_smooth) * nn.functional.softplus(d_p).mean() + \
                       label_smooth * nn.functional.softplus(-d_p).mean()
    else:
        expert_loss = nn.functional.softplus(-d_e).mean()
        policy_loss = nn.functional.softplus(d_p).mean()

    # Symmetric R1 gradient penalty on BOTH distributions
    grad_penalty = torch.tensor(0.0, device=expert.device)
    if grad_penalty_coef > 0:
        # Policy samples
        policy_g = policy.detach().requires_grad_(True)
        d_pg = disc(policy_g)
        grads_p = torch.autograd.grad(
            outputs=d_pg.sum(), inputs=policy_g,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp_p = (grads_p.pow(2).reshape(grads_p.shape[0], -1).sum(-1)).mean()

        # Expert samples (symmetric penalty)
        expert_g = expert.detach().requires_grad_(True)
        d_eg = disc(expert_g)
        grads_e = torch.autograd.grad(
            outputs=d_eg.sum(), inputs=expert_g,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp_e = (grads_e.pow(2).reshape(grads_e.shape[0], -1).sum(-1)).mean()

        grad_penalty = 0.5 * (gp_p + gp_e)

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
