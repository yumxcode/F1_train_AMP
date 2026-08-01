# SPDX-License-Identifier: BSD-3-Clause
"""AMP discriminator following NVIDIA official AMP regularization stack.

Key differences from naive implementations:
  * R1 gradient penalty on REAL/demo samples (λ configurable, default 5.0).
  * Final-logit weight regularization (disc_logit_reg, default 0.05).
  * Global weight decay (disc_weight_decay, default 1e-4).
  * BCEWithLogits loss with HARD 0/1 labels (NO label smoothing per official AMP).
  * NO spectral norm / instance noise / label smoothing (absent from all official configs).
  * Capacity [1024, 512] matches official — stability comes from penalty terms, not shrinking.

Reference: IsaacGymEnvs HumanoidAMP, ASE amp_agent.py (Peng et al. 2021).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Single-logit discriminator over AMP transition features.

    Input dim = AMP_OBS_DIM. Produces a scalar logit D(x).
    Architecture: [1024, 512] ReLU → Linear(1), matching official AMP.
    """

    def __init__(self, input_dim: int, hidden_dims=(1024, 512), activation="relu"):
        super().__init__()
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        # Separate logit head for weight regularization (disc_logit_reg)
        self.logit_head = nn.Linear(prev, 1)
        # Official init: uniform(-1, 1) for logit head
        nn.init.uniform_(self.logit_head.weight, -1.0, 1.0)
        nn.init.zeros_(self.logit_head.bias)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return self.logit_head(h)

    def compute_reward(self, policy_logit: torch.Tensor, expert_logit_ema: torch.Tensor,
                       reward_clamp: float = 0.5) -> torch.Tensor:
        """AMP style reward: r = -log(max(1 - sigmoid(D(p)), 1e-4)) * scale.

        This is the OFFICIAL AMP reward formula (Peng et al. 2021).
        Uses sigmoid of policy logit, NOT the exp-floor workaround.
        Clamp at 1e-4 prevents reward blow-up when disc is confident.
        reward_clamp=0.5 keeps style reward from dominating task reward.
        """
        prob = torch.sigmoid(policy_logit.squeeze(-1))
        r = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        return r.clamp(0.0, reward_clamp)

    def get_logit_weights(self):
        """Return logit head weights for disc_logit_reg."""
        return self.logit_head.weight

    def get_all_weights(self):
        """Return all weights for disc_weight_decay."""
        return list(self.trunk.parameters()) + list(self.logit_head.parameters())


def compute_disc_loss(disc: Discriminator, expert: torch.Tensor, policy: torch.Tensor,
                      grad_penalty_coef: float = 5.0,
                      label_smooth: float = 0.0,
                      logit_reg_coef: float = 0.05,
                      weight_decay_coef: float = 1e-4) -> Dict[str, torch.Tensor]:
    """AMP discriminator loss following official NVIDIA/ASE formulation.

    Loss = 0.5 * (BCE(fake, 0) + BCE(real, 1))
         + grad_penalty_coef * mean(||grad_x D(real)||^2)     [R1 on REAL only]
         + logit_reg_coef * sum(||w_logit||^2)                 [logit weight reg]
         + weight_decay_coef * sum(||W||^2)                    [global weight decay]

    label_smooth is accepted for API compatibility but defaults to 0.0 (hard labels).
    """
    d_e = disc(expert)   # (batch, 1) — real/demo logits
    d_p = disc(policy)   # (batch, 1) — fake/agent logits

    # BCEWithLogits with hard labels (official AMP: NO label smoothing)
    expert_loss = nn.functional.binary_cross_entropy_with_logits(d_e, torch.ones_like(d_e))
    policy_loss = nn.functional.binary_cross_entropy_with_logits(d_p, torch.zeros_like(d_p))
    bce_loss = 0.5 * (expert_loss + policy_loss)

    # R1 gradient penalty on REAL/expert samples (official: real-only)
    grad_penalty = torch.tensor(0.0, device=expert.device)
    if grad_penalty_coef > 0:
        expert_g = expert.detach().requires_grad_(True)
        d_eg = disc(expert_g)
        grads_e = torch.autograd.grad(
            outputs=d_eg.sum(), inputs=expert_g,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = (grads_e.pow(2).reshape(grads_e.shape[0], -1).sum(-1)).mean()

    # Logit weight regularization (disc_logit_reg)
    logit_reg = torch.tensor(0.0, device=expert.device)
    if logit_reg_coef > 0:
        w = disc.get_logit_weights()
        logit_reg = (w ** 2).sum()

    # Global weight decay
    weight_decay = torch.tensor(0.0, device=expert.device)
    if weight_decay_coef > 0:
        wd_sum = torch.tensor(0.0, device=expert.device)
        for p in disc.get_all_weights():
            wd_sum = wd_sum + (p ** 2).sum()
        weight_decay = wd_sum

    loss = bce_loss + grad_penalty_coef * grad_penalty + \
           logit_reg_coef * logit_reg + weight_decay_coef * weight_decay

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
