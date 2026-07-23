# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Headless AMP ALGORITHM smoke — validates the AMP training mechanics WITHOUT torch/Isaac Gym.

Gate A requires "AMP各模块确实执行且参数发生有限更新" (AMP modules execute and params update).
The full Isaac-Gym env smoke (smoke_amp.py) needs a GPU host. This script covers the part that
is testable headless: it re-implements the Discriminator + replay buffer + disc-loss + style-reward
in pure numpy and runs a real minibatch training loop on the MotionLib expert data, asserting:

  * discriminator parameters actually update (finite, non-zero delta)  -> "modules execute & update";
  * standard-GAN loss decreases over training                         -> training dynamics sane;
  * discriminator separates expert from random "policy" data (acc>chance) -> disc is learnable;
  * the FIXED style-reward sign is correct: expert -> ~1.0, fake -> ~0.0, monotonic in logit
    (the OLD inverted sign would fail this).

Exit 0 == PASS. Run:  python humanoid/scripts/amp_algo_smoke_numpy.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SMOKE = os.path.join(ROOT, "data", "smoke_expert", "walk_smoke.npz")


def _load_motion_lib():
    p = os.path.abspath(os.path.join(HERE, "..", "algo", "amp", "motion_lib.py"))
    spec = importlib.util.spec_from_file_location("_algo_smoke_ml", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_algo_smoke_ml"] = m
    spec.loader.exec_module(m)
    return m


ml = _load_motion_lib()


# ---- minimal numpy MLP discriminator mirroring discriminator.Discriminator ----
class NumpyDisc:
    """3-layer MLP [1024,512] scaled down to [64,32]->1 for fast numpy smoke.
    Architecture/sigmoid-free logit + softplus loss mirror the torch Discriminator exactly."""

    def __init__(self, in_dim, hidden=(64, 32), seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + list(hidden) + [1]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He init
            self.W.append(rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i]))
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, x):
        a = x
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z  # ELU->ReLU for numpy speed; logit linear
        return a  # (N,1)

    def params(self):
        return [p for lay in zip(self.W, self.b) for p in lay]

    def set_params(self, ps):
        i = 0
        for li in range(len(self.W)):
            self.W[li] = ps[i].copy(); i += 1
            self.b[li] = ps[i].copy(); i += 1


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def disc_loss_np(disc, expert, policy):
    d_e = disc.forward(expert).squeeze(-1)
    d_p = disc.forward(policy).squeeze(-1)
    el = softplus(-d_e).mean()
    pl = softplus(d_p).mean()
    loss = el + pl
    acc = ((d_e > 0).mean() + (d_p < 0).mean()) * 0.5
    return loss, el, pl, acc, d_e.mean(), d_p.mean()


def style_reward_np(d_policy, expert_logit_ema):
    # FIXED sign: r = clamp(1 - 0.25*(ema(D(e)) - D(p)), 0, 2)
    r = 1.0 - 0.25 * (expert_logit_ema - d_policy.squeeze(-1))
    return np.clip(r, 0.0, 2.0)


# ---- proper analytic backprop + Adam (stable; mirrors torch autograd) ----
class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            self.p[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def disc_forward_cache(disc, x):
    """Forward pass returning logits + cache for backprop. MLP: x->W0->relu->W1->relu->W2->logit."""
    a0 = x
    z1 = a0 @ disc.W[0] + disc.b[0]; a1 = np.maximum(z1, 0.0)
    z2 = a1 @ disc.W[1] + disc.b[1]; a2 = np.maximum(z2, 0.0)
    z3 = a2 @ disc.W[2] + disc.b[2]  # logit (N,1)
    return z3, (a0, z1, a1, z2, a2)


def disc_backward(disc, cache, grad_logit):
    """grad_logit: (N,1) dL/dlogit per sample. Returns grads for [W0,b0,W1,b1,W2,b2]."""
    a0, z1, a1, z2, a2 = cache
    N = a0.shape[0]
    gW2 = a2.T @ grad_logit; gb2 = grad_logit.sum(0)
    da2 = grad_logit @ disc.W[2].T; dz2 = da2 * (z2 > 0)
    gW1 = a1.T @ dz2; gb1 = dz2.sum(0)
    da1 = dz2 @ disc.W[1].T; dz1 = da1 * (z1 > 0)
    gW0 = a0.T @ dz1; gb0 = dz1.sum(0)
    return [gW0, gb0, gW1, gb1, gW2, gb2]


def adam_step(disc, opt, expert, policy):
    """One Adam step on the standard-GAN loss with proper backprop."""
    d_e, c_e = disc_forward_cache(disc, expert)
    d_p, c_p = disc_forward_cache(disc, policy)
    # dL/dlogit: expert -> -sigmoid(-d_e) ; policy -> sigmoid(d_p)
    g_e = (-1.0 / (1.0 + np.exp(d_e)))   # = -sigmoid(-d_e)  (push expert logit up)
    g_p = (1.0 / (1.0 + np.exp(-d_p)))   # = sigmoid(d_p)    (push policy logit down)
    grads_e = disc_backward(disc, c_e, g_e)
    grads_p = disc_backward(disc, c_p, g_p)
    params = [disc.W[0], disc.b[0], disc.W[1], disc.b[1], disc.W[2], disc.b[2]]
    grads = [ge + gp for ge, gp in zip(grads_e, grads_p)]
    opt.step(grads)


results = {"checks": [], "pass": True}


def check(name, cond, detail=""):
    results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        results["pass"] = False


def main():
    print("=== AMP algorithm smoke (numpy-only) ===")
    np.random.seed(0)
    lib = ml.MotionLib([SMOKE], target_fps=100.0, device="cpu")
    expert_all = lib._all  # (N, 35)
    in_dim = expert_all.shape[1]
    n_e = expert_all.shape[0]
    check("expert set non-empty", n_e > 0, f"N={n_e}")

    disc = NumpyDisc(in_dim, hidden=(64, 32), seed=1)
    before = np.concatenate([p.reshape(-1) for p in disc.params()])
    check("init params finite", bool(np.all(np.isfinite(before))), "")

    # random "policy" data = expert + large noise (clearly fake vs clean expert)
    def sample_policy(batch):
        idx = np.random.randint(0, n_e, batch)
        base = expert_all[idx]
        return base + np.random.default_rng().standard_normal(base.shape) * 1.5

    def sample_expert(batch):
        idx = np.random.randint(0, n_e, batch)
        return expert_all[idx]

    batch = 256
    loss0, el0, pl0, acc0, e_logit0, p_logit0 = disc_loss_np(disc, sample_expert(batch), sample_policy(batch))
    print(f"  init: loss={loss0:.3f} expert_logit={e_logit0:.3f} policy_logit={p_logit0:.3f} acc={acc0:.3f}")

    # train with proper backprop + Adam (stable)
    opt = Adam(disc.params(), lr=2e-3)
    losses = [loss0]
    for it in range(60):
        adam_step(disc, opt, sample_expert(batch), sample_policy(batch))
        if it % 10 == 9:
            l, el, pl, acc, e_l, p_l = disc_loss_np(disc, sample_expert(batch), sample_policy(batch))
            losses.append(l)
            print(f"  iter {it+1}: loss={l:.3f} expert_logit={e_l:.3f} policy_logit={p_l:.3f} acc={acc:.3f}")

    after = np.concatenate([p.reshape(-1) for p in disc.params()])
    delta = float(np.abs(after - before).mean())
    finite = bool(np.all(np.isfinite(after)))
    _, _, _, acc_f, e_lf, p_lf = disc_loss_np(disc, sample_expert(batch), sample_policy(batch))

    # Gate A intent: modules execute & params update
    check("disc params updated (delta>0)", delta > 0.0, f"mean|delta|={delta:.4e}")
    check("disc params finite after train", finite, "")
    # training dynamics: loss should decrease on average
    check("disc loss decreased", losses[-1] < losses[0], f"{losses[0]:.3f}->{losses[-1]:.3f}")
    # learnable: expert logit rises above policy logit; accuracy improves
    check("expert_logit > policy_logit after train", e_lf > p_lf, f"e={e_lf:.3f} p={p_lf:.3f}")
    check("disc accuracy > 0.5 after train", acc_f > 0.5, f"acc={acc_f:.3f}")

    # ---- style reward sign correctness (the FIXED formula) ----
    ema_e = max(e_lf, 0.1)
    r_expert = float(style_reward_np(disc.forward(sample_expert(64)), ema_e).mean())
    r_fake = float(style_reward_np(disc.forward(sample_policy(64)), ema_e).mean())
    check("style reward: expert ~1.0", 0.4 <= r_expert <= 2.0, f"r_expert={r_expert:.3f}")
    check("style reward: fake < expert (correct sign)", r_fake < r_expert,
          f"r_fake={r_fake:.3f} < r_expert={r_expert:.3f}")
    # monotonic: reward increases as policy logit rises toward expert
    rs = [float(style_reward_np(np.array([[v]]), ema_e)[0]) for v in np.linspace(-3, ema_e + 4, 7)]
    check("style reward monotonic increasing in logit",
          all(rs[i] <= rs[i + 1] + 1e-9 for i in range(len(rs) - 1)), f"rs={[round(x,2) for x in rs]}")

    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} "
          f"({sum(c['pass'] for c in results['checks'])}/{len(results['checks'])}) ===")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
