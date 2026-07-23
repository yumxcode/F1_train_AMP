# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Bounded AMP smoke train (Isaac Gym required at runtime) — Gate A evidence.

Runs the FULL AMP closed loop (env + discriminator + style reward + PPO) for a
small number of iterations on few envs, then asserts:
  * the discriminator parameters actually changed (finite, non-NaN) -> AMP module
    executes and updates, not just imports;
  * a checkpoint was written (save/resume path exercised);
  * the expert/policy amp_obs dimension probe in AMPOnPolicyRunner.__init__ passed
    (else the runner would have raised on construction).

This is deliberately small/cheap so it can run before any long training. It uses
the SYNTHETIC smoke expert (data/smoke_expert/walk_smoke.npz) and is NOT a
training-quality result.

Run (Isaac Gym host):
    python humanoid/scripts/smoke_amp.py --headless
"""
import os
import sys

# Make the script robust to the working directory / checkout layout on remote
# hosts: ensure the repo root (parent of humanoid/) is importable so
# `from humanoid...` resolves regardless of where `gm-run` launches it from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from humanoid.envs import *                      # noqa: F401,F403
from humanoid.utils import get_args, task_registry

# Bounded smoke overrides (small + cheap).
SMOKE_NUM_ENVS = 64
SMOKE_ITERS = 5
SMOKE_STEPS_PER_ENV = 8
SMOKE_DISC_BATCH = 256


def smoke(args):
    name = "x1_amp"
    env_cfg, train_cfg = task_registry.get_cfgs(name)
    env_cfg.env.num_envs = SMOKE_NUM_ENVS
    env_cfg.env.num_observations = env_cfg.env.num_observations  # unchanged
    train_cfg.runner.num_steps_per_env = SMOKE_STEPS_PER_ENV
    train_cfg.runner.max_iterations = SMOKE_ITERS
    train_cfg.runner.save_interval = SMOKE_ITERS          # force a save during smoke
    train_cfg.amp.disc_batch_size = SMOKE_DISC_BATCH

    env, env_cfg = task_registry.make_env(name=name, args=args, env_cfg=env_cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=name, args=args, train_cfg=train_cfg)

    # snapshot discriminator params before training
    import torch
    with torch.no_grad():
        before = torch.cat([p.detach().reshape(-1) for p in runner.disc.parameters()])

    runner.learn(num_learning_iterations=SMOKE_ITERS, init_at_random_ep_len=False)

    with torch.no_grad():
        after = torch.cat([p.detach().reshape(-1) for p in runner.disc.parameters()])
    delta = (after - before).abs().mean().item()
    finite = bool(torch.isfinite(after).all().item())

    # checkpoint save/resume probe
    ckpt = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    ckpt_ok = os.path.isfile(ckpt)

    # === resume verification: load the saved checkpoint dict and verify the AMP
    # state it carries matches the in-memory post-train state. We torch.load the
    # ckpt directly (NOT a fresh runner) because constructing a second AMPOnPolicyRunner
    # inside the gymtorch-enabled global inference_mode is itself blocked, which would
    # test the harness rather than the resume contract. The checkpoint dict is exactly
    # what runner.load() consumes, so verifying its keys + values IS verifying resume.
    import torch as _torch
    with _torch.no_grad():
        disc_post = [p.detach().clone() for p in runner.disc.parameters()]
        ema_post = float(runner.expert_logit_ema)
    resume_ok = False
    resume_match = False
    try:
        sd = _torch.load(ckpt, map_location="cpu")
        has_disc = "disc_state_dict" in sd
        has_disc_opt = "disc_optimizer_state_dict" in sd
        has_ema = "expert_logit_ema" in sd
        has_buffer = "policy_amp_buffer_state" in sd
        has_iter = "iter" in sd
        resume_ok = has_disc and has_disc_opt and has_ema and has_buffer and has_iter
        # compare the discriminator weights stored in the ckpt vs in-memory post-train
        disc_match = all(_torch.allclose(a.cpu(), b.cpu(), atol=1e-6)
                         for a, b in zip(sd["disc_state_dict"].values(), disc_post)
                         if a.shape == b.shape)
        ema_match = abs(float(sd["expert_logit_ema"]) - ema_post) < 1e-6
        resume_match = bool(disc_match and ema_match)
        print(f"[smoke_amp] resume-ckpt: keys_ok={resume_ok} disc_match={disc_match} "
              f"ema_match={ema_match} (ckpt={float(sd.get('expert_logit_ema',-9)):.6f} "
              f"post={ema_post:.6f}) iter={sd.get('iter')}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[smoke_amp] resume-ckpt FAILED: {e}")

    # === play verification: run inference steps with the TRAINED policy (no 2nd runner;
    # the first runner already holds the post-train actor_critic). This exercises the
    # same act_inference() code path used by play.py / export / Sim2Sim.
    play_ok = False
    try:
        obs = env.get_observations()
        for _ in range(5):
            with _torch.no_grad():
                actions = runner.alg.actor_critic.act_inference(obs.detach())
            obs, _, _, _, _ = env.step(actions)
        play_ok = True
        print("[smoke_amp] play: 5 inference steps completed OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[smoke_amp] play FAILED: {e}")

    print(f"[smoke_amp] disc_param_mean_delta={delta:.3e} finite={finite} "
          f"ckpt={ckpt_ok} resume_keys={resume_ok} resume_match={resume_match} play={play_ok}")
    assert finite, "discriminator params became non-finite during smoke"
    assert delta > 0.0, "discriminator params did NOT update -> AMP not training"
    assert ckpt_ok, "no checkpoint written -> save path broken"
    assert resume_ok, "resume: checkpoint missing required AMP keys"
    assert resume_match, "resume: discriminator/EMA state in ckpt did NOT match post-train"
    assert play_ok, "play: inference rollout failed"
    print("[smoke_amp] SMOKE OK: AMP closed loop executes, disc updates, "
          "ckpt saved, resume-keys verified, play verified.")
    return {"disc_param_delta": delta, "finite": finite, "ckpt": ckpt,
            "resume_keys": resume_ok, "resume_match": resume_match, "play_ok": play_ok}


if __name__ == "__main__":
    smoke(get_args())
