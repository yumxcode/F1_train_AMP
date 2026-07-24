# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Bounded PLAIN PPO baseline smoke (Isaac Gym required at runtime) — amp_conversion evidence.

This is the regression baseline for the AMP-gain comparison. It runs the UNCHANGED
plain-PPO task ``x1_dh_stand`` (DHOnPolicyRunner / DHPPO, NO discriminator, NO style
reward) for a few iterations on few envs and asserts the PPO actually trains:
  * the policy (actor_critic) parameters changed (finite, non-NaN) -> PPO executes & updates;
  * a checkpoint was written (save/resume path exercised);
  * a short inference (play) rollout completes.

It deliberately mirrors ``smoke_amp.py`` in structure and bounds so the two runs are
directly comparable (same env/robot, same smoke budget, the ONLY difference is the
runner: DHOnPolicyRunner vs AMPOnPolicyRunner).

Run (Isaac Gym host):
    python humanoid/scripts/smoke_baseline.py --headless
"""
import os
import sys

# Repo root importable regardless of the remote host launch dir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from humanoid.envs import *                      # noqa: F401,F403
from humanoid.utils import get_args, task_registry

# Bounded smoke overrides — IDENTICAL budget to the AMP smoke for comparability.
SMOKE_NUM_ENVS = 64
SMOKE_ITERS = 5
SMOKE_STEPS_PER_ENV = 8


def smoke(args):
    # Plain PPO baseline task (DHOnPolicyRunner). NOT the AMP task.
    name = "x1_dh_stand"
    env_cfg, train_cfg = task_registry.get_cfgs(name)
    env_cfg.env.num_envs = SMOKE_NUM_ENVS
    train_cfg.runner.num_steps_per_env = SMOKE_STEPS_PER_ENV
    train_cfg.runner.max_iterations = SMOKE_ITERS
    train_cfg.runner.save_interval = SMOKE_ITERS          # force a save during smoke

    env, env_cfg = task_registry.make_env(name=name, args=args, env_cfg=env_cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=name, args=args, train_cfg=train_cfg)

    import torch
    # snapshot actor_critic params before training
    with torch.no_grad():
        before = torch.cat([p.detach().reshape(-1) for p in runner.alg.actor_critic.parameters()])

    runner.learn(num_learning_iterations=SMOKE_ITERS, init_at_random_ep_len=False)

    with torch.no_grad():
        after = torch.cat([p.detach().reshape(-1) for p in runner.alg.actor_critic.parameters()])
    delta = (after - before).abs().mean().item()
    finite = bool(torch.isfinite(after).all().item())

    ckpt = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    ckpt_ok = os.path.isfile(ckpt)

    # resume probe: checkpoint must carry the plain-PPO keys (no AMP keys expected).
    import torch as _torch
    sd = _torch.load(ckpt, map_location="cpu")
    has_model = "model_state_dict" in sd
    has_opt = "optimizer_state_dict" in sd
    has_iter = "iter" in sd
    is_plain = "disc_state_dict" not in sd            # baseline MUST NOT carry AMP state
    resume_ok = has_model and has_opt and has_iter

    # play verification (same act_inference path as smoke_amp / export / Sim2Sim)
    play_ok = False
    try:
        obs = env.get_observations()
        for _ in range(5):
            with _torch.no_grad():
                actions = runner.alg.actor_critic.act_inference(obs.detach())
            obs, _, _, _, _ = env.step(actions)
        play_ok = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[smoke_baseline] play FAILED: {e}")

    print(f"[smoke_baseline] task=x1_dh_stand (plain PPO, runner={type(runner).__name__}) "
          f"ppo_param_mean_delta={delta:.3e} finite={finite} "
          f"ckpt={ckpt_ok} resume_keys={resume_ok} plain_no_amp_state={is_plain} play={play_ok}")
    assert finite, "actor_critic params became non-finite during baseline smoke"
    assert delta > 0.0, "PPO params did NOT update -> plain PPO not training"
    assert ckpt_ok, "no checkpoint written -> save path broken"
    assert resume_ok, "resume: checkpoint missing required plain-PPO keys"
    assert is_plain, "baseline checkpoint unexpectedly carries AMP discriminator state"
    assert play_ok, "play: inference rollout failed"
    print("[smoke_baseline] SMOKE OK: plain PPO baseline trains (params update), "
          "ckpt saved, resume-keys verified, no AMP state, play verified.")
    return {"ppo_param_delta": delta, "finite": finite, "ckpt": ckpt,
            "resume_keys": resume_ok, "plain_no_amp_state": is_plain, "play_ok": play_ok}


if __name__ == "__main__":
    smoke(get_args())
