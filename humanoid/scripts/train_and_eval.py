# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Combined train+eval entry for Gate C (single gm-run, no checkpoint-resume dependency).

Trains x1_amp for a bounded number of iterations, then runs the fixed 5-seed
eval on the just-trained checkpoint IN THE SAME PROCESS (the checkpoint lives in
the local logs dir, so task_registry.get_load_path() finds it). This sidesteps
the Gradmotion checkpoint-resume mount-path integration gap.

Usage (remote):
    gm-run F1_train_AMP/humanoid/scripts/train_and_eval.py --task=x1_amp --headless \
        --seed=5 --num_envs=4096 --max_iterations=2000
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# isaacgym MUST be imported before torch.
from isaacgym.torch_utils import *  # noqa: F401,F403
import numpy as np
import torch

from humanoid.envs import *                      # noqa: F401,F403
from humanoid.utils import get_args, task_registry
from humanoid.algo.amp.motion_lib import X1_JOINT_LIMITS

OUT = os.path.join(_REPO, "data", "amp_eval", "train_eval_report.json")
EVAL_SEEDS = [5, 17, 42, 123, 2024]
EVAL_EPISODES = 4
EPISODE_LEN_S = 24
NOMINAL_VX = 0.5


def train_phase(args, name, max_iter):
    """Train x1_amp and return (env, runner, train_cfg, log_dir)."""
    env_cfg, train_cfg = task_registry.get_cfgs(name)
    train_cfg.runner.max_iterations = max_iter
    env, _ = task_registry.make_env(name=name, args=args, env_cfg=env_cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=name, args=args, train_cfg=train_cfg)
    runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=False)
    return env, runner, train_cfg, log_dir


def evaluate(env, runner, retarget_joints, seed):
    """Fixed-seed eval (Gate C). CRITICAL autograd-context fix:

    ``env.step`` is run inside ``torch.inference_mode()`` -- the SAME context as
    the training rollout (amp_on_policy_runner.py:105-108). The env's *internal*
    auto-reset (``post_physics_step -> reset_idx``), which inplace-updates gym
    inference tensors (``dof_pos[env_ids]=...`` etc.), is therefore LEGAL.

    The prior harness called ``env.reset_idx(...)`` MANUALLY wrapped in
    ``torch.inference_mode(False)``. ``inference_mode(False)`` EXITS inference
    mode, so the manual reset's inplace updates were illegal and crashed with
    "Inplace update to inference tensor outside InferenceMode" -- which is why
    TASK_20260724_067 and TASK_20260724_079 both terminated right at
    "evaluating seed=5". We now (a) use ``inference_mode()`` (enter), (b) rely on
    the env's internal auto-reset, and (c) never call ``reset_idx`` ourselves.

    Episodes are segmented by ``env.reset_buf[0]``; a fall = a reset with
    ``time_out_buf[0] == False``. Nominal dynamics (DR/noise off, fixed vx=0.5),
    matching the proven play.py path.
    """
    dev = env.device
    dt = env.cfg.control.decimation * env.cfg.sim.dt
    lims = np.array([[lo, hi] for lo, hi in X1_JOINT_LIMITS])
    clip_len = retarget_joints.shape[0]

    # --- nominal-dynamics eval config (mirror play.py) ---
    for attr in ["randomize_friction", "randomize_base_mass", "randomize_com",
                 "randomize_gains", "randomize_torque", "randomize_link_mass",
                 "randomize_motor_offset", "randomize_joint_friction",
                 "randomize_joint_damping", "randomize_joint_armature",
                 "randomize_coulomb_friction", "add_lag", "add_dof_lag",
                 "add_imu_lag", "add_dof_pos_vel_lag", "push_robots",
                 "continuous_push", "randomize_lag_timesteps"]:
        if hasattr(env.cfg.domain_rand, attr):
            setattr(env.cfg.domain_rand, attr, False)
    env.cfg.noise.add_noise = False
    env.cfg.env.send_timeouts = True                      # populates extras["time_outs"]
    env.max_episode_length = int(round(EPISODE_LEN_S / dt))   # defines a full eval episode
    env.max_episode_length_s = float(EPISODE_LEN_S)
    if hasattr(env.cfg.commands, "resampling_time"):
        env.cfg.commands.resampling_time = 1e9            # keep the fixed command (no auto-resample)

    torch.manual_seed(seed)
    np.random.seed(seed)
    ac = runner.alg.actor_critic
    nfull = int(env.max_episode_length)
    metrics = {k: [] for k in ["fall", "ep_len", "vx_err", "base_h", "pitch",
                               "jp_err", "dof_viol", "contact_l", "contact_r"]}

    episodes_done = 0
    warming = True            # discard the first (warm-up) episode boundary for env[0]
    ep_step = 0
    bh = pt = ve = jpe = cl = cr = 0.0
    viol = 0
    # hard cap so a degenerate policy cannot hang the run
    step_cap = (EVAL_EPISODES + 2) * (nfull + 5)

    with torch.inference_mode():
        obs = env.get_observations()
        s = 0
        while episodes_done < EVAL_EPISODES and s < step_cap:
            actions = ac.act_inference(obs.detach())
            # fixed forward command for ALL envs every step (overrides reset resamples)
            env.commands[:, 0] = NOMINAL_VX
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            if env.commands.shape[1] > 3:
                env.commands[:, 3] = 0.0
            obs, _, _, reset_buf, extras = env.step(actions)
            s += 1
            ep_step += 1

            bh += float(env.root_states[0, 2].item())
            qw, qx, qy, qz = (env.root_states[0, 3].item(), env.root_states[0, 4].item(),
                              env.root_states[0, 5].item(), env.root_states[0, 6].item())
            sin_p = 2.0 * (qz * qx - qy * qw)
            pt += abs(float(np.degrees(np.arcsin(min(1.0, max(-1.0, sin_p))))))
            ve += abs(env.base_lin_vel[0, 0].item() - NOMINAL_VX)
            dof0 = env.dof_pos[0].detach().cpu().numpy()
            jpe += float(np.abs(dof0 - retarget_joints[ep_step % clip_len]).mean())
            cf = env.contact_forces[:, env.feet_indices, 2]
            cl += float((cf[0, 0] > 5.0).item())
            cr += float((cf[0, 1] > 5.0).item())
            viol += int(((dof0 < lims[:, 0]) | (dof0 > lims[:, 1])).sum())

            if bool(reset_buf[0].item()):
                if warming:
                    warming = False
                else:
                    to = extras.get("time_outs")
                    timed_out = bool(to[0].item()) if to is not None else (ep_step >= nfull)
                    metrics["fall"].append(int(not timed_out))
                    metrics["ep_len"].append(ep_step)
                    metrics["vx_err"].append(ve / max(ep_step, 1))
                    metrics["base_h"].append(bh / max(ep_step, 1))
                    metrics["pitch"].append(pt / max(ep_step, 1))
                    metrics["jp_err"].append(jpe / max(ep_step, 1))
                    metrics["dof_viol"].append(viol)
                    metrics["contact_l"].append(cl / max(ep_step, 1))
                    metrics["contact_r"].append(cr / max(ep_step, 1))
                    episodes_done += 1
                ep_step = 0
                bh = pt = ve = jpe = cl = cr = 0.0
                viol = 0

    if episodes_done == 0:
        # degenerate: never crossed an episode boundary (instant fall loop) -> record a hard fall
        metrics["fall"].append(1)
        metrics["ep_len"].append(0)
        for k in ["vx_err", "base_h", "pitch", "jp_err", "contact_l", "contact_r"]:
            metrics[k].append(0.0)
        metrics["dof_viol"].append(0)
    return metrics


def main():
    args = get_args()
    name = "x1_amp"
    max_iter = int(getattr(args, "max_iterations", 2000) or 2000)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rj = np.load(os.path.join(_REPO, "data", "retarget_gmr", "x1_walk_retargeted.npz"))["joint_positions"]

    print(f"[train_eval] training {name} seed={args.seed} max_iter={max_iter} ...")
    env, runner, train_cfg, log_dir = train_phase(args, name, max_iter)
    print(f"[train_eval] training done. log_dir={log_dir}")

    # eval across the fixed seed set on the just-trained checkpoint
    all_results = {}
    for eseed in EVAL_SEEDS:
        print(f"[train_eval] evaluating seed={eseed} ...", flush=True)
        try:
            all_results[eseed] = evaluate(env, runner, rj, eseed)
            r = all_results[eseed]
            print(f"[train_eval]   seed={eseed} done: "
                  f"falls={sum(r['fall'])}/{len(r['fall'])} "
                  f"mean_ep_len={float(np.mean(r['ep_len'])):.0f} "
                  f"vx_err={float(np.mean(r['vx_err'])):.3f} "
                  f"jp_err={float(np.mean(r['jp_err'])):.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[train_eval]   seed={eseed} EVAL FAILED: {e}", flush=True)
            all_results[eseed] = None

    def agg(key):
        vals = []
        for sd in EVAL_SEEDS:
            if all_results[sd] is not None:
                vals.extend(all_results[sd][key])
        vals = np.array(vals, dtype=float)
        hi = "err" in key or "fall" in key or "viol" in key or "pitch" in key
        return {"n": int(vals.size),
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std()) if vals.size else float("nan"),
                "worst": float(vals.max() if hi else vals.min()) if vals.size else float("nan")}

    n_ok = sum(1 for sd in EVAL_SEEDS if all_results[sd] is not None)
    report = {
        "task": name, "train_seed": int(args.seed), "train_max_iter": max_iter,
        "eval_seeds": EVAL_SEEDS, "episodes_per_seed": EVAL_EPISODES,
        "nominal_vx": NOMINAL_VX, "episode_len_s": EPISODE_LEN_S,
        "seeds_completed": n_ok, "seeds_total": len(EVAL_SEEDS),
        "log_dir": log_dir,
        "fall_rate": agg("fall"), "episode_length_steps": agg("ep_len"),
        "vx_track_err_mps": agg("vx_err"), "base_height_m": agg("base_h"),
        "base_pitch_deg": agg("pitch"), "joint_pos_err_ref_rad": agg("jp_err"),
        "dof_limit_viol_total": int(sum(int(np.sum(all_results[sd]["dof_viol"]))
                                        for sd in EVAL_SEEDS if all_results[sd] is not None)),
        "foot_contact_l_frac": agg("contact_l"), "foot_contact_r_frac": agg("contact_r"),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train_eval] wrote {OUT}")
    print(f"  seeds_completed={n_ok}/{len(EVAL_SEEDS)} "
          f"fall_rate={report['fall_rate']['mean']:.3f} "
          f"vx_err={report['vx_track_err_mps']['mean']:.3f} "
          f"jp_err={report['joint_pos_err_ref_rad']['mean']:.3f} "
          f"base_h={report['base_height_m']['mean']:.3f} "
          f"dof_viol={report['dof_limit_viol_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
