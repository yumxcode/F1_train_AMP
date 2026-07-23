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
    """Train x1_amp and return (runner, train_cfg, log_dir)."""
    env_cfg, train_cfg = task_registry.get_cfgs(name)
    train_cfg.runner.max_iterations = max_iter
    env, _ = task_registry.make_env(name=name, args=args, env_cfg=env_cfg)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=name, args=args, train_cfg=train_cfg)
    runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=False)
    return env, runner, train_cfg, log_dir


def evaluate(env, runner, retarget_joints, args, seed):
    """Run eval episodes for one seed; return dict of metric arrays."""
    policy = runner.alg.actor_critic
    dt = env.cfg.control.decimation * env.cfg.sim.dt
    lims = np.array([[l, h] for l, h in X1_JOINT_LIMITS])
    metrics = {k: [] for k in ["fall", "ep_len", "vx_err", "base_h", "pitch",
                               "jp_err", "dof_viol", "contact_l", "contact_r"]}
    dev = env.device
    nsteps = int(EPISODE_LEN_S / dt)
    for ep in range(EVAL_EPISODES):
        # The gymtorch extension enables a global inference mode. Run the ENTIRE
        # episode (reset + all steps + all tensor reads) under ONE sustained
        # inference_mode so no tensor is created normal-then-updated-inplace.
        with torch.inference_mode():
            env.reset_idx(torch.arange(env.num_envs, device=dev))
            obs = env.get_observations()
            steps = 0; fell = False
            bh, pt, ve, jpe, cl, cr = [], [], [], [], [], []
            viol = 0
            for _ in range(nsteps):
                actions = policy.act_inference(obs.detach())
                obs, _, rew, dones, infos = env.step(actions)
                steps += 1
                # read to python floats IMMEDIATELY (inside inference_mode)
                bh.append(float(env.root_states[0, 2]))
                q = env.root_states[0, 3:7]
                pt.append(float(np.degrees(torch.asin(torch.clamp(2*(q[3]*q[1]-q[2]*q[0]), -1, 1)).item())))
                ve.append(abs(float(env.base_lin_vel[0, 0]) - NOMINAL_VX))
                f = steps % retarget_joints.shape[0]
                jpe.append(float(torch.abs(env.dof_pos[0] - torch.as_tensor(retarget_joints[f], device=dev, dtype=torch.float)).mean()))
                cf = env.contact_forces[:, env.feet_indices, 2]
                cl.append(float(cf[0, 0] > 5.0)); cr.append(float(cf[0, 1] > 5.0))
                dof = env.dof_pos[0]
                viol += int(((dof < lims[:, 0]) | (dof > lims[:, 1])).sum())
                if dones[0]:
                    fell = True; break
        metrics["fall"].append(int(fell)); metrics["ep_len"].append(steps)
        metrics["vx_err"].append(float(np.mean(ve))); metrics["base_h"].append(float(np.mean(bh)))
        metrics["pitch"].append(float(np.mean(np.abs(pt)))); metrics["jp_err"].append(float(np.mean(jpe)))
        metrics["dof_viol"].append(viol); metrics["contact_l"].append(float(np.mean(cl)))
        metrics["contact_r"].append(float(np.mean(cr)))
    return metrics


def main():
    args = get_args()
    name = "x1_amp"
    max_iter = int(getattr(args, "max_iterations", 2000) or 2000)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rj = np.load(os.path.join(_REPO, "data", "retarget", "x1_walk_retargeted.npz"))["joint_positions"]

    print(f"[train_eval] training {name} seed={args.seed} max_iter={max_iter} ...")
    env, runner, train_cfg, log_dir = train_phase(args, name, max_iter)
    print(f"[train_eval] training done. log_dir={log_dir}")

    # eval across the fixed seed set on the just-trained checkpoint
    all_results = {}
    for eseed in EVAL_SEEDS:
        print(f"[train_eval] evaluating seed={eseed} ...")
        all_results[eseed] = evaluate(env, runner, rj, args, eseed)

    def agg(key):
        vals = []
        for s in EVAL_SEEDS:
            vals.extend(all_results[s][key])
        vals = np.array(vals, dtype=float)
        hi = "err" in key or "fall" in key or "viol" in key or "pitch" in key
        return {"mean": float(vals.mean()), "std": float(vals.std()),
                "worst": float(vals.max() if hi else vals.min())}

    report = {
        "task": name, "train_seed": int(args.seed), "train_max_iter": max_iter,
        "eval_seeds": EVAL_SEEDS, "episodes_per_seed": EVAL_EPISODES,
        "nominal_vx": NOMINAL_VX, "log_dir": log_dir,
        "fall_rate": agg("fall"), "episode_length_steps": agg("ep_len"),
        "vx_track_err_mps": agg("vx_err"), "base_height_m": agg("base_h"),
        "base_pitch_deg": agg("pitch"), "joint_pos_err_ref_rad": agg("jp_err"),
        "dof_limit_viol_total": int(sum(sum(all_results[s]["dof_viol"]) for s in EVAL_SEEDS)),
        "foot_contact_l_frac": agg("contact_l"), "foot_contact_r_frac": agg("contact_r"),
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train_eval] wrote {OUT}")
    print(f"  fall_rate={report['fall_rate']['mean']:.3f} vx_err={report['vx_track_err_mps']['mean']:.3f} "
          f"jp_err={report['joint_pos_err_ref_rad']['mean']:.3f} base_h={report['base_height_m']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
