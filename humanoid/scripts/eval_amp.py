# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Fixed evaluation of a trained x1_amp checkpoint (Gate C).

Loads the latest x1_amp checkpoint, runs the frozen eval protocol
(task_spec.md §1): fixed seeds, fixed commands, measures stability, velocity
tracking, imitation error vs the retargeted reference, and writes a JSON report.

Run (remote Gradmotion host, after training):
    python humanoid/scripts/eval_amp.py --task=x1_amp --headless --load_run=-1
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# isaacgym MUST be imported before torch. Import it first.
from isaacgym.torch_utils import *  # noqa: F401,F403  (ensures isaacgym loads first)
import numpy as np
import torch

from humanoid.envs import *                      # noqa: F401,F403
from humanoid.utils import get_args, task_registry

OUT = os.path.join(_REPO, "data", "amp_eval", "eval_report.json")

# Fixed eval protocol (must match task_spec.md §1)
EVAL_SEEDS = [5, 17, 42, 123, 2024]
EVAL_EPISODES = 4                # per seed (bounded for remote cost)
EPISODE_LEN_S = 24               # env episode length (config); we measure within
NOMINAL_VX = 0.5


def evaluate(env, runner, retarget_joints, retarget_fps):
    """Run eval episodes; collect metrics. Returns dict of metric arrays."""
    policy = runner.get_inference_policy(device=env.device) if hasattr(runner, "get_inference_policy") \
        else runner.alg.actor_critic
    dt = env.cfg.control.decimation * env.cfg.sim.dt
    metrics = {
        "fall": [], "episode_length_steps": [], "vx_track_err": [],
        "base_h": [], "base_pitch_deg": [], "joint_pos_err_ref": [],
        "dof_pos_limit_viol": [], "foot_contact_l": [], "foot_contact_r": [],
        "amp_style_reward": [],
    }
    for ep in range(EVAL_EPISODES):
        obs = env.get_observations()
        steps = 0
        fell = False
        base_h_traj, pitch_traj, vx_err = [], [], []
        jp_err = []
        contact_l, contact_r = [], []
        for _ in range(int(EPISODE_LEN_S / dt)):
            with torch.no_grad():
                actions = policy.act_inference(obs.detach()) if hasattr(policy, "act_inference") \
                    else policy(obs.detach())
            obs, _, rew, dones, infos = env.step(actions)
            steps += 1
            base_h_traj.append(float(env.root_states[0, 2]))
            # base pitch from quat (xyzw): pitch ~ asin(2*(w*y - z*x))
            q = env.root_states[0, 3:7]
            pitch = torch.asin(torch.clamp(2 * (q[3] * q[1] - q[2] * q[0]), -1, 1)).item()
            pitch_traj.append(np.degrees(pitch))
            # velocity tracking error vs nominal vx
            vx_err.append(abs(float(env.base_lin_vel[0, 0]) - NOMINAL_VX))
            # joint pos error vs reference (cycle through retarget frames)
            f = steps % retarget_joints.shape[0]
            jp_err.append(float(np.abs(env.dof_pos[0].cpu().numpy() - retarget_joints[f]).mean()))
            # foot contact
            cf = env.contact_forces[:, env.feet_indices, 2]
            contact_l.append(float((cf[0, 0] > 5.0)))
            contact_r.append(float((cf[0, 1] > 5.0)))
            # dof limit violation
            dof = env.dof_pos[0].cpu().numpy()
            lims = np.array([[l, h] for l, h in __import__("humanoid.algo.amp.motion_lib", fromlist=["X1_JOINT_LIMITS"]).X1_JOINT_LIMITS])
            viol = int(np.sum((dof < lims[:, 0]) | (dof > lims[:, 1])))
            metrics["dof_pos_limit_viol"].append(viol)
            if dones[0]:
                fell = True
                break
        metrics["fall"].append(int(fell))
        metrics["episode_length_steps"].append(steps)
        metrics["vx_track_err"].append(float(np.mean(vx_err)))
        metrics["base_h"].append(float(np.mean(base_h_traj)))
        metrics["base_pitch_deg"].append(float(np.mean(np.abs(pitch_traj))))
        metrics["joint_pos_err_ref"].append(float(np.mean(jp_err)))
        metrics["foot_contact_l"].append(float(np.mean(contact_l)))
        metrics["foot_contact_r"].append(float(np.mean(contact_r)))
    return metrics


def main():
    args = get_args()
    name = "x1_amp"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # load retarget reference for imitation error
    rj = np.load(os.path.join(_REPO, "data", "retarget", "x1_walk_retargeted.npz"))["joint_positions"]
    rf = float(np.load(os.path.join(_REPO, "data", "retarget", "x1_walk_retargeted.npz"))["fps"])

    seed_results = {}
    for seed in EVAL_SEEDS:
        env_cfg, train_cfg = task_registry.get_cfgs(name)
        env_cfg.env.num_envs = 1
        env_cfg.noise.add_noise = True
        train_cfg.seed = seed
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = -1
        env, _ = task_registry.make_env(name=name, args=args, env_cfg=env_cfg)
        runner, _, _ = task_registry.make_alg_runner(env=env, name=name, args=args, train_cfg=train_cfg)
        m = evaluate(env, runner, rj, rf)
        seed_results[seed] = m
        env.gym.destroy_sim(env.sim)

    # aggregate
    def agg(key, op="mean"):
        vals = []
        for s in EVAL_SEEDS:
            vals.extend(seed_results[s][key])
        vals = np.array(vals, dtype=float)
        if op == "mean":
            return {"mean": float(vals.mean()), "std": float(vals.std()), "worst": float(vals.max() if "err" in key or "fall" in key or "viol" in key else vals.min())}
        return {"mean": float(vals.mean())}

    report = {
        "task": name, "eval_seeds": EVAL_SEEDS, "episodes_per_seed": EVAL_EPISODES,
        "nominal_vx": NOMINAL_VX, "episode_len_s": EPISODE_LEN_S,
        "fall_rate": agg("fall"), "episode_length_steps": agg("episode_length_steps"),
        "vx_track_err_mps": agg("vx_track_err"), "base_height_m": agg("base_h"),
        "base_pitch_deg": agg("base_pitch_deg"), "joint_pos_err_ref_rad": agg("joint_pos_err_ref"),
        "dof_limit_viol_total": int(sum(sum(seed_results[s]["dof_pos_limit_viol"]) for s in EVAL_SEEDS)),
        "foot_contact_l_frac": agg("foot_contact_l"), "foot_contact_r_frac": agg("foot_contact_r"),
    }
    report["summary"] = {
        "fall_rate": report["fall_rate"]["mean"],
        "episode_length_steps_mean": report["episode_length_steps"]["mean"],
        "vx_track_err_mean": report["vx_track_err_mps"]["mean"],
        "joint_pos_err_ref_mean": report["joint_pos_err_ref_rad"]["mean"],
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] wrote {OUT}")
    print(f"  fall_rate={report['fall_rate']['mean']:.3f} "
          f"vx_track_err={report['vx_track_err_mps']['mean']:.3f} "
          f"joint_pos_err_ref={report['joint_pos_err_ref_rad']['mean']:.3f} "
          f"base_h={report['base_height_m']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
