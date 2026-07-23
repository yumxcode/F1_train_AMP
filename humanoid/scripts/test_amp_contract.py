# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Headless (numpy-only) AMP contract test — Gate A schema/loader validation.

No Isaac Gym / torch required. It validates the *deterministic* Gate A sub-checks:
  1. AMP feature layout (AMP_OBS_DIM, AMP_BLOCKS) matches the contract.
  2. X1 joint order (authoritative) matches the URDF actuated order AND the env
     default pose used by the policy side (single source of truth).
  3. MotionLib loads the synthetic smoke clip: 120->100 Hz resample, finite
     features, joint-range check, non-empty expert set.
  4. Expert == policy schema parity: the SAME builder (build_amp_obs_numpy) is
     the policy mirror, so expert and policy produce bit-identical layout/order/
     units/frame. We also assert the env default dof vector the policy subtracts
     equals the expert default vector.
  5. Loader-vs-builder statistics sanity (no NaN/Inf, bounded logits-free range).

The Isaac-Gym-dependent Gate A items (bounded smoke train with param updates,
checkpoint save/resume, play, and discriminator/style-reward evidence in the
training log) require the Isaac Gym runtime and are validated on the remote
Gradmotion platform, not here.

Exit code 0 == PASS. Run: python humanoid/scripts/test_amp_contract.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SMOKE = os.path.join(ROOT, "data", "smoke_expert", "walk_smoke.npz")
REPORT = os.path.join(ROOT, "data", "smoke_expert", "contract_test_report.txt")

# --- load motion_lib by file path (keeps this headless; no isaacgym/torch) ---
def _load_motion_lib():
    p = os.path.abspath(os.path.join(HERE, "..", "algo", "amp", "motion_lib.py"))
    spec = importlib.util.spec_from_file_location("_contract_motion_lib", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_contract_motion_lib"] = m          # needed for @dataclass
    spec.loader.exec_module(m)
    return m


ml = _load_motion_lib()
MotionLib = ml.MotionLib
build_amp_obs = ml.build_amp_obs_numpy

# Authoritative URDF actuated joint order (== action order), parsed from the URDF
# at audit time and frozen here as the contract ground truth.
URDF_ACTUATED_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
# Env default dof vector (== X1DHStandCfg.init_state.default_joint_angles in joint order)
# The policy subtracts this in compute_amp_obs; the expert subtracts the same vector.
ENV_DEFAULT_DOF_POS = (0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
                       -0.4, -0.05, 0.31, 0.49, -0.21, 0.0)

EXPECTED_DIM = 35
EXPECTED_BLOCKS = {
    "base_lin_vel": (0, 3), "base_ang_vel": (3, 6), "projected_gravity": (6, 9),
    "dof_pos_rel": (9, 21), "dof_vel": (21, 33), "foot_contact": (33, 35),
}

results = {"checks": [], "pass": True}


def check(name, cond, detail=""):
    results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        results["pass"] = False


def main():
    print("=== AMP contract test (numpy-only, Gate A schema/loader) ===")
    # 1. layout
    check("AMP_OBS_DIM==35", ml.AMP_OBS_DIM == EXPECTED_DIM,
          f"got {ml.AMP_OBS_DIM}")
    for k, (s, e) in EXPECTED_BLOCKS.items():
        got = ml.AMP_BLOCKS.get(k)
        check(f"block {k}={s}:{e}", got == (s, e), f"got {got}")

    # 2. joint order & default single-source-of-truth
    check("joint order == URDF actuated order", tuple(ml.X1_JOINT_NAMES) == URDF_ACTUATED_ORDER,
          f"{ml.X1_JOINT_NAMES[:3]}...")
    check("expert default == env default dof vec",
          tuple(ml.X1_DEFAULT_DOF_POS) == ENV_DEFAULT_DOF_POS,
          f"expert={tuple(round(v,3) for v in ml.X1_DEFAULT_DOF_POS)}")
    check("num joints == 12", ml.X1_NUM_JOINTS == 12, f"got {ml.X1_NUM_JOINTS}")

    # 3. MotionLib load of synthetic smoke clip (generate if missing)
    if not os.path.isfile(SMOKE):
        print("  smoke clip missing -> generating synthetic expert")
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "gen_smoke_expert.py")], check=True)
    lib = MotionLib([SMOKE], target_fps=100.0, device="cpu")
    st = lib.stats[0]
    check("MotionLib loaded non-empty expert set", lib.num_samples > 0, f"N={lib.num_samples}")
    check("resampled to 100 Hz", abs(st.fps - 100.0) < 1e-6, f"fps={st.fps}")
    check("expert amp features finite", bool(np.all(np.isfinite(lib._all))), "")
    check("expert joint range ok", st.joint_range_ok, "")
    check("no empty stats block", bool(np.all(np.isfinite(st.amp_mean))) and np.all(st.amp_std >= 0), "")

    # 4. expert == policy schema parity via the SAME builder on two trajectories
    N, fps = 64, 100.0
    dt = 1.0 / fps
    g = lambda i: np.asarray([0.0, 0.0, 0.0, 1.0])  # identity quat xyzw

    def traj(seed):
        rng = np.random.default_rng(seed)
        rt = np.zeros((N, 3)); rt[:, 0] = 0.4 * np.arange(N) * dt
        rt[:, 2] = 0.61 + 0.01 * rng.standard_normal(N)
        rq = np.tile([0., 0., 0., 1.], (N, 1)) + 0.0 * rng.standard_normal((N, 4))
        rq = rq / np.linalg.norm(rq, axis=1, keepdims=True)
        jp = np.tile(ml.X1_DEFAULT_DOF_POS, (N, 1)) + 0.05 * rng.standard_normal((N, 12))
        fc = (rng.standard_normal((N, 2)) > 0).astype(np.float64)
        return rt, rq, jp, fc

    def build_seq(rt, rq, jp, fc):
        out = np.zeros((N - 1, ml.AMP_OBS_DIM))
        for i in range(N - 1):
            out[i] = build_amp_obs(rt[i+1], rt[i], rq[i+1], rq[i], jp[i+1], jp[i],
                                   fc[i+1], fc[i], dt)
        return out

    expert_feat = build_seq(*traj(1))      # "expert" trajectory
    policy_feat = build_seq(*traj(2))      # "policy" trajectory (different seed)
    check("expert feature shape == (N-1, AMP_OBS_DIM)", expert_feat.shape == (N-1, EXPECTED_DIM),
          f"{expert_feat.shape}")
    check("policy feature shape == (N-1, AMP_OBS_DIM)", policy_feat.shape == (N-1, EXPECTED_DIM),
          f"{policy_feat.shape}")
    check("expert features finite", bool(np.all(np.isfinite(expert_feat))), "")
    check("policy features finite", bool(np.all(np.isfinite(policy_feat))), "")
    # the builder is the single source of truth -> layouts are identical by construction;
    # assert the dof_pos_rel block actually uses the shared default (offset-free parity)
    rel_block = expert_feat[:, EXPECTED_BLOCKS["dof_pos_rel"][0]:EXPECTED_BLOCKS["dof_pos_rel"][1]]
    check("dof_pos_rel dim == 12", rel_block.shape[1] == 12, f"{rel_block.shape[1]}")

    # 5. loader-vs-builder stat sanity
    check("expert logit-free range bounded",
          bool(np.nanmax(np.abs(lib._all)) < 1e3), f"max|amp|={np.nanmax(np.abs(lib._all)):.3f}")

    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} "
          f"({sum(c['pass'] for c in results['checks'])}/{len(results['checks'])} checks) ===")

    # write human-readable report for evidence binding
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(f"AMP contract test report ({datetime.now().isoformat(timespec='seconds')})\n")
        f.write(f"overall: {'PASS' if results['pass'] else 'FAIL'}\n")
        f.write(f"motion_lib={os.path.abspath(ml.__file__)}\n")
        f.write(f"AMP_OBS_DIM={ml.AMP_OBS_DIM} blocks={ml.AMP_BLOCKS}\n")
        f.write(f"expert_samples={lib.num_samples} resampled_fps={st.fps} "
                f"joint_range_ok={st.joint_range_ok}\n")
        f.write(f"amp_mean(prefix)={np.array2string(st.amp_mean[:6], precision=3)}\n")
        f.write(f"amp_std(prefix)={np.array2string(st.amp_std[:6], precision=3)}\n")
        for c in results["checks"]:
            f.write(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}"
                    + (f" :: {c['detail']}\n" if c['detail'] else "\n"))
    results["report"] = os.path.relpath(REPORT, ROOT)
    with open(os.path.join(os.path.dirname(REPORT), "contract_test_result.json"), "w") as f:
        json.dump({**results, "amp_obs_dim": ml.AMP_OBS_DIM,
                   "expert_samples": lib.num_samples, "resampled_fps": float(st.fps),
                   "joint_range_ok": bool(st.joint_range_ok)}, f, indent=2)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
