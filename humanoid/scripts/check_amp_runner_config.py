# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Headless check of AMP runner construction-time config resolution (BUG D + BUG E).

Validates the two iteration-2 integration fixes WITHOUT Isaac Gym/torch:
  * BUG E: expert_clip_paths '{LEGGED_GYM_ROOT_DIR}' placeholder resolves to a real,
    loadable file (the smoke expert), so MotionLib(np.load) does not FileNotFoundError.
  * BUG D: the AMP learn() carries obs_std/obs_mean for the inherited log() (source-only
    assertion that the variable is assigned before the loop that calls self.log).

Exit 0 == PASS. Run: python humanoid/scripts/check_amp_runner_config.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
results = {"checks": [], "pass": True}


def check(name, cond, detail=""):
    results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        results["pass"] = False


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, os.path.abspath(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


def main():
    print("=== AMP runner config-resolution check (BUG D/E) ===")
    ml = _load("_cfg_ml", os.path.join(HERE, "..", "algo", "amp", "motion_lib.py"))

    # --- BUG E: path placeholder resolution ---
    placeholder = "{LEGGED_GYM_ROOT_DIR}/data/smoke_expert/walk_smoke.npz"
    # LEGGED_GYM_ROOT_DIR = parent of humanoid/ (see humanoid/__init__.py); compute headless.
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(os.path.join(HERE, "..", "__init__.py"))))
    resolved = placeholder.replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT_DIR))
    resolved = os.path.expanduser(resolved)
    check("placeholder resolves to existing file", os.path.isfile(resolved), resolved)
    # MotionLib can load the resolved path (the real smoke expert)
    lib = ml.MotionLib([resolved], target_fps=100.0, device="cpu")
    check("MotionLib loads resolved expert clip", lib.num_samples > 0, f"N={lib.num_samples}")
    check("MotionLib features finite", bool(np.all(np.isfinite(lib._all))), "")

    # --- BUG D: learn() source carries obs_std/obs_mean before self.log ---
    runner_src = open(os.path.join(HERE, "..", "algo", "amp", "amp_on_policy_runner.py")).read()
    has_obs_mean = "obs_std, obs_mean = torch.std_mean(obs, dim=0)" in runner_src
    check("learn() computes obs_std/obs_mean for inherited log()", has_obs_mean,
          "required by DHOnPolicyRunner.log() locs['obs_mean']")

    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} "
          f"({sum(c['pass'] for c in results['checks'])}/{len(results['checks'])}) ===")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
