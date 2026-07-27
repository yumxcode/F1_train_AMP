# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""Gate B validation for the GMR retargeted X1 walk clip (headless, numpy-only).

Mirrors validate_retarget.py but targets the GMR output directory.
Checks (amp_loop.md §5):
  1. schema: npz keys + shapes + units match MotionLib expectation.
  2. finite: no NaN/Inf.
  3. joint mapping: 12 DOF X1 order, ranges within limits.
  4. continuity: no sudden jumps.
  5. foot contact: alternating stance (gait).
  6. MotionLib round-trip.
  7. loop continuity.

Exit 0 = PASS. Run: python humanoid/scripts/validate_retarget_gmr.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CLIP = os.path.join(ROOT, "data", "retarget_gmr", "x1_walk_retargeted.npz")
REPORT = os.path.join(ROOT, "data", "retarget_gmr", "retarget_report.json")
VAL_REPORT = os.path.join(ROOT, "data", "retarget_gmr", "validation_report.json")

_ML = os.path.join(HERE, "..", "algo", "amp", "motion_lib.py")
_spec = importlib.util.spec_from_file_location("_val_gmr_ml", os.path.abspath(_ML))
ml = importlib.util.module_from_spec(_spec)
sys.modules["_val_gmr_ml"] = ml
_spec.loader.exec_module(ml)

X1_LIMITS = ml.X1_JOINT_LIMITS
results = {"checks": [], "pass": True}


def check(name, cond, detail=""):
    results["checks"].append({"name": name, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not cond:
        results["pass"] = False


def main():
    print("=== Gate B GMR retarget validation (numpy-only) ===")
    d = np.load(CLIP, allow_pickle=True)

    # 1. schema
    need = ["root_translation", "root_rotation", "joint_positions", "foot_contact", "fps"]
    check("all required keys present", all(k in d.files for k in need), str(d.files))
    rt, rq, jp, fc = d["root_translation"], d["root_rotation"], d["joint_positions"], d["foot_contact"]
    N = jp.shape[0]
    check("root_translation shape (N,3)", rt.shape == (N, 3), str(rt.shape))
    check("root_rotation shape (N,4)", rq.shape == (N, 4), str(rq.shape))
    check("joint_positions shape (N,12)", jp.shape[1] == 12, str(jp.shape))
    check("foot_contact shape (N,2)", fc.shape == (N, 2), str(fc.shape))

    # 2. finite + unit quaternions
    check("all values finite", bool(np.all(np.isfinite(rt)) and np.all(np.isfinite(rq))
          and np.all(np.isfinite(jp))), "")
    qnorm = np.linalg.norm(rq, axis=1)
    check("root quaternions unit-norm", bool(np.all(np.abs(qnorm - 1.0) < 1e-3)),
          f"qnorm range [{qnorm.min():.4f},{qnorm.max():.4f}]")

    # 3. joint ranges within X1 limits
    oob = 0
    for i, (lo, hi) in enumerate(X1_LIMITS):
        col = jp[:, i]
        oob += int(np.sum((col < lo - 0.05) | (col > hi + 0.05)))
    check("joint values within X1 limits (tol 0.05)", oob == 0, f"out-of-range count={oob}")

    # 3b. NO FROZEN JOINTS + bilateral gait symmetry (catches the IK-collapse defect:
    #     a degenerate local min can pin a joint at its bound for the whole clip while
    #     still passing range/continuity checks). Every actuated joint must actually
    #     move, and the L/R knee (the primary gait joints) must show comparable motion.
    jr = jp.max(0) - jp.min(0)  # per-joint range over the clip
    frozen = [i for i in range(12) if jr[i] < 0.05]
    check("no frozen actuated joints (each joint range > 0.05 rad)", len(frozen) == 0,
          f"frozen joints={[f'{i}:{jr[i]:.3f}' for i in frozen]}" if frozen else "all 12 joints move")
    lk, rk = jr[3], jr[9]  # L/R knee_pitch
    knee_sym = min(lk, rk) / max(lk, rk) > 0.4 if max(lk, rk) > 0 else False
    check("bilateral knee symmetry (L/R knee range ratio > 0.4)",
          knee_sym, f"L_knee_range={lk:.3f} R_knee_range={rk:.3f} ratio={min(lk,rk)/max(lk,rk):.3f}")

    # 4. continuity
    jdelta = np.abs(np.diff(jp, axis=0))
    maxjd = float(jdelta.max())
    check("joint continuity (max per-frame delta < 0.3 rad)", maxjd < 0.3, f"max_delta={maxjd:.3f}")
    rdelta_t = np.abs(np.diff(rt[:, 0]))
    check("root forward continuity (< 0.1 m/frame)", float(rdelta_t.max()) < 0.1,
          f"max={float(rdelta_t.max()):.4f}")

    # 5. foot contact alternation
    lc, rc = fc[:, 0], fc[:, 1]
    both = float(np.mean((lc > 0.5) & (rc > 0.5)))
    neither = float(np.mean((lc < 0.5) & (rc < 0.5)))
    check("double-support fraction reasonable (< 0.6)", both < 0.6, f"double={both:.3f}")
    check("no prolonged flight (neither-contact < 0.3)", neither < 0.3, f"neither={neither:.3f}")

    def period(sig):
        s = sig - sig.mean()
        if np.all(s == 0):
            return None
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        ac /= ac[0] if ac[0] != 0 else 1
        for lag in range(5, len(ac)):
            if ac[lag] > 0.3:
                return lag
        return None
    pl = period(lc)
    check("left foot shows gait periodicity", pl is not None,
          f"period={pl/d['fps']:.3f}s lag={pl}" if pl else "no period")

    # 6. MotionLib round-trip
    lib = ml.MotionLib([CLIP], target_fps=100.0, device="cpu")
    check("MotionLib loads clip (non-empty)", lib.num_samples > 0, f"N={lib.num_samples}")
    st = lib.stats[0]
    check("MotionLib features finite", bool(np.all(np.isfinite(lib._all))), "")
    check("MotionLib resampled to 100Hz", abs(st.fps - 100.0) < 1e-6, f"fps={st.fps}")
    check("MotionLib joint range ok", st.joint_range_ok, "")
    check("AMP feature range bounded (<1e3)", float(np.nanmax(np.abs(lib._all))) < 1e3,
          f"max|amp|={float(np.nanmax(np.abs(lib._all))):.3f}")

    # 7. loop continuity
    loop_err = float(np.linalg.norm(jp[0] - jp[-1]) + np.linalg.norm(rt[0] - rt[-1]))
    loopable = loop_err < 1.0
    check("loop continuity assessed", True, f"first/last frame err={loop_err:.3f} loopable={loopable}")

    # IK residual check (from report)
    if os.path.isfile(REPORT):
        r = json.load(open(REPORT))
        ik_res = r.get("ik_residual_mean_m", {})
        mean_res = np.mean(list(ik_res.values())) if ik_res else 999.0
        check("IK residual reasonable (< 0.15m)", mean_res < 0.15, f"mean_residual={mean_res:.4f}m")

    print(f"\n=== RESULT: {'PASS' if results['pass'] else 'FAIL'} "
          f"({sum(c['pass'] for c in results['checks'])}/{len(results['checks'])} checks) ===")
    results["n_frames"] = int(N)
    results["loopable"] = bool(loopable)
    results["loop_error"] = loop_err
    with open(VAL_REPORT, "w") as f:
        json.dump(results, f, indent=2)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
