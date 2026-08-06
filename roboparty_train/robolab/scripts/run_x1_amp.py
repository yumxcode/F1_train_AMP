#!/usr/bin/env python3
"""Entry script for X1 AMP training on Gradmotion.

Pipeline:
  1. Retarget GMR motion data (rpo format) → X1 Isaac Lab format (if not done)
  2. Train X1 AMP policy

Usage (via gm-run):
    gm-run roboparty_train/robolab/scripts/run_x1_amp.py --headless [--max_iterations N]
"""

import os
import sys
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """Return the robolab submodule root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def _motion_data_exists(motion_dir: Path) -> bool:
    if not motion_dir.exists():
        return False
    pkls = list(motion_dir.glob("*.pkl"))
    return len(pkls) > 0


def run_retargeting(robolab_root: Path, extra_args: list[str]):
    """Run GMR → X1 retargeting via dataset_retarget.py as a subprocess."""
    script = robolab_root / "scripts" / "tools" / "retarget" / "dataset_retarget.py"
    input_dir = robolab_root / "data" / "motions" / "rpo_gmr"
    output_dir = robolab_root / "data" / "motions" / "x1_lab"
    config_file = robolab_root / "scripts" / "tools" / "retarget" / "config" / "x1.yaml"

    cmd = [
        sys.executable, str(script),
        "--robot", "x1",
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--config_file", str(config_file),
        "--loop", "clamp",
        "--headless",
    ]
    cmd.extend(extra_args)
    print("[run_x1_amp] Retargeting command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(robolab_root.parent))


def run_training(robolab_root: Path, extra_args: list[str]):
    """Run AMP training via train.py as a subprocess."""
    script = robolab_root / "scripts" / "rsl_rl" / "train.py"
    cmd = [
        sys.executable, str(script),
        "--task", "X1-AMP",
    ]
    cmd.extend(extra_args)
    print("[run_x1_amp] Training command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(robolab_root.parent))


def main():
    robolab_root = _repo_root()
    # Forward any unknown args (e.g. --headless, --max_iterations, --num_envs)
    extra_args = sys.argv[1:]

    motion_dir = robolab_root / "data" / "motions" / "x1_lab"

    # ------------------------------------------------------------------
    # Step 1: Retargeting (skip if motion data already exists)
    # ------------------------------------------------------------------
    if _motion_data_exists(motion_dir):
        print(f"[run_x1_amp] Motion data already exists in {motion_dir}, skipping retargeting.")
    else:
        print("=" * 60)
        print("STEP 1: Retargeting GMR motion data to X1 format")
        print("=" * 60)
        run_retargeting(robolab_root, extra_args)

    # ------------------------------------------------------------------
    # Step 2: Training
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Training X1 AMP")
    print("=" * 60)
    run_training(robolab_root, extra_args)


if __name__ == "__main__":
    main()
