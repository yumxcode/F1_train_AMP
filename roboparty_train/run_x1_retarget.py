#!/usr/bin/env python3
"""X1 retarget entry: install deps then run retarget.

Called via gm-run with mainWorkDir=roboparty_train.
Working directory is roboparty_train/, so paths are relative from there.
"""
import subprocess
import sys
import os


def run(cmd, cwd=None):
    """Run a command and stream output."""
    print(f"[run_x1_retarget] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"[run_x1_retarget] FAILED exit={result.returncode}", flush=True)
        sys.exit(result.returncode)


def main():
    # mainWorkDir = roboparty_train, so cwd is roboparty_train/
    cwd = os.getcwd()
    print(f"[run_x1_retarget] CWD={cwd}", flush=True)
    print(f"[run_x1_retarget] Contents: {os.listdir(cwd)}", flush=True)

    # Step 1: Install robolab
    print("=" * 60, flush=True)
    print("STEP 1: Installing robolab + rsl_rl", flush=True)
    print("=" * 60, flush=True)

    robolab_dir = os.path.join(cwd, "robolab")
    rsl_rl_dir = os.path.join(cwd, "rsl_rl")
    print(f"robolab_dir exists: {os.path.isdir(robolab_dir)} ({robolab_dir})", flush=True)
    print(f"rsl_rl_dir exists: {os.path.isdir(rsl_rl_dir)} ({rsl_rl_dir})", flush=True)

    if os.path.isdir(robolab_dir):
        run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q"])
    if os.path.isdir(rsl_rl_dir):
        run([sys.executable, "-m", "pip", "install", "-e", rsl_rl_dir, "-q"])

    # Step 2: Run retarget
    print("=" * 60, flush=True)
    print("STEP 2: Running X1 retarget", flush=True)
    print("=" * 60, flush=True)

    retarget_script = os.path.join(cwd, "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
    config_file = os.path.join(cwd, "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
    input_dir = os.path.join(cwd, "robolab", "data", "motions", "rpo_gmr")
    output_dir = os.path.join(cwd, "robolab", "data", "motions", "x1_lab")

    print(f"retarget_script exists: {os.path.isfile(retarget_script)}", flush=True)
    print(f"config_file exists: {os.path.isfile(config_file)}", flush=True)
    print(f"input_dir exists: {os.path.isdir(input_dir)}", flush=True)

    run([
        sys.executable, retarget_script,
        "--robot", "x1",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--config_file", config_file,
        "--loop", "clamp",
        "--headless",
    ])

    print("=" * 60, flush=True)
    print("RETARGET COMPLETE!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
