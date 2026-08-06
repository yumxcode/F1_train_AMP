#!/usr/bin/env python3
"""X1 retarget entry for F1_train_AMP repo. Installs deps then runs retarget."""
import subprocess, sys, os

def run(cmd):
    print(f"[x1_retarget] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

cwd = os.getcwd()
print(f"[x1_retarget] CWD={cwd}", flush=True)
print(f"[x1_retarget] Contents: {os.listdir('.')}", flush=True)

# Install robolab and rsl_rl
run([sys.executable, "-m", "pip", "install", "-e", os.path.join(cwd, "roboparty_train", "robolab"), "-q"])
run([sys.executable, "-m", "pip", "install", "-e", os.path.join(cwd, "roboparty_train", "rsl_rl"), "-q"])

# Run retarget
script = os.path.join(cwd, "roboparty_train", "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
config = os.path.join(cwd, "roboparty_train", "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
input_dir = os.path.join(cwd, "roboparty_train", "robolab", "data", "motions", "rpo_gmr")
output_dir = os.path.join(cwd, "roboparty_train", "robolab", "data", "motions", "x1_lab")

run([sys.executable, script, "--robot", "x1", "--input_dir", input_dir,
     "--output_dir", output_dir, "--config_file", config, "--loop", "clamp", "--headless"])

print("[x1_retarget] RETARGET COMPLETE!", flush=True)
