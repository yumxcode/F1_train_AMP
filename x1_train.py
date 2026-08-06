#!/usr/bin/env python3
"""X1 AMP training entry: install deps, verify motion data, run training."""
import subprocess, sys, os

def run(cmd):
    print(f"[x1_train] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

cwd = os.getcwd()
print(f"[x1_train] CWD={cwd}", flush=True)

# Find roboparty_train
repo_base = None
for p in [os.path.join(cwd, "roboparty_train"),
          os.path.join(cwd, "F1_train_AMP", "roboparty_train"),
          "/workspace/isaaclab/F1_train_AMP/roboparty_train"]:
    if os.path.isdir(os.path.join(p, "robolab")):
        repo_base = p
        break

if repo_base is None:
    for root, dirs, _ in os.walk("/workspace", topdown=True, maxdepth=5):
        if "robolab" in dirs and "setup.py" in os.listdir(os.path.join(root, "robolab")):
            repo_base = root
            break
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '_isaac_sim']]

if repo_base is None:
    print("[x1_train] FATAL: roboparty_train not found!", flush=True)
    sys.exit(1)

print(f"[x1_train] Found: {repo_base}", flush=True)

# Install robolab with --no-deps
robolab_dir = os.path.join(repo_base, "robolab")
rsl_rl_dir = os.path.join(repo_base, "rsl_rl")

# Use minimal setup for robolab
import shutil
egg_info = os.path.join(robolab_dir, "robolab.egg-info")
if os.path.isdir(egg_info):
    shutil.rmtree(egg_info)

original_setup = os.path.join(robolab_dir, "setup.py")
with open(original_setup, 'w') as f:
    f.write('from setuptools import setup, find_packages\nsetup(name="robolab", packages=find_packages(), version="1.0.0", install_requires=["joblib>=1.2.0", "prettytable", "pyyaml"])')

run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q", "--no-deps"])
run([sys.executable, "-m", "pip", "install", "joblib", "prettytable", "pyyaml", "-q"])
run([sys.executable, "-m", "pip", "install", "-e", rsl_rl_dir, "-q", "--no-deps"])

# Verify motion data exists
x1_lab = os.path.join(repo_base, "robolab", "data", "motions", "x1_lab")
if os.path.isdir(x1_lab):
    pkls = [f for f in os.listdir(x1_lab) if f.endswith('.pkl')]
    print(f"[x1_train] Found {len(pkls)} motion files in x1_lab/", flush=True)
else:
    print(f"[x1_train] WARNING: x1_lab not found at {x1_lab}", flush=True)

# Run training
train_script = os.path.join(repo_base, "robolab", "scripts", "rsl_rl", "train.py")
run([sys.executable, train_script, "--task", "X1-AMP", "--headless",
     "--num_envs", "4096", "--max_iterations", "2000"])

print("[x1_train] TRAINING COMPLETE!", flush=True)
