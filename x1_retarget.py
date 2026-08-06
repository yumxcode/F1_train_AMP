#!/usr/bin/env python3
"""X1 retarget entry - installs robolab with minimal deps (no mujoco/opencv/onnxruntime)."""
import subprocess, sys, os

def run(cmd):
    print(f"[x1_retarget] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

cwd = os.getcwd()
print(f"[x1_retarget] CWD={cwd}", flush=True)

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
    print("[x1_retarget] FATAL: roboparty_train not found!", flush=True)
    sys.exit(1)

print(f"[x1_retarget] Found: {repo_base}", flush=True)

# Install robolab with minimal deps
robolab_dir = os.path.join(repo_base, "robolab")
# Temporarily replace setup.py with minimal version
original_setup = os.path.join(robolab_dir, "setup.py")
minimal_setup = os.path.join(robolab_dir, "setup_minimal.py")
with open(original_setup, 'r') as f:
    original_content = f.read()
with open(minimal_setup, 'w') as f:
    f.write('''from setuptools import setup, find_packages
setup(name="robolab", packages=find_packages(), version="1.0.0",
      install_requires=["joblib>=1.2.0", "prettytable", "pyyaml"])''')

try:
    # Remove egg-info to force rebuild
    import shutil
    egg_info = os.path.join(robolab_dir, "robolab.egg-info")
    if os.path.isdir(egg_info):
        shutil.rmtree(egg_info)
    
    run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q", "--no-deps"])
    # Install only needed deps manually
    run([sys.executable, "-m", "pip", "install", "joblib", "prettytable", "pyyaml", "-q"])
finally:
    # Restore original setup.py
    os.rename(original_setup, original_setup + ".bak")
    os.rename(minimal_setup, original_setup)

# Install rsl_rl with --no-deps
rsl_rl_dir = os.path.join(repo_base, "rsl_rl")
run([sys.executable, "-m", "pip", "install", "-e", rsl_rl_dir, "-q", "--no-deps"])

# Run retarget
script = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
config = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
input_dir = os.path.join(repo_base, "robolab", "data", "motions", "rpo_gmr")
output_dir = os.path.join(repo_base, "robolab", "data", "motions", "x1_lab")

run([sys.executable, script, "--robot", "x1", "--input_dir", input_dir,
     "--output_dir", output_dir, "--config_file", config, "--loop", "clamp", "--headless"])

print("[x1_retarget] RETARGET COMPLETE!", flush=True)
