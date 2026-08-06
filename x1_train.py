#!/usr/bin/env python3
"""X1 AMP ALL-IN-ONE: retarget + train in same container.

This script runs in a single gradmotion container:
  1. Install deps (robolab, rsl_rl)
  2. Run retarget (GMR -> X1 format)
  3. Run AMP training
"""
import subprocess, sys, os, shutil

def run(cmd):
    print(f"[x1_all] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"[x1_all] FAILED exit={r.returncode}", flush=True)
        sys.exit(r.returncode)

cwd = os.getcwd()
print(f"[x1_all] CWD={cwd}", flush=True)
print(f"[x1_all] Contents: {os.listdir(cwd)}", flush=True)

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
    print("[x1_all] FATAL: roboparty_train not found!", flush=True)
    sys.exit(1)

print(f"[x1_all] Found: {repo_base}", flush=True)
robolab_dir = os.path.join(repo_base, "robolab")
rsl_rl_dir = os.path.join(repo_base, "rsl_rl")

# Step 1: Install deps with minimal requirements
print("=" * 60, flush=True)
print("STEP 1: Install dependencies", flush=True)
print("=" * 60, flush=True)

# Write minimal setup.py for robolab
with open(os.path.join(robolab_dir, "setup.py"), 'w') as f:
    f.write('from setuptools import setup, find_packages\n'
            'setup(name="robolab", packages=find_packages(), version="1.0.0",\n'
            '      install_requires=["joblib>=1.2.0", "prettytable", "pyyaml"])')

egg_info = os.path.join(robolab_dir, "robolab.egg-info")
if os.path.isdir(egg_info):
    shutil.rmtree(egg_info)

run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q", "--no-deps"])
run([sys.executable, "-m", "pip", "install", "joblib", "prettytable", "pyyaml", "-q"])
run([sys.executable, "-m", "pip", "install", "-e", rsl_rl_dir, "-q", "--no-deps"])

# Step 2: Run retarget (if x1_lab doesn't exist)
print("=" * 60, flush=True)
print("STEP 2: Retarget GMR -> X1", flush=True)
print("=" * 60, flush=True)

x1_lab = os.path.join(repo_base, "robolab", "data", "motions", "x1_lab")
if os.path.isdir(x1_lab) and len([f for f in os.listdir(x1_lab) if f.endswith('.pkl')]) > 0:
    pkls = [f for f in os.listdir(x1_lab) if f.endswith('.pkl')]
    print(f"[x1_all] x1_lab already has {len(pkls)} files, skipping retarget", flush=True)
else:
    retarget_script = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
    config_file = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
    input_dir = os.path.join(repo_base, "robolab", "data", "motions", "rpo_gmr")
    output_dir = x1_lab

    run([sys.executable, retarget_script,
         "--robot", "x1",
         "--input_dir", input_dir,
         "--output_dir", output_dir,
         "--config_file", config_file,
         "--loop", "clamp",
         "--headless"])

    # Verify output
    if os.path.isdir(x1_lab):
        pkls = [f for f in os.listdir(x1_lab) if f.endswith('.pkl')]
        print(f"[x1_all] Retarget produced {len(pkls)} pkl files", flush=True)
    else:
        print("[x1_all] FATAL: x1_lab not created!", flush=True)
        sys.exit(1)

# Step 3: Fix train.py import error (handle_deprecated_rsl_rl_cfg)
print("=" * 60, flush=True)
print("STEP 3: Patch train.py for IsaacLab 2.3.1 compatibility", flush=True)
print("=" * 60, flush=True)

train_script = os.path.join(repo_base, "robolab", "scripts", "rsl_rl", "train.py")
with open(train_script, 'r') as f:
    train_content = f.read()

# Remove problematic import and usage
train_content = train_content.replace(
    "from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg",
    "from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper"
)
train_content = train_content.replace(
    "    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)\n",
    ""
)
# Also remove the version check block that references unsupported fields
train_content = train_content.replace(
    """    # rsl-rl 3.x 的 PPO/PPOAMP 不支持 Isaac Lab 2.3 为 5.0 预留的字段
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)""",
    """    # rsl-rl 3.x compatibility: remove unsupported fields if present
    for key in ("optimizer", "share_cnn_encoders"):
        if hasattr(agent_cfg.algorithm, key):
            delattr(agent_cfg.algorithm, key)"""
)

with open(train_script, 'w') as f:
    f.write(train_content)
print("[x1_all] train.py patched", flush=True)

# Step 4: Run training
print("=" * 60, flush=True)
print("STEP 4: AMP Training", flush=True)
print("=" * 60, flush=True)

run([sys.executable, train_script, "--task", "X1-AMP", "--headless",
     "--num_envs", "4096", "--max_iterations", "2000"])

print("[x1_all] ALL DONE!", flush=True)
