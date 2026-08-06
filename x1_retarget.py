#!/usr/bin/env python3
"""X1 retarget entry for F1_train_AMP repo. Installs deps then runs retarget."""
import subprocess, sys, os

def run(cmd):
    print(f"[x1_retarget] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

# CWD is /workspace, repo is cloned to /workspace/F1_train_AMP/
# But mainWorkDir=F1_train_AMP so gm-run cd's into /workspace/F1_train_AMP/
# Let's detect the correct path
cwd = os.getcwd()
print(f"[x1_retarget] CWD={cwd}", flush=True)
print(f"[x1_retarget] Contents: {os.listdir(cwd)}", flush=True)

# Find roboparty_train
candidates = [
    os.path.join(cwd, "roboparty_train"),
    os.path.join(cwd, "F1_train_AMP", "roboparty_train"),
    "/workspace/F1_train_AMP/roboparty_train",
    "/workspace/roboparty_train",
]
repo_base = None
for p in candidates:
    if os.path.isdir(os.path.join(p, "robolab")):
        repo_base = p
        break

if repo_base is None:
    # Search for it
    for root, dirs, files in os.walk("/workspace", topdown=True, maxdepth=4):
        if "robolab" in dirs and "setup.py" in os.listdir(os.path.join(root, "robolab")):
            repo_base = root
            break
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]

if repo_base is None:
    print("[x1_retarget] FATAL: Could not find roboparty_train/robolab directory!", flush=True)
    sys.exit(1)

print(f"[x1_retarget] Found roboparty_train at: {repo_base}", flush=True)

# Install robolab and rsl_rl
run([sys.executable, "-m", "pip", "install", "-e", os.path.join(repo_base, "robolab"), "-q"])
run([sys.executable, "-m", "pip", "install", "-e", os.path.join(repo_base, "rsl_rl"), "-q"])

# Run retarget
script = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
config = os.path.join(repo_base, "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
input_dir = os.path.join(repo_base, "robolab", "data", "motions", "rpo_gmr")
output_dir = os.path.join(repo_base, "robolab", "data", "motions", "x1_lab")

run([sys.executable, script, "--robot", "x1", "--input_dir", input_dir,
     "--output_dir", output_dir, "--config_file", config, "--loop", "clamp", "--headless"])

print("[x1_retarget] RETARGET COMPLETE!", flush=True)
