#!/bin/bash
# X1 retarget entry script: install deps then run retarget
set -e

echo "=== Installing robolab and rsl_rl ==="
cd /workspace/roboparty_train/robolab
pip install -e . -q 2>&1 | tail -3
echo "robolab installed"

cd /workspace/roboparty_train/rsl_rl
pip install -e . -q 2>&1 | tail -3
echo "rsl_rl installed"

cd /workspace

echo "=== Running X1 retarget ==="
python roboparty_train/robolab/scripts/tools/retarget/dataset_retarget.py \
    --robot x1 \
    --input_dir roboparty_train/robolab/data/motions/rpo_gmr \
    --output_dir roboparty_train/robolab/data/motions/x1_lab \
    --config_file roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml \
    --loop clamp \
    --headless

echo "=== Retarget complete ==="
