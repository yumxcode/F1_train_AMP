#!/bin/bash
# F1 X1 Training Script — OMA v2
# Usage: bash train_f1_v2.sh [run_name]
# 
# Prerequisites:
#   - Python 3.8 env with isaacgym, pytorch 1.13, cuda 11.7
#   - This repo cloned at latest commit
#
# Steps:
#   1. git pull
#   2. Activate conda env
#   3. Run training

set -e

RUN_NAME=${1:-"f1_v2_$(date +%m%d_%H%M)"}
TASK="x1_dh_stand"
EXPERIMENT_NAME="x1_dh_stand"
MAX_ITER=30000

echo "=========================================="
echo "  F1 X1 Training — OMA v2"
echo "  Task:      ${TASK}"
echo "  Run:       ${RUN_NAME}"
echo "  Max iter:  ${MAX_ITER}"
echo "=========================================="

# Activate conda environment (adjust path as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate isaacgym

# Run training
python scripts/train.py \
    --task=${TASK} \
    --run_name=${RUN_NAME} \
    --experiment_name=${EXPERIMENT_NAME} \
    --headless \
    --max_iterations=${MAX_ITER} \
    --rl_device=cuda:0 \
    --num_envs=4096

echo "✅ Training complete: ${RUN_NAME}"
echo "Logs at: logs/${EXPERIMENT_NAME}/${RUN_NAME}/"
