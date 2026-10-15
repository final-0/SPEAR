#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120
#SBATCH --job-name=clean-eval
#SBATCH -o logs/%x_%j.log
# Clean 2-frame EPE on Spring train/val.  Env: CKPT, SPLIT (train|val|all), N_PAIRS.
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}"
CKPT="${CKPT:-checkpoints/clean-step4-spring/waft-a2/dav2/42/20000.pth}"
spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SPRING_ROOT="${SPRING_ROOT}" PYTHONPATH="${SPEAR_ROOT}" \
    "${SPEAR_PYTHON}" -u scripts/evaluate_spring.py --ckpt "${CKPT}" --split "${SPLIT:-all}" --n_pairs "${N_PAIRS:-5}"
