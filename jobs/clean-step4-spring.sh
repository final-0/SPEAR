#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=clean-step4-spring
#SBATCH -o logs/%x_%j.log
# Clean step 4: Spring train+val at native 1080p, 20k steps -> checkpoints/clean-step4-spring/ (the SPEAR parent)
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}"
DATA="$(spear_stage "${SPRING_ROOT}" spring train val)"
spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SPRING_ROOT="${DATA}" \
    "${SPEAR_PYTHON}" -u train.py --cfg config/clean-step4-spring.json
