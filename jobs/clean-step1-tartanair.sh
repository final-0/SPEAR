#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=clean-step1-tartanair
#SBATCH -o logs/%x_%j.log
# Clean step 1: TartanAir, 100k steps from the WAFT release checkpoint -> checkpoints/clean-step1-tartanair/
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${TARTAN_ROOT:?}" "${SPRING_ROOT:?}"
spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TARTAN_ROOT="${TARTAN_ROOT}" SPRING_ROOT="${SPRING_ROOT}" \
    "${SPEAR_PYTHON}" -u train.py --cfg config/clean-step1-tartanair.json
