#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=clean-step2-tartanair
#SBATCH -o logs/%x_%j.log
# Clean step 2: TartanAir, 200k more steps at lr 1e-4 -> checkpoints/clean-step2-tartanair/
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${TARTAN_ROOT:?}" "${SPRING_ROOT:?}"
spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TARTAN_ROOT="${TARTAN_ROOT}" SPRING_ROOT="${SPRING_ROOT}" \
    "${SPEAR_PYTHON}" -u train.py --cfg config/clean-step2-tartanair.json
