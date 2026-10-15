#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=clean-step3-flyingthings
#SBATCH -o logs/%x_%j.log
# Clean step 3: FlyingThings3D (cleanpass), 200k-step schedule; step 4 uses its 100000.pth -> checkpoints/clean-step3-flyingthings/
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${THINGS_ROOT:?}" "${SPRING_ROOT:?}"
spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 THINGS_ROOT="${THINGS_ROOT}" SPRING_ROOT="${SPRING_ROOT}" \
    "${SPEAR_PYTHON}" -u train.py --cfg config/clean-step3-flyingthings.json
