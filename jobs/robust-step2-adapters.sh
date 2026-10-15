#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=robust-step2-adapters
#SBATCH -o logs/%x_%j.log
# Robust step 2: train expert + temporal adapters from the step-1 router -> $SPEAR_OUTPUT/robust-step2-adapters/
# Env: INIT (router checkpoint).
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
INIT="${INIT:-${SPEAR_OUTPUT:?}/robust-step1-router/step-010000.pt}"
spear_robust_train "${SPEAR_OUTPUT}/robust-step2-adapters" --stage adapters --init "${INIT}" --steps 5000 --save-every 5000
