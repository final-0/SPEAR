#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4320
#SBATCH --job-name=robust-step1-router
#SBATCH -o logs/%x_%j.log
# Robust step 1: train the K=3 token router on the frozen parent -> $SPEAR_OUTPUT/robust-step1-router/
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
spear_robust_train "${SPEAR_OUTPUT:?}/robust-step1-router" --stage router --steps 10000 --save-every 5000
