#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=60
#SBATCH --job-name=prepare-step2-spring-val
#SBATCH -o logs/%x_%j.log
# Moves the first frames of every Spring train sequence into $SPRING_ROOT/val (run once; no GPU work).
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}"
spear_run env SPRING_ROOT="${SPRING_ROOT}" "${SPEAR_PYTHON}" scripts/split_spring_val.py
