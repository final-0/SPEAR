#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=180
#SBATCH --job-name=robust-eval
#SBATCH -o logs/%x_%j.log
# Spring-val robustness (2-frame parent vs SPEAR).  Env: CKPT (empty = parent only), CONDITIONS, HEAVY, LIGHT, OUT.
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}" "${WEATHER_OVERLAY_DIR:?}" "${SPEAR_OUTPUT:?}" "${SPEAR_VENV:?}"
CKPT="${CKPT-${SPEAR_OUTPUT}/robust-step2-adapters/step-005000.pt}"
OUT="${OUT:-${SPEAR_OUTPUT}/robust-eval/eval-${SLURM_JOB_ID:-local}.json}"
mkdir -p "$(dirname "${OUT}")"
spear_run env PYTHONUNBUFFERED=1 WEATHER_OVERLAY_DIR="${WEATHER_OVERLAY_DIR}" \
    "${SPEAR_PYTHON}" -m robust_finetune.evaluate_val --spring-root "${SPRING_ROOT}" --output "${OUT}" \
    --conditions "${CONDITIONS:-all20}" --heavy-samples "${HEAVY:-50}" --light-samples "${LIGHT:-30}" \
    --cameras left --directions FW ${CKPT:+--checkpoint "${CKPT}"}
echo "OUTPUT: ${OUT}"
