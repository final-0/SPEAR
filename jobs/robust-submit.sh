#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=1440
#SBATCH --job-name=robust-submit
#SBATCH -o logs/%x_%j.log
# RoCo test inference: clean + 20 corruptions -> flo5 (~1.1 TB).  Re-running overwrites OUTPUT.  Env: CKPT, OUTPUT, BRANCHES.
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}" "${ROBUST_SPRING_ROOT:?}" "${SPEAR_OUTPUT:?}" "${SPEAR_VENV:?}"
CKPT="${CKPT:-${SPEAR_OUTPUT}/robust-step2-adapters/step-005000.pt}"
OUTPUT="${OUTPUT:-${SPEAR_OUTPUT}/robust-submission}"
GPUS="${SLURM_GPUS_ON_NODE:-8}"
mkdir -p "${OUTPUT}"
SPRING_DATA="$(spear_stage "${SPRING_ROOT}" spring test)"
ROBUST_DATA="$(spear_stage "${ROBUST_SPRING_ROOT}" robust_spring .)"
pids=()
for i in $(seq 0 $((GPUS - 1))); do
    spear_run env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${i}" \
        "${SPEAR_PYTHON}" -m robust_finetune.submit_robust --checkpoint "${CKPT}" \
        --spring-root "${SPRING_DATA}" --robust-root "${ROBUST_DATA}" --output "${OUTPUT}" \
        --num-shards "${GPUS}" --shard-id "${i}" ${BRANCHES:+--branches ${BRANCHES}} &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done; exit "${rc}"
