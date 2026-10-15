#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=720
#SBATCH --job-name=clean-submit
#SBATCH -o logs/%x_%j.log
# Clean 2-frame Spring test inference (flo5).  Env: CKPT, OUTPUT.  Shards over all allocated GPUs, e.g.
#   sbatch --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 jobs/clean-submit.sh
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}" "${SPEAR_OUTPUT:?}"
CKPT="${CKPT:-checkpoints/clean-step4-spring/waft-a2/dav2/42/20000.pth}"
OUTPUT="${OUTPUT:-${SPEAR_OUTPUT}/clean-submission}"
GPUS="${SLURM_GPUS_ON_NODE:-1}"
mkdir -p "${OUTPUT}"
pids=()
for i in $(seq 0 $((GPUS - 1))); do
    spear_run env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SPRING_ROOT="${SPRING_ROOT}" CUDA_VISIBLE_DEVICES="${i}" \
        "${SPEAR_PYTHON}" -u submission.py --cfg config/clean-step4-spring.json --ckpt "${CKPT}" \
        --output_path "${OUTPUT}" --num_shards "${GPUS}" --shard_id "${i}" &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done; exit "${rc}"
