# Sourced by every Slurm job; all machine-specific settings come from spear_env.sh (see README).
: "${SPEAR_ROOT:?}" "${SPEAR_CONTAINER:?}"
# Optional site setup, e.g. SPEAR_SETUP="module load apptainer" (see spear_env.sh.example).
if [ -n "${SPEAR_SETUP:-}" ]; then
    eval "${SPEAR_SETUP}"
fi
SPEAR_PYTHON="${SPEAR_VENV:+${SPEAR_VENV}/bin/}python"

spear_exec() {
    singularity exec --nv ${SPEAR_BIND:+--bind "${SPEAR_BIND}"} "${SPEAR_CONTAINER}" "$@"
}

# spear_run CMD...: run CMD inside the container, from the repository root.
spear_run() {
    spear_exec bash -c 'cd "$1" && shift && exec "$@"' _ "${SPEAR_ROOT}" "$@"
}

# spear_stage SRC NAME SUBDIR...: copy SUBDIRs of SRC to node-local scratch and print the staged root.
# Prints SRC unchanged when SPEAR_LOCAL_SCRATCH is unset (datasets are then read in place).
spear_stage() {
    local src="$1" name="$2" sub dst
    shift 2
    if [ -z "${SPEAR_LOCAL_SCRATCH:-}" ]; then echo "${src}"; return 0; fi
    dst="${SPEAR_LOCAL_SCRATCH}/${SLURM_JOB_ID:-$$}/${name}"
    for sub in "$@"; do
        mkdir -p "${dst}/${sub}"
        rsync -a "${src}/${sub}/" "${dst}/${sub}/"
        [ "$(find "${src}/${sub}" -type f | wc -l)" -eq "$(find "${dst}/${sub}" -type f | wc -l)" ] \
            || { echo "ERROR: incomplete copy of ${src}/${sub}" >&2; return 1; }
    done
    echo "${dst}"
}

# spear_robust_train RUN_DIR ARGS...: SPEAR stage-2 training, one process per GPU, global batch 8.
spear_robust_train() {
    local run_dir="$1" data
    shift
    : "${SPRING_ROOT:?}" "${WEATHER_OVERLAY_DIR:?}" "${SPEAR_VENV:?run jobs/prepare-step1-venv.sh first}"
    data="$(spear_stage "${SPRING_ROOT}" spring train)"   # stage 2 reads train/ only
    mkdir -p "$(dirname "${run_dir}")"
    spear_run env PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native}" WEATHER_OVERLAY_DIR="${WEATHER_OVERLAY_DIR}" \
        "${SPEAR_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-8}" \
        --module robust_finetune.train --spring-root "${data}" --output "${run_dir}" \
        --global-batch 8 --micro-batch 1 --precision auto --workers 2 "$@"
}
