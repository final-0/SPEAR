#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120
#SBATCH --job-name=prepare-step3-weather-overlays
#SBATCH -o logs/%x_%j.log
# Extracts the RobustSpring frost / rain / snow layers into $WEATHER_OVERLAY_DIR (no GPU work).
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPRING_ROOT:?}" "${ROBUST_SPRING_ROOT:?}" "${WEATHER_OVERLAY_DIR:?}"
for corruption in frost rain snow; do
    spear_run "${SPEAR_PYTHON}" scripts/extract_weather_overlays.py --corruption "${corruption}" --num 120 \
        --clean-root "${SPRING_ROOT}/test" --robust-root "${ROBUST_SPRING_ROOT}" --out-dir "${WEATHER_OVERLAY_DIR}"
done
# Must match the overlays used for our training runs.
( cd "${WEATHER_OVERLAY_DIR}" && sha256sum --check --quiet "${SPEAR_ROOT}/config/weather_overlays.sha256" )
echo "weather overlays match config/weather_overlays.sha256"
