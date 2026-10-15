#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120
#SBATCH --job-name=prepare-step1-venv
#SBATCH -o logs/%x_%j.log
# Creates $SPEAR_VENV inside the container, then runs the CPU unit tests as an environment check.
set -eux
source "${SPEAR_ROOT:?source spear_env.sh first (see README)}/jobs/common.sh"
: "${SPEAR_VENV:?}"
spear_exec python -m venv --system-site-packages "${SPEAR_VENV}"
spear_exec "${SPEAR_VENV}/bin/python" -m pip install --no-cache-dir \
    "scikit-image==0.24.0" "imagecorruptions==1.1.2" "opencv-python==4.11.0.86" "setuptools>=65,<81"
spear_run "${SPEAR_VENV}/bin/python" -c \
    "import torch,timm,numpy,h5py,skimage,cv2,PIL,imagecorruptions; print('imports OK; torch', torch.__version__)"
spear_run "${SPEAR_VENV}/bin/python" -m unittest robust_finetune.test_native
