# SPEAR container: NGC PyTorch 25.04 (CUDA 12.8, PyTorch 2.7, Python 3.12).
# Stage-2 packages (scikit-image, imagecorruptions) are added by jobs/prepare-step1-venv.sh.
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --no-deps keeps the NGC torch build (xformers would otherwise pull a different torch).
RUN pip install xformers==0.0.30 --no-deps

# Versions used for our runs.
RUN pip install \
      "h5py==3.16.0" \
      "opencv-python-headless==4.11.0.86" \
      "timm==1.0.28"

CMD ["bash"]
