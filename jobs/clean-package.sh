#!/bin/bash
# Packs clean-submit.sh output into <OUTPUT>/_uploads/flow_submission.hdf5.  Run with bash (e.g. in tmux), not sbatch.
set -euo pipefail
OUT="${1:-${SPEAR_OUTPUT:?}/clean-submission}"
TOOLS="${SUBSAMPLING_TOOLS:?set SUBSAMPLING_TOOLS (see README)}"
mkdir -p "${OUT}/_uploads"
chmod +x "${TOOLS}"/flow_*subsampling || true
( cd "${TOOLS}" && ./flow_subsampling "${OUT}" )
mv -f "${TOOLS}/flow_submission.hdf5" "${OUT}/_uploads/flow_submission.hdf5"
ls -la "${OUT}/_uploads"
