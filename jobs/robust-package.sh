#!/bin/bash
# Packs robust-submit.sh output into <OUTPUT>/_uploads/{flow_submission,flow_robustness}.hdf5.
# Run with bash in tmux (several hours, single-threaded); do not interrupt.  Delete the flo5 directories afterwards.
set -euo pipefail
OUT="${1:-${SPEAR_OUTPUT:?}/robust-submission}"
OUT="${OUT%/}"
TOOLS="${SUBSAMPLING_TOOLS:?set SUBSAMPLING_TOOLS (see README)}"
mkdir -p "${OUT}/_uploads"
chmod +x "${TOOLS}"/flow_*subsampling || true
( cd "${TOOLS}" && ./flow_subsampling "${OUT}/clean/test" )
mv -f "${TOOLS}/flow_submission.hdf5" "${OUT}/_uploads/flow_submission.hdf5"
( cd "${TOOLS}" && ./flow_robust_subsampling "${OUT}" )
mv -f "${TOOLS}/flow_robustness.hdf5" "${OUT}/_uploads/flow_robustness.hdf5"
ls -la "${OUT}/_uploads"
