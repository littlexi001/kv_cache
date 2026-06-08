#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR="${OUTDIR:-fdong_embedding_dim/outputs/toy3d_controlled}"
STEPS="${STEPS:-2000}"
RECORD_EVERY="${RECORD_EVERY:-20}"
SEED="${SEED:-0}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp}"

run_one() {
  local name="$1"
  local probs="$2"
  local centers="$3"
  echo "================================================================================"
  echo "Running ${name}"
  "${PYTHON_BIN}" -u fdong_embedding_dim/scripts/toy_bigram_spectral_dim.py \
    --experiment_name "${name}" \
    --group_probs "${probs}" \
    --tail_centers="${centers}" \
    --dim 3 \
    --steps "${STEPS}" \
    --record_every "${RECORD_EVERY}" \
    --seed "${SEED}" \
    --outdir "${OUTDIR}"
}

# 1.1: common + 3 uniform tail groups in 3D.
run_one \
  "toy3d_tail3_uniform_spread" \
  "common:0.70,tail1:0.10,tail2:0.10,tail3:0.10" \
  "0,1,0;0,-1,0;0,0,1"

run_one \
  "toy3d_tail3_uniform_packed_common" \
  "common:0.70,tail1:0.10,tail2:0.10,tail3:0.10" \
  "1,0,0;1,0,0;1,0,0"

run_one \
  "toy3d_tail3_uniform_packed_negative_common" \
  "common:0.70,tail1:0.10,tail2:0.10,tail3:0.10" \
  "-1,0,0;-1,0,0;-1,0,0"

# 1.2: common + 3 Zipf tail groups in 3D.
run_one \
  "toy3d_tail3_zipf_spread" \
  "common:0.70,tail1:0.20,tail2:0.07,tail3:0.03" \
  "0,1,0;0,-1,0;0,0,1"

run_one \
  "toy3d_tail3_zipf_packed_common" \
  "common:0.70,tail1:0.20,tail2:0.07,tail3:0.03" \
  "1,0,0;1,0,0;1,0,0"

echo "================================================================================"
echo "Toy 3D experiments finished. Results: ${REPO_ROOT}/${OUTDIR}"
