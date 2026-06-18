#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTDIR="${OUTDIR:-fdong_embedding_dim/outputs/common_hurts_tail_2d_control}"
STEPS="${STEPS:-2000}"
RECORD_EVERY="${RECORD_EVERY:-1}"
SEED="${SEED:-0}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp}"

run_one() {
  local name="$1"
  local probs="$2"
  echo "================================================================================"
  echo "Running ${name}"
  "${PYTHON_BIN}" -u fdong_embedding_dim/scripts/two_dimension_testnew.py \
    --experiment_name "${name}" \
    --group_probs "${probs}" \
    --init_layout spread \
    --steps "${STEPS}" \
    --record_every "${RECORD_EVERY}" \
    --seed "${SEED}" \
    --outdir "${OUTDIR}"
}

# Strict control: same model, initialization, seed, steps, and four data rules.
# Only the training frequency distribution changes.
run_one \
  "four_group_uniform_1_1_1_1_spread" \
  "common:0.25,tail1:0.25,tail2:0.25,tail3:0.25"

run_one \
  "four_group_zipf_7_1_1_1_spread" \
  "common:0.70,tail1:0.10,tail2:0.10,tail3:0.10"

echo "================================================================================"
echo "Finished. Results: ${REPO_ROOT}/${OUTDIR}"
