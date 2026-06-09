#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
RECORD_EVERY="${RECORD_EVERY:-20}"
OUTDIR="${OUTDIR:-fdong_embedding_dim/outputs/controlled_sweeps}"
SEED="${SEED:-0}"
INIT_NOISE="${INIT_NOISE:-0.001}"
RUN_FILTER="${RUN_FILTER:-}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp}"

run_one() {
  local name="$1"
  local probs="$2"
  local layout="$3"
  local centers="${4:-}"
  local noise="${5:-${INIT_NOISE}}"

  if [[ -n "${RUN_FILTER}" && "${name}" != *"${RUN_FILTER}"* ]]; then
    return 0
  fi

  echo "================================================================================"
  echo "Running ${name}"
  "${PYTHON_BIN}" -u fdong_embedding_dim/scripts/two_dimension_testnew.py \
    --experiment_name "${name}" \
    --group_probs "${probs}" \
    --init_layout "${layout}" \
    --tail_centers="${centers}" \
    --init_noise "${noise}" \
    --steps "${STEPS}" \
    --record_every "${RECORD_EVERY}" \
    --seed "${SEED}" \
    --outdir "${OUTDIR}"
}

# Family 1: common + 3 tail groups.
TAIL3_UNIFORM="common:0.70,tail1:0.10,tail2:0.10,tail3:0.10"
TAIL3_ZIPF="common:0.70,tail1:0.20,tail2:0.07,tail3:0.03"

run_one "tail3_uniform_spread" "${TAIL3_UNIFORM}" "spread" "0,1;0,-1;-1,0" "0.0"
run_one "tail3_uniform_packed_x_pos" "${TAIL3_UNIFORM}" "packed_x_pos" "1,0;1,0;1,0"
run_one "tail3_uniform_packed_x_neg" "${TAIL3_UNIFORM}" "packed_x_neg" "-1,0;-1,0;-1,0"
run_one "tail3_uniform_packed_y_pos" "${TAIL3_UNIFORM}" "packed_y_pos" "0,1;0,1;0,1"

run_one "tail3_zipf_spread" "${TAIL3_ZIPF}" "spread" "0,1;0,-1;-1,0" "0.0"
run_one "tail3_zipf_packed_x_pos" "${TAIL3_ZIPF}" "packed_x_pos" "1,0;1,0;1,0"
run_one "tail3_zipf_packed_x_neg" "${TAIL3_ZIPF}" "packed_x_neg" "-1,0;-1,0;-1,0"
run_one "tail3_zipf_packed_y_pos" "${TAIL3_ZIPF}" "packed_y_pos" "0,1;0,1;0,1"

# Family 2: common + 4 tail groups, five total groups in 2D.
TAIL4_UNIFORM="common:0.60,tail1:0.10,tail2:0.10,tail3:0.10,tail4:0.10"
TAIL4_ZIPF="common:0.60,tail1:0.25,tail2:0.09,tail3:0.04,tail4:0.02"
TAIL4_SPREAD_CENTERS="0,1;0,-1;-1,0;-0.707,0.707"

run_one "tail4_uniform_spread" "${TAIL4_UNIFORM}" "spread" "${TAIL4_SPREAD_CENTERS}" "0.0"
run_one "tail4_uniform_packed_x_pos" "${TAIL4_UNIFORM}" "packed_x_pos" "1,0;1,0;1,0;1,0"
run_one "tail4_zipf_spread" "${TAIL4_ZIPF}" "spread" "${TAIL4_SPREAD_CENTERS}" "0.0"
run_one "tail4_zipf_packed_x_pos" "${TAIL4_ZIPF}" "packed_x_pos" "1,0;1,0;1,0;1,0"

# Family 3: one common group plus one tail group.
SINGLE_TAIL="common:0.90,tail1:0.10"

run_one "single_tail_init_x_pos" "${SINGLE_TAIL}" "packed_x_pos" "1,0"
run_one "single_tail_init_x_neg" "${SINGLE_TAIL}" "packed_x_neg" "-1,0"
run_one "single_tail_init_y_pos" "${SINGLE_TAIL}" "packed_y_pos" "0,1"
run_one "single_tail_init_y_neg" "${SINGLE_TAIL}" "packed_y_neg" "0,-1"

echo "================================================================================"
echo "All embedding-dim sweeps finished. Results: ${REPO_ROOT}/${OUTDIR}"
