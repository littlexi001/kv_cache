#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/outputs/attention_value_decomposition_v4/value_pairwise_hist_by_head.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_value_decomposition_v4/plots/pairwise_hist_bins}"

python "${SCRIPT_DIR}/plot_pairwise_hist_bins.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --plot_dpi "${PLOT_DPI:-160}" \
  --pairs "${PAIRS:-top0p01|tail0p1,top0p1|tail0p1,top0p9|tail0p1}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}"
