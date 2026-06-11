#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/outputs/attention_value_decomposition/value_pairwise_by_head.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_value_decomposition/plots/pairwise_cos}"

python "${SCRIPT_DIR}/plot_pairwise_cos.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --top_n "${TOP_N:-0}"
