#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/outputs/attention_value_decomposition/value_pairwise_per_token.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_value_decomposition/plots/pairwise_token_hist}"

python "${SCRIPT_DIR}/plot_pairwise_token_hist.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --bins "${BINS:-60}" \
  --plot_dpi "${PLOT_DPI:-160}" \
  --pairs "${PAIRS:-}" \
  --layers "${LAYERS:-}" \
  --heads "${HEADS:-}"
