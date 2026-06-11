#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_CSV="${INPUT_CSV:-${PROJECT_DIR}/outputs/attention_value_decomposition/value_vectors_by_head.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_value_decomposition/plots/vector_frequency}"

python "${SCRIPT_DIR}/plot_vector_frequency.py" \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --metrics "${METRICS:-mean_norm,mean_attention_mass,mean_token_count}" \
  --vectors "${VECTORS:-}" \
  --bins "${BINS:-50}" \
  --plot_dpi "${PLOT_DPI:-160}" \
  --log_y "${LOG_Y:-false}"
