#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_pruning_cos_ppl}"
CSV_PATH="${CSV_PATH:-${OUTPUT_DIR}/cos_per_token.csv}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${OUTPUT_DIR}/plots/cos_distribution_by_ratio}"

python "${SCRIPT_DIR}/plot_cos_distribution_by_ratio.py" \
  --csv_path "${CSV_PATH}" \
  --output_dir "${PLOT_OUTPUT_DIR}" \
  --ratios "${RATIOS:-0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20}" \
  --bins "${COS_DIST_BINS:-100}" \
  --hist_min "${COS_DIST_MIN:-0.0}" \
  --hist_max "${COS_DIST_MAX:-1.0}" \
  --dpi "${PLOT_DPI:-180}"
