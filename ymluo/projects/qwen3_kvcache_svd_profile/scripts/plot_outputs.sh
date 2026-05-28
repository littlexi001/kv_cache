#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kvcache_svd_profile}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_DIR}/plots_from_csv}"

python "${PROJECT_DIR}/src/plot_kvcache_svd_outputs.py" \
  --output_dir "${OUTPUT_DIR}" \
  --plot_dir "${PLOT_DIR}" \
  --plot_dpi "${PLOT_DPI:-160}" \
  --max_rank "${MAX_RANK:-128}" \
  --cache_kinds "${CACHE_KINDS:-key,value}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --heatmap_metrics "${HEATMAP_METRICS:-top1_singular_value,rank0_energy_fraction,u_cosine_mean,right_singular_vector_cosine_mean}"
