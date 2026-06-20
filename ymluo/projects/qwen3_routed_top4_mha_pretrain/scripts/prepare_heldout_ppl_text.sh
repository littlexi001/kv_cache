#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${HELDOUT_DATASET:-wikitext103_validation}"
OUTPUT_DIR="${HELDOUT_TEXT_DIR:-${PROJECT_DIR}/output/heldout_text}"
MAX_CHARS="${HELDOUT_MAX_CHARS:-5000000}"
MIN_LINE_CHARS="${HELDOUT_MIN_LINE_CHARS:-1}"

python "${PROJECT_DIR}/eval/prepare_heldout_text.py" \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_chars "${MAX_CHARS}" \
  --min_line_chars "${MIN_LINE_CHARS}"
