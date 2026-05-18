#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PROJECT_DIR}/src/generate_quadruple_table.py" \
  --output_file "${QUADRUPLE_FILE:-${PROJECT_DIR}/data/random_quadruples_1000_100000.pt}" \
  --token_min "${TOKEN_MIN:-1}" \
  --token_max "${TOKEN_MAX:-1000}" \
  --quadruple_len "${QUADRUPLE_LEN:-4}" \
  --num_quadruples "${NUM_QUADRUPLES:-100000}" \
  --quadruple_seed "${QUADRUPLE_SEED:-20260518}" \
  --force "${FORCE:-false}" \
  "$@"
