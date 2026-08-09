#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${1:?usage: merge_result_bundles.sh BUNDLE_DIR OUTPUT_DIR}"
OUTPUT_DIR="${2:?usage: merge_result_bundles.sh BUNDLE_DIR OUTPUT_DIR}"
"${PYTHON_BIN:-python}" "${PROJECT_ROOT}/src/merge_result_bundles.py" \
  --bundle-dir "${BUNDLE_DIR}" --output-dir "${OUTPUT_DIR}"

