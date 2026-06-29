#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
python "${PROJECT_DIR}/src/build_topic_texts.py" --target_repetitions 4 --overwrite
python -m py_compile \
  "${PROJECT_DIR}/src/build_topic_texts.py" \
  "${PROJECT_DIR}/src/run_quality_suite.py" \
  "${PROJECT_DIR}/src/evaluate_needle_generation.py"

echo "local smoke check complete"

