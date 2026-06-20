#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${EVAL_DATA_DIR:-${PROJECT_DIR}/output/eval_data}"
TASKS="${TASKS:-piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq}"
SPLIT="${SPLIT:-validation}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
SEED="${SEED:-1234}"

python "${PROJECT_DIR}/eval/prepare_eval_jsonl.py" \
  --output_dir "${OUTPUT_DIR}" \
  --tasks "${TASKS}" \
  --split "${SPLIT}" \
  --max_examples "${MAX_EXAMPLES}" \
  --seed "${SEED}"
