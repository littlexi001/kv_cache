#!/usr/bin/env bash
set -euo pipefail

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
OUTPUT="$BASE/outputs/local_global_rope_heldout24_gpu7_20260730"
ANALYSIS="$OUTPUT/analysis"

while [[ ! -f "$OUTPUT/launcher.done" && ! -f "$OUTPUT/launcher.failed" ]]; do
    sleep 20
done

if [[ -f "$OUTPUT/launcher.failed" ]]; then
    cat "$OUTPUT/launcher.failed" >&2
    exit 1
fi

mkdir -p "$ANALYSIS"
"$PYTHON" "$BASE/src/analyze_local_global_rope_heldout.py" \
    --rows-csv "$OUTPUT/rows.csv" \
    --output-dir "$ANALYSIS" \
    --bootstrap-samples 20000 \
    > "$ANALYSIS/run.log" 2>&1

date --iso-8601=seconds > "$ANALYSIS/analysis.done"
