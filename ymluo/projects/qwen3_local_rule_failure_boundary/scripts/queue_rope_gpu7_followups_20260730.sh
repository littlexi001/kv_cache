#!/usr/bin/env bash
set -euo pipefail

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
MAIN="$BASE/outputs/local_global_rope_probe_gpu7_20260730"
cd "$BASE"

while [[ ! -f "$MAIN/launcher.done" && ! -f "$MAIN/launcher.failed" ]]; do
    sleep 10
done

if [[ -f "$MAIN/launcher.failed" ]]; then
    cat "$MAIN/launcher.failed" >&2
    exit 1
fi

bash scripts/run_first_layer_rope_phase_gpu7_20260730.sh
bash scripts/run_local_global_rope_validation_gpu7_20260730.sh
date --iso-8601=seconds > outputs/rope_gpu7_followups_20260730.done
