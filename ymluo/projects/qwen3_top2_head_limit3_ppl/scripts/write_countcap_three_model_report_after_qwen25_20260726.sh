#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PARENT_RUN=$ROOT/results/20260726_countcap_qwen25_7b_longbench_m100_prompt7500
LOG=$PARENT_RUN/logs/three_model_report.log

cd "$ROOT"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    printf '%s Qwen2.5 parent failed\n' "$(date -Is)" >"$LOG"
    exit 1
  fi
  if ! pgrep -f \
    "launch_countcap_qwen25_7b_longbench_after_final_4gpu_20260726.sh" \
    >/dev/null; then
    printf '%s Qwen2.5 parent stopped without ALL_COMPLETE\n' \
      "$(date -Is)" >"$LOG"
    exit 1
  fi
  sleep 300
done

"$PYTHON" src/write_countcap_three_model_validation_report_20260726.py \
  --comparison \
    results/20260726_final_direct_multimodel_m100_prompt7500/comparison/comparison.json \
  --qwen25_csv \
    results/20260726_countcap_qwen25_7b_longbench_m100_prompt7500/merged/sample_results.csv \
  --long_speed \
    results/20260726_countcap_final_long_speed_multimodel_4gpu/analysis/summary.json \
  --output \
    docs/20260726_countcap_three_model_final_validation_zh.md \
  >"$LOG" 2>&1

test -s docs/20260726_countcap_three_model_final_validation_zh.md
touch "$PARENT_RUN/REPORT_COMPLETE"
