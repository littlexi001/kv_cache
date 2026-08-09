#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PARENT_RUN=$ROOT/results/20260726_final_direct_multimodel_m100_prompt7500
LOG=$PARENT_RUN/logs/final_report.log

cd "$ROOT"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    printf '%s parent failed\n' "$(date -Is)" >"$LOG"
    exit 1
  fi
  if ! pgrep -f \
    "launch_countcap_final_direct_multimodel_prompt7500_after_speed_4gpu_20260726.sh" \
    >/dev/null; then
    printf '%s parent stopped without ALL_COMPLETE\n' "$(date -Is)" >"$LOG"
    exit 1
  fi
  sleep 300
done

"$PYTHON" src/write_countcap_final_research_report_20260726.py \
  --comparison \
    results/20260726_final_direct_multimodel_m100_prompt7500/comparison/comparison.json \
  --llama_logit \
    results/20260726_countcap_logit_stability_32k_4gpu/merged/summary.json \
  --qwen_logit \
    results/20260726_countcap_logit_stability_qwen3_4b_32k_4gpu/merged/summary.json \
  --crossing \
    results/20260726_pca48_int4_delta_invariance_32k_v4/summary.json \
  --qk_spectrum \
    results/20260726_qk_matrix_spectrum_multimodel_32k/analysis_centered/summary.json \
  --fixed_basis \
    results/20260726_countcap_online_vs_fixed_basis_m20_4gpu/analysis/summary.json \
  --actual_budget \
    results/20260726_countcap_actual_budget_m4_4gpu/analysis/summary.json \
  --long_speed \
    results/20260726_countcap_final_long_speed_multimodel_4gpu/analysis/summary.json \
  --output \
    docs/20260726_countcap_final_theory_multimodel_results_zh.md \
  >"$LOG" 2>&1

test -s docs/20260726_countcap_final_theory_multimodel_results_zh.md
touch "$PARENT_RUN/REPORT_COMPLETE"
