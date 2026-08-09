#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RUN_ROOT=$ROOT/results/20260726_final_direct_multimodel_m100_ctx7500
BASELINES=$ROOT/outputs/kvcache_factory_aligned_b1024_20260713_m100_v1/analysis/method_summary.csv
OUTPUT=$RUN_ROOT/comparison
LOG=$RUN_ROOT/logs/comparison_postprocess.log

cd "$ROOT"
while [[ ! -f "$RUN_ROOT/ALL_COMPLETE" ]]; do
  if ! pgrep -f "20260726_final_direct_multimodel_m100_ctx7500" >/dev/null; then
    printf '%s parent stopped without ALL_COMPLETE\n' "$(date -Is)" >"$LOG"
    exit 1
  fi
  sleep 300
done

"$PYTHON" src/summarize_final_direct_multimodel_comparison_20260726.py \
  --llama_csv "$RUN_ROOT/llama31_8b/merged/sample_results.csv" \
  --qwen_csv "$RUN_ROOT/qwen3_4b/merged/sample_results.csv" \
  --baseline_method_summary "$BASELINES" \
  --output_dir "$OUTPUT" \
  --bootstrap_samples 5000 \
  >"$LOG" 2>&1

touch "$OUTPUT/ALL_COMPLETE"
