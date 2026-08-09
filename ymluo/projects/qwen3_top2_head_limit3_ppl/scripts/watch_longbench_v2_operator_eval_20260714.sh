#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

SAMPLES="${SAMPLES:-503}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-32000}"
STAMP="${STAMP:-20260714_longbench_v2}"
FULL="outputs/${STAMP}_full_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}/task_results.csv"
V466="outputs/${STAMP}_v466_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}/task_results.csv"
DIRECT_OFF="outputs/${STAMP}_v466_direct_off_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}/task_results.csv"
OUT="outputs/${STAMP}_comparison_m${SAMPLES}_c${MAX_CONTEXT_TOKENS}"
mkdir -p "$OUT"

while [[ ! -s "$FULL" || ! -s "$V466" || ! -s "$DIRECT_OFF" ]]; do
  echo "WAIT LongBench-v2 full=$([[ -s "$FULL" ]] && echo 1 || echo 0) v466=$([[ -s "$V466" ]] && echo 1 || echo 0) direct_off=$([[ -s "$DIRECT_OFF" ]] && echo 1 || echo 0) $(date -Is)"
  sleep 120
done

python scripts/summarize_longbench_v2_operator_eval_20260714.py \
  --full "$FULL" \
  --v466 "$V466" \
  --direct-off "$DIRECT_OFF" \
  --output-dir "$OUT" \
  > "$OUT/summary.log" 2>&1

cat "$OUT/summary.log"
