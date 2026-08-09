#!/usr/bin/env bash
set -u

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$BASE/outputs/incremental_twohop_first_token_98k_120k_20260727"
LOG="$OUT/experiment.log"

mkdir -p "$OUT"
printf '%s\n' "$$" > "$OUT/launcher.pid"
rm -f "$OUT/launcher.done"

cd "$BASE" || exit 1
echo "[$(date --iso-8601=seconds)] start: one 98K shared prefill, then 22,528 one-token increments" >> "$LOG"

CUDA_VISIBLE_DEVICES=4,5,6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
"$PY" -u src/run_incremental_twohop_first_token_8b.py \
  --model-name-or-path "$MODEL" \
  --output-dir "$OUT" \
  --start-total-tokens 100352 \
  --end-total-tokens 122880 \
  --max-distractors 3839 \
  --device-map balanced \
  --fixed-rope-factor 8 \
  --fixed-max-position-embeddings 163840 \
  --checkpoint-every 100 \
  >> "$LOG" 2>&1
status=$?

if (( status == 0 )); then
  mkdir -p "$OUT/analysis"
  "$PY" -u src/analyze_incremental_twohop_first_token.py \
    --points-csv "$OUT/points.csv" \
    --output-dir "$OUT/analysis" \
    >> "$LOG" 2>&1
  status=$?
fi

echo "$status" > "$OUT/launcher.done"
echo "[$(date --iso-8601=seconds)] exit status $status" >> "$LOG"
exit "$status"
