#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${SCORE_TOP_SWEEP_WORKER:-0}" != "1" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  SWEEP_DIR="${SWEEP_DIR:-fdong_seq_compress/outputs/score_top_ratio_sweep_${TIMESTAMP}}"
  mkdir -p "$SWEEP_DIR"
  nohup env SCORE_TOP_SWEEP_WORKER=1 SWEEP_DIR="$SWEEP_DIR" bash "$0" \
    > "$SWEEP_DIR/run.log" 2>&1 &
  PID=$!
  echo "$PID" > "$SWEEP_DIR/run.pid"
  echo "Started PID $PID"
  echo "Log: $SWEEP_DIR/run.log"
  echo "Output: $SWEEP_DIR"
  echo "Follow: tail -f $SWEEP_DIR/run.log"
  exit 0
fi

read -r -a RATIOS <<< "${SCORE_RATIOS:-0.005 0.01 0.02 0.04 0.06 0.08 0.10}"

for RATIO in "${RATIOS[@]}"; do
  SLUG="$(printf '%s' "$RATIO" | tr '.' 'p')"
  OUTPUT_DIR="$SWEEP_DIR/ratio_${SLUG}"
  mkdir -p "$OUTPUT_DIR"
  echo "[$(date '+%F %T')] Starting score ratio $RATIO"
  python3 -u fdong_seq_compress/src/run_score_top_task_ablation.py \
    --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
    --data-path "${DATA_PATH:-ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}" \
    --sample-id "${SAMPLE_ID:-niah_len2000_depth25}" \
    --device "${DEVICE:-mps}" \
    --dtype "${DTYPE:-float16}" \
    --score-ratio "$RATIO" \
    --position-ratio "${POSITION_RATIO:-0.01}" \
    --excluded-categories "" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --output-dir "$OUTPUT_DIR"
  echo "[$(date '+%F %T')] Finished score ratio $RATIO"
done

echo "Sweep complete: $SWEEP_DIR"
