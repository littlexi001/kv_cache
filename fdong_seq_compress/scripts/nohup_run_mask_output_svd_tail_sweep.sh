#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-fdong_seq_compress/outputs/mask_output_svd_tail_sweep}"
mkdir -p "$OUTPUT_ROOT"

if [[ "${RUN_SWEEP_WORKER:-0}" != "1" ]]; then
  nohup env RUN_SWEEP_WORKER=1 OUTPUT_ROOT="$OUTPUT_ROOT" bash "$0" \
    > "$OUTPUT_ROOT/sweep.log" 2>&1 &
  PID=$!
  echo "$PID" > "$OUTPUT_ROOT/sweep.pid"
  echo "Started PID $PID"
  echo "Output: $OUTPUT_ROOT"
  echo "Follow: tail -f $OUTPUT_ROOT/sweep.log"
  exit 0
fi

# keep_ratio = 1 - removed_tail_ratio
KEEP_RATIOS=(0.8 0.5 0.2 0.02)
TAIL_LABELS=(tail20 tail50 tail80 tail98)

for index in "${!KEEP_RATIOS[@]}"; do
  keep_ratio="${KEEP_RATIOS[$index]}"
  label="${TAIL_LABELS[$index]}"
  case_dir="$OUTPUT_ROOT/$label"
  mkdir -p "$case_dir"
  echo "[$(date '+%F %T')] Starting $label: keep top $keep_ratio" 
  python3 -u fdong_seq_compress/src/analyze_mask_output_svd_shift.py \
    --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
    --artifact-dir "${ARTIFACT_DIR:-fdong_seq_compress/artifacts/output_svd_qwen3_0p6b}" \
    --data-path "${DATA_PATH:-ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}" \
    --sample-id "${SAMPLE_ID:-niah_len2000_depth25}" \
    --device "${DEVICE:-mps}" \
    --dtype "${DTYPE:-float16}" \
    --score-ratio "$keep_ratio" \
    --position-ratio "${POSITION_RATIO:-0.01}" \
    --excluded-categories none \
    --ig-steps "${IG_STEPS:-33}" \
    --output-dir "$case_dir" \
    > "$case_dir/run.log" 2>&1
  echo "[$(date '+%F %T')] Finished $label"
done

echo "[$(date '+%F %T')] Sweep complete: $OUTPUT_ROOT"
