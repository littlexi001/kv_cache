#!/usr/bin/env bash
set -u

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$BASE/outputs/fixed50_onehop_twohop_age_20260727"
LOG="$OUT/experiment.log"

mkdir -p "$OUT"
echo "[$(date --iso-8601=seconds)] waiting for one free GPU among 4,5,6,7" >> "$LOG"

while true; do
  selected=""
  for gpu in 4 5 6 7; do
    memory_used="$(
      nvidia-smi -i "$gpu" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null |
        tr -d ' '
    )"
    utilization="$(
      nvidia-smi -i "$gpu" \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null |
        tr -d ' '
    )"
    if [[ "$memory_used" =~ ^[0-9]+$ ]] &&
       [[ "$utilization" =~ ^[0-9]+$ ]] &&
       (( memory_used < 1000 && utilization < 10 )); then
      selected="$gpu"
      break
    fi
  done

  if [[ -z "$selected" ]]; then
    sleep 20
    continue
  fi

  echo "[$(date --iso-8601=seconds)] starting on physical GPU $selected" >> "$LOG"
  rm -f "$OUT/result.json" "$OUT/records.csv" "$OUT/report.md"
  if (
    cd "$BASE" &&
      CUDA_VISIBLE_DEVICES="$selected" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=src \
      "$PY" -u src/run_fixed50_onehop_twohop_age_8b.py \
        --model-name-or-path "$MODEL" \
        --output-dir "$OUT" \
        --samples 64 \
        --total-tokens 50 \
        --batch-size 16 \
        --generation-max-new-tokens 12 \
        --device-map none \
        >> "$LOG" 2>&1
  ); then
    echo "[$(date --iso-8601=seconds)] completed on physical GPU $selected" >> "$LOG"
    exit 0
  fi

  echo "[$(date --iso-8601=seconds)] attempt on GPU $selected failed; retrying" >> "$LOG"
  sleep 20
done
