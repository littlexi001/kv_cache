#!/usr/bin/env bash
set -u

BASE=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$BASE/outputs/twohop_age_boundary_coarse_20260727"
LOG="$OUT/experiment.log"
POINTS="32768:1023,65536:2047,98304:3071,114688:3583,131072:4095,139264:4351,147456:4607,163840:5119"

mkdir -p "$OUT"
echo "[$(date --iso-8601=seconds)] waiting for three free GPUs among 4,5,6,7" >> "$LOG"

while true; do
  free_gpus=()
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
      free_gpus+=("$gpu")
    fi
  done
  if (( ${#free_gpus[@]} >= 3 )); then
    break
  fi
  sleep 30
done

selected="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]}"
echo "[$(date --iso-8601=seconds)] starting coarse scan on physical GPUs $selected" >> "$LOG"
rm -f "$OUT"/manifest.json "$OUT"/design.json "$OUT"/tokens_*.json

cd "$BASE" || exit 1
CUDA_VISIBLE_DEVICES="$selected" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
"$PY" -u src/run_twohop_age_distractor_failure_boundary_8b.py \
  --model-name-or-path "$MODEL" \
  --output-dir "$OUT" \
  --points "$POINTS" \
  --answer-only \
  --device-map balanced \
  --fixed-rope-factor 8 \
  --fixed-max-position-embeddings 163840 \
  >> "$LOG" 2>&1
status=$?
echo "[$(date --iso-8601=seconds)] coarse scan exited with status $status" >> "$LOG"
exit "$status"
