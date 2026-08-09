#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/fixed300_age_distractor_qk_qwen3_8b_20260724}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FREE_MEMORY_USED_MB="${FREE_MEMORY_USED_MB:-512}"
FREE_UTILIZATION="${FREE_UTILIZATION:-10}"

mkdir -p "$OUT"
exec 8>"$OUT/launcher.lock"
if ! flock -n 8; then
  echo "another launcher already owns $OUT/launcher.lock"
  exit 0
fi
if [[ -f "$OUT/launcher.done" ]]; then
  echo "already complete: $OUT"
  exit 0
fi
rm -f "$OUT/launcher.done" "$OUT/launcher.failed"
echo "$$" >"$OUT/launcher.pid"

choose_free_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, \
      -v max_mem="$FREE_MEMORY_USED_MB" \
      -v max_util="$FREE_UTILIZATION" \
      '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (($1 + 0) >= 4 && ($1 + 0) <= 7 &&
            ($2 + 0) < max_mem && ($3 + 0) < max_util) {
          print $1;
          exit;
        }
      }'
}

gpu=""
while [[ -z "$gpu" ]]; do
  gpu="$(choose_free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    echo "$(date -Is) waiting: no idle GPU among 4-7"
    sleep "$POLL_SECONDS"
  fi
done

echo "$(date -Is) selected physical GPU $gpu"
nvidia-smi --id="$gpu" \
  --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"$OUT/gpu_before_launch.txt"

export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if "$PY" -u "$PROJECT/src/run_fixed300_age_distractor_qk_8b.py" \
  --model-name-or-path "$MODEL" \
  --output-dir "$OUT" \
  --total-tokens 300 \
  --distractor-counts 0,1,2,3,4,5,6,7,8,9 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --device cuda \
  --device-map none \
  --attn-implementation sdpa \
  --original-max-position-embeddings 32768 \
  >"$OUT/experiment.log" 2>&1; then
  date -Is >"$OUT/launcher.done"
  echo "$(date -Is) complete: $OUT"
else
  status=$?
  date -Is >"$OUT/launcher.failed"
  exit "$status"
fi
