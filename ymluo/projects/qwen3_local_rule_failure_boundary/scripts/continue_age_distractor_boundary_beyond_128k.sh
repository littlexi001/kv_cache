#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
WAIT_FOR_OUT="${WAIT_FOR_OUT:-$PROJECT/outputs/age_distractor_distance_96k128k_20260724}"
OUT_160="${OUT_160:-$PROJECT/outputs/age_distractor_distance_160k_answer_only_20260724}"
OUT_192_256="${OUT_192_256:-$PROJECT/outputs/age_distractor_distance_192k256k_answer_only_20260724}"
POLL_SECONDS="${POLL_SECONDS:-30}"

wait_for_pid_file() {
  local pid_file="$1"
  while [[ ! -f "$pid_file" ]]; do
    sleep "$POLL_SECONDS"
  done
  local pid
  pid="$(cat "$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
}

last_point_is_correct() {
  "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
completed = manifest.get("completed", [])
raise SystemExit(0 if completed and completed[-1]["full_vocab_correct"] else 1)
PY
}

wait_for_gpus_4_to_7() {
  while true; do
    local busy
    busy="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, '{
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if (($1 + 0) >= 4 && ($1 + 0) <= 7 && ($2 + 0) >= 512) print $1;
        }'
    )"
    if [[ -z "$busy" ]]; then
      return
    fi
    sleep "$POLL_SECONDS"
  done
}

run_scan() {
  local out="$1"
  local points="$2"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$PROJECT/src/run_age_distractor_failure_boundary_8b.py" \
      --model-name-or-path "$MODEL" \
      --output-dir "$out" \
      --points "$points" \
      --prefill-chunk-size 128 \
      --dtype bfloat16 \
      --device cuda \
      --device-map balanced \
      --attn-implementation sdpa \
      --original-max-position-embeddings 40960 \
      --answer-only \
      --stop-on-failure \
      >"$out/experiment.log" 2>&1
}

wait_for_pid_file "$WAIT_FOR_OUT/pid"
if ! last_point_is_correct "$WAIT_FOR_OUT/manifest.json"; then
  echo "128K already failed; extended scan is unnecessary."
  exit 0
fi

wait_for_gpus_4_to_7
run_scan "$OUT_160" "163840:5119"
if ! last_point_is_correct "$OUT_160/manifest.json"; then
  echo "First extended failure found at 160K."
  exit 0
fi

wait_for_gpus_4_to_7
run_scan "$OUT_192_256" "196608:6143,229376:7167,262144:8191"
echo "Extended scan complete."
