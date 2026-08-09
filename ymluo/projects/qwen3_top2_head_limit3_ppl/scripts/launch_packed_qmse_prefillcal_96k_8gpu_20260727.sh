#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_packed_qmse_prefillcal_96k}"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

run_topic() {
  local devices="$1"
  local topic="$2"
  local output="$OUTPUT/packed_l96000_${topic}"
  if [[ -s "$output/case_summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices 0 \
    --methods direct_countcap \
    --history_tokens 96000 \
    --eval_tokens 64 \
    --window_stride_tokens 96512 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --direct_score_mode pca_hierarchical_autoqmsetotal15z_packed_direct \
    --prefill_chunk_tokens 1024 \
    --cache_mode preallocated \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"$LOGS/packed_l96000_${topic}.log" 2>&1
}

(
  run_topic "0,1" sports
  run_topic "0,1" computer
) &
(
  run_topic "2,3" medicine
  run_topic "2,3" religion
) &
(
  run_topic "4,5" space
) &
(
  run_topic "6,7" politics
) &

wait
echo "ALL_COMPLETE"
