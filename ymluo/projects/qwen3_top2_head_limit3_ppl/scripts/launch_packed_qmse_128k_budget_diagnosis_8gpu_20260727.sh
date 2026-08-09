#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_packed_qmse_128k_budget_diagnosis}"
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

run_case() {
  local devices="$1"
  local variant="$2"
  local topic="$3"
  local window="$4"
  local output="$OUTPUT/${variant}/${topic}_w${window}"
  local log="$LOGS/${variant}_${topic}_w${window}.log"
  shift 4

  if [[ -s "$output/case_summary.json" ]]; then
    echo "SKIP ${variant} ${topic} window ${window}"
    return
  fi

  echo "START ${variant} ${topic} window ${window} on GPUs ${devices}"
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -u "$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics "$topic" \
    --window_indices "$window" \
    --history_tokens 128000 \
    --eval_tokens 256 \
    --window_stride_tokens 128512 \
    --prefill_chunk_tokens 1024 \
    --cache_mode preallocated \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    "$@" \
    >"$log" 2>&1
  echo "DONE ${variant} ${topic} window ${window}"
}

run_phase() {
  local variant="$1"
  shift
  run_case "0,1" "$variant" mixed_a 0 "$@" &
  pid0=$!
  run_case "2,3" "$variant" mixed_a 1 "$@" &
  pid1=$!
  run_case "4,5" "$variant" mixed_b 0 "$@" &
  pid2=$!
  run_case "6,7" "$variant" mixed_b 1 "$@" &
  pid3=$!
  wait "$pid0"
  wait "$pid1"
  wait "$pid2"
  wait "$pid3"
}

run_phase exact_top1 \
  --methods exact_top_fraction \
  --exact_fraction 0.01

run_phase packed_qmse_top2 \
  --methods direct_countcap \
  --direct_fraction 0.06 \
  --direct_min_tokens 256 \
  --direct_max_tokens 2560 \
  --sample_count 256 \
  --candidate_overfetch 1.0 \
  --direct_score_mode pca_hierarchical_autoqmsetotal15z_packed_direct

echo "ALL_COMPLETE"
