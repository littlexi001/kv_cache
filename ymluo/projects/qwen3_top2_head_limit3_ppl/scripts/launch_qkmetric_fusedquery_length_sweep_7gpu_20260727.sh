#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qkmetric_fusedquery_length_sweep}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/logs"

run_case() {
  local gpu="$1"
  local length="$2"
  local output="$OUTPUT/${length}"
  local log="$OUTPUT/logs/${length}.log"
  if [[ -s "$output/case_summary.json" ]]; then
    echo "SKIP ${length}"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics mixed_a \
    --window_indices 0 \
    --methods full_attention,direct_countcap \
    --history_tokens "$length" \
    --eval_tokens 64 \
    --window_stride_tokens "$((length + 512))" \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --direct_score_mode \
      pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct \
    --qk_metric_query_shrinkage 0.75 \
    --prefill_chunk_tokens 1024 \
    --cache_mode preallocated \
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --collect_logit_stability \
    >"$log" 2>&1
}

run_case 0 4096 &
pid0=$!
run_case 1 8192 &
pid1=$!
run_case 2 16384 &
pid2=$!
run_case 3 24576 &
pid3=$!
run_case 4 32768 &
pid4=$!
run_case 5 49152 &
pid5=$!
run_case 6 65536 &
pid6=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
wait "$pid4"
wait "$pid5"
wait "$pid6"
touch "$OUTPUT/ALL_COMPLETE"
echo "ALL_COMPLETE"
