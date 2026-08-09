#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=results/20260716_128k_hot_cache_sweep
SUMMARY=${OUT}_summary
REFERENCE_PPL=15.16051075145507

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT" "$SUMMARY" outputs/logs

run_case() {
  local devices=$1
  local cpus=$2
  local cache_fraction=$3
  local tag=${cache_fraction/./p}
  local name="pca64_top1_s2_cache${tag}"
  local output="$OUT/${name}.json"
  local log="outputs/logs/20260716_128k_hot_cache_${name}.log"
  if [[ -s "$output" ]]; then
    echo "skip existing $output"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$devices" taskset -c "$cpus" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic religion \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 32 \
    --window_index 0 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.01 \
    --attention_fraction 0.01 \
    --candidate_selection_mode per_head_stream \
    --rerank_selection_mode shared_sum \
    --exact_cache_fraction "$cache_fraction" \
    --stream_group_size 2 \
    --candidate_refresh_interval 1 \
    --host_append_mode async \
    --conversion_mode async \
    --directory_backend fused \
    --known_reference_ppl "$REFERENCE_PPL" \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$log" 2>&1
}

run_pair() {
  local left_fraction=$1
  local right_fraction=$2
  run_case 0,1,2,3 0-23,48-71 "$left_fraction" &
  local left_pid=$!
  run_case 4,5,6,7 24-47,72-95 "$right_fraction" &
  local right_pid=$!
  wait "$left_pid" "$right_pid"
}

# A fresh 3.2% control is included to absorb run-to-run timing variance.
run_pair 0.032 0.06
run_pair 0.10 0.20
run_pair 0.40 0.80

"$PYTHON" src/summarize_128k_speed_pareto_20260716.py \
  --input_dir "$OUT" \
  --output_dir "$SUMMARY"

