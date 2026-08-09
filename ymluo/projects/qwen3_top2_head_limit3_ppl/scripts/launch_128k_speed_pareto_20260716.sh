#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=results/20260716_128k_speed_pareto

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT" outputs/logs

run_full_case() {
  local devices=$1
  local cpus=$2
  local topic=$3
  local output="$OUT/full_${topic}.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$devices" taskset -c "$cpus" "$PYTHON" \
    src/run_full_cache_ppl_baseline_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 32 \
    --window_index 0 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "outputs/logs/20260716_128k_speed_pareto_full_${topic}.log" 2>&1
}

run_case() {
  local devices=$1
  local cpus=$2
  local name=$3
  local topic=$4
  local reference_ppl=$5
  local projection_dim=$6
  local candidate_fraction=$7
  local exact_cache_fraction=$8
  local stream_group_size=$9
  local host_append_mode=${10}
  local conversion_mode=${11}
  local candidate_refresh_interval=${12:-1}
  local output="$OUT/${name}.json"
  local log="outputs/logs/20260716_128k_speed_pareto_${name}.log"
  if [[ -s "$output" ]]; then
    echo "skip existing $output"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$devices" taskset -c "$cpus" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 32 \
    --projection_dim "$projection_dim" \
    --index_bits 4 \
    --candidate_fraction "$candidate_fraction" \
    --attention_fraction "$candidate_fraction" \
    --candidate_selection_mode per_head_stream \
    --rerank_selection_mode shared_sum \
    --exact_cache_fraction "$exact_cache_fraction" \
    --stream_group_size "$stream_group_size" \
    --candidate_refresh_interval "$candidate_refresh_interval" \
    --host_append_mode "$host_append_mode" \
    --conversion_mode "$conversion_mode" \
    --directory_backend fused \
    --known_reference_ppl "$reference_ppl" \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$log" 2>&1
}

run_wave() {
  run_case 0,1,2,3 0-23,48-71 "${@:1:9}" &
  local left=$!
  run_case 4,5,6,7 24-47,72-95 "${@:10:9}" &
  local right=$!
  wait "$left" "$right"
}

run_full_case 0,1,2,3 0-23,48-71 religion &
full_left=$!
run_full_case 4,5,6,7 24-47,72-95 computer &
full_right=$!
wait "$full_left" "$full_right"

# Wave 1: isolate the per-layer D2H synchronization cost.
run_wave \
  pca64_top1_s2_cache3p2_async religion 15.16051075145507 64 0.01 0.032 2 async async \
  pca64_top1_s2_cache3p2_sync religion 15.16051075145507 64 0.01 0.032 2 sync async

# Wave 2: trade index bytes for a larger exact residency cache under ~10% state.
run_wave \
  pca56_top1p5_s2_cache4p0 computer 60.43246449071301 56 0.015 0.040 2 async async \
  pca48_top1p5_s2_cache4p5 computer 60.43246449071301 48 0.015 0.045 2 async async

# Wave 3: same PCA64 index, more residency, to measure the attainable speed ceiling.
run_wave \
  pca64_top1_s4_cache4p1 religion 15.16051075145507 64 0.01 0.041 4 async async \
  pca64_top1p5_s2_cache4p1 computer 60.43246449071301 64 0.015 0.041 2 async async

# Wave 4: isolate multi-GPU asynchronous conversion from ordinary run variance.
run_wave \
  pca64_top1_s2_cache3p2_conversion_async_repeat religion 15.16051075145507 64 0.01 0.032 2 async async \
  pca64_top1_s2_cache3p2_conversion_sync religion 15.16051075145507 64 0.01 0.032 2 async sync

# Wave 5: amortize global PCA scan/top-k across adjacent decode positions.
run_case 0,1,2,3 0-23,48-71 \
  pca64_top1_s2_cache3p2_refresh2 religion 15.16051075145507 \
  64 0.01 0.032 2 async async 2 &
refresh2_pid=$!
run_case 4,5,6,7 24-47,72-95 \
  pca64_top1_s2_cache3p2_refresh4 religion 15.16051075145507 \
  64 0.01 0.032 2 async async 4 &
refresh4_pid=$!
wait "$refresh2_pid" "$refresh4_pid"

"$PYTHON" src/summarize_128k_speed_pareto_20260716.py \
  --input_dir "$OUT" \
  --output_dir "${OUT}_summary"
