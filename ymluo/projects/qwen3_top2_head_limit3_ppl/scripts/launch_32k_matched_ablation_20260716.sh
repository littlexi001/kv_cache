#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=results/20260716_32k_matched_ablation_m64

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT" outputs/logs

cpu_set() {
  if [[ "$1" -lt 4 ]]; then echo 0-23,48-71; else echo 24-47,72-95; fi
}

run_full() {
  local gpu=$1
  local output="$OUT/full_kv.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$(cpu_set "$gpu")" "$PYTHON" \
    src/run_full_cache_ppl_baseline_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic religion \
    --history_tokens 32000 \
    --query_tokens 256 \
    --eval_tokens 64 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > outputs/logs/20260716_32k_ablation_full.log 2>&1
}

run_sparse() {
  local gpu=$1
  local spec=$2
  local name projection bits candidate attention candidate_mode rerank_mode hot stream directory
  IFS='|' read -r name projection bits candidate attention candidate_mode rerank_mode hot stream directory <<< "$spec"
  local output="$OUT/${name}.json"
  [[ -s "$output" ]] && return 0
  CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$(cpu_set "$gpu")" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic religion \
    --history_tokens 32000 \
    --query_tokens 256 \
    --eval_tokens 64 \
    --projection_dim "$projection" \
    --index_bits "$bits" \
    --candidate_fraction "$candidate" \
    --attention_fraction "$attention" \
    --candidate_selection_mode "$candidate_mode" \
    --rerank_selection_mode "$rerank_mode" \
    --exact_cache_fraction "$hot" \
    --stream_group_size "$stream" \
    --directory_backend "$directory" \
    --host_append_mode async \
    --conversion_mode async \
    --known_reference_ppl 1.0 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "outputs/logs/20260716_32k_ablation_${name}.log" 2>&1
}

CASES=(
  'rank16|16|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'rank32|32|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'rank48|48|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'rank56|56|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'rank64_per_head|64|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'rank96|96|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'int8_rank64|64|8|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'shared_mean|64|4|0.015|0.015|shared_sum|shared_sum|0.032|1|fused'
  'shared_max|64|4|0.015|0.015|shared_max|shared_max|0.032|1|fused'
  'stream1|64|4|0.010|0.010|per_head_stream|shared_sum|0.041|1|fused'
  'stream2|64|4|0.010|0.010|per_head_stream|shared_sum|0.041|2|fused'
  'stream4|64|4|0.010|0.010|per_head_stream|shared_sum|0.041|4|fused'
  'hot1p1|64|4|0.010|0.010|per_head_stream|shared_sum|0.011|1|fused'
  'hot2p1|64|4|0.010|0.010|per_head_stream|shared_sum|0.021|1|fused'
  'hot3p2|64|4|0.010|0.010|per_head_stream|shared_sum|0.032|1|fused'
  'hot4p1|64|4|0.010|0.010|per_head_stream|shared_sum|0.041|1|fused'
  'fixed1|64|4|0.010|0.010|per_head_stream|shared_sum|0.032|2|fused'
  'fixed1p5|64|4|0.015|0.015|per_head_stream|shared_sum|0.032|2|fused'
  'fixed2|64|4|0.020|0.020|per_head_stream|shared_sum|0.032|1|fused'
  'directory_sorted|64|4|0.015|0.015|shared_sum|shared_sum|0.032|1|sorted'
  'directory_fused|64|4|0.015|0.015|shared_sum|shared_sum|0.032|1|fused'
)

pids=()
run_full 0 & pids+=("$!")
for index in $(seq 0 6); do
  gpu=$((index + 1))
  run_sparse "$gpu" "${CASES[$index]}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

next=7
while [[ "$next" -lt "${#CASES[@]}" ]]; do
  pids=()
  for gpu in $(seq 0 7); do
    index=$((next + gpu))
    [[ "$index" -ge "${#CASES[@]}" ]] && break
    run_sparse "$gpu" "${CASES[$index]}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  next=$((next + 8))
done

"$PYTHON" src/summarize_32k_matched_ablation_20260716.py \
  --input_dir "$OUT" \
  --output_dir "${OUT}_summary" \
  --expected_methods 21
