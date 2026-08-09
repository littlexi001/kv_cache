#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
PREREQUISITE=$ROOT/results/20260727_qk_variable_physical_32k_paired_8gpu
REFERENCE=$ROOT/results/20260727_qkmetric_qscale_128k_holdout
RUN_ROOT=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
  if ! pgrep -f '^bash scripts/launch_qk_variable_physical_32k_paired_8gpu_20260727.sh$' >/dev/null; then
    echo "32K prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

topics=(mixed_a mixed_a mixed_b mixed_b)
windows=(2 3 2 3)
gpus=(0 1 2 3)
cpus=(0-23,48-71 0-23,48-71 24-47,72-95 24-47,72-95)
pids=()

for index in "${!topics[@]}"; do
  topic=${topics[$index]}
  window=${windows[$index]}
  stem=${topic}_w${window}
  output=$RUN_ROOT/${stem}_qkphysical.json
  if [[ -s "$output" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=${gpus[$index]} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "${cpus[$index]}" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index "$window" \
    --window_stride_tokens 128512 \
    --index_mode qk_variable \
    --qk_metric_query_shrinkage 0.75 \
    --variable_rate_budget 15 \
    --candidate_fraction 0.06 \
    --candidate_min_tokens 256 \
    --candidate_max_tokens 1280 \
    --attention_fraction 0.06 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 1 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --prefill_cache_mode offloaded_exact \
    --prefill_chunk_tokens 4096 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/${stem}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more physical 128K workers failed" >&2
  exit 1
fi

sampled_pids=()
for index in "${!topics[@]}"; do
  topic=${topics[$index]}
  window=${windows[$index]}
  stem=${topic}_w${window}
  output=$RUN_ROOT/${stem}_qksampled.json
  if [[ -s "$output" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=${gpus[$index]} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "${cpus[$index]}" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index "$window" \
    --window_stride_tokens 128512 \
    --index_mode qk_variable \
    --qk_metric_query_shrinkage 0.75 \
    --variable_rate_budget 15 \
    --candidate_fraction 0.06 \
    --candidate_min_tokens 256 \
    --candidate_max_tokens 1280 \
    --retrieval_backend sampled_compact \
    --sampled_candidate_multiplier 1.5 \
    --attention_fraction 0.06 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 1 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --prefill_cache_mode offloaded_exact \
    --prefill_chunk_tokens 4096 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/${stem}_sampled.log" 2>&1 &
  sampled_pids+=("$!")
done

failed=0
for pid in "${sampled_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more sampled physical 128K workers failed" >&2
  exit 1
fi

fixed_pids=()
for index in "${!topics[@]}"; do
  topic=${topics[$index]}
  window=${windows[$index]}
  stem=${topic}_w${window}
  output=$RUN_ROOT/${stem}_qkfixed4421sampled.json
  if [[ -s "$output" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES=${gpus[$index]} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c "${cpus[$index]}" "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output "$output" \
    --topic "$topic" \
    --history_tokens 128000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --window_index "$window" \
    --window_stride_tokens 128512 \
    --index_mode qk_variable \
    --qk_metric_query_shrinkage 0.75 \
    --variable_rate_budget 15 \
    --fixed_bit_allocation 4,4,2,1,0,0,0,0 \
    --candidate_fraction 0.06 \
    --candidate_min_tokens 256 \
    --candidate_max_tokens 1280 \
    --retrieval_backend sampled_compact \
    --sampled_candidate_multiplier 1.5 \
    --attention_fraction 0.06 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 1 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --prefill_cache_mode offloaded_exact \
    --prefill_chunk_tokens 4096 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/${stem}_fixed4421_sampled.log" 2>&1 &
  fixed_pids+=("$!")
done

failed=0
for pid in "${fixed_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more fixed4421 sampled physical workers failed" >&2
  exit 1
fi

"$PYTHON" src/summarize_qk_variable_physical_128k_20260727.py \
  --physical_root "$RUN_ROOT" \
  --reference_root "$REFERENCE" \
  --output_dir "$RUN_ROOT/summary" \
  > "$LOG_ROOT/summary.log" 2>&1
touch "$RUN_ROOT/ALL_COMPLETE"
