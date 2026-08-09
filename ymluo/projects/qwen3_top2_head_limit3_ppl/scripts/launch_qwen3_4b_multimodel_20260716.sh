#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
LM_EVAL=/home/fdong/lm-evaluation-harness
LOG_ROOT=$ROOT/outputs/logs/20260716_qwen3_4b_multimodel
LB_PREFIX=20260716_qwen3_4b_longbench_m10_shard
RULER_PREFIX=20260716_qwen3_4b_ruler_m5_shard
RULER_EXAMPLES=$ROOT/data/ruler_generated/qwen3_4b_4k16k_m5_seed42.jsonl

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$LOG_ROOT"
cd "$ROOT"

lb_pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" taskset -c 0-23,48-71 "$PYTHON" \
    src/run_hierarchical_longbench_probe_20260715.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "outputs/${LB_PREFIX}${shard}" \
    --tasks narrativeqa,hotpotqa,passage_retrieval_en,lcc \
    --max_samples_per_task 10 \
    --num_shards 4 \
    --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.025 \
    --exact_cache_fraction 0.032 \
    --stream_group_size 1 \
    --candidate_refresh_interval 1 \
    --hierarchical_prompt_mode full_prompt_then_compress \
    --prefill_cache_mode dynamic \
    --prompt_wrapper qwen3 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    > "$LOG_ROOT/longbench_shard${shard}.log" 2>&1 &
  lb_pids+=("$!")
done

(
  CUDA_VISIBLE_DEVICES=4 taskset -c 24-47,72-95 "$PYTHON" \
    src/run_full_cache_ppl_baseline_20260715.py \
    --model_name_or_path "$MODEL" \
    --output results/20260716_qwen3_4b_physical_32k_full.json \
    --topic medicine \
    --history_tokens 32000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --prefill_chunk_tokens 2048 \
    --dtype float16 --device cuda --device_map auto
  CUDA_VISIBLE_DEVICES=4 taskset -c 24-47,72-95 "$PYTHON" \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --model_name_or_path "$MODEL" \
    --output results/20260716_qwen3_4b_physical_32k_sparse.json \
    --topic medicine \
    --history_tokens 32000 \
    --query_tokens 256 \
    --eval_tokens 256 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --attention_fraction 0.015 \
    --candidate_selection_mode per_head_stream \
    --stream_group_size 2 \
    --exact_cache_fraction 0.032 \
    --directory_backend fused \
    --prefill_chunk_tokens 2048 \
    --dtype float16 --device cuda --device_map auto
) > "$LOG_ROOT/physical_32k.log" 2>&1 &
physical_pid=$!

wait "${lb_pids[@]}" "$physical_pid"
"$PYTHON" src/summarize_hierarchical_longbench_shards_20260716.py \
  --input_glob "outputs/${LB_PREFIX}*/sample_results.csv" \
  --output_dir outputs/20260716_qwen3_4b_longbench_m10_merged \
  --expected_tasks 4 \
  --expected_samples_per_method 40 \
  --auto_gate_prompt_tokens 16384 \
  > "$LOG_ROOT/longbench_summary.log" 2>&1

"$PYTHON" src/prepare_hierarchical_ruler_data_20260716.py \
  --model_name_or_path "$MODEL" \
  --lm_eval_path "$LM_EVAL" \
  --output "$RULER_EXAMPLES" \
  --ruler_tasks niah_single_1,niah_multikey_1,niah_multiquery,qa_hotpot \
  --ruler_lengths 4096,8192,16384 \
  --max_samples_per_task 5 \
  --seed 42 \
  > "$LOG_ROOT/ruler_prepare.log" 2>&1

ruler_pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" taskset -c 0-23,48-71 "$PYTHON" \
    src/run_hierarchical_ruler_probe_20260716.py \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --examples_jsonl "$RULER_EXAMPLES" \
    --output_dir "outputs/${RULER_PREFIX}${shard}" \
    --ruler_tasks niah_single_1,niah_multikey_1,niah_multiquery,qa_hotpot \
    --ruler_lengths 4096,8192,16384 \
    --max_samples_per_task 5 \
    --num_shards 4 \
    --shard_index "$shard" \
    --seed 42 \
    --prefill_chunk_tokens 2048 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --exact_cache_fraction 0.032 \
    --stream_group_size 2 \
    --candidate_refresh_interval 1 \
    --hierarchical_prompt_mode full_prompt_then_compress \
    --prefill_cache_mode dynamic \
    --prompt_wrapper none \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/ruler_shard${shard}.log" 2>&1 &
  ruler_pids+=("$!")
done
wait "${ruler_pids[@]}"
"$PYTHON" src/summarize_hierarchical_ruler_shards_20260716.py \
  --input_glob "outputs/${RULER_PREFIX}*/sample_results.csv" \
  --output_dir outputs/20260716_qwen3_4b_ruler_m5_merged \
  --expected_task_lengths 12 \
  --expected_samples_per_method 60 \
  --auto_gate_requested_length 16384 \
  > "$LOG_ROOT/ruler_summary.log" 2>&1

"$PYTHON" src/audit_20260716_experiment_completion.py \
  --root "$ROOT" \
  --output results/20260716_experiment_completion_audit.json \
  > "$LOG_ROOT/completion_audit.log" 2>&1

touch results/20260716_qwen3_4b_multimodel_COMPLETE
