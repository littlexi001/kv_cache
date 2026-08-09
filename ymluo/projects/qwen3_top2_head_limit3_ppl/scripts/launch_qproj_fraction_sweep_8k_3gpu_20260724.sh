#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qproj_fraction_sweep_8k_3gpu
METHODS=full_kv,countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

pids=()
for spec in "0:0.04" "1:0.05" "2:0.06"; do
  gpu=${spec%%:*}
  fraction=${spec##*:}
  label=${fraction/./p}
  out="$RUN_ROOT/fraction${label}"
  mkdir -p "$out"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$out" \
      --tasks gov_report \
      --methods "$METHODS" \
      --sample_offset_per_task 115 \
      --max_samples_per_task 1 \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens 8192 \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --countcap_direct_fraction_override "$fraction" \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/fraction${label}.log" 2>&1
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
touch "$RUN_ROOT/ALL_COMPLETE"
