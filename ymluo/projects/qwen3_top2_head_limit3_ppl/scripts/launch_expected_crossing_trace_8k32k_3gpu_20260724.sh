#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_expected_crossing_trace_8k32k_3gpu
METHOD=countcap_fullprompt_keypca_direct_qkvfused_prefillindex

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

for gpu in 0 1 2; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU $gpu is busy" >&2
    exit 1
  fi
done

pids=()
for spec in "0:8192" "1:16000" "2:32000"; do
  gpu=${spec%%:*}
  length=${spec##*:}
  out="$RUN_ROOT/length${length}"
  mkdir -p "$out"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$out" \
      --tasks gov_report \
      --methods "$METHOD" \
      --sample_offset_per_task 115 \
      --max_samples_per_task 1 \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens "$length" \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      --collect_attention_stats \
      > "$RUN_ROOT/logs/length${length}.log" 2>&1
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
touch "$RUN_ROOT/ALL_COMPLETE"
