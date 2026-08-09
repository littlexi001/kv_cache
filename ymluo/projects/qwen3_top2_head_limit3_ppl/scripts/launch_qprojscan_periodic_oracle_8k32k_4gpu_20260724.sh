#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qprojscan_periodic_oracle_8k32k_4gpu
BASE=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex
REUSE2=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse2_prefillindex
REUSE4=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse4_prefillindex
REUSE8=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_reuse8_prefillindex

export PATH=/home/fdong/miniconda3/bin:/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

for gpu in 0 1 2 3; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU $gpu is busy" >&2
    exit 1
  fi
done

run_method() {
  local gpu=$1
  local length=$2
  local name=$3
  local method=$4
  local rotation=$5
  local output="$RUN_ROOT/length${length}/rotation${rotation}/${name}"
  local log="$RUN_ROOT/logs/length${length}_rotation${rotation}_${name}.log"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$output" \
      --tasks gov_report \
      --methods "$method" \
      --sample_offset_per_task 115 \
      --max_samples_per_task 1 \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens "$length" \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$log" 2>&1
}

methods=("$BASE" "$REUSE2" "$REUSE4" "$REUSE8")
names=(base reuse2 reuse4 reuse8)
for length in 8192 16000 32000; do
  for rotation in 0 1; do
    pids=()
    for slot in 0 1 2 3; do
      method_index=$(((slot + rotation) % 4))
      run_method \
        "$slot" \
        "$length" \
        "${names[$method_index]}" \
        "${methods[$method_index]}" \
        "$rotation" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      exit "$failed"
    fi
  done
done
touch "$RUN_ROOT/ALL_COMPLETE"
