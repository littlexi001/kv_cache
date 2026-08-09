#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_splitauto_inplace_32k_rotation_v102_3gpu
BASE=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex
AUTO=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_prefillindex
COMBO=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_inplacecache_prefillindex

export PATH=/home/fdong/miniconda3/bin:/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
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

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/validate_preallocated_cache_stride_20260724.py \
  --output_json "$RUN_ROOT/stride_validation.json" \
  > "$RUN_ROOT/logs/stride_validation.log" 2>&1

run_order() {
  local gpu=$1
  local name=$2
  local methods=$3
  local output="$RUN_ROOT/$name"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$output" \
      --tasks gov_report \
      --methods "$methods" \
      --sample_offset_per_task 115 \
      --max_samples_per_task 1 \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens 32000 \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/$name.log" 2>&1
}

run_order 0 base_auto_combo "$BASE,$AUTO,$COMBO" &
pid0=$!
run_order 1 auto_combo_base "$AUTO,$COMBO,$BASE" &
pid1=$!
run_order 2 combo_base_auto "$COMBO,$BASE,$AUTO" &
pid2=$!
wait "$pid0" "$pid1" "$pid2"
touch "$RUN_ROOT/ALL_COMPLETE"
