#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qprojscan_split_order_rotation_16k_3gpu
BASE=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex
SPLIT2=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit2_prefillindex
SPLIT4=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit4_prefillindex

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
      --max_prompt_tokens 16000 \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/$name.log" 2>&1
}

run_order 0 base_split2_split4 "$BASE,$SPLIT2,$SPLIT4" &
pid0=$!
run_order 1 split2_split4_base "$SPLIT2,$SPLIT4,$BASE" &
pid1=$!
run_order 2 split4_base_split2 "$SPLIT4,$BASE,$SPLIT2" &
pid2=$!
wait "$pid0" "$pid1" "$pid2"
touch "$RUN_ROOT/ALL_COMPLETE"
