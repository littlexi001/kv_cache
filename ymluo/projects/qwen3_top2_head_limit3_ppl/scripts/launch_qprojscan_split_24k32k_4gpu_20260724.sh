#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qprojscan_split_24k32k_4gpu
BASE=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex
SPLIT2=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit2_prefillindex
SPLIT4=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplit4_prefillindex

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

run_case() {
  local gpu=$1
  local name=$2
  local length=$3
  local methods=$4
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
      --max_prompt_tokens "$length" \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/$name.log" 2>&1
}

run_case 0 32k_base_split2_split4 32000 "$BASE,$SPLIT2,$SPLIT4" &
pid0=$!
run_case 1 32k_split2_split4_base 32000 "$SPLIT2,$SPLIT4,$BASE" &
pid1=$!
run_case 2 32k_split4_base_split2 32000 "$SPLIT4,$BASE,$SPLIT2" &
pid2=$!
run_case 3 24k_split2_base_split4 24000 "$SPLIT2,$BASE,$SPLIT4" &
pid3=$!
wait "$pid0" "$pid1" "$pid2" "$pid3"
touch "$RUN_ROOT/ALL_COMPLETE"
