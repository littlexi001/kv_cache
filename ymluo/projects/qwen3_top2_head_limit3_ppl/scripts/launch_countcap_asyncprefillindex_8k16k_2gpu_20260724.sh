#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
HORIZON=${HORIZON:-64}
RUN_ROOT=$ROOT/results/20260724_countcap_asyncprefillindex_g${HORIZON}_2gpu
SYNC=countcap_fullprompt_keypca_direct_qkvfused_prefillindex
ASYNC=countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

for gpu in 0 1; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU $gpu is busy" >&2
    exit 1
  fi
done

pids=()
for spec in "0:8192" "1:16000"; do
  gpu=${spec%%:*}
  length=${spec##*:}
  (
    orders=(
      "full_kv,$SYNC,$ASYNC"
      "$SYNC,$ASYNC,full_kv"
      "$ASYNC,full_kv,$SYNC"
    )
    for repeat in 1 2 3; do
      out="$RUN_ROOT/length${length}/g${HORIZON}/repeat${repeat}"
      mkdir -p "$out"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        src/run_sample_calibrated_longbench_20260717.py \
        --model_name_or_path "$MODEL" \
        --longbench_data_dir "$DATA" \
        --output_dir "$out" \
        --tasks gov_report \
        --methods "${orders[$((repeat - 1))]}" \
        --sample_offset_per_task 115 \
        --max_samples_per_task 1 \
        --num_shards 1 --shard_index 0 \
        --max_prompt_tokens "$length" \
        --max_context_tokens 0 \
        --max_new_tokens_override "$HORIZON" \
        --prefill_chunk_tokens 2048 \
        --prompt_wrapper llama3 \
        --dtype float16 --device cuda --device_map auto \
        > "$RUN_ROOT/logs/length${length}_repeat${repeat}.log" 2>&1
    done
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
touch "$RUN_ROOT/ALL_COMPLETE"
