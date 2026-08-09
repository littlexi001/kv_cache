#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qkvsplit4_multitask_m2_4gpu
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
BASE=countcap_fullprompt_keypca_direct_qkvfused_prefillindex
SPLIT=countcap_fullprompt_keypca_direct_qkvsplit4_prefillindex

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

pids=()
for shard in 0 1 2 3; do
  methods="$BASE,$SPLIT"
  if (( shard % 2 == 1 )); then
    methods="$SPLIT,$BASE"
  fi
  out="$RUN_ROOT/shard${shard}"
  mkdir -p "$out"
  (
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
      src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$out" \
      --tasks "$TASKS" \
      --methods "$methods" \
      --max_samples_per_task 2 \
      --num_shards 4 --shard_index "$shard" \
      --max_prompt_tokens 32000 \
      --max_context_tokens 0 \
      --max_new_tokens_override 64 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/shard${shard}.log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -eq 0 ]]; then
  touch "$RUN_ROOT/ALL_COMPLETE"
fi
exit "$failed"
