#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260801_fier_matched_longbench_m10_4gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHOD=fier_rtn1_g32_fulltopk
MAX_SAMPLES_PER_TASK=10

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-2,3,4,5}"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "this launcher requires four GPUs" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-5]$ ]]; then
    echo "GPU $gpu is outside the allowed 0-5 range" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "duplicate GPU $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHOD" \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 7500 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override 8 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHOD" \
    --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
    --num_shards 4 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more FIER shards failed; valid rows remain" >&2
  exit 1
fi

touch "$RUN_ROOT/ALL_COMPLETE"
