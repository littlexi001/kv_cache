#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
LM_EVAL=/home/fdong/lm-evaluation-harness
SHORT_DATA=$ROOT/data/ruler_generated/llama31_8b_ruler13_4k32k_m10_seed42.jsonl
LONG_DATA=$ROOT/data/ruler_generated/llama31_8b_ruler13_64k_m1_seed42.jsonl
RUN_ROOT=${RUN_ROOT:-$ROOT/results/20260801_qksieve_targeted_ruler_6gpu}
LOG_ROOT=$RUN_ROOT/logs
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
METHODS=full_kv,qksieve_fullprompt_auto_plain_fulltopk
GPU_CSV=${QKSIEVE_GPUS:-0,1,2,3,4,5}

IFS=',' read -r -a gpus <<<"$GPU_CSV"
if [[ "${#gpus[@]}" -ne 6 ]]; then
  echo "targeted RULER requires exactly six GPUs from 0-5" >&2
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

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

"$PYTHON" src/prepare_hierarchical_ruler_data_20260716.py \
  --model_name_or_path "$MODEL" \
  --lm_eval_path "$LM_EVAL" \
  --output "$LONG_DATA" \
  --ruler_tasks "$TASKS" \
  --ruler_lengths 65536 \
  --max_samples_per_task 1 \
  --seed 42 \
  >"$LOG_ROOT/prepare_long.log" 2>&1

run_shard() {
  local visible_gpus="$1"
  local data="$2"
  local output="$3"
  local lengths="$4"
  local samples="$5"
  local shards="$6"
  local shard="$7"
  local device_map="$8"
  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$data" \
    --output_dir "$output" \
    --methods "$METHODS" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths "$lengths" \
    --max_samples_per_task "$samples" \
    --num_shards "$shards" \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map "$device_map"
}

if [[ "${QKSIEVE_SKIP_SHORT:-0}" != "1" ]]; then
  short_pids=()
  for shard in 0 1 2 3 4 5; do
    run_shard \
      "${gpus[$shard]}" "$SHORT_DATA" "$RUN_ROOT/short_shard${shard}" \
      8192,16384,32768 2 6 "$shard" auto \
      >"$LOG_ROOT/short_shard${shard}.log" 2>&1 &
    short_pids+=("$!")
  done
  failed=0
  for pid in "${short_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more short RULER shards failed; valid rows remain" >&2
    exit 1
  fi
fi

long_pids=()
for shard in 0 1 2; do
  first=$((2 * shard))
  second=$((first + 1))
  run_shard \
    "${gpus[$first]},${gpus[$second]}" "$LONG_DATA" \
    "$RUN_ROOT/long_shard${shard}" 65536 1 3 "$shard" balanced \
    >"$LOG_ROOT/long_shard${shard}.log" 2>&1 &
  long_pids+=("$!")
done
failed=0
for pid in "${long_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more 64K RULER shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "$RUN_ROOT/*_shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/merge.log" 2>&1

"$PYTHON" src/summarize_qksieve_ruler_20260728.py \
  --input_csv "$RUN_ROOT/merged/sample_results.csv" \
  --project_root "$ROOT" \
  --output "$RUN_ROOT/targeted_summary.json" \
  --expected_length_samples 8192:2,16384:2,32768:2,65536:1 \
  >"$LOG_ROOT/summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
