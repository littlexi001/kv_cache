#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
LM_EVAL=/home/fdong/lm-evaluation-harness
DATA=$ROOT/data/ruler_generated/llama31_8b_ruler13_4k32k_m10_seed42.jsonl
RUN_ROOT=$ROOT/results/20260728_qksieve_fulltopk_ruler_6gpu
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
METHODS=full_kv,qksieve_fullprompt_auto_plain_fulltopk

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "$RUN_ROOT/logs" "$(dirname "$DATA")"
cd "$ROOT"

if [[ ! -s "$DATA" ]]; then
  "$PYTHON" src/prepare_hierarchical_ruler_data_20260716.py \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --output "$DATA" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 4096,8192,16384,32768 \
    --max_samples_per_task 10 \
    --seed 42 \
    >"$RUN_ROOT/logs/prepare_short_early.log" 2>&1
fi

CUDA_VISIBLE_DEVICES=5 "$PYTHON" -u \
  src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "$MODEL" \
  --examples_jsonl "$DATA" \
  --output_dir "$RUN_ROOT/short_shard5" \
  --methods "$METHODS" \
  --ruler_tasks "$TASKS" \
  --ruler_lengths 4096,8192,16384,32768 \
  --max_samples_per_task 10 \
  --num_shards 6 \
  --shard_index 5 \
  --max_new_tokens_override 0 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$RUN_ROOT/logs/short_shard5_early.log" 2>&1
