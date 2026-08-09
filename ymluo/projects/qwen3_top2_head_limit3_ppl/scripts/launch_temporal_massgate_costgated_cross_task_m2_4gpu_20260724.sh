#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_temporal_massgate_costgated_cross_task_m2_4gpu
BASE=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex
GATE90=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate90_prefillindex
GATE94=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate94_prefillindex
GATE95=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_massgate95_prefillindex
METHODS=$BASE,$GATE90,$GATE94,$GATE95

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
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

run_tasks() {
  local gpu=$1
  local tasks=$2
  local out="$RUN_ROOT/gpu${gpu}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$out" \
    --tasks "$tasks" \
    --methods "$METHODS" \
    --max_samples_per_task 2 \
    --num_shards 1 --shard_index 0 \
    --max_prompt_tokens 32000 \
    --max_context_tokens 0 \
    --max_new_tokens_override 64 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --collect_attention_stats \
    --dtype float16 --device cuda --device_map auto \
    > "$RUN_ROOT/logs/gpu${gpu}.log" 2>&1
}

run_tasks 0 narrativeqa,hotpotqa &
pid0=$!
run_tasks 1 multi_news,qasper &
pid1=$!
run_tasks 2 passage_count &
pid2=$!
run_tasks 3 repobench-p &
pid3=$!
wait "$pid0" "$pid1" "$pid2" "$pid3"
touch "$RUN_ROOT/ALL_COMPLETE"
