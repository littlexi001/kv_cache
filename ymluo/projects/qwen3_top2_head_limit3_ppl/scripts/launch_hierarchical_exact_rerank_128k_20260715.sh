#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH="$ROOT/src"
export TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_for_gpus() {
  local first_gpu=$1
  local last_gpu=$2
  local stable=0
  while (( stable < 3 )); do
    mapfile -t free_memory < <(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    )
    local ready=1
    for ((gpu = first_gpu; gpu <= last_gpu; gpu++)); do
      if (( free_memory[gpu] < 22000 )); then
        ready=0
      fi
    done
    if (( ready == 1 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    if (( stable < 3 )); then
      sleep 30
    fi
  done
}

cd "$ROOT"
mkdir -p logs results

wait_for_gpus 4 5
CUDA_VISIBLE_DEVICES=4,5 "$PYTHON" \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --model_name_or_path "$MODEL" \
  --output results/20260715_hierarchical_exact_rerank_religion_4k_m16.json \
  --topic religion \
  --history_tokens 4096 \
  --query_tokens 64 \
  --eval_tokens 16 \
  --projection_dim 32 \
  --candidate_fraction 0.03 \
  --attention_fraction 0.02 \
  --exact_cache_fraction 0.032 \
  --directory_backend fused \
  --device_map balanced \
  --known_reference_ppl 1.0 \
  > logs/20260715_hierarchical_exact_rerank_4k_m16.log 2>&1

wait_for_gpus 4 7
CUDA_VISIBLE_DEVICES=4,5,6,7 "$PYTHON" \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --model_name_or_path "$MODEL" \
  --output results/20260715_hierarchical_exact_rerank_religion_128k_m32.json \
  --topic religion \
  --history_tokens 128000 \
  --query_tokens 256 \
  --eval_tokens 32 \
  --projection_dim 32 \
  --candidate_fraction 0.03 \
  --attention_fraction 0.02 \
  --exact_cache_fraction 0.032 \
  --directory_backend fused \
  --device_map balanced \
  --known_reference_ppl 15.16051075145507 \
  > logs/20260715_hierarchical_exact_rerank_128k_m32.log 2>&1
