#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_qprojscan256_e2e_32k_gpu0
QPROJ=countcap_fullprompt_keypca_direct_qkvfused_qproj_prefillindex
QPROJSCAN=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT"
cd "$ROOT"

if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU 0 is busy" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT" \
  --tasks gov_report \
  --methods "$QPROJ,$QPROJSCAN" \
  --sample_offset_per_task 115 \
  --max_samples_per_task 1 \
  --num_shards 1 --shard_index 0 \
  --max_prompt_tokens 32000 \
  --max_context_tokens 0 \
  --max_new_tokens_override 64 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --dtype float16 --device cuda --device_map auto
touch "$RUN_ROOT/ALL_COMPLETE"
