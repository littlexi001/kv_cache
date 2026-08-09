#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_splitauto_inplace_nsys_32k_v102
METHOD=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_inplacecache_prefillindex
PROFILE=$RUN_ROOT/profile

export PATH=/home/fdong/miniconda3/bin:/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs"
cd "$ROOT"

if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU 0 is busy" >&2
  exit 1
fi

COUNTCAP_CUDA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output="$PROFILE" \
    "$PYTHON" -u src/run_sample_calibrated_longbench_20260717.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "$RUN_ROOT/output" \
      --tasks gov_report \
      --methods "$METHOD" \
      --sample_offset_per_task 115 \
      --max_samples_per_task 1 \
      --num_shards 1 --shard_index 0 \
      --max_prompt_tokens 32000 \
      --max_context_tokens 0 \
      --max_new_tokens_override 32 \
      --prefill_chunk_tokens 2048 \
      --prompt_wrapper llama3 \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/run.log" 2>&1

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  "$PROFILE.nsys-rep" \
  > "$RUN_ROOT/profile_stats.txt"
touch "$RUN_ROOT/ALL_COMPLETE"
