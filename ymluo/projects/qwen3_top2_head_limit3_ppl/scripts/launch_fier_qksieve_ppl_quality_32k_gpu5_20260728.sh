#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260728_fier_reference_ppl_quality_32k

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

"$PYTHON" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$RUN_ROOT/fier" \
  --topics sports,medicine \
  --window_indices 0,1 \
  --methods full_attention,direct_countcap \
  --history_tokens 32000 \
  --eval_tokens 128 \
  --direct_fraction 0.06 \
  --direct_min_tokens 256 \
  --direct_max_tokens 1280 \
  --projection_dim 48 \
  --sample_count 256 \
  --candidate_overfetch 1.0 \
  --protect_recent_tokens 0 \
  --direct_score_mode fier_rtn1_g32_fulltopk \
  --qk_metric_query_shrinkage 0.75 \
  --prefill_chunk_tokens 2048 \
  --cache_mode auto \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --collect_logit_stability \
  >"$RUN_ROOT/logs/fier.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
