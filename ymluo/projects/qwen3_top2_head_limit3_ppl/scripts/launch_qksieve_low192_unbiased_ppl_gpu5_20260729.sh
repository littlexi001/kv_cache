#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260729_qksieve_low192_unbiased_ppl_32k

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export CUDA_VISIBLE_DEVICES=5
export TORCH_CUDA_ARCH_LIST=8.6
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"
rm -f "$RUN_ROOT/ALL_COMPLETE"

"$PYTHON" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$RUN_ROOT" \
  --topics sports,medicine \
  --window_indices 0,1 \
  --methods full_attention,direct_countcap \
  --history_tokens 32000 \
  --eval_tokens 128 \
  --window_stride_tokens 32512 \
  --direct_fraction 0.06 \
  --direct_min_tokens 256 \
  --direct_max_tokens 1280 \
  --projection_dim 48 \
  --sample_count 256 \
  --candidate_overfetch 1.0 \
  --protect_recent_tokens 0 \
  --direct_score_mode \
    pca_hierarchical_fixed441_qkmetric_unbiased_packed_direct \
  --qk_metric_query_shrinkage 0.75 \
  --prefill_chunk_tokens 2048 \
  --cache_mode auto \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --collect_logit_stability \
  >"$RUN_ROOT/logs/main.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
