#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
OUT_ROOT="${ROOT}/results/20260728_qksieve_fp16_topk_quality_32k"
GPU="${QKSIEVE_GPU:-5}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "${OUT_ROOT}/logs"

for variant in fp32 fp16; do
  if [[ "${variant}" == "fp32" ]]; then
    score_mode=pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk
  else
    score_mode=pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk_fp16
  fi
  out_dir="${OUT_ROOT}/${variant}"
  if [[ -f "${out_dir}/case_summary.json" ]]; then
    echo "[skip] ${variant}: case_summary.json exists"
    continue
  fi

  "${PYTHON}" -u "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${out_dir}" \
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
    --direct_score_mode "${score_mode}" \
    --qk_metric_query_shrinkage 0.75 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --collect_logit_stability \
    >"${OUT_ROOT}/logs/${variant}.log" 2>&1
done

touch "${OUT_ROOT}/ALL_COMPLETE"
