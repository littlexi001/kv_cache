#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
OUT_ROOT="${ROOT}/results/20260728_qksieve_true_fulltopk_length_smoke"
GPU="${QKSIEVE_GPU:-5}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "${OUT_ROOT}/logs"

for length in 8000 16000 32000; do
  out_dir="${OUT_ROOT}/${length}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[skip] ${length}: summary.json exists"
    continue
  fi

  "${PYTHON}" -u "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${out_dir}" \
    --topics sports \
    --window_indices 0 \
    --methods full_attention,direct_countcap \
    --history_tokens "${length}" \
    --eval_tokens 64 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --protect_recent_tokens 0 \
    --direct_score_mode pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk \
    --qk_metric_query_shrinkage 0.75 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --collect_logit_stability \
    >"${OUT_ROOT}/logs/${length}.log" 2>&1
done

touch "${OUT_ROOT}/ALL_COMPLETE"
