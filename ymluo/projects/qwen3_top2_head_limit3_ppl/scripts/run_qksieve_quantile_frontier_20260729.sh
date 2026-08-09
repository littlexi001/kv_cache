#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
DATASET_CACHE="${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/20260729_qksieve_speed_frontier}"
SCORE_MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"

export PATH="$(dirname "${PYTHON}"):${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RESULT_ROOT}"
cd "${PROJECT_ROOT}"

launch_case() {
    local gpu_list="$1"
    local history_tokens="$2"
    local topic="$3"
    local min_tail_samples="$4"
    local case_name="$5"
    local output_dir="${RESULT_ROOT}/${case_name}"

    mkdir -p "${output_dir}"
    nohup env \
        CUDA_VISIBLE_DEVICES="${gpu_list}" \
        QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES="${min_tail_samples}" \
        "${PYTHON}" src/run_direct_countcap_denseprompt_ppl_20260725.py \
        --model_name_or_path "${MODEL}" \
        --topics "${topic}" \
        --window_indices 0 \
        --methods direct_countcap \
        --history_tokens "${history_tokens}" \
        --eval_tokens 64 \
        --direct_fraction 0.06 \
        --direct_min_tokens 256 \
        --direct_max_tokens 1280 \
        --projection_dim 128 \
        --sample_count 256 \
        --direct_score_mode "${SCORE_MODE}" \
        --qk_metric_query_shrinkage 0.75 \
        --prefill_chunk_tokens 2048 \
        --cache_mode preallocated \
        --preallocated_cache_min_tokens 14000 \
        --dataset_cache_dir "${DATASET_CACHE}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --output_dir "${output_dir}" \
        >"${output_dir}/run.log" 2>&1 &
    echo "${case_name}:$!"
}

launch_case "0" 64000 medicine 16 "model_qksieve_qmin16_64k_gpu0"
launch_case "1" 64000 medicine 8 "model_qksieve_qmin8_64k_gpu1"
launch_case "2" 64000 medicine 4 "model_qksieve_qmin4_64k_gpu2"
launch_case "3,4" 120000 sports 8 "model_qksieve_qmin8_120k_gpu34"
launch_case "5,6" 120000 sports 4 "model_qksieve_qmin4_120k_gpu56"
