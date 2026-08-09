#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MAIN_SRC=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src
EXPERIMENT_SRC=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/experimental_bandef
OUTPUT_ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/ec_bandef_ppl_20260718
MODEL=/home/fdong/models/Qwen3-4B-Instruct
LOG_DIR="${OUTPUT_ROOT}/qwen4b_8k_budget_targeted"
DENSITY_LOG_DIR="${OUTPUT_ROOT}/qwen4b_8k_density_budget_targeted"

mkdir -p "${LOG_DIR}"

previous_gpu=""
while true; do
    candidate="$({
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    } | awk -F, '$2 + 0 < 2000 && $3 + 0 < 20 {gsub(/ /, "", $1); print $1; exit}')"
    if [[ -n "${candidate}" && "${candidate}" == "${previous_gpu}" ]]; then
        break
    fi
    previous_gpu="${candidate}"
    printf '[wait] %s candidate=%s\n' "$(date --iso-8601=seconds)" "${candidate:-none}"
    sleep 60
done

printf '[launch] %s gpu=%s\n' "$(date --iso-8601=seconds)" "${candidate}"
export CUDA_VISIBLE_DEVICES="${candidate}"
export PYTHONPATH="${EXPERIMENT_SRC}:${MAIN_SRC}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_case() {
    local score_mode="$1"
    local output_dir="$2"
    "${PYTHON}" -u ./run_adaptive_mass_budget_ppl_20260715.py \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output_dir}" \
        --topics sports,medicine \
        --window_indices 0 \
        --history_tokens 8192 \
        --query_tokens 64 \
        --eval_tokens 32 \
        --mass_thresholds 0.75 \
        --budget_fractions 0.005,0.01,0.02,0.03,0.04,0.06,0.08 \
        --mass_estimator qabs_sampled_tail \
        --sample_fraction 0.0025 \
        --qabs_dim_count 8 \
        --candidate_fraction 0.08 \
        --qabs_use_cuda_kernels \
        --qabs_score_mode "${score_mode}" \
        --qabs_projection_dim 64 \
        --qabs_partition_ucb_z 0 \
        --qabs_partition_overfetch_factor 2 \
        --prefill_chunk_tokens 2048 \
        --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
        --seed 20260714 \
        --dtype float16 \
        --device cuda \
        --device_map auto
}

cd "${EXPERIMENT_SRC}"
run_case pca_int4_partition_global_delta16ec95_budget "${LOG_DIR}"
run_case pca_int4_partition_global_delta16density95_budget "${DENSITY_LOG_DIR}"
