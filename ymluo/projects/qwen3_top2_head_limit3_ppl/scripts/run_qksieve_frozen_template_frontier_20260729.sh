#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
PYTHON="/home/fdong/miniconda3/envs/moe/bin/python"
MODEL="/home/fdong/models/Qwen3-4B-Instruct"
DATA="/home/fdong/ymluo/datasets/sklearn"
RUN_ROOT="${PROJECT_ROOT}/results/20260729_qksieve_frozen_template_frontier"
MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"
WMMA_MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_wmma_kappend_unbiased_packed_direct"

export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="8.6"
export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES="16"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/templates"
cd "${PROJECT_ROOT}"

run_case() {
    local visible_devices="$1"
    local topic="$2"
    local output_name="$3"
    local template_in="${4:-}"
    local template_out="${5:-}"
    local history_tokens="${6:-120000}"
    local eval_tokens="${7:-64}"
    local window_indices="${8:-0}"
    local max_tokens="${9:-1280}"
    local output_dir="${RUN_ROOT}/${output_name}"
    local log_file="${RUN_ROOT}/logs/${output_name}.log"
    local extra_args=()
    if [[ -n "${template_in}" ]]; then
        extra_args+=(--packed_qmse_template_in "${template_in}")
    fi
    if [[ -n "${template_out}" ]]; then
        extra_args+=(--packed_qmse_template_out "${template_out}")
    fi
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        "${PYTHON}" src/run_direct_countcap_denseprompt_ppl_20260725.py \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output_dir}" \
        --topics "${topic}" \
        --window_indices "${window_indices}" \
        --methods direct_countcap \
        --history_tokens "${history_tokens}" \
        --eval_tokens "${eval_tokens}" \
        --direct_fraction 0.06 \
        --direct_min_tokens 256 \
        --direct_max_tokens "${max_tokens}" \
        --projection_dim 128 \
        --sample_count 256 \
        --direct_score_mode "${MODE}" \
        --qk_metric_query_shrinkage 0.75 \
        --prefill_chunk_tokens 2048 \
        --cache_mode preallocated \
        --dataset_cache_dir "${DATA}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        "${extra_args[@]}" \
        >"${log_file}" 2>&1
}

run_fier_case() {
    local visible_devices="$1"
    local topic="$2"
    local output_name="$3"
    local history_tokens="${4:-32000}"
    local eval_tokens="${5:-128}"
    local window_indices="${6:-0}"
    local output_dir="${RUN_ROOT}/${output_name}"
    local log_file="${RUN_ROOT}/logs/${output_name}.log"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        "${PYTHON}" src/run_direct_countcap_denseprompt_ppl_20260725.py \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output_dir}" \
        --topics "${topic}" \
        --window_indices "${window_indices}" \
        --methods direct_countcap \
        --history_tokens "${history_tokens}" \
        --eval_tokens "${eval_tokens}" \
        --direct_fraction 0.06 \
        --direct_min_tokens 256 \
        --direct_max_tokens 1280 \
        --projection_dim 128 \
        --sample_count 256 \
        --direct_score_mode fier_rtn1_g32_sampled_packed_direct \
        --qk_metric_query_shrinkage 0.75 \
        --prefill_chunk_tokens 2048 \
        --cache_mode preallocated \
        --dataset_cache_dir "${DATA}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        >"${log_file}" 2>&1
}

run_full_case() {
    local visible_devices="$1"
    local topic="$2"
    local output_name="$3"
    local history_tokens="${4:-32000}"
    local eval_tokens="${5:-128}"
    local window_indices="${6:-0}"
    local output_dir="${RUN_ROOT}/${output_name}"
    local log_file="${RUN_ROOT}/logs/${output_name}.log"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        "${PYTHON}" src/run_direct_countcap_denseprompt_ppl_20260725.py \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output_dir}" \
        --topics "${topic}" \
        --window_indices "${window_indices}" \
        --methods full_attention \
        --history_tokens "${history_tokens}" \
        --eval_tokens "${eval_tokens}" \
        --prefill_chunk_tokens 2048 \
        --cache_mode preallocated \
        --dataset_cache_dir "${DATA}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        >"${log_file}" 2>&1
}

stage="${1:-calibrate}"
case "${stage}" in
    calibrate)
        run_case \
            "0,1" sports online_sports_export "" \
            "${RUN_ROOT}/templates/sports.pt" &
        pid0=$!
        run_case \
            "2,3" medicine online_medicine_export "" \
            "${RUN_ROOT}/templates/medicine.pt" &
        pid1=$!
        run_case \
            "4,5" mixed_a online_mixed_a_export "" \
            "${RUN_ROOT}/templates/mixed_a.pt" &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    cross)
        run_case \
            "0,1" sports frozen_medicine_to_sports \
            "${RUN_ROOT}/templates/medicine.pt" "" &
        pid0=$!
        run_case \
            "2,3" medicine frozen_sports_to_medicine \
            "${RUN_ROOT}/templates/sports.pt" "" &
        pid1=$!
        run_case \
            "4,5" mixed_b frozen_mixed_a_to_mixed_b \
            "${RUN_ROOT}/templates/mixed_a.pt" "" &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    short)
        run_case \
            "6" sports frozen_medicine_to_sports_32k \
            "${RUN_ROOT}/templates/medicine.pt" "" 32000 128
        ;;
    short_online)
        run_case \
            "6" sports online_sports_32k "" "" 32000 128
        ;;
    short_fier)
        run_fier_case "6" sports fier_sports_32k 32000 128
        ;;
    short_full)
        run_full_case "6" sports full_sports_32k 32000 128
        ;;
    short_full_gpu0)
        run_full_case "0" sports full_sports_32k 32000 128
        ;;
    mixed_b_online)
        run_case \
            "4,5" mixed_b online_mixed_b "" "" 120000 64
        ;;
    length8)
        run_case \
            "1" sports frozen_medicine_to_sports_8k \
            "${RUN_ROOT}/templates/medicine.pt" "" 8000 128 &
        pid0=$!
        run_case \
            "2" sports online_sports_8k "" "" 8000 128 &
        pid1=$!
        run_fier_case "3" sports fier_sports_8k 8000 128 &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    length16)
        run_case \
            "1" sports frozen_medicine_to_sports_16k \
            "${RUN_ROOT}/templates/medicine.pt" "" 16000 128 &
        pid0=$!
        run_case \
            "2" sports online_sports_16k "" "" 16000 128 &
        pid1=$!
        run_fier_case "3" sports fier_sports_16k 16000 128 &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    resident_frontier)
        run_case \
            "0" sports resident_medicine_to_sports_8k \
            "${RUN_ROOT}/templates/medicine.pt" "" 8000 128 &
        pid0=$!
        run_case \
            "1" sports resident_medicine_to_sports_16k \
            "${RUN_ROOT}/templates/medicine.pt" "" 16000 128 &
        pid1=$!
        run_case \
            "2" sports resident_medicine_to_sports_32k \
            "${RUN_ROOT}/templates/medicine.pt" "" 32000 128 &
        pid2=$!
        run_case \
            "3,6" sports resident_medicine_to_sports_64k \
            "${RUN_ROOT}/templates/medicine.pt" "" 64000 128 &
        pid3=$!
        wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
        ;;
    length16_compare)
        run_case \
            "0" sports online_sports_16k "" "" 16000 128 &
        pid0=$!
        run_fier_case "1" sports fier_sports_16k 16000 128 &
        pid1=$!
        run_full_case "2" sports full_sports_16k 16000 128 &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    warm32)
        run_case \
            "0" sports warm2_medicine_to_sports_32k \
            "${RUN_ROOT}/templates/medicine.pt" "" 32000 128 "0,1" &
        pid0=$!
        run_fier_case \
            "1" sports warm2_fier_sports_32k 32000 128 "0,1" &
        pid1=$!
        run_full_case \
            "2" sports warm2_full_sports_32k 32000 128 "0,1" &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    resident64_single)
        run_case \
            "3" sports resident_single_medicine_to_sports_64k \
            "${RUN_ROOT}/templates/medicine.pt" "" 64000 128
        ;;
    template32_compare)
        run_case \
            "0" sports warm2_sports_template_to_sports_32k \
            "${RUN_ROOT}/templates/sports.pt" "" 32000 128 "0,1" &
        pid0=$!
        run_case \
            "1" sports warm2_mixed_a_template_to_sports_32k \
            "${RUN_ROOT}/templates/mixed_a.pt" "" 32000 128 "0,1" &
        pid1=$!
        run_case \
            "2" sports warm2_online_sports_32k \
            "" "" 32000 128 "0,1" &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    calibrate_global32)
        run_case \
            "4" sports global_calibration_sports_32k \
            "" "${RUN_ROOT}/templates/global32_sports.pt" 32000 2 &
        pid0=$!
        run_case \
            "5" medicine global_calibration_medicine_32k \
            "" "${RUN_ROOT}/templates/global32_medicine.pt" 32000 2 &
        pid1=$!
        run_case \
            "6" mixed_a global_calibration_mixed_a_32k \
            "" "${RUN_ROOT}/templates/global32_mixed_a.pt" 32000 2 &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        "${PYTHON}" src/build_global_qksieve_template_20260729.py \
            --inputs \
            "${RUN_ROOT}/templates/global32_sports.pt" \
            "${RUN_ROOT}/templates/global32_medicine.pt" \
            "${RUN_ROOT}/templates/global32_mixed_a.pt" \
            --output "${RUN_ROOT}/templates/global32_3domain_runtime.pt" \
            --query_shrinkage 0.75
        ;;
    global32_test)
        run_case \
            "0" sports warm2_global3_to_sports_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" 32000 128 "0,1" &
        pid0=$!
        run_case \
            "1" medicine warm2_global3_to_medicine_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" 32000 128 "0,1" &
        pid1=$!
        run_case \
            "2" mixed_b warm2_global3_to_mixed_b_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" 32000 128 "0,1" &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    global32_full_refs)
        run_full_case \
            "4" medicine warm2_full_medicine_32k 32000 128 "0,1" &
        pid0=$!
        run_full_case \
            "5" mixed_b warm2_full_mixed_b_32k 32000 128 "0,1" &
        pid1=$!
        wait "${pid0}" "${pid1}"
        ;;
    global32_fier_refs)
        run_fier_case \
            "3" medicine warm2_fier_medicine_32k 32000 128 "0,1" &
        pid0=$!
        run_fier_case \
            "6" mixed_b warm2_fier_mixed_b_32k 32000 128 "0,1" &
        pid1=$!
        wait "${pid0}" "${pid1}"
        ;;
    budget32)
        run_case \
            "0" sports warm2_global3_sports_b1024_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            32000 128 "0,1" 1024 &
        pid0=$!
        run_case \
            "1" sports warm2_global3_sports_b896_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            32000 128 "0,1" 896 &
        pid1=$!
        run_case \
            "2" sports warm2_global3_sports_b768_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            32000 128 "0,1" 768 &
        pid2=$!
        run_case \
            "4" medicine warm2_global3_medicine_b1024_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            32000 128 "0,1" 1024 &
        pid3=$!
        run_case \
            "5" mixed_b warm2_global3_mixed_b_b1024_32k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            32000 128 "0,1" 1024 &
        pid4=$!
        wait "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}"
        ;;
    split32)
        (
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=4
            run_case \
                "0" sports warm2_global3_sports_b1024_split4_32k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                32000 128 "0,1" 1024
        ) &
        pid0=$!
        (
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
            run_case \
                "1" sports warm2_global3_sports_b1024_split8_32k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                32000 128 "0,1" 1024
        ) &
        pid1=$!
        (
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=4
            run_case \
                "2" sports warm2_global3_sports_b1152_split4_32k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                32000 128 "0,1" 1152
        ) &
        pid2=$!
        (
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
            run_case \
                "3" sports warm2_global3_sports_b1152_split8_32k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                32000 128 "0,1" 1152
        ) &
        pid3=$!
        (
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
            run_case \
                "4" sports warm2_global3_sports_b1200_split8_32k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                32000 128 "0,1" 1200
        ) &
        pid4=$!
        wait "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}"
        ;;
    global120_b1024)
        run_case \
            "5,6" sports global3_sports_b1024_120k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            120000 64 0 1024
        ;;
    global120_b1280_dual)
        run_case \
            "0,1" sports global3_sports_b1280_120k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            120000 64 0 1280 &
        pid0=$!
        run_case \
            "2,3" medicine global3_medicine_b1280_120k \
            "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
            120000 64 0 1280 &
        pid1=$!
        wait "${pid0}" "${pid1}"
        ;;
    profile120)
        (
            export QKSIEVE_PROFILE_STAGES=1
            run_case \
                "0,1" sports profile_global3_sports_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid0=$!
        (
            export QKSIEVE_PROFILE_STAGES=1
            run_fier_case \
                "2,3" sports profile_fier_sports_b1280_120k \
                120000 32 0
        ) &
        pid1=$!
        wait "${pid0}" "${pid1}"
        ;;
    quantile120)
        (
            export QKSIEVE_PROFILE_STAGES=1
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=16
            run_case \
                "0,1" sports profile_global3_qmin16_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid0=$!
        (
            export QKSIEVE_PROFILE_STAGES=1
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            run_case \
                "2,3" sports profile_global3_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid1=$!
        wait "${pid0}" "${pid1}"
        ;;
    tightcap120)
        (
            export QKSIEVE_PROFILE_STAGES=1
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            run_case \
                "0,1" sports profile_global3_qmin24_regular_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid0=$!
        (
            export QKSIEVE_PROFILE_STAGES=1
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=4
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_tightcap_packed_direct"
            run_case \
                "2,3" sports profile_global3_qmin24_tightcap_split4_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid1=$!
        (
            export QKSIEVE_PROFILE_STAGES=1
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_tightcap_packed_direct"
            run_case \
                "4,5" sports profile_global3_qmin24_tightcap_split8_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        ) &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    tightcap120_samegpu)
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            run_case \
                "0,1" sports fair_global3_qmin24_regular_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
            export QKSIEVE_ATTENTION_SPLIT_OVERRIDE=8
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_tightcap_packed_direct"
            run_case \
                "0,1" sports fair_global3_qmin24_tightcap_split8_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
        )
        ;;
    wmma120_samegpu)
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            if [[ ! -f "${RUN_ROOT}/fair_global3_scalar_qmin24_b1280_120k/case_summary.json" ]]; then
                run_case \
                    "0,1" sports fair_global3_scalar_qmin24_b1280_120k \
                    "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                    120000 64 0 1280
            fi
            MODE="${WMMA_MODE}"
            run_case \
                "0,1" sports fair_global3_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
        )
        ;;
    wmma32_quality)
        for spec in "2:sports" "3:medicine" "4:mixed_b"; do
            gpu="${spec%%:*}"
            topic="${spec#*:}"
            (
                export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
                if [[ ! -f "${RUN_ROOT}/warm2_scalar_qmin24_${topic}_32k/case_summary.json" ]]; then
                    run_case \
                        "${gpu}" "${topic}" \
                        "warm2_scalar_qmin24_${topic}_32k" \
                        "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                        32000 128 "0,1" 1280
                fi
                MODE="${WMMA_MODE}"
                run_case \
                    "${gpu}" "${topic}" \
                    "warm2_wmma_qmin24_${topic}_32k" \
                    "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                    32000 128 "0,1" 1280
            ) &
        done
        wait
        ;;
    wmma120_repeats)
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"
            run_case \
                "0,1" sports repeat0_scalar_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
            MODE="${WMMA_MODE}"
            run_case \
                "0,1" sports repeat0_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
        ) &
        pid0=$!
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            MODE="${WMMA_MODE}"
            run_case \
                "2,3" sports repeat1_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"
            run_case \
                "2,3" sports repeat1_scalar_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
        ) &
        pid1=$!
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"
            run_case \
                "4,5" sports repeat2_scalar_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
            MODE="${WMMA_MODE}"
            run_case \
                "4,5" sports repeat2_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 64 0 1280
        ) &
        pid2=$!
        wait "${pid0}" "${pid1}" "${pid2}"
        ;;
    wmma120_profile_samegpu)
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            export QKSIEVE_PROFILE_STAGES=1
            MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_qfused_gqa4_kappend_unbiased_packed_direct"
            run_case \
                "0,1" sports profile_samegpu_scalar_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
            MODE="${WMMA_MODE}"
            run_case \
                "0,1" sports profile_samegpu_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                120000 32 0 1280
        )
        ;;
    fixed441_template)
        "${PYTHON}" src/rewrite_qksieve_template_allocation_20260729.py \
            --input "${RUN_ROOT}/templates/global32_3domain_runtime.pt" \
            --output "${RUN_ROOT}/templates/global32_3domain_fixed441.pt" \
            --allocation 4,4,1,0,0,0,0,0
        ;;
    fixed441_wmma120)
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            MODE="${WMMA_MODE}"
            run_case \
                "2,3" sports fixed441_wmma_qmin24_b1280_120k \
                "${RUN_ROOT}/templates/global32_3domain_fixed441.pt" "" \
                120000 64 0 1280
        )
        ;;
    fixed441_wmma32_quality)
        for spec in "4:sports" "5:medicine" "6:mixed_b"; do
            gpu="${spec%%:*}"
            topic="${spec#*:}"
            (
                export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
                MODE="${WMMA_MODE}"
                run_case \
                    "${gpu}" "${topic}" \
                    "fixed441_wmma_qmin24_${topic}_32k" \
                    "${RUN_ROOT}/templates/global32_3domain_fixed441.pt" "" \
                    32000 128 "0,1" 1280
            ) &
        done
        wait
        ;;
    compare64_samegpu)
        run_full_case \
            "6" sports compare64_full_sports \
            64000 128 "0,1"
        run_fier_case \
            "6" sports compare64_fier_sports \
            64000 128 "0,1"
        (
            export QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=24
            MODE="${WMMA_MODE}"
            run_case \
                "6" sports compare64_wmma_sports \
                "${RUN_ROOT}/templates/global32_3domain_runtime.pt" "" \
                64000 128 "0,1" 1280
        )
        ;;
    *)
        echo "unknown stage: ${stage}" >&2
        exit 2
        ;;
esac
