#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
OUT="$ROOT/results/20260803_first_decode_allocation_6gpu_v1"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
QWEN=/home/fdong/models/Qwen3-4B-Instruct
LLAMA=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms

mkdir -p "$OUT/logs"

run_case() {
    local gpu="$1"
    local name="$2"
    local trace="$3"
    local model="$4"
    local rates="$5"
    local sources="$6"
    local source
    local rate
    for source in $sources; do
        for rate in $rates; do
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
                "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
                --trace "$trace" \
                --output_dir "$OUT/$name/${source}_rate$rate" \
                --model_name_or_path "$model" \
                --device cuda \
                --fixed_top_ks 1280 \
                --coverage_targets 0.9 \
                --minimum_top_k 1 \
                --key_rate_budget "$rate" \
                --key_quantizer plain \
                --key_allocation_objective oas_qk_mse \
                --key_allocation_query_source "$source" \
                --query_factor_source prefill \
                --query_factor_prefill_tokens 8 \
                --balanced_rss_tolerances 0.0025 \
                --rss_safety_factors 2 \
                --focus_balanced_rss \
                > "$OUT/logs/${name}_${source}_rate${rate}.log" 2>&1
        done
    done
}

run_case 0 qasper3k \
    "$ROOT/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces/qasper__9fb085a1f47673d1907f2378c90843b4b6e8622a14fe1fa9__qksieve_fullprompt_auto_plain_fulltopk.pt" \
    "$LLAMA" "19 23" "decode_first prefill_decode_first" &
P0=$!
run_case 1 religion4k \
    "$ROOT/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt" \
    "$LLAMA" "15 19" "decode_first prefill_decode_first" &
P1=$!
run_case 2 sports32k \
    "$ROOT/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/sports.pt" \
    "$QWEN" "15 19" "decode_first" &
P2=$!
run_case 3 medicine32k \
    "$ROOT/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/medicine.pt" \
    "$QWEN" "15 19" "decode_first" &
P3=$!
run_case 4 sports96k \
    "$ROOT/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/sports.pt" \
    "$QWEN" "15 19" "decode_first" &
P4=$!
run_case 5 medicine96k \
    "$ROOT/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/medicine.pt" \
    "$QWEN" "15 19" "decode_first" &
P5=$!

wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5"
touch "$OUT/ALL_COMPLETE"
