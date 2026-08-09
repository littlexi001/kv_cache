#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
OUT="${OUT:-$ROOT/results/20260803_balanced_joint_rss_96k_v1}"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
TRACE_ROOT="$ROOT/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces"

mkdir -p "$OUT/logs"

run_topic() {
    local gpu="$1"
    local topic="$2"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
        --trace "$TRACE_ROOT/$topic.pt" \
        --output_dir "$OUT/$topic" \
        --model_name_or_path "$MODEL" \
        --device cuda \
        --fixed_top_ks 1280 \
        --coverage_targets 0.9 \
        --minimum_top_k 256 \
        --key_rate_budget 15 \
        --key_quantizer plain \
        --key_allocation_objective oas_qk_mse \
        --query_factor_source prefill \
        --query_factor_prefill_tokens 8 \
        --balanced_rss_tolerances 0.0025,0.005,0.01 \
        --rss_safety_factors 1,2 \
        --focus_balanced_rss \
        > "$OUT/logs/$topic.log" 2>&1
}

run_topic 4 sports &
SPORTS_PID=$!
run_topic 5 medicine &
MEDICINE_PID=$!

printf '%s\n' "$SPORTS_PID" > "$OUT/sports.pid"
printf '%s\n' "$MEDICINE_PID" > "$OUT/medicine.pid"

wait "$SPORTS_PID"
wait "$MEDICINE_PID"
touch "$OUT/ALL_COMPLETE"
