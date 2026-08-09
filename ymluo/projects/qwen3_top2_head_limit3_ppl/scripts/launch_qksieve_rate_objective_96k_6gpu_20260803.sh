#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
OUT="$ROOT/results/20260803_qksieve_rate_objective_96k_6gpu_v1"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
TRACE_ROOT="$ROOT/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces"

mkdir -p "$OUT/logs"

run_config() {
    local gpu="$1"
    local name="$2"
    local objective="$3"
    local quantizer="$4"
    local rate="$5"
    local topic
    for topic in sports medicine; do
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
            "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
            --trace "$TRACE_ROOT/$topic.pt" \
            --output_dir "$OUT/$name/$topic" \
            --model_name_or_path "$MODEL" \
            --device cuda \
            --fixed_top_ks 1280 \
            --coverage_targets 0.9 \
            --minimum_top_k 256 \
            --key_rate_budget "$rate" \
            --key_quantizer "$quantizer" \
            --key_allocation_objective "$objective" \
            --query_factor_source prefill \
            --query_factor_prefill_tokens 8 \
            --balanced_rss_tolerances 0.0025 \
            --rss_safety_factors 1 \
            --focus_balanced_rss \
            > "$OUT/logs/${name}_${topic}.log" 2>&1
    done
}

run_config 0 key_plain_r15 key_mse plain 15 &
P0=$!
run_config 1 qk_plain_r15 qk_mse plain 15 &
P1=$!
run_config 2 oas_plain_r15 oas_qk_mse plain 15 &
P2=$!
run_config 3 oas_plain_r19 oas_qk_mse plain 19 &
P3=$!
run_config 4 qk_metric_r15 qk_mse metric 15 &
P4=$!
run_config 5 oas_metric_r15 oas_qk_mse metric 15 &
P5=$!

wait "$P0" "$P1" "$P2" "$P3" "$P4" "$P5"
touch "$OUT/ALL_COMPLETE"
