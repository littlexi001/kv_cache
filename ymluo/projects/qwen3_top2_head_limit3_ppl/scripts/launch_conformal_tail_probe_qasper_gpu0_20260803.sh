#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
OUT="$ROOT/results/20260803_tail_partition_qasper_rate15_v1"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
TRACE="$ROOT/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces/qasper__9fb085a1f47673d1907f2378c90843b4b6e8622a14fe1fa9__qksieve_fullprompt_auto_plain_fulltopk.pt"

mkdir -p "$OUT/logs"
for rate in 15; do
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
        "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
        --trace "$TRACE" \
        --output_dir "$OUT/rate$rate" \
        --model_name_or_path "$MODEL" \
        --device cuda \
        --fixed_top_ks 1280 \
        --coverage_targets 0.9 \
        --minimum_top_k 1 \
        --key_rate_budget "$rate" \
        --key_quantizer plain \
        --key_allocation_objective oas_qk_mse \
        --key_allocation_query_source basis \
        --query_factor_source prefill \
        --query_factor_prefill_tokens 8 \
        --balanced_rss_tolerances 0.0025 \
        --rss_safety_factors 2 \
        --focus_balanced_rss \
        > "$OUT/logs/rate${rate}.log" 2>&1
done
touch "$OUT/ALL_COMPLETE"
