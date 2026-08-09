#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
OUT="$ROOT/results/20260803_progressive_balanced_short_2gpu_v1"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
LLAMA=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms

mkdir -p "$OUT/logs"

run_case() {
    local gpu="$1"
    local name="$2"
    local trace="$3"
    local refinement_rate
    for refinement_rate in 19 23 27; do
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
            "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
            --trace "$trace" \
            --output_dir "$OUT/$name/base15_refine$refinement_rate" \
            --model_name_or_path "$LLAMA" \
            --device cuda \
            --fixed_top_ks 1280 \
            --coverage_targets 0.9 \
            --minimum_top_k 1 \
            --key_rate_budget 15 \
            --key_refinement_rate_budget "$refinement_rate" \
            --progressive_refinement_rounds 2 \
            --key_quantizer plain \
            --key_allocation_objective oas_qk_mse \
            --query_factor_source prefill \
            --query_factor_prefill_tokens 8 \
            --balanced_rss_tolerances 0.0025 \
            --rss_safety_factors 2 \
            --focus_progressive_balanced_rss \
            > "$OUT/logs/${name}_base15_refine${refinement_rate}.log" 2>&1
    done
}

run_case 0 qasper3k \
    "$ROOT/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces/qasper__9fb085a1f47673d1907f2378c90843b4b6e8622a14fe1fa9__qksieve_fullprompt_auto_plain_fulltopk.pt" &
P0=$!
run_case 1 religion4k \
    "$ROOT/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt" &
P1=$!

wait "$P0" "$P1"
touch "$OUT/ALL_COMPLETE"
