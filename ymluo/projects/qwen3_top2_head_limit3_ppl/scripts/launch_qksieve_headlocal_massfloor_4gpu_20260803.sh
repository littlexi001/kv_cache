#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_TRACES="${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces"
LLAMA4K_TRACE="${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt"
LLAMA128K_TRACE="${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_headlocal_massfloor_4gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

run_case() {
  local gpu="$1"
  local name="$2"
  local trace="$3"
  local model="$4"
  mkdir -p "$OUTPUT/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
    --trace "$trace" \
    --output_dir "$OUTPUT/$name" \
    --model_name_or_path "$model" \
    --device cuda \
    --fixed_top_k 1280 \
    --fixed_top_ks 640,1280,2560 \
    --coverage_targets 0.90,0.95,0.975,0.99 \
    --minimum_top_k 1280 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    --score_calibration_samples 256 \
    >"$OUTPUT/logs/$name.log" 2>&1
  touch "$OUTPUT/${name}_COMPLETE"
}

run_case 0 religion4k "$LLAMA4K_TRACE" "$LLAMA_MODEL" & pid0=$!
run_case 1 sports32k "$QWEN_TRACES/sports.pt" "$QWEN_MODEL" & pid1=$!
run_case 2 medicine32k "$QWEN_TRACES/medicine.pt" "$QWEN_MODEL" & pid2=$!
run_case 3 computer128k "$LLAMA128K_TRACE" "$LLAMA_MODEL" & pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
touch "$OUTPUT/ALL_COMPLETE"
