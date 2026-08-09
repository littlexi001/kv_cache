#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_TRACES="${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces"
LLAMA_TRACE="${ROOT}/results/20260803_llama4k_religion_qkv_trace_v1/traces/religion.pt"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_joint_output_risk_diagnostic_3gpu_v1}"

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
    --global_top_ks 1280 \
    --global_floor_fractions 0.25,0.5,0.75 \
    --global_priority_names grouprisk4,jointrmse4,jointoracle4 \
    --coverage_targets 0.90 \
    --minimum_top_k 1 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    >"$OUTPUT/logs/$name.log" 2>&1
  touch "$OUTPUT/${name}_COMPLETE"
}

run_case 0 sports "$QWEN_TRACES/sports.pt" "$QWEN_MODEL" &
pid0=$!
run_case 1 medicine "$QWEN_TRACES/medicine.pt" "$QWEN_MODEL" &
pid1=$!
run_case 2 religion4k "$LLAMA_TRACE" "$LLAMA_MODEL" &
pid2=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
touch "$OUTPUT/ALL_COMPLETE"
