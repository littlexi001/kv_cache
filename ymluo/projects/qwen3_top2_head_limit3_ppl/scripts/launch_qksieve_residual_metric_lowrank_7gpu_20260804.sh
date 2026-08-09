#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260804_qksieve_query_crossfit_multistep_8gpu_v1/traces}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_residual_metric_lowrank_7gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/home/fdong/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local trace="$3"
  local model="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_residual_metric_lowrank_20260804.py \
    --trace "${trace}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --evaluation_stride 16 \
    --timing_iterations 20 \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
}

run_case 1 qwen_medicine32k "${TRACE_ROOT}/qwen_medicine32k.pt" "${QWEN_MODEL}" & p1=$!
run_case 2 qwen_religion32k "${TRACE_ROOT}/qwen_religion32k.pt" "${QWEN_MODEL}" & p2=$!
run_case 3 qwen_computer32k "${TRACE_ROOT}/qwen_computer32k.pt" "${QWEN_MODEL}" & p3=$!
run_case 4 qwen_sports96k "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/sports.pt" "${QWEN_MODEL}" & p4=$!
run_case 5 qwen_medicine96k "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/medicine.pt" "${QWEN_MODEL}" & p5=$!
run_case 6 llama_religion4k "${TRACE_ROOT}/llama_religion4k.pt" "${LLAMA_MODEL}" & p6=$!
run_case 7 llama_computer32k "${TRACE_ROOT}/llama_computer32k.pt" "${LLAMA_MODEL}" & p7=$!

failed=0
for pid in "${p1}" "${p2}" "${p3}" "${p4}" "${p5}" "${p6}" "${p7}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
