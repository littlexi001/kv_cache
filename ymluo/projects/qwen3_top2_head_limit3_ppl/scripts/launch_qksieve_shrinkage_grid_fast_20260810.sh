#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/py312/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${ROOT}/results/20260810_qksieve_shrinkage_sensitivity_v1}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_shrinkage_sensitivity_v2_fast}"
GPUS="${GPUS:-4,5,6,7}"
ANALYZER="${ROOT}/src/analyze_qksieve_shrinkage_grid_20260810.py"
EQUIVALENCE_VERIFIER="${ROOT}/src/verify_qksieve_shrinkage_grid_equivalence_20260810.py"
SUMMARIZER="${ROOT}/src/summarize_qksieve_shrinkage_sensitivity_20260810.py"
LABELS="qwen3_4b_sports32k,qwen3_4b_medicine32k,llama31_8b_sports32k,llama31_8b_medicine32k"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "The registered grid requires exactly four GPUs." >&2
  exit 2
fi
if [[ -f "${RUN_ROOT}/ALL_COMPLETE" ]]; then
  echo "Already complete: ${RUN_ROOT}"
  exit 0
fi

labels=(
  qwen3_4b_sports32k
  qwen3_4b_medicine32k
  llama31_8b_sports32k
  llama31_8b_medicine32k
)
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED"
for label in "${labels[@]}"; do
  if [[ ! -s "${SOURCE_RUN}/traces/${label}.pt" ]]; then
    echo "Missing frozen trace: ${label}" >&2
    exit 2
  fi
done

{
  echo "schema=qksieve_shrinkage_fast_grid_protocol_v1"
  echo "source_run=${SOURCE_RUN}"
  echo "gpus=${GPUS}"
  echo "query_shrinkages=0,0.25,0.5,0.75,0.9"
  echo "selected_fractions=0.01,0.02,0.04"
  echo "calibration_source=prefill_tail"
  echo "calibration_steps=8"
  echo "sample_stride=32"
  sha256sum "${ANALYZER}" "${EQUIVALENCE_VERIFIER}" "${SUMMARIZER}"
  for label in "${labels[@]}"; do
    sha256sum "${SOURCE_RUN}/traces/${label}.pt"
  done
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader,nounits
} >"${RUN_ROOT}/manifest.txt"

pids=()
for index in 0 1 2 3; do
  label="${labels[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu_list[$index]}" "${PYTHON}" -u "${ANALYZER}" \
    --trace_path "${SOURCE_RUN}/traces/${label}.pt" \
    --output_root "${RUN_ROOT}/analysis/${label}" \
    --label "${label}" \
    --query_shrinkages 0,0.25,0.5,0.75,0.9 \
    --selected_fractions 0.01,0.02,0.04 \
    --sample_stride 32 \
    --calibration_steps 8 \
    --calibration_source prefill_tail \
    --total_rate_budget 15 \
    --top_fraction 0.02 \
    --device cuda \
    >"${RUN_ROOT}/logs/${label}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi

"${PYTHON}" "${EQUIVALENCE_VERIFIER}" \
  --reference_dir \
    "${SOURCE_RUN}/analysis/qwen3_4b_sports32k/lambda_0p00" \
  --candidate_dir \
    "${RUN_ROOT}/analysis/qwen3_4b_sports32k/lambda_0p00" \
  --output "${RUN_ROOT}/equivalence_audit.json" \
  >"${RUN_ROOT}/logs/equivalence.log" 2>&1 || {
    touch "${RUN_ROOT}/FAILED"
    exit 1
  }

"${PYTHON}" "${SUMMARIZER}" \
  --run_root "${RUN_ROOT}" \
  --labels "${LABELS}" \
  --shrinkages 0,0.25,0.5,0.75,0.9 \
  --fractions 0.01,0.02,0.04 \
  --bootstrap_samples 10000 \
  --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1 || {
    touch "${RUN_ROOT}/FAILED"
    exit 1
  }

touch "${RUN_ROOT}/ALL_COMPLETE"
echo "ALL_COMPLETE: ${RUN_ROOT}"
