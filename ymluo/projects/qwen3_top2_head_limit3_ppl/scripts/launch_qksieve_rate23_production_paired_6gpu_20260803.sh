#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_rate23_production_paired_6gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

RATE15="pca_hierarchical_autoqmsetotal15z_qkmetric_valuesketch16i4shared_wometric_residualrisk4_prefixrss25e4_safety1_packed_fulltopk_oas"
RATE23="pca_hierarchical_autoqmsetotal23z_qkmetric_valuesketch16i4shared_wometric_residualrisk4_prefixrss25e4_safety2_packed_fulltopk_oas"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_PROFILE_STAGES=1
export QKSIEVE_RECORD_LAYER_CANDIDATE_STATS=1

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local topic="$4"
  local history_tokens="$5"
  local score_mode="$6"
  local output_dir="${OUTPUT}/${name}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "${model}" \
    --output_dir "${output_dir}" \
    --topics "${topic}" \
    --window_indices 0 \
    --methods full_attention,direct_countcap \
    --history_tokens "${history_tokens}" \
    --eval_tokens 8 \
    --direct_fraction 0.02 \
    --direct_min_tokens 1 \
    --direct_max_tokens 8192 \
    --sample_count 256 \
    --protect_recent_tokens 0 \
    --direct_score_mode "${score_mode}" \
    --prefill_chunk_tokens 1024 \
    --cache_mode preallocated \
    --preallocated_cache_min_tokens 1 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --seed 20260803 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --collect_logit_stability \
    >"${OUTPUT}/logs/${name}.log" 2>&1
  touch "${OUTPUT}/${name}_COMPLETE"
}

run_case 0 religion4k_rate15 "${LLAMA_MODEL}" religion 4096 "${RATE15}" & pid0=$!
run_case 1 religion4k_rate23 "${LLAMA_MODEL}" religion 4096 "${RATE23}" & pid1=$!
run_case 2 sports32k_rate15 "${QWEN_MODEL}" sports 32000 "${RATE15}" & pid2=$!
run_case 3 sports32k_rate23 "${QWEN_MODEL}" sports 32000 "${RATE23}" & pid3=$!
run_case 4 medicine96k_rate15 "${QWEN_MODEL}" medicine 96000 "${RATE15}" & pid4=$!
run_case 5 medicine96k_rate23 "${QWEN_MODEL}" medicine 96000 "${RATE23}" & pid5=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
