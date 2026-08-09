#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260803_condres_crossing_ppl_6gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/home/fdong/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QKSIEVE_PROFILE_STAGES=1
export QKSIEVE_CROSSING_MAX_RESCUE=2048

CONDRES_MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_valuesketch16i4shared_wometric_condres8global_packed_fulltopk_oas"
CROSSING_MODE="pca_hierarchical_autoqmsetotal15z_qkmetric_valuesketch16i4shared_wometric_condres8global_keyrisk4_crossempirical99_cal256_packed_fulltopk_oas"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local topic="$3"
  local history_tokens="$4"
  local mode="$5"
  local output="${RUN_ROOT}/${name}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "${MODEL}" \
    --output_dir "${output}" \
    --topics "${topic}" \
    --window_indices 0 \
    --methods full_attention,direct_countcap \
    --history_tokens "${history_tokens}" \
    --eval_tokens 16 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --sample_count 256 \
    --protect_recent_tokens 0 \
    --direct_score_mode "${mode}" \
    --qk_metric_query_shrinkage 0.5 \
    --prefill_chunk_tokens 512 \
    --cache_mode preallocated \
    --collect_logit_stability \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

run_case 0 medicine4k_condres medicine 4096 "${CONDRES_MODE}" & p0=$!
run_case 1 sports4k_condres sports 4096 "${CONDRES_MODE}" & p1=$!
run_case 2 sports32k_condres sports 32768 "${CONDRES_MODE}" & p2=$!
run_case 3 medicine4k_crossing medicine 4096 "${CROSSING_MODE}" & p3=$!
run_case 4 sports4k_crossing sports 4096 "${CROSSING_MODE}" & p4=$!
run_case 5 sports32k_crossing sports 32768 "${CROSSING_MODE}" & p5=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
