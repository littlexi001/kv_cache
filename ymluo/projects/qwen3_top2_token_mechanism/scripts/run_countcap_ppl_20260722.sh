#!/usr/bin/env bash
set -euo pipefail

GPU_LIST=${1:?'usage: GPU_LIST TOPIC HISTORY_TOKENS EVAL_TOKENS OUTPUT_DIR'}
TOPIC=${2:?'topic is required'}
HISTORY_TOKENS=${3:?'history length is required'}
EVAL_TOKENS=${4:?'evaluation length is required'}
OUTPUT_DIR=${5:?'output directory is required'}

PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
PROJECT=${PROJECT:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism}

read -r CANDIDATE_FRACTION FINAL_FRACTION < <(
    PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}" "${PYTHON}" -c '
import sys
from count_capped_support_policy import count_capped_support

policy = count_capped_support(int(sys.argv[1]))
print(policy.candidate_fraction, policy.final_fraction)
' "${HISTORY_TOKENS}"
)

bash "${PROJECT}/scripts/run_fused_sampleq_ppl_20260722.sh" \
    "${GPU_LIST}" \
    pca_int4_chunked_logscale16_qkmetric_sampleq_autosplit \
    "${TOPIC}" \
    "${HISTORY_TOKENS}" \
    "${EVAL_TOKENS}" \
    "${OUTPUT_DIR}" \
    48 \
    "${CANDIDATE_FRACTION}" \
    "${FINAL_FRACTION}"
