#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export PROJECT_DIR
export STAMP
export CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/real_longbench_docqa_10m_clean_record64}"
export PROFILE_DIR="${PROFILE_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile}"
export OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_prerope_colbert_scaling_${STAMP}}"
export FORCE_PREPARE="${FORCE_PREPARE:-false}"
export FORCE_PROFILE="${FORCE_PROFILE:-false}"
export PROFILE_SPACE="pre_rope_record_qk"
export QUERY_VECTOR_TOKENS="${QUERY_VECTOR_TOKENS:-16}"
export QUERY_VECTOR_MODE="question_content"
export METHODS="${METHODS:-colbert128,colbert32,colbert64,colbert32_rerank}"
export NLL_METHODS="${NLL_METHODS:-colbert128,colbert32,colbert64,colbert32_rerank}"
export RUN_NLL="${RUN_NLL:-true}"

exec bash "${SCRIPT_DIR}/run_real_qk_pipeline_server.sh"
