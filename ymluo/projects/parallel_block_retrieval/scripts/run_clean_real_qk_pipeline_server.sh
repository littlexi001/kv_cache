#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export PROJECT_DIR
export STAMP
export DATASETS="${DATASETS:-hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en,qmsum,gov_report}"
export QUERY_DATASETS="${QUERY_DATASETS:-hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en}"
export CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/real_longbench_docqa_10m_clean_record64}"
export PROFILE_DIR="${PROFILE_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_clean_postrope_qk64_profile}"
export OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_clean_scaling_${STAMP}}"
export RUN_NLL="${RUN_NLL:-true}"

exec bash "${SCRIPT_DIR}/run_real_qk_pipeline_server.sh"
