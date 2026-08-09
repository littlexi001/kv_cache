#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

state_root="outputs/longmemeval_10m_temporal_multivalue_state_ku_v1"
reader_root="outputs/longmemeval_10m_temporal_multivalue_reader_ku_v1"
mkdir -p "${state_root}" "${reader_root}"

run_parallel_phase() {
  local phase="$1"
  local pids=()
  for partition in $(seq 0 7); do
    if [[ "${phase}" == "state" ]]; then
      CUDA_VISIBLE_DEVICES="${partition}" \
        /home/fdong/miniconda3/envs/moe/bin/python \
        src/evaluate_longmemeval_10m_evidence_conditioned_state.py \
        --data_dir "data/longmemeval_10m_partition${partition}_v1" \
        --output_dir "${state_root}/part${partition}" \
        --device cuda:0 \
        --question_types knowledge-update \
        --state_prompt_mode temporal_multivalue \
        --state_tokens 64 \
        --state_prefixes 16,32,64 \
        > "${state_root}/part${partition}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${partition}" \
        /home/fdong/miniconda3/envs/moe/bin/python \
        src/evaluate_longmemeval_10m_selected_reader.py \
        --data_dir "data/longmemeval_10m_partition${partition}_v1" \
        --selection_rows "${state_root}/part${partition}/rows.jsonl" \
        --output_dir "${reader_root}/part${partition}" \
        --device cuda:0 \
        --question_types knowledge-update \
        > "${reader_root}/part${partition}.log" 2>&1 &
    fi
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

run_parallel_phase state
run_parallel_phase reader
