#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

selection_root="outputs/longmemeval_10m_evidence_state_all500_v1"
output_root="outputs/longmemeval_10m_all500_dates_chronological_reader_v1"
mkdir -p "${output_root}"
IFS=',' read -r -a partitions <<< "${PARTITION_LIST:?set PARTITION_LIST}"
IFS=',' read -r -a gpu_ids <<< "${GPU_LIST:?set GPU_LIST}"
if [[ "${#partitions[@]}" -ne "${#gpu_ids[@]}" ]]; then
  echo "PARTITION_LIST and GPU_LIST must have the same length" >&2
  exit 2
fi

pids=()
for slot in "${!gpu_ids[@]}"; do
  partition="${partitions[slot]}"
  if [[ -s "${output_root}/part${partition}/summary.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu_ids[slot]}" \
    /home/fdong/miniconda3/envs/moe/bin/python \
    src/evaluate_longmemeval_10m_selected_reader.py \
    --data_dir "data/longmemeval_10m_partition${partition}_v1" \
    --selection_rows "${selection_root}/part${partition}/rows.jsonl" \
    --output_dir "${output_root}/part${partition}" \
    --device cuda:0 \
    --include_page_dates \
    --page_order chronological \
    > "${output_root}/part${partition}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
