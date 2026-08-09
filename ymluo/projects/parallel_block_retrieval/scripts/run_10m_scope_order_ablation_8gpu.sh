#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

domain="${DOMAIN:?set DOMAIN to pg19 or code}"
if [[ "${domain}" == "pg19" ]]; then
  data_dir="data/pg19_past_only_10m_q77_v2"
  selection_rows="outputs/pg19_past_only_10m_q77_rag_ppl_s512_v1/rows.jsonl"
  expected_queries=77
elif [[ "${domain}" == "code" ]]; then
  data_dir="data/longbench_code_10m_continuation_memory_v1"
  selection_rows="outputs/longbench_code_10m_retrieval_ppl_locality_v1/rows.jsonl"
  expected_queries=30
else
  echo "unsupported DOMAIN=${domain}" >&2
  exit 2
fi

output_dir="outputs/${domain}_10m_scope_order_ablation_v1"
mkdir -p "${output_dir}"
IFS=',' read -r -a gpu_ids <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
world_size="${#gpu_ids[@]}"
pids=()
for rank in "${!gpu_ids[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[rank]}" \
    /home/fdong/miniconda3/envs/moe/bin/python \
    src/evaluate_10m_scope_order_ablation.py \
    --data_dir "${data_dir}" \
    --selection_rows "${selection_rows}" \
    --output_dir "${output_dir}" \
    --method bm25_e5_rrf \
    --device cuda:0 \
    --rank "${rank}" \
    --world_size "${world_size}" \
    > "${output_dir}/rank${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if ((status != 0)); then
  exit "${status}"
fi

/home/fdong/miniconda3/envs/moe/bin/python \
  src/analyze_10m_scope_order_ablation.py \
  --input_dir "${output_dir}" \
  --world_size "${world_size}" \
  --expected_queries "${expected_queries}" \
  --domain "${domain}" \
  --output "${output_dir}/analysis.json" \
  > "${output_dir}/analysis.log"
