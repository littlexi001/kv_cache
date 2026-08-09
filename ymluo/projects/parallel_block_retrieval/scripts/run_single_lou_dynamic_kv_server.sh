#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
CORPUS="${CORPUS:-${ROOT}/data/real_longbench_docqa_10m_clean_record64}"
OUTPUT="${OUTPUT:-${ROOT}/outputs/single_lou_dynamic_kv_v1}"

mapfile -t FREE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) < 1000 && ($3 + 0) < 10) print $1}'
)
if ((${#FREE_GPUS[@]} == 0)); then
  echo "No idle GPU is available" >&2
  exit 1
fi

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

COMMON=(
  "${ROOT}/src/run_single_query_dynamic_kv_generation.py"
  --corpus_dir "${CORPUS}"
  --query_id 0
  --max_new_tokens 128
)
CONFIGS=(
  "full_free|--mode full_source"
  "question_only_free|--mode question_only"
  "dynamic_c2k2_free|--mode dynamic --retrieval_interval 2 --blocks_per_refresh 2"
  "dynamic_c2k2_seed|--mode dynamic --retrieval_interval 2 --blocks_per_refresh 2 --seed_hop1"
  "dynamic_c1k1_free|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 1"
  "dynamic_c1k2_free|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 2"
  "dynamic_c1k3_free|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 3"
  "dynamic_c2k1_free|--mode dynamic --retrieval_interval 2 --blocks_per_refresh 1"
  "dynamic_c2k3_free|--mode dynamic --retrieval_interval 2 --blocks_per_refresh 3"
  "dynamic_c3k1_free|--mode dynamic --retrieval_interval 3 --blocks_per_refresh 1"
  "dynamic_c3k2_free|--mode dynamic --retrieval_interval 3 --blocks_per_refresh 2"
  "dynamic_c3k3_free|--mode dynamic --retrieval_interval 3 --blocks_per_refresh 3"
  "full_hop2_query|--mode full_source --seed_hop2_query"
  "question_only_hop2_query|--mode question_only --seed_hop2_query"
  "dynamic_k1_hop2_query|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 1 --seed_hop2_query"
  "dynamic_k2_hop2_query|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 2 --seed_hop2_query"
  "dynamic_k3_hop2_query|--mode dynamic --retrieval_interval 1 --blocks_per_refresh 3 --seed_hop2_query"
)

pids=()
for index in "${!CONFIGS[@]}"; do
  IFS='|' read -r name raw_args <<< "${CONFIGS[$index]}"
  gpu="${FREE_GPUS[$((index % ${#FREE_GPUS[@]}))]}"
  read -r -a extra <<< "${raw_args}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${COMMON[@]}" \
    --output_path "${OUTPUT}/${name}.json" "${extra[@]}" \
    > "${OUTPUT}/${name}.log" 2>&1 &
  pids+=("$!")
  if (((${#FREE_GPUS[@]} < ${#CONFIGS[@]}) && ((index + 1) % ${#FREE_GPUS[@]} == 0))); then
    wait "${pids[@]}"
    pids=()
  fi
done
if ((${#pids[@]})); then
  wait "${pids[@]}"
fi

"${PYTHON}" - "${OUTPUT}" <<'PY'
import glob
import json
import os
import sys

output = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(output, "*.json"))):
    data = json.load(open(path, encoding="utf-8"))
    rows.append(
        {
            "run": os.path.basename(path).removesuffix(".json"),
            "answer_hit_128": data["answer_hit_128"],
            "first_answer_end_normalized_token": data["first_answer_end_normalized_token"],
            "gold_ever_retrieved": data["gold_ever_retrieved"],
            "unique_block_count": data["unique_block_count"],
            "generated_tokens": data["generated_tokens"],
            "dayton_first_token_probability": data["dayton_first_token_probability"],
            "dayton_first_token_rank": data["dayton_first_token_rank"],
            "generated_text": data["generated_text"],
        }
    )
with open(os.path.join(output, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
