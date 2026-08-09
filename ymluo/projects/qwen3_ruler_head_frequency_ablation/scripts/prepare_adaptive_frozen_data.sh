#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
TOP2=${TOP2:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
LM_EVAL=${LM_EVAL:-/home/fdong/lm-evaluation-harness}
HOTPOT=${HOTPOT:-/home/fdong/ymluo/datasets/ruler_sources/hotpotqa/distractor/validation-00000-of-00001.parquet}
OUT=${OUT:-$ROOT/outputs/multiseed_frequency_scaling_20260806/adaptive_frozen_data}
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
SEEDS=${SEEDS:-60,61,62}
MAX_SAMPLES_PER_TASK=${MAX_SAMPLES_PER_TASK:-2}

mkdir -p "$OUT"

prepare_seed() {
  local seed=$1
  local output="$OUT/ruler32k_seed${seed}_m${MAX_SAMPLES_PER_TASK}.jsonl"
  "$PY" "$TOP2/src/prepare_hierarchical_ruler_data_20260716.py" \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --output "$output" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 32768 \
    --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
    --seed "$seed" \
    --ruler_hotpot_parquet "$HOTPOT" \
    >"$OUT/seed${seed}.log" 2>&1
}

pids=()
for seed in ${SEEDS//,/ }; do
  prepare_seed "$seed" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
touch "$OUT/data.done"
