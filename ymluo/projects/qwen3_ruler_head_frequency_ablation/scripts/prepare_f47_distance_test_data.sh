#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
TOP2=${TOP2:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
LM_EVAL=${LM_EVAL:-/home/fdong/lm-evaluation-harness}
HOTPOT=${HOTPOT:-/home/fdong/ymluo/datasets/ruler_sources/hotpotqa/distractor/validation-00000-of-00001.parquet}
RUN=${RUN:-$ROOT/outputs/multiseed_frequency_scaling_20260806/f47_distance_bf16}
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot

mkdir -p "$RUN/data"

prepare_seed() {
  local seed=$1
  local output="$RUN/data/ruler32k_seed${seed}_m2.jsonl"
  "$PY" "$TOP2/src/prepare_hierarchical_ruler_data_20260716.py" \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --output "$output" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 32768 \
    --max_samples_per_task 2 \
    --seed "$seed" \
    --ruler_hotpot_parquet "$HOTPOT" \
    >"$RUN/data/seed${seed}.log" 2>&1
}

prepare_seed 54 & p0=$!
prepare_seed 55 & p1=$!
prepare_seed 56 & p2=$!
wait "$p0" "$p1" "$p2"
touch "$RUN/data.done"
