#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_smoke}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
PY="${PY:-python}"
VARIANTS="${VARIANTS:-casual_recent,temporal_fact,multihop_bridge,summary_theme,compare_score}"
TASKS_PER_VARIANT="${TASKS_PER_VARIANT:-4}"
DISTRACTOR_PAGES="${DISTRACTOR_PAGES:-16}"
TOPK_BUDGETS="${TOPK_BUDGETS:-1,3,5}"
MAX_ABLATE_PAGES="${MAX_ABLATE_PAGES:-12}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
SEED="${SEED:-2026070307}"

OUT="outputs/oracle_regret_memory_planner_v7_${TASKS_PER_VARIANT}x_b${TOPK_BUDGETS//,/}_${STAMP}"
LOG="outputs/logs/oracle_regret_memory_planner_v7_${TASKS_PER_VARIANT}x_b${TOPK_BUDGETS//,/}_${STAMP}.log"
mkdir -p outputs/logs "$OUT"

"$PY" src/run_oracle_regret_memory_planner_v7.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --variants "$VARIANTS" \
  --tasks_per_variant "$TASKS_PER_VARIANT" \
  --train_fraction "$TRAIN_FRACTION" \
  --distractor_pages "$DISTRACTOR_PAGES" \
  --topk_budgets "$TOPK_BUDGETS" \
  --max_ablate_pages "$MAX_ABLATE_PAGES" \
  --seed "$SEED" \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  2>&1 | tee "$LOG"

echo "$OUT"
