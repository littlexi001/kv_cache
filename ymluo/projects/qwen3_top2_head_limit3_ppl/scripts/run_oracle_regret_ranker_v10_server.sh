#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

STAMP="${STAMP:-20260704_v10}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/bin/python}"
INPUT_KIND="${INPUT_KIND:-expanded}"
EPOCHS="${EPOCHS:-250}"
HIDDEN_DIM="${HIDDEN_DIM:-12}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PAIR_MARGIN="${PAIR_MARGIN:-0.1}"
LISTWISE_WEIGHT="${LISTWISE_WEIGHT:-0.15}"
POINTWISE_WEIGHT="${POINTWISE_WEIGHT:-0.15}"
FALLBACK_MARGINS="${FALLBACK_MARGINS:-0,0.05,0.1,0.2,0.35,0.5,0.8,1.2}"

case "$INPUT_KIND" in
  expanded)
    INPUT_DIR="outputs/oracle_regret_memory_planner_v7_20x_b1358_20260703_v7_expanded_5x20"
    OUT="outputs/oracle_regret_ranker_v10_${STAMP}_expanded"
    ;;
  hardnoise)
    INPUT_DIR="outputs/oracle_regret_memory_planner_v7_10x_b358_20260703_v7_hardnoise_5x10"
    OUT="outputs/oracle_regret_ranker_v10_${STAMP}_hardnoise"
    ;;
  *)
    echo "Unknown INPUT_KIND=$INPUT_KIND. Use expanded or hardnoise." >&2
    exit 2
    ;;
esac

OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$PYTHON" src/train_oracle_regret_ranker_v10.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUT" \
  --epochs "$EPOCHS" \
  --hidden_dim "$HIDDEN_DIM" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --pair_margin "$PAIR_MARGIN" \
  --listwise_weight "$LISTWISE_WEIGHT" \
  --pointwise_weight "$POINTWISE_WEIGHT" \
  --feature_policy learned_causal_proxy \
  --full_fallback_margins "$FALLBACK_MARGINS"
