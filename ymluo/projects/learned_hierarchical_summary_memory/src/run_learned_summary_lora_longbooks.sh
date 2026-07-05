#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong}"
MODEL_PATH="${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s${TRAIN_STEPS}_20260703}"

TEXT_PATHS="$PROJECT_ROOT/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt,\
$PROJECT_ROOT/ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt"
DATASET_NAMES="warpeace,montecristo"

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES

python ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_lora_adaptation.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --text_paths "$TEXT_PATHS" \
  --dataset_names "$DATASET_NAMES" \
  --eval_methods full_raw,recent_only,static_hier,static_sum100,static_sum1000 \
  --train_method static_hier \
  --train_steps "$TRAIN_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --grad_accum_steps 1 \
  --prefill_tokens 8192 \
  --target_tokens 128 \
  --block_tokens 2048 \
  --recent_tokens 512 \
  --summary10_words 10 \
  --summary100_words 100 \
  --summary1000_words 900 \
  --max_text_tokens 240000 \
  --train_start_tokens 0 \
  --train_span_tokens 120000 \
  --eval_start_tokens 150000 \
  --eval_samples_per_dataset 8 \
  --eval_stride_tokens 2048 \
  --seed 2026070309 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --device cuda \
  --dtype float16 \
  --attn_implementation sdpa \
  --summary_backend learned \
  --learned_summary_train_tokens 120000 \
  --learned_summary_epochs 8 \
  --learned_summary_hidden_dim 32 \
  --learned_summary_lr 3e-3 \
  --learned_summary_max_sentences 20000 \
  --learned_summary_seed 2026070307
