#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
EXP="$ROOT/experiments/temporal_representation_fusion_20260804"
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
CRITICAL="$ROOT/outputs/age_distractor_prerope_ablation_v2_20260726/counterfactual_heads.csv"
OUT="$ROOT/outputs/temporal_representation_fusion_20260804_phase_bank_v3"

mkdir -p "$OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6,7 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" "$EXP/src/run_temporal_representation_fusion_8b.py" \
    --model-name-or-path "$MODEL" \
    --critical-heads-csv "$CRITICAL" \
    --output-dir "$OUT" \
    --candidate-offsets 1,2,4,8,16,32,48,64,96,128,192,256,384,512 \
    --layers 20,22,23 \
    --alphas 0.25,0.5,0.75,1.0 \
    --modes residual_linear,q_pre_current_phase,q_native_phase \
    --strategies offset64,diverse1,diverse2,diverse4 \
    --prefill-chunk-size 128 \
    --dtype bfloat16 \
    --device-map long_context_2gpu \
    --attn-implementation sdpa \
    --original-max-position-embeddings 40960 \
    --fixed-rope-factor 4.0 \
    --fixed-max-position-embeddings 147456 \
    > "$OUT/run.log" 2>&1

printf 'ok\n' > "$OUT/done.txt"
