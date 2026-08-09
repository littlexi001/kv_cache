#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/candidate_margin_dense_34_100_20260724}"
GPU="${GPU:-4}"

mkdir -p "$OUT/data"
exec 8>"$OUT/single_gpu_launcher.lock"
flock -n 8 || {
  echo "single-GPU candidate-margin launcher already running"
  exit 0
}
rm -f "$OUT/launcher.done" "$OUT/launcher.failed"
trap 'date -Is >"$OUT/launcher.failed"' ERR

read -r memory utilization < <(
  nvidia-smi -i "$GPU" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    tr -d ','
)
if (( GPU < 4 || GPU > 7 || memory >= 2000 || utilization >= 10 )); then
  echo "GPU $GPU is unavailable: memory=$memory utilization=$utilization"
  exit 1
fi

lengths="$(seq 34 100 | paste -sd, -)"
echo "$(date -Is) selected GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u \
  "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --lengths "$lengths" \
  --seed 0 \
  --code_mode english_single_token \
  --placement middle \
  --query_mode full2 \
  --prompt_style legacy \
  --max_top 100 \
  --prefill_chunk_size 128 \
  --dtype float16 \
  --device cuda \
  --device_map none \
  --attn_implementation sdpa \
  --original_max_position_embeddings 40960 \
  --global_max_position 130000 \
  --shard_label "gpu${GPU}_single" \
  >"$OUT/single_gpu.log" 2>&1

"$PY" "$PROJECT/src/analyze_single_sample_failure_trace.py" \
  --input_dir "$OUT/data" \
  --output_dir "$OUT/analysis" \
  >"$OUT/analysis.log" 2>&1

rm -f "$OUT/launcher.failed"
date -Is >"$OUT/launcher.done"
echo "$(date -Is) complete: $OUT"
