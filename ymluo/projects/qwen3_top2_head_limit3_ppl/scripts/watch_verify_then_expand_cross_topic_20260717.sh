#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
OUTPUT_ROOT="$PROJECT/results/20260717_verify_then_expand_cross_topic"
LOG_ROOT="$PROJECT/logs"
POLL_SECONDS=20
STABLE_SECONDS=20

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '$2 + 0 < 1000 && $3 + 0 < 10 {gsub(/ /, "", $1); print $1}'
}

while true; do
  mapfile -t first < <(free_gpus)
  if (( ${#first[@]} >= 2 )); then
    sleep "$STABLE_SECONDS"
    mapfile -t second < <(free_gpus)
    stable=()
    for gpu in "${first[@]}"; do
      if printf '%s\n' "${second[@]}" | grep -qx "$gpu"; then
        stable+=("$gpu")
      fi
    done
    if (( ${#stable[@]} >= 2 )); then
      gpu_a=${stable[0]}
      gpu_b=${stable[1]}
      break
    fi
  fi
  sleep "$POLL_SECONDS"
done

cd "$PROJECT"
export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH

CUDA_VISIBLE_DEVICES="$gpu_a" "$PYTHON" \
  src/analyze_verify_then_expand_pca_20260717.py \
  --calibration_trace results/20260715_real_qk_traces_32k/sports.pt \
  --test_trace results/20260715_real_qk_traces_32k/medicine.pt \
  --calibration_topic sports \
  --test_topic medicine \
  --output_dir "$OUTPUT_ROOT/sports_to_medicine" \
  > "$LOG_ROOT/verify_then_expand_sports_to_medicine.log" 2>&1 &
pid_a=$!

CUDA_VISIBLE_DEVICES="$gpu_b" "$PYTHON" \
  src/analyze_verify_then_expand_pca_20260717.py \
  --calibration_trace results/20260715_real_qk_traces_32k/medicine.pt \
  --test_trace results/20260715_real_qk_traces_32k/sports.pt \
  --calibration_topic medicine \
  --test_topic sports \
  --output_dir "$OUTPUT_ROOT/medicine_to_sports" \
  > "$LOG_ROOT/verify_then_expand_medicine_to_sports.log" 2>&1 &
pid_b=$!

wait "$pid_a"

for history_tokens in 4096 32000 64000 128000; do
  CUDA_VISIBLE_DEVICES="$gpu_a" "$PYTHON" \
    src/benchmark_residual_sentinel_pipeline_20260717.py \
    --history_tokens "$history_tokens" \
    --iterations 20 \
    --warmup 5 \
    --output "$OUTPUT_ROOT/speed_${history_tokens}.json" \
    > "$LOG_ROOT/verify_then_expand_speed_${history_tokens}.log" 2>&1
done

wait "$pid_b"
