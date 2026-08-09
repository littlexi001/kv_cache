#!/usr/bin/env bash
set -euo pipefail

EXP=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining/experiments/natural95_synth5_v1
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
TRAIN="$EXP/data/train_150m.uint16"
VALIDATION="$EXP/data/validation_16m.uint16"
META="$EXP/data/tokenizer.meta.json"

launch() {
  local variant=$1
  local gpu_list=$2
  local port=$3
  local output="$EXP/outputs/smoke10m_${variant}"
  if [[ -e "$output/config.json" || -e "$output/DONE" ]]; then
    echo "refusing to reuse $output" >&2
    return 1
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu_list" OMP_NUM_THREADS=1 nohup "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 --master_port="$port" \
    "$EXP/src/train_mixed_pe.py" \
    --variant "$variant" --output-dir "$output" \
    --train-bin "$TRAIN" --validation-bin "$VALIDATION" --tokenizer-meta "$META" \
    --tokens 10000000 --sequence-length 2048 --micro-batch 2 --grad-accum 4 \
    --synthetic-fraction 0.05 --training-queries 32 --answer-weight 16 \
    --learning-rate 3e-4 --warmup-steps 50 --log-every 10 \
    --eval-every 150 --save-every 1000 --eval-samples 8 --natural-eval-samples 8 \
    > "$output/launcher.log" 2>&1 < /dev/null &
  echo $! > "$output/launcher.pid"
  echo "$variant pid=$(cat "$output/launcher.pid") GPUs=$gpu_list"
}

launch deep_highfreq_taper 0,1 29750
launch layerwise_slow_rope 2,3 29751

