#!/usr/bin/env bash
set -euo pipefail

EXP=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining/experiments/natural95_synth5_v1
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python

launch() {
  local variant=$1 gpu_list=$2 port=$3 label=$4
  local output="$EXP/outputs/fadeproto10m_${label}"
  if [[ -e "$output/config.json" || -e "$output/DONE" ]]; then
    echo "refusing to reuse $output" >&2; return 1
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu_list" OMP_NUM_THREADS=1 nohup "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=4 --master_port="$port" "$EXP/src/train_mixed_pe.py" \
    --variant "$variant" --output-dir "$output" \
    --train-bin "$EXP/data/train_150m.uint16" \
    --validation-bin "$EXP/data/validation_16m.uint16" \
    --tokenizer-meta "$EXP/data/tokenizer.meta.json" \
    --tokens 10000000 --sequence-length 512 --micro-batch 1 --grad-accum 4 \
    --synthetic-fraction 0.05 --training-queries 32 --answer-weight 16 \
    --learning-rate 3e-4 --warmup-steps 100 --log-every 20 \
    --eval-every 600 --save-every 2000 \
    --eval-lengths 512,1024 --final-eval-lengths 512,1024,2048 \
    --eval-samples 16 --natural-eval-samples 16 \
    > "$output/launcher.log" 2>&1 < /dev/null &
  echo $! > "$output/launcher.pid"
  echo "$label pid=$(cat "$output/launcher.pid") GPUs=$gpu_list"
}

launch native_band8_reference 0,1,2,3 29770 native_reference
launch fade_rope_band8 4,5,6,7 29771 fade_rope

