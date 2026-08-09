#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
mkdir -p "$PROJECT/outputs/native_v2" "$PROJECT/outputs/smooth_layer_frequency_v2"

CUDA_VISIBLE_DEVICES=0,1 nohup "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29632 \
  "$PROJECT/src/train_synthetic_pe.py" \
  --variant native \
  --output-dir "$PROJECT/outputs/native_v2" \
  --tokens 20000000 --sequence-length 2048 --micro-batch 2 --grad-accum 4 \
  > "$PROJECT/outputs/native_v2/launcher.log" 2>&1 < /dev/null &
echo $! > "$PROJECT/outputs/native_v2/launcher.pid"

CUDA_VISIBLE_DEVICES=2,3 nohup "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29633 \
  "$PROJECT/src/train_synthetic_pe.py" \
  --variant smooth_layer_frequency \
  --output-dir "$PROJECT/outputs/smooth_layer_frequency_v2" \
  --tokens 20000000 --sequence-length 2048 --micro-batch 2 --grad-accum 4 \
  > "$PROJECT/outputs/smooth_layer_frequency_v2/launcher.log" 2>&1 < /dev/null &
echo $! > "$PROJECT/outputs/smooth_layer_frequency_v2/launcher.pid"

echo "server31 jobs launched"
