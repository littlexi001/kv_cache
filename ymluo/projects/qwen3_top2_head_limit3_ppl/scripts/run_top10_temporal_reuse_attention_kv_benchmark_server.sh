#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_attention_kv_section65_20260705_v1}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TOKENIZERS_PARALLELISM=false

python -u src/benchmark_top10_temporal_reuse_attention_kv.py \
  --output_dir "$OUT" \
  --full_kv_len "${FULL_KV_LEN:-4097}" \
  --batch_count "${BATCH_COUNT:-1}" \
  --layer_count "${LAYER_COUNT:-28}" \
  --head_count "${HEAD_COUNT:-16}" \
  --head_dim "${HEAD_DIM:-64}" \
  --top_fraction "${TOP_FRACTION:-0.10}" \
  --sink_tokens "${SINK_TOKENS:-64}" \
  --recent_tokens "${RECENT_TOKENS:-512}" \
  --include_self "${INCLUDE_SELF:-true}" \
  --refresh_intervals "${REFRESH_INTERVALS:-1,16,64}" \
  --steps "${STEPS:-1,16,64,256,1024}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --warmup "${WARMUP:-50}" \
  --repeat "${REPEAT:-200}" \
  --seed "${SEED:-0}" \
  --use_cuda_final_attention "${USE_CUDA_FINAL_ATTENTION:-false}" \
  2>&1 | tee "$OUT/run.log"

echo "output $OUT"
