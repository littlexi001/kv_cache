#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/lowrank_dot_ppl_downstream_0630}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/run_lowrank_dot_ppl_downstream.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --output_dir "$OUT" \
  --topic_text_dir "${TOPIC_TEXT_DIR:-/home/fdong/ymluo/projects/qabs8cand3reuse_quality_suite/data/topic_texts}" \
  --topics "${TOPICS-finance,history,literature,science,software,mixed_qa}" \
  --ppl_prefill_tokens "${PPL_PREFILL_TOKENS:-2048}" \
  --ppl_eval_tokens "${PPL_EVAL_TOKENS:-128}" \
  --downstream_variants "${DOWNSTREAM_VARIANTS-compact_kv,json_kv,needle_sentence,topic_table}" \
  --downstream_tasks_per_variant "${DOWNSTREAM_TASKS_PER_VARIANT:-16}" \
  --records_per_task "${RECORDS_PER_TASK:-16}" \
  --modes "${MODES:-baseline,lrsvd32attn,lrsvd64attn}" \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --eval_chunk_size "${EVAL_CHUNK_SIZE:-1}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --log_every "${LOG_EVERY:-4}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
