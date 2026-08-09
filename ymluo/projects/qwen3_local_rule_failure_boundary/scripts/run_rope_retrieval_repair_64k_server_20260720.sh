#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/rope_retrieval_repair_qwen3_8b_64k_20260720}"

mkdir -p "$OUT/logs"
rm -f "$OUT/done.txt" "$OUT/launcher.done" "$OUT/launcher.failed"

failure() {
  date -Is >"$OUT/launcher.failed"
}
trap failure ERR

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
"$PY" -u "$PROJECT/src/run_rope_retrieval_repair_8b.py" \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --target_context_tokens 64000 \
  --placement prefix \
  --prompt_style chat_concise \
  --seed_start "${SEED_START:-0}" \
  --num_seeds "${NUM_SEEDS:-16}" \
  --ratio "${RATIO:-0.02}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
  --prefill_chunk_size 128 \
  --dtype float16 \
  --device_map balanced \
  --attn_implementation sdpa \
  --original_max_position_embeddings 40960 \
  --global_max_position 130000 \
  >"$OUT/logs/run.log" 2>&1

date -Is >"$OUT/launcher.done"
echo "completed: $OUT/summary.json"
