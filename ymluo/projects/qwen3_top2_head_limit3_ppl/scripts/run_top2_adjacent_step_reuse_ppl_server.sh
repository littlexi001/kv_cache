#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_reuse_ppl_war_4k_eval512_20260703}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/evaluate_qwen3_top2_head_limit3_ppl.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --text_path "${TEXT_PATH:-data/war_and_peace_pg2600.txt}" \
  --output_dir "$OUT" \
  --prefill_tokens "${PREFILL_TOKENS:-4096}" \
  --eval_tokens "${EVAL_TOKENS:-512}" \
  --chunk_size "${CHUNK_SIZE:-64}" \
  --eval_chunk_size "${EVAL_CHUNK_SIZE:-64}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens false \
  --append_eos false \
  --require_total_tokens true \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --protect_sink_tokens "${PROTECT_SINK_TOKENS:-64}" \
  --protect_recent_tokens "${PROTECT_RECENT_TOKENS:-512}" \
  --always_keep_self true \
  --modes "${MODES:-adjtop2rs2attn,adjtop2rs3attn,adjtop2rs4attn,top2,baseline}" \
  --adjacent_top2_whitelist_path "${ADJACENT_TOP2_WHITELIST_PATH:-outputs/top2_adjacent_step_position_sharing_remote_war_4k_s64_r512_20260703_v1/layer_head_adjacent_step_stats.csv}" \
  --qabs_cuda_final_kernel "${QABS_CUDA_FINAL_KERNEL:-false}" \
  --qabs_cuda_candidate_kernel "${QABS_CUDA_CANDIDATE_KERNEL:-false}" \
  --qabs_cuda_reuse_select_kernel "${QABS_CUDA_REUSE_SELECT_KERNEL:-false}" \
  --reuse_prefill_cache "${REUSE_PREFILL_CACHE:-true}" \
  --baseline_last "${BASELINE_LAST:-true}" \
  --disable_sparse_stats "${DISABLE_SPARSE_STATS:-true}" \
  --log_every "${LOG_EVERY:-1}" \
  --make_plots "${MAKE_PLOTS:-false}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
