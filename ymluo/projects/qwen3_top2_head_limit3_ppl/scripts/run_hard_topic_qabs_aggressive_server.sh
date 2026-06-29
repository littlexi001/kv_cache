#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

DATA=/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
MODEL=/home/fdong/hrj/prove/Qwen3-0.6B

run_case() {
  local pct="$1"
  local top="$2"
  local mode="$3"
  local out="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hard_topic_v2_${mode}_tf${pct}_p2048_e256"
  rm -rf "$out"
  echo "RUN ${mode} top_fraction=${top} output=${out}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u src/evaluate_qwen3_top2_head_limit3_ppl.py \
      --model_name_or_path "$MODEL" \
      --text_path "$DATA" \
      --output_dir "$out" \
      --modes "${mode},baseline" \
      --prefill_tokens 2048 \
      --eval_tokens 256 \
      --chunk_size 8 \
      --eval_chunk_size 1 \
      --max_chars 80000000 \
      --add_special_tokens false \
      --append_eos false \
      --require_total_tokens true \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --attn_implementation eager \
      --top_fraction "$top" \
      --protect_sink_tokens 10 \
      --protect_recent_tokens 10 \
      --always_keep_self true \
      --qabs_fast_path true \
      --qabs_cuda_final_kernel true \
      --qabs_cuda_candidate_kernel true \
      --qabs_cuda_reuse_select_kernel false \
      --reuse_prefill_cache true \
      --baseline_last true \
      --disable_sparse_stats true \
      --log_every 128 \
      --make_plots false
  echo "DONE ${mode}"
}

run_case 2 0.02 qabs8cand2reuse
run_case 3 0.03 qabs8cand3reuse
run_case 5 0.05 qabs8cand5reuse
