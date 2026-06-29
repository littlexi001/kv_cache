#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL=/home/fdong/hrj/prove/Qwen3-0.6B
GEN=/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/src/build_hard_topic_eval_text.py

python "$GEN" --output /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v3.txt --paragraphs 420 --seed 202606292
python "$GEN" --output /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v4.txt --paragraphs 420 --seed 202606293

run_ppl() {
  local name="$1"
  local text_path="$2"
  local out="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/broad_ppl_${name}_qabs8cand5_tf5_p2048_e256"
  rm -rf "$out"
  echo "RUN ${name}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u src/evaluate_qwen3_top2_head_limit3_ppl.py \
      --model_name_or_path "$MODEL" \
      --text_path "$text_path" \
      --output_dir "$out" \
      --modes qabs8cand5reuse,baseline \
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
      --top_fraction 0.05 \
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
}

run_ppl hard_v2 /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt
run_ppl hard_v3 /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v3.txt
run_ppl hard_v4 /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v4.txt
run_ppl topic_stress /home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/topic_stress_eval.txt
run_ppl war /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt
run_ppl monte /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt
