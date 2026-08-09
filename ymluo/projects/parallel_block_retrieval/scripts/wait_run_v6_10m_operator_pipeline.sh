#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
CORPUS="${ROOT}/data/real10m_controlled_v6_mix_seed20260715"
STEPS="${ROOT}/data/real10m_controlled_v6_mix_seed20260715_step_labels_v1/step_queries.jsonl"
SIDECAR="${ROOT}/outputs/real10m_controlled_v6_sentence_sidecar_v1"
ANCHOR="${ROOT}/outputs/real10m_controlled_v6_anchor_goldstate_devtest_v1"
PROFILE="${ROOT}/outputs/real10m_controlled_v6_operator4_qk64_profile_v1"
GLOBAL="${ROOT}/outputs/real10m_controlled_v6_operator4_global_step_devtest_v1"
SENTENCE="${ROOT}/outputs/real10m_controlled_v6_operator4_anchor_sentence_devtest_v1"
GENERATION="${ROOT}/outputs/real10m_controlled_v6_step_branch6_test_v1"

cd "$ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) print $1}'
  )
  if ((${#free_gpus[@]} >= 3)); then
    free_gpus=("${free_gpus[@]:0:3}")
    break
  fi
  echo "$(date -Is) waiting for 3 idle GPUs; found ${#free_gpus[@]}" >&2
  sleep 60
done

gpu_list="$(IFS=,; echo "${free_gpus[*]}")"
echo "$(date -Is) using idle GPUs: $gpu_list"

if [[ ! -f "$PROFILE/summary.json" ]]; then
  rm -rf "$PROFILE"
  CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_DIR/torchrun" --standalone --nproc_per_node=3 \
    src/profile_real_qk.py \
    --corpus_dir "$CORPUS" \
    --profile_dir "$PROFILE" \
    --pairs 3:10,21:9,6:13,6:12 \
    --svd_rank 64 \
    --calibration_blocks 32 \
    --query_vector_tokens 16 \
    --query_vector_mode question_content \
    --profile_space pre_rope_record_qk \
    --skip_query_profiles \
    --dtype float16 \
    --log_every 50
fi

rm -rf "$GLOBAL"
CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_DIR/torchrun" --standalone --nproc_per_node=3 \
  src/run_global_step_block_retrieval.py \
  --corpus_dir "$CORPUS" \
  --profile_dir "$PROFILE" \
  --step_queries_path "$STEPS" \
  --output_dir "$GLOBAL" \
  --splits dev,test \
  --task_types multihop \
  --exclude_query_ids 375 \
  --svd_rank 32 \
  --candidate_blocks 512 \
  --target_blocks 16 \
  --resolve_bridge_profiles 0,1 \
  --resolve_answer_profiles 2,3

until [[ -f "$SIDECAR/summary.json" ]]; do
  echo "$(date -Is) waiting for v6 sentence sidecar" >&2
  sleep 30
done

rm -rf "$SENTENCE"
CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_DIR/torchrun" --standalone --nproc_per_node=3 \
  src/run_global_candidate_sentence_kv_rerank.py \
  --corpus_dir "$CORPUS" \
  --profile_dir "$PROFILE" \
  --sidecar_dir "$SIDECAR" \
  --step_queries_path "$STEPS" \
  --candidate_rows_path "$ANCHOR/rows.jsonl" \
  --candidate_field anchor_candidates \
  --output_dir "$SENTENCE" \
  --splits dev,test \
  --task_types multihop \
  --exclude_query_ids 375 \
  --query_tokens 16 \
  --score_chunk 32 \
  --resolve_bridge_profiles 0,1 \
  --resolve_answer_profiles 2,3 \
  --branch_blocks 3 \
  --spans_per_block 3

rm -rf "$GENERATION"
CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_DIR/torchrun" --standalone --nproc_per_node=3 \
  src/evaluate_global_step_branch_generation.py \
  --corpus_dir "$CORPUS" \
  --step_queries_path "$STEPS" \
  --retrieval_rows_path "$SENTENCE/rows.jsonl" \
  --output_dir "$GENERATION" \
  --split test \
  --exclude_query_ids 375 \
  --max_new_tokens 24 \
  --max_retrieval_branches 6

"$PYTHON_DIR/python" src/analyze_branch_transition_verifier.py \
  --rows_path "$GENERATION/rows.jsonl" \
  --step_queries_path "$STEPS" \
  --output_path "$GENERATION/verifier.json"

echo "$(date -Is) v6 operator pipeline complete: $GENERATION"
