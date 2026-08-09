#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
RAW_REFERENCE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_top16_query480_20260714_v1/per_head_topk.npz"
OUTPUT="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
BM25="$PROJECT/outputs/real_longbench_docqa_10m_record480_bm25_20260714_v1"
LOG_ROOT="$PROJECT/outputs/logs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6,7}"
WORLD_SIZE="${WORLD_SIZE:-7}"

mkdir -p "$OUTPUT" "$LOG_ROOT"
cd "$PROJECT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$WORLD_SIZE" \
  src/run_all_head_prior_debiased_retrieval.py \
    --corpus_dir "$CORPUS" \
    --profile_dir "$PROFILE" \
    --query_profiles "$QUERY_PROFILE" \
    --output_dir "$OUTPUT" \
    --top_per_head 16 \
    --query_batch 8 \
    --block_chunk 64 \
    --exclude_block_prefix_tokens 16 \
    --folds 6 \
    --fold_strategy dataset_leave_one_out \
    --seed 20260714 \
    --std_epsilon 1e-4 \
  >"$LOG_ROOT/retrieval.log" 2>&1

"$PYTHON" - "$RAW_REFERENCE" "$OUTPUT/raw/per_head_topk.npz" \
  >"$LOG_ROOT/raw_identity.log" 2>&1 <<'PY'
import json
import sys

import numpy as np

reference = np.load(sys.argv[1])
candidate = np.load(sys.argv[2])
id_mismatch = reference["block_ids"] != candidate["block_ids"]
result = {
    "scores_exact": bool(np.array_equal(reference["scores"], candidate["scores"])),
    "score_max_abs_error": float(
        np.max(np.abs(reference["scores"].astype(np.float64) - candidate["scores"]))
    ),
    "id_mismatch_slots": int(id_mismatch.sum()),
    "id_mismatch_fraction": float(id_mismatch.mean()),
    "all_id_mismatches_have_equal_rank_score": bool(
        np.all(reference["scores"][id_mismatch] == candidate["scores"][id_mismatch])
    ),
}
print(json.dumps(result, indent=2))
if not result["scores_exact"] or not result["all_id_mismatches_have_equal_rank_score"]:
    raise SystemExit("raw scores do not reproduce the frozen reference")
PY

"$PYTHON" src/analyze_dataset_lodo_head_responsiveness.py \
  --raw_topk_npz "$OUTPUT/raw/per_head_topk.npz" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --output_dir "$OUTPUT/dataset_lodo_head_responsiveness" \
  --head_sizes 1,4,16,64 \
  --depth 16 \
  --target_blocks 39 \
  --num_blocks 39062 \
  --block_tokens 256 \
  --random_subsets_per_fold 200 \
  --seed 20260714 \
  >"$LOG_ROOT/dataset_lodo_head_responsiveness.log" 2>&1

for method in raw centered zscore; do
  "$PYTHON" src/analyze_head_sparse_decomposition.py \
    --topk_npz "$OUTPUT/$method/per_head_topk.npz" \
    --queries_jsonl "$CORPUS/queries.jsonl" \
    --output_dir "$OUTPUT/property_$method" \
    --num_blocks 39062 \
    --depths 1,2,4,8,16 \
    --cv_splits 500 \
    --random_subsets_per_split 20 \
    --overlap_samples 500000 \
    --seed 20260714 \
    >"$LOG_ROOT/property_$method.log" 2>&1
done

"$PYTHON" src/summarize_head_prior_debiasing.py \
  --retrieval_dir "$OUTPUT" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --output_dir "$OUTPUT/comparison" \
  --num_blocks 39062 \
  --target_blocks 39 \
  --depths 1,2,4,8,16 \
  --bootstrap_samples 20000 \
  --seed 20260714 \
  >"$LOG_ROOT/comparison.log" 2>&1

"$PYTHON" src/analyze_unsupervised_head_gate.py \
  --raw_topk_npz "$OUTPUT/raw/per_head_topk.npz" \
  --candidate_topk_npz "$OUTPUT/zscore/per_head_topk.npz" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --output_dir "$OUTPUT/unsupervised_head_gate" \
  --head_sizes 1,2,4,8,16,32,64 \
  --depths 8,16 \
  --target_blocks 39 \
  --block_tokens 256 \
  --random_subsets_per_fold 200 \
  --seed 20260714 \
  >"$LOG_ROOT/unsupervised_head_gate.log" 2>&1

"$PYTHON" src/compare_head_gate_to_bm25.py \
  --raw_topk_npz "$OUTPUT/raw/per_head_topk.npz" \
  --candidate_topk_npz "$OUTPUT/zscore/per_head_topk.npz" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --bm25_query_results "$BM25/query_results.csv" \
  --output_dir "$OUTPUT/matched_bm25_comparison" \
  --gate_feature raw_top1_block_diversity \
  --heads 16 \
  --depth 16 \
  --target_blocks 39 \
  --bootstrap_samples 20000 \
  --seed 20260714 \
  >"$LOG_ROOT/matched_bm25_comparison.log" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
