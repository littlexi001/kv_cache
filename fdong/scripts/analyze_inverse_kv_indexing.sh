#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

python3 fdong/scripts/analyze_inverse_kv_indexing.py \
  --config_dir fdong/Qwen3-0.6B \
  --checkpoint_root fdong/checkpoints \
  --runs "${RUNS:-inverse-kv-supervised-gate-zipf-high-hash,inverse-kv-mlp-gate-hidden-supervised-zipf-high-hash,inverse-kv-mlp-gate-attention_output-supervised-zipf-high-hash}" \
  --checkpoint_step "${CHECKPOINT_STEP:-5000}" \
  --top_m_values "${TOP_M_VALUES:-1,2,3,4}" \
  --num_samples "${NUM_SAMPLES:-256}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}" \
  --output_path "${OUTPUT_PATH:-fdong/experiments/inverse_kv_indexing_step5000.json}"
