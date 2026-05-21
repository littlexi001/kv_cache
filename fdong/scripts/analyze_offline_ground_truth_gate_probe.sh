#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

python3 fdong/scripts/analyze_offline_ground_truth_gate_probe.py \
  --config_dir fdong/Qwen3-0.6B \
  --checkpoint_root fdong/checkpoints \
  --runs "${RUNS:-inverse-kv-zipf-baseline,ground-truth-zipf-higher-hash,inverse-kv-supervised-gate-zipf-high-hash}" \
  --checkpoint_step "${CHECKPOINT_STEP:-5000}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}" \
  --ground_truth_routing_strategy "${GROUND_TRUTH_ROUTING_STRATEGY:-hash}" \
  --ground_truth_routing_feature_layer "${GROUND_TRUTH_ROUTING_FEATURE_LAYER:-1}" \
  --num_train_samples "${NUM_TRAIN_SAMPLES:-512}" \
  --num_test_samples "${NUM_TEST_SAMPLES:-256}" \
  --probe_epochs "${PROBE_EPOCHS:-120}" \
  --output_path "${OUTPUT_PATH:-fdong/experiments/offline_ground_truth_gate_probe_zipf_high_hash_step5000.json}"
