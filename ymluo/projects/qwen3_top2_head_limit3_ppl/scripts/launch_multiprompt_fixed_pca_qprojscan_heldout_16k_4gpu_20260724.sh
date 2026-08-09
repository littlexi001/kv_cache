#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_multiprompt_fixed_pca_qprojscan_heldout_16k_4gpu
SCORE_MODE=pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan
CALIBRATION=gov_report:114,narrativeqa:114,qasper:114,repobench-p:114

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

for gpu in 0 1 2 3; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU $gpu is busy" >&2
    exit 1
  fi
done

pids=()
for spec in "0:hotpotqa" "1:musique" "2:lcc" "3:multi_news"; do
  gpu=${spec%%:*}
  task=${spec##*:}
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
      src/run_fixed_pca_basis_cross_prompt_probe_20260724.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_path "$RUN_ROOT/${task}.json" \
      --calibration_specs "$CALIBRATION" \
      --calibration_max_new_tokens 1 \
      --task "$task" \
      --test_offset 115 \
      --max_prompt_tokens 16000 \
      --max_new_tokens 64 \
      --repeats 2 \
      --prefill_chunk_tokens 2048 \
      --score_mode "$SCORE_MODE" \
      --dtype float16 --device cuda --device_map auto \
      > "$RUN_ROOT/logs/${task}.log" 2>&1
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
touch "$RUN_ROOT/ALL_COMPLETE"
