#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

SAMPLES="${SAMPLES:-20}"
TASKS="${TASKS:-hotpotqa,musique,qasper}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAMP="${STAMP:-20260709_v92_graph_bridge}"

LABEL="v92_graph_bridge" \
POLICY="configs/riskkv_task_policy_v92_graph_bridge_multihop_20260709.json" \
SAMPLES="$SAMPLES" \
TASKS="$TASKS" \
GPUS="$GPUS" \
STAMP="$STAMP" \
bash scripts/run_riskkv_task_policy_v19_one_20260709.sh
