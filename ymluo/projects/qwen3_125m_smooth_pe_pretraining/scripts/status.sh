#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining
for variant in native deep_highfreq_drop slow_rope smooth_layer_frequency; do
  if [[ -d "$PROJECT/outputs/$variant" ]]; then
    echo "===== $variant ====="
    if [[ -f "$PROJECT/outputs/$variant/DONE" ]]; then
      echo "DONE"
    elif [[ -f "$PROJECT/outputs/$variant/launcher.pid" ]]; then
      pid=$(cat "$PROJECT/outputs/$variant/launcher.pid")
      ps -p "$pid" -o pid=,stat=,etime=,cmd= || true
    fi
    tail -n 2 "$PROJECT/outputs/$variant/train.jsonl" 2>/dev/null || true
    tail -n 4 "$PROJECT/outputs/$variant/launcher.log" 2>/dev/null || true
  fi
done

