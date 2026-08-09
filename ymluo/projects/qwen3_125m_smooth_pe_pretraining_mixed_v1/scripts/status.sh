#!/usr/bin/env bash
set -euo pipefail

EXP=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining/experiments/natural95_synth5_v1
for directory in \
  "$EXP"/outputs/smoke10m_* \
  "$EXP"/outputs/pilot100m_* \
  "$EXP"/outputs/fadeproto10m_* \
  "$EXP"/outputs/scale2p5b_*; do
  [[ -d "$directory" ]] || continue
  name=$(basename "$directory")
  if [[ -f "$directory/DONE" ]]; then
    state=DONE
  elif [[ -f "$directory/launcher.pid" ]] && kill -0 "$(cat "$directory/launcher.pid")" 2>/dev/null; then
    state=RUNNING
  else
    state=STOPPED_OR_FAILED
  fi
  last=$(tail -1 "$directory/train.jsonl" 2>/dev/null || true)
  printf '%-42s %s\n' "$name" "$state"
  [[ -n "$last" ]] && printf '  %s\n' "$last"
done
