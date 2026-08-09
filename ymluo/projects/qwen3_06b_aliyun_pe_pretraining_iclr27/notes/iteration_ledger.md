# Iteration ledger

## Iteration 0: package specification

- Conjecture: reducing deep high-frequency remote phase accumulation can improve
  retrieval while preserving local language modeling.
- Operationalization: two direct ablations and two continuous PE candidates,
  all compared against base and matched-token native continued pretraining.
- Profiling plan: training stability, DCLM validation PPL, controlled retrieval,
  official LongBench subset, per-sample outputs, and runtime evidence.
- Current result: no remote training has been run from this package.
- Next uncertainty: which, if any, candidate beats native RoPE at matched tokens
  without violating the PPL guardrail.

## Iteration 1: 16-way from-scratch pretraining protocol

- Goal change: replace the four-condition continued-pretraining screen with 16
  matched Qwen3-0.6B runs initialized from the same random seed.
- Operationalization: 100B processed tokens per task, global batch 256, 8K
  sequences, peak LR `1e-4`, seven token checkpoints, and one eight-GPU machine
  per condition.
- Method decomposition: native/NoPE controls; layer-only interventions;
  high/middle/low frequency interventions; uniform scales; smooth layer,
  layer-frequency, and remote-position functions; period-aware and
  phase-diverse candidates.
- Profiling: TensorBoard receives loss, LR, gradient norm, tokens, progress, PPL,
  controlled retrieval, and LongBench checkpoint metrics. JSONL and result
  bundles retain per-stage evidence.
- Validation result: all 16 assignments pass dry-run contract validation; all
  16 PE implementations pass tiny Qwen3 forward and cached generation; a real
  tiny from-scratch training step writes a resumable checkpoint and TensorBoard
  token-progress event. No 100B result has been run yet.
- Next uncertainty: whether any selective phase function beats native RoPE at
  matched tokens and whether data-file coverage is sufficient to avoid harmful
  repetition over 100B processed tokens.
