# Experiments required for a submission-ready paper

This list separates the **verified controlled result** already used in the
draft from the experiments still needed to support broader claims.

## P0 — required before submission

1. **Natural long-context benchmarks.** Evaluate Full, exact post-RoPE Top-2%,
   and SAGE-Post on RULER retrieval/multi-hop subsets and LongBench v2 or an
   equivalent public suite. Report per-task paired confidence intervals, not
   only a macro average.
2. **Cross-model replication.** Repeat the mechanism and method experiments on
   at least one additional RoPE architecture (for example Llama-3.1-8B or a
   native 128K Qwen variant). Separate native-window and position-extended
   settings.
3. **Local-language control.** Measure short-context language-model PPL and
   local order/syntax tasks to test that keeping the local window avoids
   damaging order-sensitive behavior.
4. **Efficient index.** Replace the full pre-RoPE scan with the intended
   approximate index. Report proposal recall, end-to-end latency, throughput,
   peak memory, index memory, and build/update cost at matched batch sizes.
5. **More seeds at 64K.** The current paired NLL interval against Full and
   exact Top-2% crosses zero. Increase sample count under a frozen protocol.

## P1 — strengthens the mechanism claim

1. **Phase intervention at fixed distractor set.** Hold token identities and
   softmax cardinality fixed while changing only effective relative phase
   (position remapping or controlled RoPE rotation).
2. **Denominator-only control.** Keep evidence post-RoPE score fixed and vary
   the number/strength of distractors to isolate softmax competition.
3. **Query/key/content decomposition across seeds.** Quantify the three terms
   in the first-order score change (query drift, key drift, and phase) with
   uncertainty across examples and layers.
4. **Activation patching across examples.** The present patch result is a
   high-resolution causal case study. Repeat on a registered set of successful
   and failed transitions.
5. **Distance sweeps without extension confounds.** Run dense sweeps entirely
   inside each model's native training/evaluation window.

## P2 — method ablations

1. Candidate budget: fixed 0.5/1/2/4% and capped length-dependent budgets.
2. Local window and sink sizes: 0/64/128/256 and 0/4/16.
3. Proposal choices: pre-RoPE only, post-RoPE only, their union, and calibrated
   score mixtures.
4. Per-head gating and confidence: show whether the method can turn itself off
   at short range, where current gains are not uniformly strong.
5. Block retrieval and position repair: test whether retrieving evidence blocks
   while preserving within-block order improves multi-token reasoning.
6. Conflict and semantic distractors: distinguish ordinary filler, plausible
   distractors, and contradictory chains at a matched total length.

## Claims that must remain scoped until these TODOs are complete

- Do not call the phenomenon universal across RoPE models.
- Do not equate the configured context window with the training distribution.
- Do not claim end-to-end speedup while the prototype performs a full pre-RoPE
  scan.
- Do not say that relevance decreases monotonically with distance; the phase
  model predicts oscillation.
- Do not say remote position is useless. The current hypothesis is narrower:
  fine-grained distance should not dominate remote candidate relevance.

