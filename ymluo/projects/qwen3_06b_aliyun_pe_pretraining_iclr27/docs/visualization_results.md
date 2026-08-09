# Training monitoring and result visualization

## Current state

The 16-condition 100B-token runs have not been executed from this package yet,
so this document defines the monitoring contract and contains no performance
claim. Actual results must be copied here only from saved event files and
evaluation summaries.

## TensorBoard live view

Each task writes to
`$RUN_ROOT/<strategy>/tensorboard/` and starts a TensorBoard server on port 6006.
The first plots to inspect are:

1. `train/loss` versus optimizer step. The x-axis can be converted to tokens by
   multiplying by 2,097,152. A falling finite curve indicates optimization is
   proceeding; it does not show downstream quality.
2. `train/learning_rate` versus step. It should warm to \(10^{-4}\) by step 500
   and then follow the shared cosine schedule.
3. `train/grad_norm` versus step. Isolated spikes require inspection; sustained
   non-finite or rapidly increasing values are a training failure.
4. Throughput/runtime scalars when available. Compare hardware health, not PE
   quality, unless all machines are hardware-matched.
5. `eval/dclm_ppl`, `eval/controlled_qa_f1_percent`, and
   `eval/longbench_qa_f1_percent`. These update after token checkpoints and are
   the first live downstream indicators; compare only matched checkpoint steps.

TensorBoard alone cannot decide the winner because training loss may improve
without long retrieval. Checkpoint evaluation tables remain required.

## Result plot contracts

No result plot should be produced until all fields below are available.

### Plot A: held-out PPL versus processed tokens

- Question: does a PE preserve ordinary language modeling?
- Metric: `exp(mean held-out token NLL)`, unitless; lower is better.
- X-axis: actual processed tokens from `token_schedule.json`.
- Lines: 16 named conditions, with native RoPE visually emphasized.
- Allowed conclusion: relative language-modeling quality at matched tokens.
- Limitation: does not measure retrieval.

### Plot B: long-context score change versus native

- Question: which PE improves retrieval at matched compute?
- Metric: candidate QA F1 minus native QA F1, percentage points.
- X-axis: actual processed tokens; separate panels for controlled and LongBench.
- Zero line: no change from native. Positive is better.
- Allowed conclusion: screening improvement on the measured tasks.
- Limitation: the small LongBench subset is not an official leaderboard result.

### Plot C: PPL–retrieval tradeoff at 100B

- X-axis: PPL percent change versus native; left is better.
- Y-axis: long-context F1 change versus native in percentage points; up is
  better.
- Each point: one condition. Mark the 2% PPL guardrail.
- Allowed conclusion: candidates that improve retrieval inside the guardrail.
- Limitation: one seed does not establish statistical significance.

## Required audit before presenting plots

Verify that manifests and token counts match, units are visible, no evaluation
rows are missing, native is present, axes do not mix units, key numbers appear
beside each plot, and every caption states what the plot does not prove.
