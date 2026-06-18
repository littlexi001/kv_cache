# Experiment Design

## Purpose

Compare original attention, top-2% historical-token attention, and top-2%
historical-token attention with a maximum of 3 heads per historical token.

## Conditions

- `baseline`: full original attention.
- `top2`: each head keeps only its top 2% historical tokens. The query token's
  own key/value is kept.
- `top2limit3`: same as `top2`, but a historical token can be kept by at most
  3 heads in the same layer and query.

## Metrics

- `loss`: mean next-token cross entropy on evaluation tokens.
- `ppl`: `exp(loss)`.
- `final_kept_per_query_mean`: mean number of historical tokens kept by a
  layer/head after the limit3 rule.
- `kept_fraction_of_original_top2`: final kept historical links divided by
  original top2 historical links.
- `removed_per_query_mean`: mean number of top2 historical links removed from a
  layer/head by the limit3 rule.

## Evidence Plan

Stage 1: smoke test.

- Run a short prefill/eval setting.
- Pass: all three modes finish and produce finite PPL.

Stage 2: main run.

- Run the default setting in `README.md`.
- Pass: `ppl_by_mode.csv`, `top2limit3_load_by_head.csv`, and all plots exist.

Stage 3: interpretation.

- If `top2limit3` PPL is close to `top2`, the max-3 rule mainly removes
  duplicate links without large extra PPL cost.
- If `top2limit3` PPL is much worse than `top2`, the removed duplicate
  head-token links carry useful information.
- If head loads vary strongly, the random max-3 rule creates head imbalance.

## Profiling Evidence

`ppl_by_mode.csv` records runtime seconds per mode. This separates model/PPL
quality from implementation runtime.

## Remaining Uncertainty

The rule chooses 3 heads randomly. A deterministic choice based on score,
head group, or previous-layer load may produce a different PPL and load
distribution.

## Sink/Recent Protection Follow-up

Purpose: test whether the PPL collapse comes from deleting attention sink tokens
or recent local-context tokens across heads.

Additional condition:

- `top2limit3protectsSrR`: after top2 selection, protect the first `S`
  historical positions and the most recent `R%` historical positions. Protected
  tokens keep all selected heads. Unprotected tokens keep only the 3 selected
  heads with largest pre-softmax score.

Follow-up sweeps:

- Sink/recent ablation: `S = 0, 4, 16, 32`, `R = 0% or 1%`.
- Wider sink/recent sweep: `S = 64, 128, 256`, `R = 1%, 2%, 4%, 8%, 16%`.
- Threshold sweep around the useful region: `S = 32, 64, 128`, `R = 10%, 12%,
  14%, 16%`.

Pass condition:

- A protect rule is considered non-collapsing if PPL is within `1.05x` of top2
  on the same evaluation window.

Failure condition:

- If PPL remains orders of magnitude above top2, protecting only sink/recent
  tokens is insufficient and the deleted non-protected duplicate links still
  contain important information.

Current outcome:

- `top2limit3protects64r16p0` passes: PPL `36.462`, ratio to top2 `1.029`.
- Protecting only recent `1%` fails: PPL remains above `10,000` without sink and
  remains in the thousands with small sink protection.

## Head-Count Position Distribution Diagnostic

Purpose: measure whether tokens selected by many heads are located near the
query, near the sequence start, or both.

Procedure:

1. Run full attention with `output_attentions=true`.
2. For every layer and query, reconstruct each head's top2 historical-token
   set from the attention ranking.
3. For every historical token selected by at least one head, count how many
   heads selected it.
4. Group token cases by selected-head count `1..16`.
5. Save relative-position histograms, quantiles, sink/recent fractions, and
   example rows.

Pass condition:

- The summary contains all selected-head counts `1..16`, the total case count
  is positive, and each plot states the position metric and unit.

Interpretation condition:

- If high selected-head counts concentrate near recent positions, it supports
  protecting broad recent context.
- If the all-head group concentrates near position zero, it supports treating
  sink tokens separately from recent tokens.
