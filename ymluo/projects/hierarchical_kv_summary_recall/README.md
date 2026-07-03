# Hierarchical KV Summary Recall Smoke Test

This is a small synthetic experiment for the idea:

```text
for each 10k-token KV block, keep:
  10-token summary
  100-token summary
  1000-token summary
  original raw KV tokens

at decode time, route each query to the cheapest level that can answer it,
and drill down to raw tokens only when the query requires exact details.
```

The experiment is intentionally lightweight and does not require GPU,
Transformers, or real model inference. It validates the information-routing
condition before wiring the idea into a real Qwen KV patch.

## Synthetic Setup

Each block represents 10k original tokens. The block has three compressed
summary levels:

- `tiny10`: block label, dominant theme, domain.
- `small100`: block label, policy-level facts, sample projects.
- `medium1000`: compact records for all keys, with score/color/action, but no
  exact verification code.
- `raw`: original token window for an exact record, including the rare code.

Queries are mixed across five granularities:

- `theme`: answerable from `tiny10`.
- `policy`: answerable from `small100`.
- `score`: answerable from `medium1000`.
- `color`: answerable from `medium1000`.
- `code`: requires raw-token drill-down.

## Methods

- `full_raw_scan`: upper-bound baseline that keeps all original tokens visible.
- `tiny10_only`, `small100_only`, `medium1000_only`: fixed-level baselines.
- `hier_narrow_top1`: cheap but risky hierarchical routing; it only follows the
  top block from the 10-token digest.
- `hier_adaptive`: query-aware routing; it stops early for coarse questions and
  broadens summary search before raw drill-down for rare-key questions.

## Run

```bash
python ymluo/projects/hierarchical_kv_summary_recall/src/run_hierarchical_kv_summary_recall.py
```

Outputs:

```text
ymluo/projects/hierarchical_kv_summary_recall/outputs/default/
  summary.csv
  summary.json
  by_kind.csv
  trials.csv
  examples.jsonl
```

## Interpretation

A positive result is:

- `hier_adaptive` matches `full_raw_scan` accuracy on the synthetic tasks.
- `hier_adaptive` uses much lower average token cost.
- fixed coarse summaries are cheap but lose fine/exact queries.
- `hier_narrow_top1` shows the failure mode: if a tiny summary does not preserve
  enough routing information, rare-key queries need broader summary search or a
  better learned router.

This does not prove real KV summaries will work. It only checks the minimum
claim that mixed query granularities can benefit from hierarchical memory when
the summary hierarchy preserves the right information.
