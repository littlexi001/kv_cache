# Head-Routed Heterogeneous KV Retrieval

This project tests a precise hypothesis:

> Different attention heads need different kinds of evidence, so an external
> retriever should be selected per head instead of asking every head to use one
> homogeneous token-selection rule.

The first pilot treats per-layer, per-query-head oracle attention Top-2% as the
teacher.  It compares strictly equal-budget retrievers that never read QK
attention scores:

- `position`: attention sinks plus the most recent tokens;
- `lexical`: block retrieval from exact token overlap and inverse block frequency;
- `semantic`: block retrieval from Qwen input-embedding mean similarity;
- `format`: block retrieval from punctuation/newline/digit/word-shape features;
- `repeat`: repeated suffix / induction-style token retrieval;
- `hybrid_*`: the same specialist retriever with a fixed 50% position scaffold;
- `random`: deterministic calibration baseline.

Every policy still returns exactly Top-2% history tokens.  The hybrid scaffold is
an internal split of the same budget, not an extra budget and not a different
budget for different heads.

The first half of evaluation queries is used only to assign a retriever to each
layer/query-head.  The second half is held out for reporting.  This is a pilot
split within one document; final evidence must split by document and task.

The pilot reports both query-head selection quality and the physical GQA cost.
Qwen3-0.6B has 16 query heads and 8 KV heads, so the KV set loaded for a group is
the union of the two assigned query-head sets.

## Remote pilot

```bash
bash ymluo/projects/qwen3_head_routed_retrieval/scripts/run_pilot_server.sh
```

Important outputs:

- `head_retriever_metrics.csv`: train/test metrics for every layer, head, retriever;
- `head_assignments.csv`: retriever learned on train queries and held-out result;
- `aggregate_retriever_metrics.csv`: homogeneous versus head-routed comparison;
- `gqa_union_by_layer_group.csv`: actual GQA KV-union expansion;
- `summary.json`: machine-readable headline results and exact configuration.

The pilot measures imitation, not yet behavioral equivalence.  A positive pilot
must next be followed by masked-attention NLL/PPL evaluation using the retrieved
sets.

The first War-and-Peace 4K remote pilot is summarized in
[`docs/pilot_war4k_results_20260714.md`](docs/pilot_war4k_results_20260714.md).
The result finds a stable repeat/induction-head subgroup, but only a small global
recall gain because the current semantic/lexical/format retrievers miss most
remote oracle positions.
