# Qwen3 KV Retrieval Energy Analysis

This project implements phase A of the KV retrieval experiment: evaluate how
much true attention energy is covered by a query-time candidate set built from
K-cache L2 neighbor graphs.

No model loss is measured here. This project only measures attention-energy
coverage.

## Candidate Set

For query token `t`, layer `l`, and KV head `h`, build one candidate set shared
by the attention heads mapped to that KV head.

For Qwen3-0.6B:

```text
num_attention_heads = 16
num_key_value_heads = 8
kv_head = attention_head // 2
```

The candidate set is:

```text
S_t = prefix_1% union recent_1% union expanded_middle_seeds
```

where percentages are computed from the current visible context length `t + 1`.

Definitions:

- `prefix_1%`: first 1% visible tokens.
- `recent_1%`: most recent 1% visible tokens.
- `middle`: visible tokens excluding prefix and recent.
- seeds: for each attention head sharing the KV head, use the previous query
  token `t-1` attention distribution and select the highest-attention 1% tokens
  from `middle`.
- expanded seeds: each seed plus its previous-only top-20 K-L2 neighbors from
  the same `(layer, KV head)`.

All candidates are deduplicated and constrained to `j <= t`.

## Metrics

For each attention head separately, compute:

```text
method_energy = sum_{j in S_t} attention_t(j)
```

Oracle baseline:

```text
oracle_energy = sum of top-|S_t| current attention weights over all visible tokens
```

Prefix/recent baseline:

```text
prefix_recent_energy = sum attention over prefix_1% union recent_1%
```

The oracle baseline uses the same candidate count as the method, but it is not
online because it sorts the current query's true attention.

If this sort is too slow, disable it:

```bash
COMPUTE_ORACLE_BASELINE=false \
bash ymluo/projects/qwen3_kv_retrieval_energy_analysis/scripts/run_analysis.sh
```

When disabled, `oracle_energy` and `energy_gap_to_oracle` are left blank in the
CSV files, and the plot only shows method energy, prefix/recent energy, and
candidate percentage.

## Run

```bash
bash ymluo/projects/qwen3_kv_retrieval_energy_analysis/scripts/run_analysis.sh
```

Useful focused smoke test:

```bash
MAX_TOKENS=256 CHUNK_SIZE=64 LAYERS=0 KV_HEADS=0 \
bash ymluo/projects/qwen3_kv_retrieval_energy_analysis/scripts/run_analysis.sh
```

Default parameters:

```text
BOUNDARY_FRACTION=0.01
SEED_FRACTION=0.01
NEIGHBOR_COUNT=20
COMPUTE_ORACLE_BASELINE=true
PLOT_SMOOTHING_WINDOW=500
```

`PLOT_SMOOTHING_WINDOW` controls an extra centered rolling-mean plot. The raw
plot is still saved. For 100k-token runs, `500` or `1000` is usually easier to
read; for 5k-token runs, `50` to `200` is usually enough.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_kv_retrieval_energy_analysis/outputs/retrieval_energy/
```

Main files:

- `retrieval_energy_by_token.csv`: token-level candidate size and energy
  coverage rows.
- `summary_by_head.csv`: aggregate metrics per `(layer, KV head, attention
  head)`.
- `summary.json`: run metadata and output paths.

Per-head plots:

```text
plots/
  layer_00/
    kv_head_00/
      attention_head_00/
        energy_and_candidate_fraction_by_token.png
        energy_and_candidate_fraction_smoothed_w500.png
      attention_head_01/
        energy_and_candidate_fraction_by_token.png
        energy_and_candidate_fraction_smoothed_w500.png
```

Plot axes:

- x-axis: query token index.
- left y-axis: method energy, oracle energy, and prefix/recent energy.
- right y-axis: candidate set size as a percentage of visible tokens.

## Interpretation

If `method_energy` is close to `oracle_energy`, the K-L2 graph retrieval method
captures the important attention mass with nearly the same budget as the oracle.

If `method_energy` is much higher than `prefix_recent_energy`, the expanded
middle seeds are useful beyond the fixed prefix/recent policy.

If candidate fraction is small but energy remains high, the retrieval method is
promising for later loss/PPL experiments.
