# Qwen3 KV Tree Retrieval Energy Analysis

This project evaluates a hierarchical K-center tree retrieval method by true
attention-energy coverage. It is a separate phase-A experiment; it does not
measure model loss.

## Candidate Set

For query token `t`, layer `l`, KV head `h`, and attention head `a`, the default
candidate set is:

```text
S_t = prefix_1% union recent_1% union tree_retrieved_middle
```

Percentages are computed from the current visible context length `t + 1`.

- `prefix_1%`: first 1% visible tokens.
- `recent_1%`: most recent 1% visible tokens.
- `middle`: visible tokens excluding prefix and recent.
- `tree_retrieved_middle`: tokens selected from the middle region by a
  hierarchical tree over K-cache vectors.

The tree is built per `(layer, KV head)` from contiguous K blocks:

```text
leaf: continuous LEAF_FRACTION of total tokens
middle node: TREE_FANOUT leaves
big node: TREE_FANOUT middle nodes
root children: big nodes
```

With the default 5000-token run:

```text
LEAF_FRACTION=0.001 -> leaf size = 5 tokens
TREE_FANOUT=10
TREE_BRANCH_COUNTS=5,5,5
```

The query score for a node is:

```text
score(node) = q_t · center(node)
```

where `center(node)` is the mean K vector over the node's valid middle tokens.
The query vector is extracted from the same layer and attention head after
`q_proj`, optional `q_norm`, and RoPE, so it is comparable to the cached K
vectors.

Tree traversal:

```text
root -> top-5 big nodes
each big node -> top-5 middle nodes
each middle node -> top-5 leaf nodes
selected leaves -> all tokens inside those leaves
```

The branch counts are configurable.

## Candidate Granularity

Default:

```text
CANDIDATE_GRANULARITY=attention_head
```

Each attention head uses its own Q vector to retrieve candidates.

Optional:

```text
CANDIDATE_GRANULARITY=kv_head_union
```

For a KV head shared by multiple attention heads, retrieve candidates for each
shared attention head and take the union as one shared candidate set.

For Qwen3-0.6B:

```text
num_attention_heads = 16
num_key_value_heads = 8
kv_head = attention_head // 2
```

## Metrics

For each attention head separately:

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
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_analysis.sh
```

When disabled, `oracle_energy` and `energy_gap_to_oracle` are left blank in the
CSV files, and the plot only shows method energy, prefix/recent energy, and
candidate percentage.

## Run

```bash
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_analysis.sh
```

Focused smoke test:

```bash
MAX_TOKENS=256 CHUNK_SIZE=64 LAYERS=0 KV_HEADS=0 \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_analysis.sh
```

Useful parameters:

```text
BOUNDARY_FRACTION=0.01
LEAF_FRACTION=0.001
LEAF_SIZE=0                 # 0 means derive from LEAF_FRACTION * total tokens
TREE_FANOUT=10
TREE_BRANCH_COUNTS=5,5,5
CANDIDATE_GRANULARITY=attention_head
COMPUTE_ORACLE_BASELINE=true
PLOT_SMOOTHING_WINDOW=500
```

`PLOT_SMOOTHING_WINDOW` controls an extra centered rolling-mean plot. The raw
plot is still saved. For 100k-token runs, `500` or `1000` is usually easier to
read; for 5k-token runs, `50` to `200` is usually enough.

Examples:

```bash
LAYERS=15 KV_HEADS=2 TREE_BRANCH_COUNTS=5,5,5 \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_analysis.sh
```

```bash
LAYERS=15 KV_HEADS=2 CANDIDATE_GRANULARITY=kv_head_union \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_analysis.sh
```

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/outputs/tree_retrieval_energy/
```

Main files:

- `retrieval_energy_by_token.csv`: token-level candidate size and energy rows.
- `summary_by_head.csv`: aggregate metrics per `(layer, KV head, attention head)`.
- `summary.json`: run metadata and output paths.

Per-head plots:

```text
plots/
  layer_00/
    kv_head_00/
      attention_head_00/
        energy_and_candidate_fraction_by_token.png
        energy_and_candidate_fraction_smoothed_w500.png
```

Plot axes:

- x-axis: query token index.
- left y-axis: method energy, oracle energy, and prefix/recent energy.
- right y-axis: candidate set size as a percentage of visible tokens.

## Interpretation

If `method_energy` is close to `oracle_energy`, the tree retrieval captures most
of the important attention mass with the same candidate budget.

If `method_energy` is much higher than `prefix_recent_energy`, the tree-selected
middle tokens contribute useful attention energy beyond the fixed prefix/recent
policy.

If `candidate_fraction` is low while `method_energy` remains high, this tree
structure is a promising candidate for the later loss/PPL masking experiment.
