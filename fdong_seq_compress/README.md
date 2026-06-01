# Qwen3 KV Cache Geometry Diagnostics

This folder contains a minimal first-pass experiment for studying how one fixed
sequence's KV-cache geometry changes as the prefix length grows.

The first version intentionally avoids indexing, pruning, training, and
attention/loss-based selection. It treats each layer/head KV cache as a growing
high-dimensional point cloud:

```text
K_t = [k_1, ..., k_t]
V_t = [v_1, ..., v_t]
```

and measures whether this point cloud expands, stabilizes, drifts, clusters, or
stays low-dimensional as `t` increases.

## Files

```text
src/model_loader.py          Load tokenizer and Qwen3 model.
src/text_loader.py           Read one text file and tokenize one sequence.
src/geometry_metrics.py      SVD, anisotropy, novelty, temporal, and block metrics.
src/run_prefix_geometry.py   Main prefix-sweep experiment entry.
src/run_k_similarity_graph_probe.py
                            Causal top-k K-cache similarity distribution probe.
scripts/run_prefix_geometry.sh
scripts/run_k_similarity_graph_probe.sh
data/synthetic_texts/long_english_article_01.txt
```

## Quick Smoke Test

```bash
MAX_TOKENS=256 \
PREFIX_LENGTHS=64,128,256 \
LAYERS=0,1 \
HEADS=0,1 \
bash fdong_seq_compress/scripts/run_prefix_geometry.sh
```

## Default Run

```bash
bash fdong_seq_compress/scripts/run_prefix_geometry.sh
```

Default model path:

```text
fdong/Qwen3-0.6B
```

Default outputs:

```text
fdong_seq_compress/outputs/prefix_geometry_<timestamp>/
```

Main output files:

```text
metrics_by_prefix_layer_head.csv
block_metrics.csv
singular_values.csv
tokens.csv
summary.json
```

## Core Metrics

- SVD spectrum and cumulative energy ranks.
- Effective rank and stable rank.
- Centered and uncentered cosine/anisotropy statistics.
- Adjacent-token delta norm and cosine.
- Incremental novelty against the previous prefix's top-r subspace.
- Top-r subspace overlap against the previous prefix.
- Multi-scale block within/between variance ratio.

## K Similarity Graph Probe

This is the first sanity check for a K-cache graph-index idea. For each selected
layer, the default mode concatenates all KV heads into one token-level K vector:

```text
k_i(layer) = concat_h k_i(layer, h)
```

It then computes causal nearest neighbors:

```text
top-k similarity(k_i, k_j), j < i
```

so self-similarity and future tokens are excluded.

Default first experiment:

```bash
bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```

Default settings:

```text
MAX_TOKENS=1000
TOP_K=5
SIMILARITY=cos
CENTER_TOKENS=1
ANALYSIS_LEVEL=token
```

Supported K-neighbor metrics:

```text
SIMILARITY=cos   Higher is closer; uses cosine on centered/normalized K.
SIMILARITY=dot   Higher is closer; diagnostic only because K norm can create hubs.
SIMILARITY=l2    Lower is closer; directly bounds qK score differences.
```

For `l2`, the `summary_by_layer.csv` columns still use the historical
`similarity` field name, but the values are L2 distances, so smaller values are
better/closer. The script also records:

```text
model_max_position_embeddings
seq_len_within_model_max_position_embeddings
```

and fails by default if the tokenized sequence is longer than the model's
configured position range, because position/RoPE handling could otherwise
dominate the geometry.

Main output files:

```text
summary_by_layer.csv     Per-layer top-k distribution statistics.
histograms.csv           Per-layer histogram buckets.
histogram_global.csv     Histogram buckets aggregated across selected layers.
histogram_global.svg     Dependency-free SVG plot of the global distribution.
distance_histogram_global.svg
                         Global top-k edge token-distance distribution.
indegree_histogram_global.svg
                         Global top-k graph in-degree distribution.
plots/*.svg              Per-layer or per-head histogram plots.
tokens.csv               Token index/id/text audit.
summary.json             Run configuration and global summary.
```

Useful variants:

```bash
CENTER_TOKENS=1 bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
SIMILARITY=dot bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
SIMILARITY=l2 bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
ANALYSIS_LEVEL=head SAVE_NEIGHBORS=1 bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```
