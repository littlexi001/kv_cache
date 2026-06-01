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

## Document Structure

The clean current-state framework lives outside this experiment folder:

```text
main_seq_compress/project_overview.md
  Current-state research framework: conjecture, priors, math models,
  implementation contracts, evidence boundary, and next falsification tests.
```

This folder keeps the iteration ledgers and runnable experiment code.

The root documents are intentionally reduced to round-level research notes:

```text
k_cache_graph_round1_findings.md
  Consolidated Round1 understanding: KV geometry, K/V role split, centered K
  graph evidence, common-direction analysis, and remaining gaps.

k_cache_graph_round2_handoff.md
  Round2 findings: L2 metric sweep, seq-len scaling, layer/head selection,
  domain robustness, and the next query-attention recall gate.
```

Code and runtime folders:

```text
src/
  model_loader.py                    Load tokenizer and Qwen3 model.
  text_loader.py                     Read one text file and tokenize one sequence.
  geometry_metrics.py                SVD, anisotropy, novelty, temporal, and block metrics.
  run_prefix_geometry.py             Prefix-growth KV geometry entrypoint.
  run_k_similarity_graph_probe.py    K-cache graph metric probe.
  run_qk_common_direction_probe.py   QK common-direction sanity check.

scripts/
  run_prefix_geometry.sh
  run_mps_geometry_long.sh
  nohup_run_mps_geometry_long.sh
  run_k_similarity_graph_probe.sh
  run_k_graph_metric_sweep.sh
  nohup_run_k_graph_metric_sweep.sh
  run_k_seq_len_scaling_sweep.sh
  run_k_transform_sweep.sh
  run_k_graph_construction_sweep.sh
  run_qk_common_direction_probe.sh
  generate_long_synthetic_text.py

data/
  synthetic_texts/                   Local long English synthetic texts.

outputs/
  Local experiment outputs; ignored by git except .gitignore/.gitkeep.

logs/
  Local nohup logs; ignored by git except .gitignore/.gitkeep.
```

Older Round1 notes were consolidated into `k_cache_graph_round1_findings.md` to
avoid having several partially overlapping conclusion files.

## Available Synthetic Texts

These files are generated locally and are meant for geometry diagnostics, not
for benchmark claims.

```text
long_english_12000_words.txt
  battery / microgrid technical report; about 43.9k Qwen tokens.

long_textbook_distributed_systems.txt
  textbook-style distributed systems chapter; about 19.7k Qwen tokens.

long_codebase_query_engine.txt
  codebase / API / bug-report style document; about 21.8k Qwen tokens.

long_news_supply_chain_dossier.txt
  multi-article news dossier; about 16.5k Qwen tokens.

long_dialogue_tool_transcript.txt
  agent conversation plus tool outputs; about 18.4k Qwen tokens.

long_english_article_01.txt
  short smoke-test article; about 788 Qwen tokens.
```

Use the long files for `MAX_TOKENS=1000,2000,4000,8000,12000` sweeps. Use the
short file only for quick smoke tests.

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

Supported K transforms:

```text
KEY_TRANSFORM=raw
KEY_TRANSFORM=center
KEY_TRANSFORM=remove_pc PC_REMOVE_COUNT=1
KEY_TRANSFORM=remove_pc PC_REMOVE_COUNT=4
KEY_TRANSFORM=whiten
```

Supported graph construction modes:

```text
GRAPH_MODE=topk
GRAPH_MODE=radius RADIUS_THRESHOLD=0.5
```

Every run writes `graph_structure_summary_by_layer.csv`, which summarizes edge
count, average out-degree, weak component structure, local edge fraction, and
long-range edge fraction.

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
KEY_TRANSFORM=remove_pc PC_REMOVE_COUNT=4 bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
GRAPH_MODE=radius RADIUS_THRESHOLD=0.6 bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```

Round2 sweep helpers:

```bash
bash fdong_seq_compress/scripts/run_k_seq_len_scaling_sweep.sh
bash fdong_seq_compress/scripts/run_k_transform_sweep.sh
bash fdong_seq_compress/scripts/run_k_graph_construction_sweep.sh
```
