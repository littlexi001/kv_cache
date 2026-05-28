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
scripts/run_prefix_geometry.sh
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

