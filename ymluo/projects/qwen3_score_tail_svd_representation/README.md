# Qwen3 V-Basis Projection Analysis

This project implements the section 6.2 SVD representation-basis analysis from:

```text
common_doc/top1_context_research_questions.md
```

The goal is to first build a fixed V representation basis, then project
attention outputs from different score-ranked token selections onto that basis.

## Method

This project uses a two-stage design.

### 1. Build the V Basis

Sample V vectors from ordinary forward passes. `BASIS_TOKENS` is the source
token pool size, and `MAX_BASIS_VECTORS_PER_LAYER_HEAD` is the number of V
vectors sampled per layer/head from that pool.

For each `(layer, head)`, build:

```text
M = [v_1; v_2; ...; v_n]
```

By default the script samples up to `n=5000` V vectors per layer/head. Sampling
is random over the whole source token pool, not a contiguous chunk. Then compute:

```text
M_centered = U S V^T
```

The rows of `V^T` are saved as the fixed V representation basis for that
layer/head.

### 2. Project Score-Selected Attention Outputs

During a normal forward pass, each query row sorts valid keys by raw masked QK
score. The script selects:

```text
top1, top2, top4, top8, top16, top30, top50, top90
tail10, tail30, tail50
```

For each selection, it recomputes a conditional softmax over only the selected
scores and forms a weighted V output:

```text
v_top1 = softmax(scores[top1]) @ V[top1]
```

Then it projects that output onto the saved V basis:

```text
projection = (v_top1 - mean_V) @ V_basis
energy_i = projection_i^2 / sum_j projection_j^2
```

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_score_tail_svd_representation/outputs/score_tail_svd_representation/
```

Main files:

- `svd_basis.pt`
- `singular_value_energy.csv`
- `projection_energy_by_layer_head.csv`
- `projection_energy_by_layer.csv`
- `projection_energy_global.csv`
- `plots/`
- `summary.json`

Plots include:

- Per-layer/per-head projection energy curves.
- Per-layer summary curves averaged over heads.
- Global summary curves averaged over all layers/heads.
- Singular-value energy distribution plots.
- Curve subset plots:
  - `all`
  - `top_sparse`: `top1/top4/top16/top50/top90`
  - `top_low`: `top1/top2/top4/top8/top16`
  - `tail_only`: `tail10/tail30/tail50`

Top-ratio curves use a blue gradient. Tail-ratio curves use a red gradient.

## Run

```bash
bash ymluo/projects/qwen3_score_tail_svd_representation/scripts/run_analysis.sh
```

Smoke test:

```bash
PREFILL_TOKENS=128 EVAL_TOKENS=32 CHUNK_SIZE=16 LAYERS=0 HEADS=0 \
MAX_QUERY_ROWS_PER_LAYER_HEAD=32 MAX_BASIS_VECTORS_PER_LAYER_HEAD=256 \
bash ymluo/projects/qwen3_score_tail_svd_representation/scripts/run_analysis.sh
```

Useful parameters:

```text
BASIS_TOKENS=5000
BASIS_SAMPLE_MODE=random
BASIS_SAMPLE_SEED=0
PREFILL_TOKENS=5000
EVAL_TOKENS=1024
CHUNK_SIZE=128
LAYERS=all
HEADS=all
SVD_COMPONENTS=16
QUERY_STRIDE=8
MAX_QUERY_ROWS_PER_LAYER_HEAD=512
MAX_BASIS_VECTORS_PER_LAYER_HEAD=5000
TOP_RATIOS=0.01,0.02,0.04,0.08,0.16,0.30,0.50,0.90
TAIL_RATIOS=0.10,0.30,0.50
MAKE_PLOTS=true
MAKE_HEAD_PLOTS=true
```

For a more stable high-dimensional basis, use a larger source token pool than
the final sample count, for example:

```bash
BASIS_TOKENS=64000 MAX_BASIS_VECTORS_PER_LAYER_HEAD=5000 SVD_COMPONENTS=128 \
bash ymluo/projects/qwen3_score_tail_svd_representation/scripts/run_analysis.sh
```
