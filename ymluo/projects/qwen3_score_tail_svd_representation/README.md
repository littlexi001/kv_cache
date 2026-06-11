# Qwen3 Score-Tail SVD Representation Analysis

This project implements the section 6.2 SVD representation-basis analysis from:

```text
common_doc/top1_context_research_questions.md
```

The goal is to compare where different score-ranked token groups live in
K/V/weighted-V representation space.

## Token Groups

For each layer, head, and query row, valid key tokens are sorted by raw masked
QK score.

- `score_top_1pct`: highest 1% by raw QK score.
- `score_top_90pct`: highest 90% by raw QK score.
- `score_tail_10pct`: lowest 10% by raw QK score.

`score_top_90pct` is score-rank based. It is not the same as the attention-mass
`top90` used by `qwen3_attention_value_decomposition`.

## Representations

Default representation types:

- `key`: selected key vectors.
- `value`: selected value vectors after KV heads are repeated to attention heads.
- `weighted_value`: selected value vectors multiplied by their full-attention
  softmax weight for that query row.

The script builds an SVD basis per `(layer, head, representation)` from sampled
vectors across the three groups, then projects each group onto the first `k`
singular directions.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_score_tail_svd_representation/outputs/score_tail_svd_representation/
```

Main files:

- `svd_projection_by_group.csv`
- `centroid_similarity_by_group.csv`
- `score_distribution_by_group.csv`
- `score_top_90pct_distribution.csv`
- `singular_value_energy.csv`
- `summary.json`

`score_top_90pct_distribution.csv` is the dedicated distribution requested for
the top 90% score-ranked token set.

## Run

```bash
bash ymluo/projects/qwen3_score_tail_svd_representation/scripts/run_analysis.sh
```

Smoke test:

```bash
PREFILL_TOKENS=128 EVAL_TOKENS=32 CHUNK_SIZE=16 LAYERS=0 HEADS=0 \
MAX_QUERY_ROWS_PER_LAYER_HEAD=32 MAX_VECTORS_PER_GROUP=256 \
bash ymluo/projects/qwen3_score_tail_svd_representation/scripts/run_analysis.sh
```

Useful parameters:

```text
PREFILL_TOKENS=5000
EVAL_TOKENS=1024
CHUNK_SIZE=128
LAYERS=all
HEADS=all
SVD_COMPONENTS=8
QUERY_STRIDE=8
MAX_QUERY_ROWS_PER_LAYER_HEAD=512
MAX_VECTORS_PER_GROUP=4096
REPRESENTATIONS=key,value,weighted_value
```
