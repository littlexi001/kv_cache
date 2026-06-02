# Qwen3 K-cache L2 Neighbor Analysis

This project studies query-agnostic K-vector similarity. For every selected
`(layer, KV head)`, it computes pairwise L2 distances among token K vectors:

```text
distance(i, j) = ||k_i - k_j||_2
```

For each token `i`, the script selects the top `NEIGHBOR_COUNT` nearest
neighbors by smallest L2 distance, excluding itself.

Default setup:

```text
model: /mnt/workspace/Qwen3-0.6B
tokens: 5000
neighbor_count: 5
neighbor_scope: all
rope_max_position_embeddings: 8192
```

`neighbor_scope=all` means each token can choose any other token. Set
`NEIGHBOR_SCOPE=previous` to restrict neighbors to earlier tokens `j < i`.

## Run

```bash
bash ymluo/projects/qwen3_kcache_l2_neighbor_analysis/scripts/run_analysis.sh
```

Useful focused run:

```bash
LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_l2_neighbor_analysis/scripts/run_analysis.sh
```

Raw and mean-centered K vectors:

```bash
VARIANTS=raw,centered \
bash ymluo/projects/qwen3_kcache_l2_neighbor_analysis/scripts/run_analysis.sh
```

## RoPE Length

The script loads the model config first and ensures:

```text
config.max_position_embeddings >= ROPE_MAX_POSITION_EMBEDDINGS
```

The default is:

```bash
ROPE_MAX_POSITION_EMBEDDINGS=8192
```

This is greater than the default 5000-token analysis length. If the model config
already supports a longer context, it is left unchanged.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_kcache_l2_neighbor_analysis/outputs/k_l2_neighbors/
```

Main files:

- `nearest_neighbors_by_token.csv`: one row per `(layer, head, token, neighbor
  rank)`.
- `summary_by_head.csv`: mean nearest-neighbor L2 distance and index-distance
  statistics per `(layer, head)`.
- `profile_timings.csv`: forward pass chunk timings.
- `summary.json`: run metadata and output paths.

Plots are organized by cache type, variant, layer, and head:

```text
plots/
  k/
    raw/
      layer_00/
        head_00/
          index_distance_by_rank_tokens.png
          l2_distance_by_rank_tokens.png
          pairwise_l2_distance_heatmap.png
```

`index_distance_by_rank_tokens.png`:

- x-axis: current token index `i`
- y-axis: absolute sequence index gap `abs(i - j)`
- colors: nearest-neighbor rank by L2 distance, where `top1` is the closest K
  vector and `top5` is the fifth closest

`l2_distance_by_rank_tokens.png`:

- x-axis: current token index `i`
- y-axis: `||k_i - k_j||_2`
- colors: nearest-neighbor rank by L2 distance

`pairwise_l2_distance_heatmap.png`:

- x-axis: token index `j`
- y-axis: token index `i`
- color: pairwise L2 distance `||k_i - k_j||_2`
- use `HEATMAP_MAX_TOKENS` to control downsampling for large contexts

Example:

```bash
HEATMAP_MAX_TOKENS=1500 \
bash ymluo/projects/qwen3_kcache_l2_neighbor_analysis/scripts/run_analysis.sh
```

## Notes

- This analysis does not use query vectors. It is a conservative, query-agnostic
  way to connect K vectors that are hard to distinguish for arbitrary bounded
  queries.
- L2 distance is stricter than cosine similarity because it includes both
  direction and norm differences.
- A horizontal band in `index_distance_by_rank_tokens.png` means many tokens
  have nearest K neighbors at a fixed sequence lag, e.g. `abs(i - j) ~= 1000`.
