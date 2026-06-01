# Qwen3 K-cache Cosine Heatmap

This project profiles Qwen3-0.6B on a DCLM text prefix, extracts the final K
cache, and computes token-token cosine similarity for each selected
`(layer, KV head)` K matrix.

Default inputs:

```text
model: /mnt/workspace/Qwen3-0.6B
text:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
tokens: 5000
```

For a selected layer/head, the K cache is reshaped to:

```text
[tokens, head_dim]
```

The script L2-normalizes the token vectors and computes:

```text
cosine_matrix = normalized_k @ normalized_k.T
```

so each heatmap is a `tokens x tokens` pairwise cosine matrix. The diagonal is
self-similarity and should be near `1.0`.

## Run

```bash
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

Useful overrides:

```bash
MAX_TOKENS=5000 \
CHUNK_SIZE=512 \
MODEL_PATH=/mnt/workspace/Qwen3-0.6B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

For a quick smoke test on one layer/head:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

To generate one full 5k-token heatmap for a single layer/head:

```bash
LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

To generate every layer/head heatmap, leave `LAYERS=all HEADS=all` as the
default. This can produce many large PNGs.

## Compression Diagnostics

Run the extended diagnostics requested for KV-cache compression:

```bash
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_compression_diagnostics.sh
```

Useful one-head smoke test:

```bash
MAX_TOKENS=128 CHUNK_SIZE=32 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_compression_diagnostics.sh
```

This extended script does the following for every selected `(layer, KV head)`:

- recomputes raw K-cache cosine and mean-centered K-cache cosine;
- analyzes V-cache in the same raw/centered way;
- writes raw and centered SVD singular values for K and V;
- plots singular values and cumulative PCA energy;
- samples query positions and validates low-rank K/V reconstructions with
  `|q dot (k_hat - k)|`, attention KL, top-1 match, and output-vector error
  when RoPE-aligned query capture is available.

Exact loss/PPL change is not reported by this script. Measuring that correctly
requires injecting the compressed K/V representation inside each attention layer
during the model forward; the diagnostics here are intended as a safer
pre-screening step before that model-patching experiment.

## Outputs

By default, files are written to:

```text
ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/
```

Main files:

- `plots/layer_XX_head_YY_cosine.png`: token-token cosine heatmap for one
  `(layer, KV head)`.
- `plots/layer_head_offdiag_mean_heatmap.png`: layer/head overview of mean
  off-diagonal cosine.
- `plots/layer_head_offdiag_std_heatmap.png`: layer/head overview of
  off-diagonal cosine standard deviation.
- `summary_by_head.csv`: per `(layer, head)` summary statistics for the full
  cosine matrix and the off-diagonal entries.
- `histogram_by_head.csv` and `histogram_global.csv`: binned cosine
  distributions with `count` and `probability`. Use `scope=offdiag` to ignore
  diagonal self-similarity.
- `distance_summary_by_head.csv`: per `(cache_type, layer, head)` summary of
  pairwise L2 distances between K/V token vectors.
- `distance_histogram_by_head.csv` and `distance_histogram_global.csv`: binned
  pairwise L2 distance distributions for K/V vectors.
- `top_p_previous_distance_summary_by_head.csv`: for each `(cache_type, layer,
  head)`, summarizes how far back in sequence the most similar previous K/V
  vectors are.
- `top_p_previous_distance_by_token.csv`: one row per token with the selected
  previous token indices, cosine similarities, and average index distance.
- `cluster_summary_by_head.csv`: optional K-means clustering summary for K/V
  token vectors, one row per `(cache_type, layer, head)`.
- `cluster_assignments_by_token.csv`: optional per-token cluster assignment
  output when `SAVE_CLUSTER_ASSIGNMENTS=true`.
- `profile_timings.csv`: elapsed time for each forward chunk used to build the
  K cache.
- `tokens.csv`: token index, token id, tokenizer piece, and decoded text.
- `summary.json`: run metadata and output paths.

Extended diagnostic outputs are written by default to:

```text
ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kv_compression_diagnostics/
```

Main extended files:

- `compression_summary_by_head.csv`: raw and mean-centered K/V cosine summaries.
- `singular_values.csv`: one row per K/V singular value for raw and centered
  matrices.
- `svd_summary_by_head.csv`: PCA cumulative energy and ranks needed to reach
  configured energy thresholds.
- `attention_validation_by_head_rank.csv`: sampled low-rank validation metrics
  for `k_only` and `kv` compression variants when query capture succeeds.
- `plots/k_centered_cosine/*.png`: centered K-cache cosine heatmaps.
- `plots/v_raw_cosine/*.png` and `plots/v_centered_cosine/*.png`: V-cache
  heatmaps.
- `plots/svd/*.png`: singular value and cumulative-energy curves.

Optional tensor dump:

```bash
SAVE_SIMILARITY_TENSORS=true LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

This writes `similarity_tensors/layer_XX_head_YY_cosine.pt`. A single 5000 x
5000 float16 matrix is about 50 MB, so avoid enabling this for all heads unless
you have enough disk space.

## Options

Layer/head selection:

```bash
LAYERS=0,7,15 HEADS=0-3 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

Plot downsampling:

```bash
PLOT_MAX_TOKENS=1500 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

The cosine matrix is still computed on all tokens; only the rendered heatmap is
strided if the token count exceeds `PLOT_MAX_TOKENS`. Set
`PLOT_MAX_TOKENS=0` to force plotting every token.

Top-p previous-neighbor distance:

```bash
TOP_P_PREVIOUS_COUNT=5 TOP_P_PREVIOUS_CACHE_TYPES=k \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

For each token position `i`, this metric looks only at previous positions
`j < i`, selects the top `p` vectors by cosine similarity to the current vector,
and computes:

```text
mean_index_distance(i) = mean(abs(i - selected_j))
```

If fewer than `p` previous vectors exist, all previous vectors are selected. The
main aggregate column is
`top_p_previous_distance_summary_by_head.csv::mean_index_distance_mean`. Per
token values are in
`top_p_previous_distance_by_token.csv::mean_index_distance`.

By default this uses raw vectors. To compare raw and mean-centered vectors:

```bash
TOP_P_PREVIOUS_VARIANTS=raw,centered \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

The output CSVs include a `variant` column. `centered` means the mean vector of
the current `(cache_type, layer, head)` matrix is subtracted before computing
cosine neighbors.

The token-level file also reports two relative forms:

```text
mean_index_distance_percent_of_history = 100 * mean_index_distance / token_index
mean_index_distance_percent_of_context = 100 * mean_index_distance / tokens
```

Use `percent_of_history` when asking how far back the selected neighbors are
relative to the amount of available prefix for that token. Use
`percent_of_context` when comparing all tokens against the fixed context length.

To plot the result without drawing thousands of token points:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_top_p_previous_distance.py \
  --summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_summary_by_head.csv \
  --token_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_by_token.csv
```

This writes layer/head heatmaps for mean, median, and p95 distance. If the token
CSV exists, it also bins token positions and plots only representative short-
and long-range heads. This is usually more readable than plotting all 5000
tokens for every head.

To draw separate per-token scatter plots for selected heads, add
`--plot_token_points`. For example:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_top_p_previous_distance.py \
  --summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_summary_by_head.csv \
  --token_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_by_token.csv \
  --selected_heads k:6:3,k:14:4 \
  --plot_token_points
```

This writes one plot per selected `(cache_type, layer, head)` and metric. The
most useful plots are `mean_index_distance_tokens.png` and
`mean_index_distance_percent_of_history_tokens.png`.

To plot the selected top-p neighbors separately by rank instead of averaging
them first:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_top_p_previous_distance.py \
  --summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_summary_by_head.csv \
  --token_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_by_token.csv \
  --selected_heads k:6:3,k:14:4 \
  --plot_token_rank_points
```

This draws one color per similarity rank: `top1` is the most similar previous
token, `top2` is the second most similar, and so on. The distance is still the
absolute sequence-index gap from the current token.

To draw every layer/head, use `--plot_all_heads`. Per-head plots are written
under separate folders:

```text
top_p_previous_plots/
  k/
    raw/
      layer_00/
        head_00/
        head_01/
        ...
    centered/
      layer_00/
        head_00/
        ...
```

Example:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_top_p_previous_distance.py \
  --summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_summary_by_head.csv \
  --token_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/top_p_previous_distance_by_token.csv \
  --plot_all_heads \
  --plot_token_rank_points \
  --plot_token_points
```

The script also writes `head_plot_index.csv`, which maps each `(cache_type,
layer, head)` to its output folder.

K/V clustering:

```bash
COMPUTE_CACHE_CLUSTERS=true CLUSTER_CACHE_TYPES=k,v CLUSTER_COUNT=32 \
bash ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/run_analysis.sh
```

The clustering uses K-means independently for each selected `(cache_type, layer,
head)` matrix. `cluster_summary_by_head.csv` reports inertia, mean squared
distance to centroids, cluster-size imbalance, largest-cluster fraction, entropy,
and centroid norms. Enable `SAVE_CLUSTER_ASSIGNMENTS=true` only when token-level
cluster labels are needed, because it can write many rows.

Cluster plots:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_cluster_summary.py \
  --cluster_summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/cluster_summary_by_head.csv
```

If cluster assignments were saved:

```bash
python ymluo/projects/qwen3_kcache_cosine_heatmap/scripts/plot_cluster_summary.py \
  --cluster_summary_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/cluster_summary_by_head.csv \
  --cluster_assignments_csv ymluo/projects/qwen3_kcache_cosine_heatmap/outputs/kcache_cosine_heatmap/cluster_assignments_by_token.csv \
  --selected_heads k:6:3,v:6:3
```

The cluster plot script writes layer/head heatmaps, layer-average curves, cluster
size bar charts for representative heads, and token-index cluster assignment
scatter plots when assignment rows are available.

## Current Top-p Previous Distance Result

The run saved in `top_p_previous_distance_summary_by_head.csv` used:

```text
cache_type: k
tokens: 5000
head_dim: 128
top_p: 5
layers: 0-27
KV heads: 0-7
```

Key observations:

- Across all 224 `(layer, head)` rows, the average of
  `mean_index_distance_mean` is `147.25` tokens; the median head is `87.21`
  tokens.
- Most heads are local: `55.4%` of heads have
  `mean_index_distance_mean <= 100`, `76.3%` are `<= 200`, and `95.1%` are
  `<= 500`.
- The shortest-range heads are very local. Examples:
  layer 14 head 4 has mean distance `7.26`, layer 4 head 0 has `8.05`, and
  layer 7 head 1 has `9.29`.
- A small number of heads retrieve much farther back. The largest mean distance
  is layer 6 head 3 at `1044.84`; other large heads include layer 2 head 6
  (`866.95`) and layer 3 head 5 (`852.65`).
- By layer average, the strongest long-range layers are layer 2 (`476.57`),
  layer 0 (`346.91`), and layer 6 (`279.05`). The most local layers are layer 7
  (`28.58`), layer 17 (`36.92`), layer 19 (`42.28`), and layer 10 (`43.84`).

Interpretation: for this 5k-token prefix and `p=5`, most K heads choose their
nearest cosine neighbors from relatively nearby sequence positions, but a few
heads consistently point hundreds to about one thousand tokens back. Those
high-distance heads are the most direct candidates for long-range retrieval or
repetition-sensitive behavior.

## Notes

- Heads are KV heads, not query attention heads. Qwen-style GQA models can have
  fewer KV heads than query heads.
- The default summary percentile columns sample up to `SUMMARY_SAMPLE_SIZE`
  values per matrix to avoid spending most of the run sorting 25M entries per
  head. Mean, std, min, max, and RMS are computed on the full matrix.
- `SIMILARITY_DEVICE=auto` uses CUDA when available; otherwise it falls back to
  CPU.
