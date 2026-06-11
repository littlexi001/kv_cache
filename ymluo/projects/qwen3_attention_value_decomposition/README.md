# Qwen3 Attention Value Decomposition

This project studies attention-weighted value outputs under different attention
splits.

For each selected layer/head/query token, the script computes:

```text
full = sum(all attention weights * V)
topX = sum(selected high-attention weights * V)
tailY = sum(selected low-attention weights * V)
```

The split is configurable.

```text
SPLIT_MODE=mass
```

means `top0p9` selects the smallest high-attention token set whose cumulative
attention mass reaches `0.9`; `tail0p1` selects the lowest-attention token set
whose cumulative attention mass reaches `0.1`.

```text
SPLIT_MODE=token_fraction
```

means `top0p9` selects the top 90% tokens by attention weight; `tail0p1` selects
the bottom 10% tokens by attention weight.

By default selected vectors are not renormalized, so in mass mode
`top0p9 + tail0p1` is close to `full` when the values are complementary.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_attention_value_decomposition/outputs/attention_value_decomposition/
```

Main files:

- `value_vectors_by_head.csv`
- `value_pairwise_by_head.csv`
- `ppl_by_attention_value_mode.csv`
- `summary.json`

`value_vectors_by_head.csv` contains one row per `(layer, head, vector)`:

- mean vector norm
- mean selected attention mass
- mean selected token count

`value_pairwise_by_head.csv` contains all pairwise comparisons within each
`(layer, head)`:

- `mean_cosine`
- `mean_l2`

Set `SAVE_PAIRWISE_PER_TOKEN=true` to also write `value_pairwise_per_token.csv`.
That file keeps one row per `(layer, head, query token, vector pair)` and is used
for histogram/frequency plots over the 5k evaluation tokens.

For example, with `TOP_VALUES=0.5,0.9` and `TAIL_VALUES=0.01,0.1`, the table
compares:

```text
full, top0p5, top0p9, tail0p01, tail0p1
```

against each other horizontally.

`ppl_by_attention_value_mode.csv` evaluates behavior when selected layers/heads
use a chosen vector mode instead of full attention output. By default:

```text
PPL_MODES=full,top0p9,tail0p1
```

Set `PPL_RENORMALIZE_SELECTED=true` to renormalize selected top/tail weights
before computing `attn @ V`.

## Plot Pairwise Cosine

After `value_pairwise_by_head.csv` is generated, plot the mean cosine for every
vector pair:

```bash
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/plot_pairwise_cos.sh
```

If the CSV is in a custom path:

```bash
INPUT_CSV=/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_attention_value_decomposition/outputs/attention_value_decomposition/value_pairwise_by_head.csv \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/plot_pairwise_cos.sh
```

This writes:

```text
plots/pairwise_cos/
  pairwise_mean_cosine_summary.csv
  pairwise_mean_cosine_bar.png
  pairwise_mean_cosine_heatmap.png
  plot_summary.json
```

To plot token-level cosine frequency histograms, first rerun analysis with
per-token rows enabled:

```bash
SAVE_PAIRWISE_PER_TOKEN=true \
COMPUTE_PPL=false \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Then plot histograms:

```bash
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/plot_pairwise_token_hist.sh
```

For a focused pair/layer/head subset:

```bash
PAIRS=full\|top0p9 LAYERS=0,1 HEADS=0,4 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/plot_pairwise_token_hist.sh
```

This writes:

```text
plots/pairwise_token_hist/
  pairwise_token_hist_summary.csv
  <pair>/
    all_layers_heads.png
    layer_00_head_00.png
    ...
```

## Run

```bash
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Smoke test:

```bash
PREFILL_TOKENS=128 EVAL_TOKENS=64 CHUNK_SIZE=32 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Multiple top/tail splits:

```bash
TOP_VALUES=0.5,0.9,0.99 \
TAIL_VALUES=0.01,0.05,0.1 \
PPL_MODES=full,top0p9,tail0p1 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Token-count split instead of attention-mass split:

```bash
SPLIT_MODE=token_fraction \
TOP_VALUES=0.1,0.5,0.9 \
TAIL_VALUES=0.1 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Vector analysis only:

```bash
COMPUTE_PPL=false \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Useful parameters:

```text
PREFILL_TOKENS=5000
EVAL_TOKENS=5000
CHUNK_SIZE=128
LAYERS=all
HEADS=all
SPLIT_MODE=mass
TOP_VALUES=0.9
TAIL_VALUES=0.1
PPL_MODES=full,top0p9,tail0p1
PPL_RENORMALIZE_SELECTED=false
```
