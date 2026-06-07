# Qwen3 Attention Pruning Cos/PPL Analysis

This project evaluates how close top-ratio attention pruning is to full
attention on Qwen3-0.6B.

The default experiment uses:

```text
prefill context: first 5000 tokens
evaluation tokens: next 5000 tokens
ratios: 0.1%, 0.5%, 1%, 2%, 4%, 6%, 8%, 10%, 15%, 20%
```

## Cosine Metric

For each evaluation query token, layer, and attention head:

1. Compute full attention softmax weights.
2. Select the top `k = max(1, ceil(ratio * current_key_len))` tokens by QK
   score. This is equivalent to selecting top attention weights before pruning.
3. Set non-selected QK scores to `-inf`.
4. Compute the pruned softmax distribution.
5. Compute cosine similarity:

```text
cos(softmax(full_qk_scores), softmax(pruned_qk_scores))
```

The implementation avoids materializing raw QK scores for the cosine metric.
Because top-QK and top-attention order are the same, the cosine can be computed
from full attention weights:

```text
cos = sqrt(sum(top_k_attention_weight^2) / sum(all_attention_weight^2))
```

The final plotted value is the mean cosine over all evaluation query tokens.

## PPL Metric

For PPL, the script patches Qwen3 eager attention so that, before softmax,
non-top-k attention scores are set to `-inf` at every layer and attention head.

`ppl_by_ratio.csv` contains 11 points:

- 10 pruning ratios
- 1 full-attention baseline

## Run

```bash
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/run_analysis.sh
```

Fast smoke test:

```bash
PREFILL_TOKENS=128 EVAL_TOKENS=128 CHUNK_SIZE=64 LAYERS=0 HEADS=0 COMPUTE_PPL=false \
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/run_analysis.sh
```

Focused layer/head run:

```bash
LAYERS=15 HEADS=4 COMPUTE_PPL=false \
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/run_analysis.sh
```

Evaluate PPL only on the last 10 tokens of the tokenized text:

```bash
MAX_CHARS=0 EVAL_LAST_TOKENS_ONLY=true EVAL_TOKENS=10 COMPUTE_COS=false COMPUTE_PPL=true \
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/run_analysis.sh
```

Useful parameters:

```text
PREFILL_TOKENS=5000
EVAL_TOKENS=5000
EVAL_LAST_TOKENS_ONLY=false
RATIOS=0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20
LAYERS=all
HEADS=all
SAVE_COS_PER_TOKEN=true
COMPUTE_COS=true
COMPUTE_PPL=true
```

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_attention_pruning_cos_ppl/outputs/attention_pruning_cos_ppl/
```

Main files:

- `cos_per_token.csv`: token-level cosine rows for every selected
  `(layer, head, ratio)`.
- `cos_summary_by_head.csv`: mean cosine over evaluation tokens for every
  `(layer, head, ratio)`.
- `ppl_by_ratio.csv`: baseline and pruned PPL values.
- `summary.json`: run metadata and output paths.

Plots:

```text
plots/
  layer_00/
    head_00/
      mean_cosine_by_keep_ratio.png
      mean_cosine_by_keep_ratio_logx.png
  ppl_by_keep_ratio.png
  ppl_by_keep_ratio_logx.png
```

Cosine distribution plots from an existing `cos_per_token.csv`:

```bash
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/plot_cos_distribution_by_ratio.sh
```

If the analysis output is in a custom directory:

```bash
OUTPUT_DIR=/path/to/attention_pruning_cos_ppl_output \
bash ymluo/projects/qwen3_attention_pruning_cos_ppl/scripts/plot_cos_distribution_by_ratio.sh
```

This writes:

```text
plots/cos_distribution_by_ratio/
  cos_distribution_ratio_0p001.png
  cos_distribution_ratio_0p001_logy.png
  cos_distribution_ratio_0p005.png
  cos_distribution_ratio_0p005_logy.png
  ...
  cos_distribution_ratio_0p2.png
  cos_distribution_ratio_0p2_logy.png
  cos_distribution_summary_by_ratio.csv
```

For Qwen3-0.6B, `LAYERS=all HEADS=all` gives `28 * 16` per-head cosine plots.

## Notes

The cosine metric is diagnostic: it measures distribution similarity at each
layer/head under full attention weights.

The PPL metric is behavioral: it reruns the model with pruned attention applied
inside every layer/head during the evaluation window.
