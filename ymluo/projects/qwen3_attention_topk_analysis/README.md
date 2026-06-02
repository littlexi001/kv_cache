# Qwen3 Attention Top-k Analysis

This project streams Qwen3 attention weights and plots the top-k attended key
positions for every selected `(layer, attention head)`.

Important: this script uses Hugging Face `output_attentions=True`, so the value
being ranked is the post-softmax attention weight, not raw `q dot k` logits.

For each query token `i`, layer, and attention head, it selects:

```text
top-k key tokens j by attention_weight(i, j)
```

Then it plots:

```text
index_distance = abs(i - j)
attention_weight
```

## Run

```bash
bash ymluo/projects/qwen3_attention_topk_analysis/scripts/run_analysis.sh
```

Focused smoke test:

```bash
MAX_TOKENS=256 CHUNK_SIZE=64 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_attention_topk_analysis/scripts/run_analysis.sh
```

Exclude self-attention when selecting top-k:

```bash
INCLUDE_SELF=false \
bash ymluo/projects/qwen3_attention_topk_analysis/scripts/run_analysis.sh
```

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_attention_topk_analysis/outputs/attention_topk/
```

Main files:

- `attention_topk_by_token.csv`: one row per `(layer, head, query token, rank)`.
- `summary_by_head.csv`: aggregate index-distance and attention-weight
  statistics per `(layer, head)`.
- `profile_timings.csv`: chunked forward pass timings.
- `summary.json`: run metadata and output paths.

Plots are organized by layer and attention head:

```text
plots/
  layer_00/
    head_00/
      index_distance_by_rank_tokens.png
      attention_weight_by_rank_tokens.png
      attention_weight_heatmap.png
```

`index_distance_by_rank_tokens.png`:

- x-axis: query token index `i`
- y-axis: `abs(i - j)`, where `j` is a top-k attended key token
- color: attention rank, where `top1` has the largest attention weight

`attention_weight_by_rank_tokens.png`:

- x-axis: query token index `i`
- y-axis: post-softmax attention weight
- color: attention rank

`attention_weight_heatmap.png`:

- x-axis: key token index `j`
- y-axis: query token index `i`
- color: post-softmax attention weight
- use `HEATMAP_MAX_TOKENS` to control downsampling for 5000-token contexts

Example:

```bash
HEATMAP_MAX_TOKENS=1500 \
bash ymluo/projects/qwen3_attention_topk_analysis/scripts/run_analysis.sh
```

## Notes

- Heads are attention heads, not KV heads.
- Qwen-style GQA can have more attention heads than KV heads.
- The script uses chunked generation with KV cache, so it does not store a full
  `[layers, heads, tokens, tokens]` attention tensor.
- Use `ATTN_IMPLEMENTATION=eager` if the backend does not return attentions.
- If raw `q dot k` logits are needed instead of softmax weights, that requires a
  separate hook into the attention module before softmax.
