# Design

## Falsifiable Conjecture

If the main useful historical attention tokens are redundant across heads, then
limiting each historical token to at most 3 heads after top-2% selection should
increase PPL less than plain top-2% pruning does.

## Physical Prior

The previous experiment showed that many top-2% historical tokens are selected
by more than one head. If those repeated selections are partly redundant, the
model may tolerate removing some duplicate head-token links while keeping each
head's strongest historical tokens and keeping self-attention unchanged.

## Mathematical Model

For layer `l`, head `h`, query token `q`, and historical key token `j < q`, let
`s[l,h,q,j]` be the pre-softmax attention score after causal masking.

`top2` keeps:

```text
S(l,h,q) = top ceil(0.02 * q) historical tokens by s[l,h,q,j]
```

The current token `j = q` is kept unconditionally when `always_keep_self=true`.

`top2limit3` first computes all `S(l,h,q)`. For a fixed historical token `j`,
define:

```text
H(l,q,j) = {h : j in S(l,h,q)}
```

If `|H(l,q,j)| <= 3`, all those head-token links are kept. If
`|H(l,q,j)| > 3`, a random subset of 3 heads is kept and the other links are
masked out before softmax.

## Implementation Contract

Inputs:

- Local Qwen3-0.6B model.
- Text prefix used for prefill and evaluation.
- `prefill_tokens`, `eval_tokens`, and `chunk_size`.
- `top_fraction = 0.02`.
- `max_heads_per_token = 3`.
- `seed = 1234`.

Algorithm:

1. Load the model with Qwen3 eager attention.
2. Patch `eager_attention_forward`.
3. For `baseline`, run normal attention and compute next-token loss.
4. For `top2`, mask all historical tokens except each head's top 2%; keep the
   current token.
5. For `top2limit3`, compute the top2 mask, then limit every historical token
   to at most 3 selecting heads using seeded random selection.
6. Compute PPL on the same evaluation tokens for all three modes.
7. During `top2limit3`, record per-layer and per-head load:
   original top2 kept count, final kept count, removed count, and kept fraction.
8. Save CSV files and plots.

Pass conditions:

- All three modes produce finite loss and PPL.
- `top2limit3_load_by_head.csv` has one row for every layer/head.
- The kept fraction after limit3 is in `[0, 1]`.
- Plots render without clipped legends or ambiguous axes.

Fail conditions:

- The model does not use Qwen3 eager attention.
- The tokenizer has fewer tokens than `prefill_tokens + eval_tokens`.
- The pruned attention masks all positions for a query/head.

## Claim Boundary

This experiment measures PPL on one text prefix. It does not prove that the
same rule works on all data, and it does not measure generation quality beyond
next-token likelihood. The random limit3 rule is one operationalization of the
broader idea of reducing cross-head duplicate KV use.

## Sink/Recent Protection Extension

Updated prior: some cross-head duplicate links are not redundant when they point
to attention sink tokens or to recent local context. These token positions
should be protected before applying the top3-head cap.

For each query with `q` historical tokens:

```text
protected(j) =
  j < sink_tokens
  or j >= q - ceil(recent_fraction * q)
```

`top2limit3protectsSrR` keeps every original top2-selected head for protected
historical tokens. For unprotected historical tokens, if more than 3 heads
selected the same token, it keeps the 3 selected heads with largest pre-softmax
score and masks the rest before softmax.

The current best tested setting is:

```text
sink_tokens = 64
recent_fraction = 0.16
mode = top2limit3protects64r16p0
```

This setting tests whether a broad local recent band, not only the most recent
1%, is required for stable PPL.

## Head-Count Position Distribution Diagnostic

For each layer, query token, and historical token, define:

```text
selected_head_count(l, q, j) =
  number of heads h where token j is in head h's top2 historical tokens.
```

The diagnostic groups historical tokens by `selected_head_count = 1..16` and
records:

- absolute key token position `j`;
- distance from query `q - j`;
- relative key position `j / (q - 1)`, where `0` is sequence start and `1` is
  query-near;
- membership in fixed sink and recent probes, such as `key < 64` and recent
  `1%`, `8%`, or `16%`.

This diagnostic measures where shared top2 selections occur. It does not apply
any top3, score-gap, or protect pruning rule.
