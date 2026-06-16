# Qwen3 Cluster KV-Cache Retrieval

This project evaluates a generation-time KV-cache selection scheme for
Qwen3-0.6B:

1. Split historical KV tokens into fixed clusters of 50 tokens.
2. For each new decode query, compute cosine similarity against all cluster
   centers on GPU.
3. Select `ceil(0.02 * num_clusters)` clusters, with the first and last cluster
   forced into the selection budget by default.
4. Mask attention scores outside selected clusters, then run softmax/value matmul.

For a 100k-token history, this gives roughly 2k clusters and keeps 40 clusters.

## Default Paths

```text
model: /mnt/workspace/Qwen3-0.6B
data:  /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
```

## Run

```bash
bash ymluo/projects/qwen3_cluster_kvcache_retrieval/scripts/run_eval.sh
```

Smoke test:

```bash
PREFILL_TOKENS=2048 EVAL_TOKENS=32 PREFILL_CHUNK_SIZE=256 \
bash ymluo/projects/qwen3_cluster_kvcache_retrieval/scripts/run_eval.sh
```

Only run the sparse method:

```bash
MODES=cluster bash ymluo/projects/qwen3_cluster_kvcache_retrieval/scripts/run_eval.sh
```

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_cluster_kvcache_retrieval/outputs/qwen3_cluster_kvcache_retrieval/
```

Main files:

- `summary.csv`: baseline and cluster PPL, prefill time, decode time, tokens/sec.
- `summary.json`: full metadata plus attention profile timing buckets.
- `token_timings_baseline.csv`: per-token baseline decode latency.
- `token_timings_cluster.csv`: per-token sparse decode latency.

`summary.json` includes the comparison:

```json
{
  "ppl_delta_cluster_minus_baseline": 0.0,
  "loss_delta_cluster_minus_baseline": 0.0,
  "decode_speedup_vs_baseline": 1.0
}
```

## Timing Notes

The eval path uses KV-cache decoding, not full-sequence training forward:

- prefill builds the full cache in chunks;
- eval runs one token at a time;
- PPL is computed from the previous token's logits;
- decode timing excludes the first `WARMUP_EVAL_TOKENS` tokens by default.

When `PROFILE_ATTENTION=true`, CUDA events also record attention-level buckets:

- `cluster_center_score_topk_ms`
- `gather_selected_kv_ms`
- `sparse_qk_softmax_value_ms`

These buckets are useful for later optimization, especially to see whether
cluster-center construction/topk/masking dominates the sparse path.

The optimized sparse path is used for `q_len=1` decode calls. Prefill remains
full attention and is timed separately. Cluster centers are cached per attention
layer after the first decode call and then updated incrementally as each new
token enters the KV cache.
