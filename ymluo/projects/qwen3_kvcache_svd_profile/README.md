# Qwen3 KV Cache SVD Profile

This project profiles Qwen KV cache tensors at several prefix lengths and writes
numerical diagnostics for each selected layer/head.

Default lengths:

```text
1k, 10k, 100k, 1M tokens
```

The script builds the cache once up to the largest requested length, then analyzes
prefix views of that same cache. This keeps the text and token alignment identical
across lengths.

## Run

```bash
bash ymluo/projects/qwen3_kvcache_svd_profile/scripts/run_profile.sh
```

Quick smoke test:

```bash
CACHE_LENGTHS=128,256 \
LAYERS=0 \
HEADS=0 \
CHUNK_SIZE=64 \
bash ymluo/projects/qwen3_kvcache_svd_profile/scripts/run_profile.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/workspace/Qwen3-8B \
TEXT_PATH=/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt \
CACHE_LENGTHS=1k,10k,100k,1M \
LAYERS=0-3,10,20 \
HEADS=all \
CACHE_KINDS=key,value \
SVD_DEVICE=cuda \
MAX_SVD_RANK=128 \
bash ymluo/projects/qwen3_kvcache_svd_profile/scripts/run_profile.sh
```

Use all 8 GPUs for the SVD stage:

```bash
SVD_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
bash ymluo/projects/qwen3_kvcache_svd_profile/scripts/run_profile.sh
```

`DEVICE_MAP=auto` controls multi-GPU model placement during KV-cache profiling.
`SVD_DEVICES` controls parallel SVD jobs after the cache has been built. When
`SVD_DEVICES` is set, it overrides `SVD_DEVICE`.

`OFFLOAD_CACHE_TO_CPU=true` is the default. The script moves KV cache tensors to
CPU and releases model weights before SVD so the SVD workers can use GPU memory.
Set it to `false` only if you intentionally want to keep cache tensors on their
original devices.

For a full 1M run, use a text file with at least 1M tokens after tokenization.
Set `REQUIRE_MAX_LENGTH=false` if you want the script to continue with whatever
lengths are available.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_kvcache_svd_profile/outputs/kvcache_svd_profile/
```

Main files:

- `tokens.csv`: token ids/text for the profiled prefix.
- `profile_timings.csv`: forward/cache generation timing per chunk.
- `dimension_stats.csv`: one row per `(cache kind, length, layer, head, dim)`.
  Includes signed value stats, absolute amplitude stats, and `dim_l2_norm`.
- `token_norm_stats.csv`: per-token vector norm summaries for each
  `(cache kind, length, layer, head)`.
- `singular_values.csv`: saved singular values and energy fractions.
- `svd_vector_cosines.csv`: pairwise length comparisons for corresponding SVD
  vectors. `u_cosine` compares left singular vectors on the shared token prefix;
  `right_singular_vector_cosine` compares right singular vectors directly.
- `plots/`: singular-value distribution curves and SVD-vector cosine curves.
- `summary.json`: metadata and output path index.

By default, cosine values are sign-invariant: `abs(cos)` is used because SVD
vector signs are arbitrary.

## Notes

- `key` means the K cache tensor; `value` means the V cache tensor.
- `right_singular_vector_cosine` refers to the SVD right-singular vector, not the
  transformer value cache.
- `SAVE_SVD_TENSORS=false` by default. Enabling it saves truncated `U`,
  singular values, and `Vh` tensors and can consume a lot of disk for 1M-token
  runs.
- Large runs are expensive. A 1M-token cache can require substantial GPU memory
  during profiling and substantial CPU/GPU memory during SVD.
