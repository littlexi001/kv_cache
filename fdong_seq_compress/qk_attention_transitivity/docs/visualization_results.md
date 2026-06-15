# QK Attention Transitivity: Results

## Setup

- Model: `fdong/Qwen3-0.6B`
- Text: `fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt`
- Sequence length: `1024`
- Layers: `[0, 13, 27]`
- Query heads: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- Relation definition: strict-history score top `2%` per query row
- Actual mean selected ratio after integer rounding: `2.11%`

## Weight-kernel diagnostics

| layer | symmetry cosine | skew ratio | negative quadratic fraction | row-space cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.101 | 1.340 | 0.346 | 0.366 |
| 13 | 0.087 | 1.349 | 0.219 | 0.442 |
| 27 | 0.005 | 1.411 | 0.195 | 0.355 |

## Directed two-hop closure

| layer | closure | uniform baseline | distance baseline | distance+popularity baseline | lift vs strongest baseline | endpoint percentile |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.270 | 0.011 | 0.087 | 0.184 | 1.70 | 0.899 |
| 13 | 0.340 | 0.013 | 0.089 | 0.127 | 2.95 | 0.928 |
| 27 | 0.306 | 0.012 | 0.147 | 0.159 | 1.98 | 0.898 |

## Similar-query retrieval stability

| layer | nearest query cosine | retrieval Jaccard | random Jaccard | Jaccard lift | attention JS | random JS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.826 | 0.342 | 0.195 | 2.09 | 0.215 | 0.323 |
| 13 | 0.820 | 0.445 | 0.242 | 1.92 | 0.149 | 0.311 |
| 27 | 0.927 | 0.484 | 0.261 | 1.87 | 0.018 | 0.046 |

## Interpretation contract

- Closure lift must be judged against the distance-matched baseline, not against zero.
- A block-shaped heatmap alone does not prove semantic transitivity.
- Positive results support local QK bucket geometry; they do not prove that Attention and MoE must share one partition.
- Inspect per-head CSV files before making a model-wide claim.

## Exact Top-2% Relation Strength

| layer | scaled QK threshold | threshold z-score | QK cosine threshold | selected softmax mass |
|---:|---:|---:|---:|---:|
| 0 | 9.09 | 1.85 | 0.035 | 59.4% |
| 13 | 10.25 | 2.10 | 0.278 | 69.7% |
| 27 | 1.94 | 2.08 | 0.028 | 93.1% |

这里的高相关性不是统一的 cosine 阈值。不同 layer/head 的 Q/K 映射和 score scale 不同，因此最可比较的量是 row-relative z-score：top-2% 的最低 score 平均位于本行均值以上约 `1.85-2.10` 个标准差，确实属于 QK score distribution 的高尾。

top-2% 承载的 softmax mass 随层变化明显：浅层约 `59%`，中层约 `70%`，深层约 `93%`。因此固定 `2%` budget 对浅层更激进，对深层更合适。

## Result Interpretation

严格 top-2% relation 的两跳 closure 为 `27%-34%`，相对距离和 attention-target popularity 匹配基线仍有 `1.70x-2.95x` lift。相似 query 的 retrieval-set overlap 也高于位置匹配随机 query，lift 为 `1.87x-2.09x`。

因此，现有模型的传递性可以支持：

```text
用局部 bucket / graph 生成 coarse KV candidates，
然后在候选内部继续做 exact qK attention。
```

但它还不足以支持：

```text
把 top-2% QK relation 当成严格等价类，
仅通过一次传递闭包决定全部 KV。
```

因为约三分之二的两跳路径并不闭合。更合理的结构是把 bucket 当 coarse index，而不是直接替代 exact attention score。

## Comparison With Previous Top-2% Mass Result

此前实验报告 top-2% 平均承载约 `87.6%` mass；本次三个层的结果为 `59%-93%`。两者采样协议不同：此前集中在五个代表层和末尾少量 query，本实验覆盖三个层的全部 16 个 heads，并从位置 128 到 1024 持续采样。

这不是结论冲突，而是说明 attention sparsity 具有明显的 layer/head/query 异质性，未来 bucket 宽度应当自适应，而不应全模型固定为同一个比例。

## Claim Boundary

当前可以说，Qwen3-0.6B 的高尾 QK relation 具有显著局部闭包，足以支持 coarse indexing 的研究假设。当前不能说，现有模型已经形成了可以直接替代 attention 的严格 feature partition。

训练一个训推一致的新模型仍然合理：训练目标可以显式提高 bucket 内 QK closure、attention mass recall 和 routing margin。但它必须与独立 K-index、非共享 expert router 和 full attention 基线比较，才能说明训练出的共享 bucket 确实有额外价值。
