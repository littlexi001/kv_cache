# QK Attention Transitivity: Experiment Design

## Experiment A: Weight Kernel Diagnostics

### Algorithm

对每个 layer/query-head：

1. 取对应的 `W_Q,h` 和 GQA 映射后的 `W_K,g`；
2. 计算 `M_h = W_Q,h^T W_K,g`；
3. 测量 `M_h` 与 `M_h^T` 的 cosine、skew-energy ratio；
4. 对随机单位向量测量二次型 `u^T M_h u` 的负值比例；
5. 测量 `W_Q,h` 与 `W_K,g` row space 的 principal-angle cosine。

### Interpretation

- 高 symmetry、低 negative quadratic fraction、row-space 高重合：支持共同内积空间近似；
- 否则，参数本身不支持全局传递性，但数据流形上的经验闭包仍可能存在。

## Experiment B: Directed Two-Hop Closure

严格历史 attention 图定义为：

```text
E_k = {(i, j): j is in top-k of row i and j < i}
```

对每条两跳路径：

```text
l -> i -> j
```

检查是否存在闭包边：

```text
l -> j
```

指标：

1. `closure_rate`：两跳 endpoint `j` 也在 `l` 的 top-k 中的比例；
2. `uniform_baseline`：从 `i` 的全部历史位置均匀采样 endpoint 时的期望 closure；
3. `distance_matched_baseline`：从与 `i-j` 距离相同的 log2 distance bucket 中采样 endpoint；
4. `distance_popularity_matched_baseline`：同时匹配距离 bucket 和 endpoint 的 attention in-degree bucket，排除 sink / hub 造成的伪闭包；
5. `closure_lift_vs_distance_popularity`；
6. endpoint 在 `l` attention row 中的 percentile 与 softmax mass。

默认参数：

| 参数 | 值 | 原因 |
|---|---:|---|
| `top_k` | 16 | 1024-token micro-test 中足够稀疏，同时提供足够两跳路径 |
| `min_query_index` | 128 | 避免短 prefix 的 top-k 比例过大 |
| `query_stride` | 8 | 控制运行时间，同时覆盖不同位置 |
| `distance_samples` | 4 | 为每条路径提供低成本位置匹配基线 |

## Experiment C: Similar-Query Retrieval Stability

对每个 query `l`：

1. 在过去 query 中寻找 post-RoPE query cosine 最近的 `i`；
2. 把 `l` 和 `i` 的 attention row 限制到共同历史 `[0, i)`；
3. 比较两者 top-k retrieval set 的 Jaccard 与 attention distribution JS divergence；
4. 与时间距离匹配的随机 query 比较。

这个实验直接测试：后来的 query 若与旧 query 相似，是否会重新检索旧 query 使用过的 token。

## Pass / Fail / Insufficient Evidence

### Pass

至少中层或深层的一组 heads 同时满足：

1. two-hop closure 相对 distance-and-popularity-matched baseline 的 lift 明显大于 1；
2. similar-query retrieval overlap 明显高于随机 query；
3. 结果不是只由 self、最近邻或单一 attention sink 产生；
4. heatmap 与定量结果一致。

### Fail

大多数 heads 的 closure 与 distance-matched baseline 接近，且 query-neighbor retrieval overlap 不高于随机。

### Insufficient evidence

只有少数 query/head 为正，或者结果对 top-k、文本、层选择非常敏感。

## Debug Artifacts

- `weight_kernel_metrics.csv`
- `transitivity_metrics.csv`
- `query_stability_metrics.csv`
- `representative_paths.csv`
- `attention_heatmap_layer*_head*.png`
- `summary.json`
- `docs/visualization_results.md`
