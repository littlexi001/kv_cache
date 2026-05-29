# Qwen3 K-cache Top-k Similarity Graph Probe Findings

Date: 2026-05-29

## 0. 文档定位

这个文档记录第一轮 K-cache similarity graph probe 的结果。它对应的问题是：

> 如果把每层 K-cache 看成 token 节点集合，那么每个 token 在历史 token 中是否存在相似的 nearest neighbors？这种相似性是否足够强，值得继续研究 K graph / cluster / index routing？

本轮不是在验证最终 KV compression，也不是在验证 query-time attention selector。它只回答一个更前置的 sanity check：

```text
K-cache 的 causal nearest-neighbor similarity 是否明显非零？
raw similarity 中有多少可能来自 common direction？
top-k 和 centering 会怎样改变分布？
```

## 1. 实验配置

输出目录：

```text
fdong_seq_compress/outputs/k_similarity_batch_1000_20260529_123656
```

实验设置：

```text
model: fdong/Qwen3-0.6B
text: fdong_seq_compress/data/synthetic_texts/long_english_article_01.txt
requested max length: 1000 tokens
actual sequence length: 788 tokens
layers: 0-27
analysis level: token-level
token-level K: concat over all KV heads within a layer
similarity: cosine
top-k: 5, 10, 20
centering: off / on
causal constraint: for token i, only compare with j < i
self similarity: excluded
```

实际全层样本数：

```text
top-5:  109900
top-10: 219100
top-20: 435400
```

这些数值来自 28 层乘以每个 token 在 causal past 中可用的 top-k 数量。注意：这次 batch 名称里有 `1000`，但默认文本 `long_english_article_01.txt` 只有 788 tokens，因此本轮结果应被理解为 788-token probe。后续脚本默认文本已改为 `long_english_12000_words.txt`，用于保证能真实截取 1000 tokens。

## 2. 最重要结论

第一结论：

> K-cache 的 token-level causal nearest-neighbor similarity 不是接近 0。即使去中心化之后，top-1 / top-5 邻居仍然有明显正 cosine。因此，K similarity graph 方向没有被第一轮 sanity check 否掉。

第二结论：

> raw K cosine 非常高，但 centering 后大幅下降。这说明 raw nearest-neighbor 分布强烈受到 common direction / anisotropy 影响，不能直接把 raw 高相似解释成有意义的 graph cluster。

第三结论：

> top-k 增大时，平均相似度按预期下降；但 centered top-20 仍有正相似结构。也就是说，K 空间里可能不只是单个最近邻，而是存在一定宽度的 local neighborhood，但这个 neighborhood 是否能预测 attention 还没有验证。

## 3. 全局统计

下面统计聚合了所有 28 层。

| setting | count | mean | p05 | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top-5, raw | 109900 | 0.8619 | 0.7112 | 0.8727 | 0.9850 | 0.9989 |
| top-10, raw | 219100 | 0.8410 | 0.6735 | 0.8515 | 0.9831 | 0.9989 |
| top-20, raw | 435400 | 0.8182 | 0.6324 | 0.8258 | 0.9812 | 0.9989 |
| top-5, centered | 109900 | 0.5584 | 0.3571 | 0.5449 | 0.8100 | 0.9962 |
| top-10, centered | 219100 | 0.4955 | 0.2989 | 0.4757 | 0.7719 | 0.9962 |
| top-20, centered | 435400 | 0.4264 | 0.2352 | 0.4005 | 0.7254 | 0.9962 |

Interpretation:

- Raw top-k cosine 很高，top-20 的中位数仍有 0.8258。
- Centering 后 top-k cosine 明显降低，但 top-5 中位数仍有 0.5449，top-20 中位数仍有 0.4005。
- 这说明 common direction 很强，但不是全部结构；去掉均值后，K-cache 仍保留可观 nearest-neighbor geometry。

## 4. 高相似样本比例

用 histogram bin 近似统计不同阈值以上的比例：

| setting | >=0.5 | >=0.7 | >=0.8 | >=0.9 | >=0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| top-5, raw | 0.999 | 0.964 | 0.716 | 0.396 | 0.228 |
| top-10, raw | 0.999 | 0.906 | 0.621 | 0.349 | 0.192 |
| top-20, raw | 0.998 | 0.811 | 0.553 | 0.318 | 0.165 |
| top-5, centered | 0.634 | 0.156 | 0.056 | 0.014 | 0.009 |
| top-10, centered | 0.428 | 0.096 | 0.036 | 0.011 | 0.007 |
| top-20, centered | 0.253 | 0.061 | 0.023 | 0.008 | 0.005 |

Interpretation:

- Raw K 中大部分 nearest neighbors 都高度相似，这和之前发现的 K anisotropy / cone effect 一致。
- Centering 后，仍有 63.4% 的 top-5 neighbor cosine 大于 0.5，但只有 15.6% 大于 0.7。
- 因此，后续如果构图，threshold 不能直接从 raw cosine 继承；centered space 里更合理的阈值可能在 0.5-0.7 区间试探。

## 5. Top-k 半径效应

按 rank 看，跨层平均的 rank-k mean 如下：

| setting | rank1 mean | rank2 mean | rank5 mean | rank10 mean | rank20 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.8992 | 0.8717 | 0.8370 | 0.8110 | 0.7846 |
| centered | 0.6713 | 0.5878 | 0.4832 | 0.4053 | 0.3240 |

Interpretation:

- Raw 空间里 top-20 仍然很高，说明 raw graph 会非常 dense，可能难以区分真正有意义的邻居。
- Centered 空间里 rank 增大后相似度下降更快，说明 centered graph 更有选择性。
- 如果做第一版 graph，应该优先从 centered top-5 或 centered top-10 开始，而不是 raw top-20。

## 6. 层间差异

Layer-level variation 很明显。

Raw top-5 mean 最高的层：

| layer | topk mean | p50 | p95 | rank1 mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9891 | 0.9899 | 0.9973 | 0.9924 |
| 1 | 0.9854 | 0.9842 | 0.9944 | 0.9879 |
| 5 | 0.9731 | 0.9745 | 0.9867 | 0.9785 |
| 2 | 0.9687 | 0.9684 | 0.9839 | 0.9750 |
| 10 | 0.9518 | 0.9538 | 0.9740 | 0.9639 |

Raw top-5 mean 最低的层：

| layer | topk mean | p50 | p95 | rank1 mean |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 0.7461 | 0.7416 | 0.8557 | 0.8124 |
| 13 | 0.7562 | 0.7555 | 0.8738 | 0.8298 |
| 26 | 0.7608 | 0.7580 | 0.8719 | 0.8198 |
| 16 | 0.7684 | 0.7660 | 0.8666 | 0.8348 |
| 20 | 0.7827 | 0.7804 | 0.8814 | 0.8446 |

Centered top-5 mean 最高的层：

| layer | topk mean | p50 | p95 | rank1 mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8280 | 0.8345 | 0.9886 | 0.8766 |
| 27 | 0.6330 | 0.6251 | 0.8198 | 0.7605 |
| 17 | 0.6315 | 0.6326 | 0.8372 | 0.7574 |
| 25 | 0.5945 | 0.5914 | 0.7941 | 0.6942 |
| 19 | 0.5905 | 0.5860 | 0.7857 | 0.7241 |

Centered top-5 mean 最低的层：

| layer | topk mean | p50 | p95 | rank1 mean |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.4644 | 0.4412 | 0.6907 | 0.5890 |
| 3 | 0.4838 | 0.4446 | 0.7915 | 0.5826 |
| 8 | 0.4844 | 0.4680 | 0.7217 | 0.6201 |
| 1 | 0.4900 | 0.4500 | 0.7654 | 0.5820 |
| 4 | 0.5019 | 0.4819 | 0.7391 | 0.6011 |

Interpretation:

- Layer 0 在 raw 和 centered 下都极高，可能主要反映非常强的局部/词法平滑，而不一定是有用的长程 retrieval structure。
- 一些中后层在 centered 后仍保持较高 top-k similarity，例如 layers 17, 19, 25, 27。这些层可能更值得做 graph retrieval probe。
- 中间层有些 centered similarity 较低，例如 layers 3, 6, 8。说明不能假设所有层共享同一种 graph index 参数。

## 7. 当前能支持的 claim

本轮结果支持：

> Qwen3-0.6B 的 K-cache 在 token-level layer representation 上存在明显 causal nearest-neighbor structure；这种结构在去均值后仍然存在，因此 K similarity graph 作为离线分析对象值得继续推进。

本轮结果也支持：

> Raw K similarity 不能直接作为 graph edge score，因为 raw top-k cosine 过高，强烈受到 common direction 影响。后续应优先使用 centered / PC-removed / whitened K 空间。

本轮结果不支持或尚不能支持：

- 不能说明相似 K token 的 V 可以被平均或替代。
- 不能说明 K-K similarity graph 能预测 q-K attention。
- 不能说明这些 nearest neighbors 是语义 cluster；它们目前只能被称为 K-address-space neighbors。
- 不能说明 graph index 会带来系统收益，因为还没有 attention mass recall / CE delta / latency 评估。

## 8. 下一步实验

### Experiment 1: Neighbor distance distribution

当前 batch 没有保存 `topk_neighbors.csv`，所以还不知道 nearest neighbors 主要是相邻 token，还是存在跨距离连接。

建议先跑：

```bash
MAX_TOKENS=1000 \
LAYERS=all \
TOP_K=5 \
CENTER_TOKENS=1 \
SAVE_NEIGHBORS=1 \
OUTPUT_DIR=fdong_seq_compress/outputs/k_similarity_neighbors_top5_centered1 \
bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```

要看：

```text
token_distance distribution
fraction of neighbors within 1/2/4/8/16/32 tokens
fraction of long-range neighbors > 128 tokens
```

如果大部分 centered neighbors 只是 adjacent/local tokens，那么 graph idea 更像 local smoothness / block compression；如果有显著 long-range neighbors，才更像 nonlocal K graph。

### Experiment 2: Random baseline

需要比较真实 K 的 top-k 分布与随机 baseline：

```text
row-permuted K
Gaussian vectors with same dimension
norm-preserving random directions
```

目标是确认 observed top-k 不是高维 nearest-neighbor 极值效应。

### Experiment 3: PC removal and whitening

Centering 只去掉均值，但 common direction 可能不止一个。下一步应比较：

```text
raw K
centered K
remove top-1 / top-4 / top-8 PCs
whitened K
```

如果 PC-removed 后仍有 stable top-k neighborhood，K graph 的证据会更强。

### Experiment 4: Head-level analysis

本轮 token-level 是 concat heads，可能掩盖 head specialization。下一步应跑：

```bash
MAX_TOKENS=1000 \
ANALYSIS_LEVEL=head \
TOP_K=5 \
CENTER_TOKENS=1 \
bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```

要找：

```text
which layers/heads retain high centered top-k similarity
whether only a few heads carry graph-like structure
whether some heads are mostly local while others are nonlocal
```

### Experiment 5: Attention alignment

最终关键不是 K-K similarity，而是 K graph 能不能帮助 query retrieval。

下一步应对真实 forward 取 Q/K，测试：

```text
for each query q_t:
  true top attention tokens from q_t · K_past
  graph candidate tokens from K nearest-neighbor clusters
  measure attention mass recall / top-token recall
```

最小版本可以先不建复杂图，只测：

```text
Does q's top K token lie in a high-similarity K neighborhood?
Can cluster representatives recall high-attention tokens?
```

如果这个实验不成立，K-K graph 即使几何上存在，也未必适合作为 attention selector。

## 9. 当前研究判断

这轮实验的结论偏正面，但不是决定性证据。

较稳的判断是：

> K-cache graph 方向值得继续做，但 edge score 需要在 centered / PC-removed / whitened 空间里定义；raw cosine 太容易形成过密图。

下一步最优先的不是马上实现 online graph index，而是先做：

```text
centered top-k neighbor distance analysis
random baseline
head-level centered analysis
attention alignment
```

其中 `neighbor distance analysis` 是最便宜、信息量最高的下一步；它能区分当前看到的 nearest-neighbor structure 是局部平滑，还是有可能形成真正的非局部 graph memory。
