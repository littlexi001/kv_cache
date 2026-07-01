# Section 41: token 类型、recent 分布与谱空间相关性分析

日期：2026-06-30

## 1. 实验目的

这组实验直接回答导师提到的几个问题：

1. `sink` token 里面主要是什么信息？
2. `recent` token 是什么分布？
3. evidence token 在谱空间里有什么特征？
4. 这些 token 和其他 token 在不同奇异向量方向上的相关性是什么？

说明：这批 retrieval 合成任务没有真实 `<think>` 段，因此这里把用户说的 `Shink/Think` 按 attention `sink` token 解释，也就是序列开头固定 token。如果后续要分析真实 `<think>...</think>`，需要换成带 reasoning trace 的数据。

## 2. 实验设置

新增代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_token_type_spectral_correlations.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_token_type_spectral_correlations_server.sh
```

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/token_type_spectral_correlations_0630_v1
```

本地输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/token_type_spectral_correlations_0630_v1
```

规模：

| item | value |
|---|---:|
| model | Qwen3-0.6B |
| resolved head_dim | 64 |
| selected layers | 0, 4, 8, 13, 20, 27 |
| selected heads | 0..15 |
| variants | compact_kv, json_kv, needle_sentence, topic_table |
| tasks | 16 |
| layer/head/query rows | 3072 |
| skipped SVD rows | 0 |
| runtime | 1656.8s |

输出文件：

```text
token_type_stats.csv
direction_stats.csv
pair_direction_correlations.csv
recent_position_bins.csv
lexical_type_stats.csv
token_text_examples.csv
summary.json
```

## 3. Token 类型定义

本实验统计以下 token groups：

| group | 含义 |
|---|---|
| sink | context 开头 16 个 token |
| recent_1_16 | 距离 query 最近的 16 个历史 token |
| recent_17_64 | 距离 query 第 17 到 64 个历史 token |
| evidence_key | 目标 key token |
| evidence_label | 目标 answer label token |
| evidence_record | 包含 key/label 的整条记录 |
| evidence_any | evidence_record 的并集 |
| top2_selected | full-QK attention score 的真 top 2% token |
| other_sample | 不属于上述结构组的其他 token 样本 |

## 4. 总体结果

| group | selected rate | attention mass | cosine | avg distance | K energy top8 | top16 | top32 | abs qk top8 | top16 | top32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sink | 7.24% | 2.89% | 0.2278 | 584.9 | 62.68% | 75.93% | 87.43% | 61.74% | 73.48% | 83.98% |
| recent_1_16 | 3.80% | 0.11% | 0.0280 | 31.4 | 64.02% | 78.30% | 89.55% | 61.19% | 74.10% | 84.89% |
| recent_17_64 | 1.34% | 0.05% | -0.0190 | 63.4 | 63.54% | 77.84% | 89.41% | 60.96% | 73.88% | 84.81% |
| evidence_key | 0.57% | 0.03% | 0.0000 | 242.7 | 62.84% | 76.73% | 88.86% | 60.79% | 73.29% | 84.43% |
| evidence_label | 9.67% | 0.60% | 0.0779 | 233.5 | 59.28% | 75.45% | 88.20% | 59.40% | 72.85% | 84.08% |
| evidence_any | 1.48% | 0.13% | 0.0046 | 277.8 | 61.67% | 76.02% | 88.23% | 59.89% | 72.67% | 83.83% |
| top2_selected | 100.00% | 6.41% | 0.3121 | 181.3 | 61.55% | 75.08% | 86.76% | 60.41% | 72.84% | 83.49% |
| other_sample | 0.00% | 0.03% | 0.1132 | 550.0 | 63.53% | 76.91% | 88.37% | 61.28% | 73.37% | 84.09% |

主要观察：

1. 各类 token 的 raw K energy CDF 很接近，top32 基本都在 `86%..90%`。所以“是不是 evidence/recent/sink”不能只靠 token 自身能量 CDF 区分。
2. 区分度主要来自 q-k alignment：`top2_selected` cosine 最高，`sink` 也高；`evidence_key/evidence_any` 接近 0。
3. evidence 里真正更像被模型用到的是 `evidence_label`，不是 `evidence_key`。`evidence_label` selected rate 是 `9.67%`，明显高于 `evidence_key` 的 `0.57%`。

## 5. Sink 里面包含什么信息？

sink token 的 lexical 分布：

| category | fraction |
|---|---:|
| number | 32.42% |
| upper_alpha | 22.27% |
| lower_alpha | 21.09% |
| punct | 16.02% |
| mixed | 3.52% |

高频 token examples：

```text
0, K, =, |, 2, 3, =>, \n, -, V, {", key
```

解释：

1. sink 不是单纯语义 evidence，而是开头记录格式、分隔符、key 前缀、数字、JSON/table 结构 token 的混合。
2. sink 的 q-k cosine `0.2278`，明显高于 evidence_key/evidence_any，也高于 recent。
3. sink 和 top2_selected 的谱空间 centroid cosine 为正，并且在不同 rank 下都稳定：

| pair | r1 | r2 | r4 | r8 | r16 | r32 | r64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sink vs top2_selected | 0.290 | 0.314 | 0.379 | 0.389 | 0.366 | 0.340 | 0.329 |
| sink vs other_sample | 0.428 | 0.487 | 0.654 | 0.661 | 0.615 | 0.580 | 0.563 |

结论：sink 更像一种全局格式/位置锚点，在谱空间上和 top2 有稳定正相关，但它不等价于目标证据。

## 6. Recent 是什么分布？

top2_selected token 的位置分布：

| position bucket | top2 count share | top2 mass share | avg attention mass | avg distance |
|---|---:|---:|---:|---:|
| sink | 9.27% | 56.85% | 39.28% | 600.4 |
| recent_1_8 | 29.10% | 28.34% | 6.24% | 3.2 |
| recent_9_16 | 7.64% | 1.32% | 1.11% | 11.7 |
| recent_17_32 | 23.46% | 6.88% | 1.88% | 21.6 |
| recent_33_64 | 4.28% | 0.63% | 0.94% | 46.4 |
| middle_65_128 | 3.98% | 0.64% | 1.03% | 87.2 |
| remote_129_plus | 22.27% | 5.36% | 1.54% | 508.4 |

解释：

1. top2 token 的数量上有明显 recent 成分：最后 8 个 token 占 `29.10%`，17-32 距离段占 `23.46%`。
2. attention mass 上最强的是 sink，占 `56.85%`；recent_1_8 占 `28.34%`。
3. recent_9_16 的 count 不低，但 mass 很低；recent_17_32 有一定 count 和少量 mass。
4. remote_129_plus 仍占 `22.27%` 的 top2 token count，但 mass 只有 `5.36%`。这说明远程 token 经常进入 top2 集合，但多数不是主要 attention mass 承载者。

结论：recent 不是简单“越近越重要”的单调分布。真正高 mass 的 recent 主要是 last 1-8；17-32 更像数量型补充；远程 evidence/token 也会进入 top2，但平均 mass 较小。

## 7. Evidence token 的谱空间特征

evidence 的主要结果：

| evidence group | selected rate | attention mass | cosine | K top32 | abs qk top32 |
|---|---:|---:|---:|---:|---:|
| evidence_key | 0.57% | 0.03% | 0.0000 | 88.86% | 84.43% |
| evidence_label | 9.67% | 0.60% | 0.0779 | 88.20% | 84.08% |
| evidence_record | 1.48% | 0.13% | 0.0046 | 88.23% | 83.83% |
| evidence_any | 1.48% | 0.13% | 0.0046 | 88.23% | 83.83% |

分层上，`evidence_label` 在 layer 20 最突出：

| layer | evidence_label selected rate | attention mass | cosine | K top32 |
|---:|---:|---:|---:|---:|
| 0 | 1.4% | 0.1% | 0.212 | 96.8% |
| 4 | 1.0% | 0.0% | -0.094 | 86.6% |
| 8 | 9.0% | 0.3% | 0.099 | 81.6% |
| 13 | 9.6% | 0.6% | 0.100 | 86.3% |
| 20 | 35.2% | 2.5% | 0.100 | 90.8% |
| 27 | 2.0% | 0.0% | 0.050 | 87.1% |

解释：

1. evidence key 本身并没有强 q-k alignment；它更多像普通结构 token。
2. answer label token 更容易被选中，尤其是 layer 20。
3. evidence 的谱能量也集中在低秩方向，但这不是 evidence 独有属性；other/recent/sink 也有类似能量 CDF。
4. evidence 是否有用，关键不在“能量是否低秩”，而在它和当前 query 的 q-k alignment 是否在这些低秩方向上被放大。

## 8. 不同 token 类型在奇异向量方向上的相关性

使用每个 group 在 K-SVD basis 下的 centroid，统计不同 rank 截断下的 cosine：

| pair | r1 | r2 | r4 | r8 | r16 | r32 | r64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| evidence_any vs top2_selected | -0.178 | -0.188 | -0.135 | -0.072 | -0.046 | -0.040 | -0.038 |
| evidence_label vs top2_selected | 0.222 | 0.283 | 0.232 | 0.222 | 0.205 | 0.191 | 0.182 |
| evidence_key vs top2_selected | -0.234 | -0.254 | -0.173 | -0.108 | -0.079 | -0.068 | -0.064 |
| recent vs top2_selected | -0.330 | -0.244 | -0.116 | -0.045 | -0.006 | -0.003 | -0.005 |
| sink vs top2_selected | 0.290 | 0.314 | 0.379 | 0.389 | 0.366 | 0.340 | 0.329 |
| evidence_any vs recent | 0.294 | 0.298 | 0.294 | 0.254 | 0.221 | 0.216 | 0.214 |
| recent vs other_sample | -0.476 | -0.562 | -0.587 | -0.511 | -0.440 | -0.415 | -0.408 |

解释：

1. sink 和 top2 在谱空间上稳定正相关。
2. evidence_any/evidence_key 和 top2 的 centroid correlation 接近 0 或为负，说明 evidence 不是作为一个统一的全局方向被 top2 捕捉。
3. evidence_label 和 top2 是正相关，说明模型更直接利用 label/value 侧 token，而不是 key 侧 token。
4. recent 和 top2 在整体 centroid 上不强相关，甚至 leading directions 为负；这说明 recent 进入 top2 更多是位置/局部上下文机制，不是和 top2 共用一个稳定语义方向。

## 9. 逐奇异方向贡献

在前 16 个方向内，绝对 q-k contribution 极度集中在第 1 个方向：

| group | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top2_selected | 88.0% | 4.6% | 1.9% | 1.0% | 0.8% | 0.5% | 0.6% | 0.4% |
| sink | 89.4% | 3.7% | 1.8% | 0.9% | 0.7% | 0.5% | 0.5% | 0.4% |
| recent | 90.3% | 3.5% | 1.4% | 0.8% | 0.6% | 0.5% | 0.5% | 0.4% |
| evidence_any | 90.4% | 3.4% | 1.4% | 0.8% | 0.7% | 0.4% | 0.5% | 0.4% |
| other_sample | 90.7% | 3.1% | 1.5% | 0.8% | 0.7% | 0.4% | 0.5% | 0.4% |

这不是说 rank1 就够恢复 top2 selection。它说明前几个方向承载了大量绝对 q-k contribution，但不同 token 类型在这些方向上的符号、centroid alignment 和 query-dependent 排序仍然不同。前面 candidate recall 实验里 rank32/rank64 明显更稳，也支持这一点。

## 10. 当前结论

1. `sink` 是结构/位置锚点，不是目标 evidence；它和 top2 谱方向稳定正相关，并承担大量 attention mass。
2. `recent` 的有效部分主要是 last 1-8 token；17-32 有数量贡献但 mass 较弱；远程 token 会进入 top2，但大多不是 mass 主体。
3. `evidence_key` 本身不显著；`evidence_label` 才是更容易进入 top2 的 evidence 类型，尤其在中深层 layer 20。
4. evidence/recent/sink 的 K 能量 CDF 都低秩，但这不是区分 token 类型的关键。真正关键的是 q-k alignment 和不同奇异方向上的 centroid correlation。
5. 如果做下一步方法，不能只“保护 evidence span 全部 token”；更合理的是 label/value-aware 或 query-conditioned evidence rescue，并和 sink + short recent 保护结合。

