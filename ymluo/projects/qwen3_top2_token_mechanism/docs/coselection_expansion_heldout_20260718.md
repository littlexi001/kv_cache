## 1. 目标

验证 Top-2% token 共选结构能否用于实际候选扩展，而不只是一项相关性统计。

实验采用严格时间切分：前 256 个 query 只用于构建共选图，后 256 个 query 只用于测试。每个 token 只保留固定数量的共选邻居，运行时只访问 `seed 数量 × 邻居数` 条边，不扫描完整 `N × N` 图。最终候选预算仍固定为历史长度的 2%、4% 或 8%。

## 2. 方法

对训练段中 token `i` 和 `j` 的共选关系，使用以下正超额作为边权：

`w(i,j) = max(P(j|i) - P(j), 0)`

给定 seed 集合后，累加所有 seed 指向同一 token 的边权，按累加分数扩展候选。候选不足时，使用训练段的 token 选择频率补齐。

比较方法：

- `frequency`：seed 之外只按训练段选择频率补齐；
- `local16`：优先补 seed 附近正负 16 个位置；
- `graph_sum`：按共选图边权之和扩展；
- `previous_graph`：不读取当前 query 的 seed，只用上一 query 的 Top-2% 做因果扩展。

## 3. 当前 query seed 的 held-out 上界

本实验随机暴露当前真实 Top-2% 中的 25% 作为 seed，即只暴露历史长度的约 0.5%。它是共选扩展的能力上界，不是当前已经落地的检索器。

| 数据 | 候选预算 | Frequency recall | Co-selection recall | 提升 |
|---|---:|---:|---:|---:|
| War and Peace，1K，全 448 heads | 2% | 44.69% | **49.37%** | **+4.68 pp** |
| War and Peace，1K，全 448 heads | 4% | 53.84% | **60.48%** | **+6.64 pp** |
| War and Peace，1K，全 448 heads | 8% | 64.02% | **70.85%** | **+6.83 pp** |
| Monte Cristo，1K，全 448 heads | 2% | 45.44% | **48.92%** | **+3.48 pp** |
| Monte Cristo，1K，全 448 heads | 4% | 54.33% | **60.08%** | **+5.75 pp** |
| Monte Cristo，1K，全 448 heads | 8% | 64.14% | **70.59%** | **+6.45 pp** |
| War and Peace，4K，16 heads | 2% | 45.91% | **49.28%** | **+3.37 pp** |
| War and Peace，4K，16 heads | 4% | 55.91% | **62.52%** | **+6.61 pp** |
| War and Peace，4K，16 heads | 8% | 66.82% | **75.92%** | **+9.10 pp** |

结果说明，共选图包含频率和局部位置无法解释的 held-out 信息，并且在两本文本以及 4K 上下文中复现。

但 25% Oracle seed 加 8% 候选仍只有约 71%--76% 的 Top-2% 位置召回，低于当前 PCA64 候选约 98%的水平。因此不能直接用共选图替换主检索器。

## 4. 完全因果的下一 query 预取

这一设置只使用上一 query 已经选中的 Top-2%，不访问当前 query 的任何 Oracle 信息。

| 数据 | 候选预算 | 上一 query + Frequency | 上一 query + Co-selection | 提升 |
|---|---:|---:|---:|---:|
| War and Peace，1K | 4% | 53.67% | **54.51%** | **+0.84 pp** |
| War and Peace，1K | 8% | 63.64% | **65.40%** | **+1.76 pp** |
| Monte Cristo，1K | 4% | **53.09%** | 52.85% | -0.24 pp |
| Monte Cristo，1K | 8% | 63.15% | **64.02%** | **+0.87 pp** |
| War and Peace，4K | 4% | 62.45% | **63.13%** | **+0.68 pp** |
| War and Peace，4K | 8% | 72.43% | **73.92%** | **+1.49 pp** |

因果增益稳定但较小。它适合用于 GPU hot KV 预取或 cache replacement priority，不能单独承担当前 query 的完整检索。

## 5. 真实 PCA seed 的 32K 结果

进一步使用 Llama-3.1-8B 的 Sports 和 Medicine 32K Q/K trace。前 8 个 decode step 建图，后 8 步测试；seed 不再来自 Oracle，而是 PCA16 或 PCA64 排名的前 0.5%。每个主题包含 5 层、32 query heads，共 1,280 个 held-out head-step。

Top-8 邻接表的关键结果如下：

| 数据与分数 | 候选预算 | Seed + Frequency | Seed + Co-selection | 图的增益 |
|---|---:|---:|---:|---:|
| Sports PCA16 | 4% | 79.74% | **79.88%** | +0.14 pp |
| Sports PCA16 | 8% | 84.21% | 84.21% | 0.00 pp |
| Sports PCA64 | 4% | 80.65% | **80.76%** | +0.11 pp |
| Sports PCA64 | 8% | 84.61% | 84.61% | 0.00 pp |
| Medicine PCA16 | 4% | 74.07% | **74.28%** | +0.21 pp |
| Medicine PCA16 | 8% | 81.18% | 81.18% | 0.00 pp |
| Medicine PCA64 | 4% | 75.32% | **75.54%** | +0.22 pp |
| Medicine PCA64 | 8% | 81.95% | 81.95% | 0.00 pp |

将 Medicine 邻接表扩大到 Top-32 后，4%候选的增益增加到 PCA16 `+0.52 pp`、PCA64 `+0.75 pp`，但 8%候选仍只有 `+0.01 pp`，2%候选还会退化。因此容量不是主要限制。

真实 seed 下，pairwise 共选图的收益远小于 Oracle seed 上界，不足以抵消邻接表存储和在线扩展成本。当前不应把它接入主 CUDA 路径。

这组实验还发现，一阶选择频率比二阶共选图更有效。例如 Medicine 的 PCA16 直接 4%候选召回为 67.30%，使用 0.5% PCA16 seed 加训练段频率补齐后达到 74.07%，而共选图只进一步达到 74.28%。该现象值得作为低成本 per-head persistent prior 单独测试，但它与 H2O 一类累计注意力思想相关，不能单独作为论文核心创新。

所有配置的 attention mass 都超过 97.6%，多数超过 99.7%。这说明位置召回损失主要来自低质量 Top-2% 边界 token，最终判断仍需 PPL 和任务分数，不能只看位置召回。

## 6. 完全因果的 seed 采样

固定采样历史长度的 0.5%，不允许读取当前 query 的 Oracle Top-2%。测试三种 seed 来源：全历史均匀采样、按训练段选择频率加权采样、从上一 query 的 Top-2% 中随机抽取四分之一。

下表为8%候选预算的结果：

| 数据 | Seed来源 | Frequency补齐 | Co-selection扩展 | 图的增益 |
|---|---|---:|---:|---:|
| War 1K | 均匀采样 | **49.26%** | 47.37% | -1.89 pp |
| War 1K | 历史加权 | **50.07%** | 48.77% | -1.30 pp |
| War 1K | 上一步子集 | 54.01% | **55.78%** | **+1.77 pp** |
| Monte 1K | 均匀采样 | **49.45%** | 47.52% | -1.93 pp |
| Monte 1K | 历史加权 | **50.19%** | 48.84% | -1.35 pp |
| Monte 1K | 上一步子集 | 53.96% | **55.74%** | **+1.78 pp** |
| War 4K | 均匀采样 | **55.06%** | 54.23% | -0.83 pp |
| War 4K | 历史加权 | **55.89%** | 55.57% | -0.32 pp |
| War 4K | 上一步子集 | 60.05% | **62.25%** | **+2.20 pp** |

随机从全历史采 seed 的命中率太低，不能激活有用的共选边。只有从上一 query 已确认的重要区域采样才有稳定收益，但在4K上使用完整上一 query Top-2% 的共选扩展可以达到73.92%召回，明显高于只采四分之一的62.25%。上一 query 的选择本来已经在系统中，因此没有必要为了降低少量图访问而丢弃其中四分之三。

### 6.1 真实 PCA 分数采样

在 Llama-3.1-8B、32K Sports/Medicine trace 上进一步测试了六种真实 seed：确定性 Top、均匀采样、PCA 分数 Gumbel 采样、70% Top + 30%分数采样、70% Top + 30%不确定区采样。所有方法保持0.5% seed和4%/8%最终候选预算。

结果是：凡是用采样替换确定性 Top seed，平均召回都会下降。最接近的混合不确定区策略在 Sports PCA16、4%候选下为79.65%，仍低于纯 Top 的79.88%；Medicine 为73.86%，低于74.28%。

随后保留全部0.5% Top seed，额外加入0.1%探索 seed，同时保持最终候选预算不变。该版本获得小幅提升，但同样数量直接取 next-highest PCA token 更好：

| 数据 | 候选 | 原0.5% Top + 图 | Top + 0.1% band采样 + 图 | Top + 0.1% next-highest + 图 |
|---|---:|---:|---:|---:|
| Sports PCA64 | 4% | 80.76% | 81.13% | **81.35%** |
| Sports PCA64 | 8% | 84.61% | 84.99% | **85.08%** |
| Medicine PCA64 | 4% | 75.54% | 75.98% | **76.40%** |
| Medicine PCA64 | 8% | 81.95% | 82.40% | **82.57%** |

尾部指标也支持确定性 next-highest。例如 Sports PCA64、8%候选的 P10 为69.95%/70.31%/70.92%，最差样本为45.47%/46.41%/48.44%，顺序分别对应原 Top、band采样和 next-highest。

因此采样没有提供独立于 PCA 排名的收益。小幅改善来自增加0.1%的高分 seed，而不是随机探索。当前不应继续增加采样器、温度或随机重复次数；更简单的做法是直接调整确定性 seed 数量。

## 7. 邻接表容量

在 4K、25% seed、8%候选预算下：

| 每 token 邻居数 | Top-2% recall |
|---:|---:|
| 频率基线 | 66.82% |
| 1 | 67.85% |
| 2 | 68.62% |
| 4 | 69.75% |
| 8 | 71.32% |
| 16 | 73.42% |
| 32 | **75.92%** |

Top-8 邻接表获得了 Top-32 大约一半以上的额外收益，可以作为第一版低内存实现。建图代码已经由稠密 `N × N` 改成 CSR 稀疏共现累积，保存阶段只有固定邻接表。

## 8. 对当前系统的判断

最合理的组合方式是：

1. 谱增量分数首先找少量高置信 seed；
2. 强共选 head 使用 Top-8 邻接表扩展候选；
3. 剩余位置由谱分数补齐；
4. 在固定候选预算内做 exact rerank；
5. 上一 query 的共选邻居只作为异步 KV 预取提示。

当前结果证明了图扩展的 held-out Oracle-seed 上界和较弱的因果预取收益；真实 PCA seed 下，图相对一阶频率先验只有不到 1 个召回点的增益。因此暂不继续实现 CUDA/CSR 在线图扩展。

这条探索对主方法仍有两个有用结论：

1. 上一 query 的共选邻居可以作为低优先级异步预取提示，但不应占用主候选预算；
2. PCA16 加 per-head 历史选择频率是更简单的候选混合方式，下一步可直接测试 PPL 和端到端速度。

## 9. 代码与结果

代码：

```text
src/evaluate_coselection_expansion.py
src/evaluate_causal_seed_sampling.py
src/evaluate_pca_coselection_hybrid.py
tests/test_coselection_expansion.py
tests/test_causal_seed_sampling.py
tests/test_pca_seed_sampling.py
```

结果：

```text
artifacts/20260718_coselection_expansion_n1024_war/
artifacts/20260718_coselection_expansion_n1024_monte/
artifacts/20260718_coselection_expansion_n4096_war/
artifacts/20260718_coselection_expansion_n4096_neighbors1/
artifacts/20260718_coselection_expansion_n4096_neighbors2/
artifacts/20260718_coselection_expansion_n4096_neighbors4/
artifacts/20260718_coselection_expansion_n4096_neighbors8/
artifacts/20260718_coselection_expansion_n4096_neighbors16/
artifacts/20260718_causal_seed_sampling_n1024/
artifacts/20260718_causal_seed_sampling_n4096/
artifacts/20260718_pca_coselection_hybrid_32k/
```
