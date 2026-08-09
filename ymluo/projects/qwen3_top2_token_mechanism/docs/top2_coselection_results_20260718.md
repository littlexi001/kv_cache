# 同一 Attention Head 内 Top-2% Token 共选与聚类实验

日期：2026-07-18
模型：Qwen3-0.6B
主文本：War and Peace、The Count of Monte Cristo

## 一句话结论

确实存在明显的 token 共选结构：对同一个 head，如果 token A 被选入 Top-2%，某些 token B 被同时选中的概率可达到 60%–100%，远高于固定 2% 随机预算约 1.96% 的基线。该现象在不同文本、正文偏移、稀疏 query 采样以及保留时间自相关的循环移位零假设下仍然存在，但强度有明显的 layer/head 差异。

## 1. 用户问题如何转化为统计量

固定一段包含 `N` 个历史 token 的 context。对同一个 layer/head，在 `Q` 个后续 query 上记录其 Top-2% 选择结果：

```text
X[q, i] = 1  当且仅当第 q 个 query 将历史 token i 选入该 head 的 Top-2%
```

然后计算完整的 `N × N` 共选矩阵：

```text
C[i, j] = sum_q X[q, i] X[q, j]
P(j | i) = C[i, j] / sum_q X[q, i]
```

其中 `P(j | i)` 就是“token i 被选中时，token j 也被选中的概率”。

只看条件概率会把两个各自都高频的 token 误判为一个 cluster，因此同时报告：

- `lift(i,j) = P(j|i) / P(j)`：校正 token j 自身的选择频率；
- `phi`：两个二值选择变量的相关系数；
- hypergeometric + BH FDR：在固定边际频率后检验正关联；
- circular-shift null：保留每个 token 自身的选择频率和连续选择段，只破坏两个 token 的同 query 相位；
- 位置距离富集：检查 cluster 是局部连续 span，还是远距离同类 token。

## 2. 主实验设置

### N=1024 全 head 实验

- context：正文中从 token offset 10,000 开始的 1,024 tokens；
- query observations：后续 512 tokens；
- 每个 head 的固定预算：`ceil(0.02 × 1024) = 21`；
- 层数：28；每层 query heads：16；总计 448 heads；
- 每个 head 都实际计算完整 `1024 × 1024` 共选矩阵；
- 为避免保存 `448 × N²` 的冗余数据，完整矩阵只保存代表 head；所有 head 保存统计摘要和 Top pair。原始 selection indices 足以重建任意 head 的完整矩阵。

固定预算均匀随机选择下，已知 A 被选中时 B 被选中的概率为：

```text
(21 - 1) / (1024 - 1) = 1.955%
```

### N=4096 验证

- context：4,096 tokens；
- query observations：512；
- 每个 head 预算：82；
- 选择4层×4 heads，共16个强弱混合 head；
- 随机条件概率基线：`81 / 4095 = 1.978%`。

## 3. N=1024 全 head 主结果

| 指标 | War and Peace | Monte Cristo |
|---|---:|---:|
| 分析 heads | 448 | 448 |
| 含显著正关联 pair 的 heads | 448 | 448 |
| 每 head 显著 pair 中位数 | 815 | 716.5 |
| 显著 pair 条件概率中位数 | 60.0% | 60.0% |
| 显著 pair lift 中位数 | 4.725× | 5.565× |
| 超出边际独立预期的共选质量 | 44.93% | 43.34% |
| 距离≤16的共选富集 | 2.149× | 2.206× |

这里最重要的是 lift：显著 pair 不只是因为 A、B 分别经常被选中；在校正二者边际频率后，它们仍比独立选择多共同出现约4.7–5.6倍。

## 4. 跨文本是否稳定

把448个 head 在两本文本上的统计逐 head 对齐：

| Head-level 指标 | Spearman 相关 |
|---|---:|
| cluster score | **0.943** |
| excess pair mass | **0.957** |
| 显著 pair 数 | **0.942** |
| 最大 component 大小 | **0.932** |
| 距离≤16富集 | **0.923** |

两本文本各自 Top-50 cluster heads 中有41个相同，Jaccard 为0.695。

因此稳定的是：

> 某个 layer/head 是否具有强共选倾向、偏局部 cluster 还是偏分散 cluster。

并不意味着不同文本的具体 token A/B 身份相同。具体 cluster 仍由当前输入决定。

![跨文本 head cluster score](../artifacts/20260718_coselection/figures_summary/cross_corpus_cluster_score.png)

![Layer/head 共选图谱](../artifacts/20260718_coselection/figures_summary/layer_head_coselection_atlas.png)

## 5. 排除相邻 query 时间自相关

512个相邻 query 会共享语境，所以普通独立样本检验会偏乐观。额外每8个 query 只取1个，共64个 observations：

| Stride-8 正文实验 | War and Peace | Monte Cristo |
|---|---:|---:|
| 仍含显著 pair 的 heads | 266/448 (59.4%) | 218/448 (48.7%) |
| 在这些 head 中，条件概率中位数 | 95.45% | 96.88% |
| 在这些 head 中，lift 中位数 | 3.467× | 3.879× |
| excess-mass head 排名跨文本相关 | \- | 0.960 |
| cluster-score head 排名跨文本相关 | \- | 0.476 |

观察数从512降到64后统计功效明显降低，所以不能拿显著 pair 数直接和主实验比较。关键是仍有约一半以上 head 能通过全 `N choose 2` 多重校正，而且强度排序的一部分可以跨文本复现。

## 6. 保留每个 token 时间连续性的零假设

对两本文本分别抽取16个代表 head：8个主实验 cluster score 最高、8个最低。对每个 token 的选择时间序列独立做循环平移：

- 保留每个 token 被选择多少次；
- 保留该 token 连续被选择的 run pattern；
- 只破坏 A 和 B 在同一个 query 上同步出现的相位。

每个 head 运行64次置换：

| 结果 | War and Peace | Monte Cristo |
|---|---:|---:|
| 测试代表 heads | 16 | 16 |
| 经验 `p ≤ 0.05` | 16/16 | 16/16 |
| low-cluster 组 observed-null 中位差 | +4.49 percentage points | +4.10 pp |
| high-cluster 组 observed-null 中位差 | +10.22 pp | +15.03 pp |
| 最小可分辨经验 p | 1/65 = 0.0154 | 1/65 = 0.0154 |

因此共选不能只用“两个 token 都很高频”或“每个 token 自己在相邻 query 中保持稳定”解释；pair 之间存在额外同步结构。

## 7. N=4096 是否仍成立

16个强弱混合 head 的4K验证结果：

| 指标 | 结果 |
|---|---:|
| 含显著正关联 pair | 16/16 |
| 显著 pair 数中位数 | 10,383 |
| 显著 pair 条件概率中位数 | 63.16% |
| 显著 pair lift 中位数 | 3.824× |
| excess pair mass 中位数 | 36.52% |
| 距离≤16富集中位数 | 1.992× |

这说明结果不是 N=1024 或预算21造成的特殊现象。N扩大到4096、预算扩大到82后，显著 pair 的典型条件概率仍约63%，远高于1.98%的随机固定预算基线。

## 8. 具体 pair 例子

下面均来自 War and Peace 正文 offset 10K，N=1024、Q=512。

| Head | Token A → B | 距离 | P(B\|A) | P(A\|B) | Lift | 类型 |
|---|---|---:|---:|---:|---:|---|
| L5H12 | `menace` → `everything` | 2 | 95.52% | 91.43% | 6.99× | 局部连续 span |
| L5H12 | `menace` → `to` | 1 | 92.54% | 92.54% | 7.07× | 局部连续 span |
| L4H13 | newline@795 → newline@924 | 129 | 100% | 100% | 19.69× | 远距离格式同类 |
| L3H10 | `after`@224 → `g`@910 | 686 | 84.85% | 96.55% | 14.98× | 远距离分散 motif |
| L10H1 | paragraph-end@331 → paragraph-end@698 | 367 | 92.63% | 92.18% | **1.16×** | 两者都高频，额外关联较弱 |

最后一行解释了为什么不能只看条件概率：虽然 `P(B|A)` 也是92.6%，但 B 本身几乎总被选中，所以 lift 只有1.16；它和前面 lift 7–20× 的 pair 不是同一种强 cluster。

## 9. 形成了哪几类 cluster

目前至少能看到三类：

1. **局部 span cluster**：相邻词或同一短语一起进入 Top-2%，L5H12 的距离≤16富集很高；
2. **格式/词法同类 cluster**：分散在文本各处的 newline、标点、`the` 等同步被选择；
3. **远距离 mixed motif**：相距几十到数百 token 的位置形成稳定小组，具体语义需要进一步结合 token role 和 head 功能标签解释。

不是所有 head 的结构都一样。正文跨文本平均 cluster score 最高的包括 L5H12、L3H10、L4H13、L16H14、L16H11；较弱的包括 L10H1、L13H13、L26H10、L25H2。弱 head 仍可能有少量稳定 pair，但关联质量和图结构规模明显更小。

## 10. 对外部 KV retrieval 的意义

结果支持把 token 选择从独立二分类改成条件化/分组问题：

```text
先找到少量 seed token A
    ↓
根据该 layer/head 的共选图扩展候选 B
    ↓
在扩展集合内精排，最后仍截断到2%预算
```

可能的实际收益：

- 局部 span head：选中一个 token 后可低成本补邻近 block；
- 格式/结构 head：可按 newline、边界、同类 token 建候选组；
- 远距离 motif head：可用 head-specific token affinity 或低秩特征扩展；
- 弱 cluster head：继续独立检索，避免无意义扩展。

但当前结果还不能证明“cluster expansion 可以替代 Oracle Top-2%”。下一步需要严格 train/test：用前半 query 学共选图，在后半 query 只给少量 seed，测 Top-2%位置召回、attention mass召回、PPL和下游准确率，并保持最终预算仍为2%。

## 11. 输出与代码

核心代码：

```text
src/coselection_analysis.py
src/run_top2_coselection.py
src/run_coselection_circular_null.py
src/compare_coselection_runs.py
src/plot_coselection_summary.py
tests/test_coselection_analysis.py
scripts/run_coselection_server.sh
```

本地聚合输出：

```text
artifacts/20260718_coselection/
```

远端完整输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/coselection_bodyoffset10k_n1024_q512_20260718
/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/coselection_bodyoffset10k_stride8_n1024_20260718
/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/coselection_n4096_selected_20260718
```

## 12. 当前最严谨的表述

> 对 Qwen3-0.6B 的同一 attention head，Oracle Top-2% 历史 token 不是条件独立选择的。大量 token pair 的同 query 共选概率和边际校正 lift 显著高于固定预算/边际独立基线，并可在不同正文文本、稀疏 query 采样和保留单-token时间结构的循环移位零假设下复现。共选强度具有稳定的 layer/head 特异性，但具体 token cluster 依赖输入；其工程价值仍需通过预算固定的跨 query 预测和 PPL/任务评估验证。
