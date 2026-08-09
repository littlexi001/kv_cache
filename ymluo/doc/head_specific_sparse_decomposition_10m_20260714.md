# 10M Context 的 Head-Specific 稀疏分解性质

> **结论：真实 10M Q/K 检索不是“所有 heads 共同指向少量 blocks”，而是“少数稳定专业 heads 提供互补证据，同时大量 heads 共享 query-invariant block hubs”；在 480 个真实问题上，train 选择的 16 个 heads 在 held-out test 上召回 53.4%，随机 16 heads 只有 21.2%，但全 448 heads 的 Top16 并集仍覆盖 11.0% 的 10M 语料，因此该性质支持 head routing 与去 hub 化，不支持直接全 head 扫描。**

**状态：** 64-query 探索和480-query扩展验证均已完成；两个实验复用逐字节相同的10M blocks和冻结K索引，只重新捕获更多query Q。

## 1. 为什么研究这个性质

此前有两条看似冲突的结果：

- 全 head 并集能够找回许多固定四通道遗漏的 gold blocks；
- 把所有 head 的结果做多数投票，最终召回反而下降；
- 在另一个模型和 7.5K LongBench 上，SnapKV 允许每个 KV head 保留不同 token 后，1024-token KV 预算达到 Full Attention 99.62% 的分数。

这提示超长上下文的相关性可能不是一个共享标量：

```text
shared RAG view:       query -> one global block ranking
head-factorized view:  query -> layer/head-specific sparse rankings
```

本轮不设计新的最终检索系统，只检查第二种结构是否真实存在、是否能跨问题泛化，以及它包含哪些失败因素。

## 2. 冻结实验口径

| 项 | 设置 |
|---|---|
| 模型 | Qwen3-0.6B |
| 语料 | 9,999,872 个真实 LongBench tokens |
| blocks | 39,062 × 256 tokens |
| 数据类型 | 2WikiMQA、HotpotQA、MuSiQue、Qasper、NarrativeQA、MultiFieldQA、QMSum、GovReport |
| queries | 480 个真实问题，按数据集轮询抽样 |
| 通道 | 28 layers × 16 query heads = 448 heads |
| GQA | 16 query heads 共享 8 KV heads |
| 表征 | 真实 pre-RoPE record-context Q/K；K 使用冻结 SVD32 profile |
| 每 head 输出 | 独立 Top16 blocks |
| gold | 仅用于事后召回；不参与冻结 QK 排名 |

480题分布为2WikiMQA 136、HotpotQA 135、MuSiQue 135、Qasper 59、MultiFieldQA 13、NarrativeQA 2；它不是严格均衡数据集。Head子集实验使用500次按数据集分层的随机train/test划分。每次只在train gold上贪心选择heads，再报告test的micro和dataset-macro recall；随机head子集使用同一个test split。表中的p05/p95是随机划分分布的5%/95%分位数，不是预注册置信区间。

## 3. 性质一：稀疏性是按 head 分解的

定义第 `l` 层第 `h` 个 head 的 Top-k block 集合：

```text
S_lh^k(q) = TopK_b score_lh(q, b)
```

如果所有 heads 学到同一检索函数，这些集合应高度重合；如果 heads 是不同稀疏专家，并集应随 head 数快速增长。

| 每 head 深度 | 总槽位 | 平均不同 blocks | 槽位冗余 | 占 10M blocks | Gold 并集召回 |
|---:|---:|---:|---:|---:|---:|
| 1 | 448 | 388.3 | 13.3% | 0.99% | 36.88% |
| 2 | 896 | 735.0 | 18.0% | 1.88% | 44.38% |
| 4 | 1,792 | 1,361.6 | 24.0% | 3.49% | 53.54% |
| 8 | 3,584 | 2,459.5 | 31.4% | 6.30% | 62.08% |
| 16 | 7,168 | 4,299.4 | 40.0% | 11.01% | 70.83% |

在单层内，16 个 heads × Top16 的 256 个槽位平均展开为 223.1 个不同 blocks，冗余只有12.9%。因此单层 heads 的选择高度互补；跨层汇总后出现更多重复，但仍不是一个小共享集合。64题探索中的并集4,299.8、召回71.88%，与480题结果非常接近。

**允许结论：** 相关性在模型通道上具有组合稀疏性。每个 head 很稀疏，但所有 heads 的全局并集并不稀疏。

**不允许结论：** 不能把70.83%写成“小预算召回”，因为它平均需要4,299个不同 blocks，即约1.10M tokens。

480题的冻结全head扫描在7张RTX 3090上用时466.4秒，批量吞吐约0.97秒/题。该数字不是单请求延迟，但已经说明“扫描全部448 heads再合并”不是1B场景的可扩展方案；head routing必须发生在全库扫描之前。

## 4. 性质二：GQA 共享 K 不会消除 query-head 专业化

| Head 对关系 | Top16 mean Jaccard | 有非零重合的比例 |
|---|---:|---:|
| 同层、共享同一 KV head 的 GQA siblings | 8.59% | 62.03% |
| 同层、不同 KV heads | 0.78% | 13.34% |
| 跨层、相同 query-head 编号 | 0.25% | 6.14% |
| 随机跨层 heads | 0.28% | 6.50% |
| 两个均匀随机 Top16 集合的理论期望 | 0.020% | - |

GQA siblings 因为共享 K 表征而明显更相似，但平均 Jaccard 仍只有8.6%。这说明不同 query heads 即使访问相同 KV head，也会因为 Q 方向不同而选择大部分不同 blocks。

同层非 sibling 和跨层 heads 的重合远低于 siblings，但仍高于均匀随机集合。这一部分额外重合来自后文的公共 block hubs，而不是纯语义一致性。

## 5. 性质三：专业 heads 可以跨问题泛化

| Head 数 | 选择后Test micro | 随机micro | 选择后dataset-macro | 随机macro |
|---:|---:|---:|---:|---:|
| 1 | 29.98% | 2.11% | 29.53% | 2.32% |
| 2 | 39.70% | 4.20% | 37.00% | 4.59% |
| 4 | 49.60% | 7.72% | 43.71% | 8.31% |
| 8 | 50.67% | 13.40% | 45.27% | 14.05% |
| 16 | **53.40%** | 21.25% | **47.95%** | 21.55% |
| 32 | 56.93% | 31.32% | 51.20% | 30.58% |
| 64 | 61.62% | 42.48% | 55.07% | 40.28% |
| 全448 heads test oracle | 约70.8% | - | - | - |

最强单head `L11/H3` 在完整480题上命中149题（31.04%）；`L14/H15`命中146题（30.42%）。它们也是64题探索中的前两名，只是顺序互换。反复分层切分后，train选出的单head在held-out test上仍有29.98% micro recall，而随机单head只有2.11%。

这支持“head usefulness 具有可泛化结构”，而不是每道题都需要从448个 heads 里事后寻找完全不同的最佳 head。但从16 heads增加到64 heads仍有收益，说明固定小集合尚不能覆盖全部专业模式。

## 6. 性质四：存在 query-invariant block hubs

全 head Top16 在480题中一共提名过36,488个不同 blocks：

- block 被提名的 query 数中位数为17；
- 2,321个 blocks 至少被一半 queries 提名；
- 81个 blocks被全部480个 queries提名；
- 这81个universal hubs中只有一个在一条query上同时是gold，另外80个没有支持任何当前gold。

因此 QK 分数可用下面的诊断模型描述：

```text
score_lh(q, b) = mu_lh(b) + delta_lh(q, b)
```

- `mu_lh(b)`：query-invariant 的 block/head 吸引力，来源可能包括公共方向、范数、位置或 SVD 归一化残余；
- `delta_lh(q,b)`：当前 query 的条件相关信号。

多数投票会反复累加大量heads共享的`mu`，所以universal hubs很容易排到前面；只被少数专业heads提名的gold block则会被投票阈值删除。480题中，被consensus选中的gold平均得到27.1个heads支持；已被某个head提名却最终丢弃的gold有666个，平均只得到2.57个heads支持。

这个结果给出一个比“增加更多 heads”更具体的下一假设：先用 calibration queries 估计每个 `head × block` 的 query-independent prior，再比较去偏分数

```text
score'_lh(q,b) = score_lh(q,b) - E_qtrain[score_lh(q,b)]
```

是否在相同 head 和 block 预算下提高 held-out gold recall。该实验现已完成：5折z-score把universal hubs从81降到0，全head Top16的RRF39从22.71%提高到38.13%。进一步用train-only Top1-block多样性选择16 heads后，RRF39达到49.79%。完整协议见 [Query-Invariant Prior 与无标签 Head Gate](query_invariant_prior_and_unsupervised_head_gate_10m_20260714.md)。

## 7. 与 RAG 的边界

本性质与 RAG 的区别不是“换一种文本 embedding”：

- RAG 对每个 query 产生一个供所有层和 heads 共享的文本 block 排名；
- head-specific QK 对每个模型内部通道产生不同排名，可以依赖当前生成状态、层深和 Q 方向；
- universal hub 去偏针对的是模型内部访问先验，不是 BM25/E5 的词法或语义相似度。

后续matched实验确认当前结果没有超过RAG：同一480题与39-block预算下，无标签16-head z-score + RRF为49.79%，BM25-block为66.67%，BM25-record39为81.04%。等权融合相对BM25-block只提高1.04pp且不显著，不能声称Q/K已提供稳定增量。

更准确的潜在作用是：RAG负责第一次全局语义 seed；少量路由后的内部 heads负责模型状态相关的候选扩展、KV加载和RAG难以文本化的隐式证据。

## 8. 对 1B 高效搜索本质的更新

当前新增的性质不是“1B context 里只有固定1K tokens有用”，而是：

> **有用信息可能在模型通道上条件分解。每个 head 只需要一个很小的 block 子集，但不同 heads 需要的子集高度不同；高效搜索的关键不是构造一个全局共享 Top-K，而是先路由少量有效 heads，再在每个被激活通道内做稀疏搜索，同时去除 query-invariant hubs。**

它与已有性质可以组合成一条待证伪链：

```text
query/state
-> route a small stable head subset                 # 本轮部分支持
-> remove head/block query-independent priors       # 本轮发现问题，尚未验证方法
-> search record-local residual-K hierarchy         # 已有10M几何证据
-> post-RoPE exact QK on small candidates           # 尚未完成
-> load head-specific KV and continue generation    # 尚未完成
```

不能把现有“16 heads”与“7.79×层次点积减少”直接相乘成端到端加速，因为两项实验的目标、候选和成本口径不同。只有在同一 query 流程中测量候选召回、真实 attention、通信和 wall-clock 后才能报告组合收益。

## 9. 下一轮可证伪实验

### E10a：480-query 稳定性复验（已完成）

- 使用完全相同的39,062 blocks和冻结K索引；
- 将平衡问题数从64扩展到480；
- 重新捕获 query Q，不重建或调节K索引；
- 重复本文全部集合、hubness和分层train/test分析；
- 结果：16个train-selected heads的test micro/macro recall为53.40%/47.95%，随机基线为21.25%/21.55%；稳定专业head假设得到进一步支持。
- 限制：NarrativeQA和MultiFieldQA样本仍少，尚未做整个dataset留出测试。

### E10b：Query-invariant prior 去偏

- 状态：已完成5折全库分数均值/方差估计；
- prior平均解释49.8%分数方差；
- z-score把Top16并集召回从71.04%提高到80.63%，RRF39从22.71%提高到38.13%；
- universal hubs从81降到0；
- 限制：仍未测最终attention和生成。

### E10c：Head routing 的无 gold 输入实现

已完成探索性5折gate：只用train queries的raw Top1-block多样性选择16 heads，held-out召回62.08%，matched random为28.68%；RRF压到39 blocks后为49.79%。该特征是在七个proxy比较后发现，下一步必须冻结到未参与探索的新dataset/query；当前结果不能写成确认性外部泛化。

## 10. 产物

- 分析器：`ymluo/projects/parallel_block_retrieval/src/analyze_head_sparse_decomposition.py`
- 64题结果：`ymluo/projects/parallel_block_retrieval/outputs/head_sparse_decomposition_10m_20260714_v1/`
- 480题结果：`ymluo/projects/parallel_block_retrieval/outputs/head_sparse_decomposition_10m_query480_20260714_v1/`
- 480题运行流程：`ymluo/projects/parallel_block_retrieval/scripts/run_head_sparse_decomposition_10m_expanded_20260714.sh`
- prior去偏与无标签gate：`ymluo/projects/parallel_block_retrieval/outputs/head_prior_debiasing_10m_query480_20260714_v1/`
- 完整后续报告：`ymluo/doc/query_invariant_prior_and_unsupervised_head_gate_10m_20260714.md`
- 冻结输入：`real_longbench_docqa_10m_allhead_consensus_20260711_v1/per_head_topk.npz`
