# Section 128：全层全 Head 独立检索与多数共识召回（2026-07-11）

## 1. 要验证的假设

本实验验证以下判断：旧 attention 检索只使用 4 个指定 layer/query-head，而不同 head 关注的信息不同，因此固定 4 heads 可能漏掉由其他专业 head 发现的答案 block。

先做最直接的全 head 共识版本：

1. 每个 layer/query-head 独立检索自己的 Top-16 blocks；
2. 不直接比较不同 head 的 raw attention 或 cosine 数值；
3. 将每个 head 的结果转换为 head 内名次；
4. 根据覆盖层数、支持 head 数和 RRF 名次分选择 39 个多数 head 共同关注的 blocks。

实验结论分为两部分：

1. **不同 head 的确找到了固定 4 heads 漏掉的答案。**全 head Top-16 候选并集的 answer-block coverage 为 71.88%，固定 4 heads 只有 35.94%。
2. **多数共识不是正确的 39-block 压缩方法。**全 head 共识 Recall@39 只有 28.13%，因为真正有用的证据经常只被 1 至 2 个专业层/head 发现，会被通用 block 的多数票淹没。

因此，“扩展到所有 head”这个方向成立，但下一步应做专业 head 路由或多样性配额，而不是要求所有 head 达成共识。

## 2. 真实 Q/K 与索引结构

模型为 Qwen3-0.6B：

| 项目 | 数值 |
| --- | ---: |
| transformer layers | 28 |
| query heads / layer | 16 |
| KV heads / layer | 8 |
| 独立 query-head 检索器 | 28 x 16 = 448 |
| 唯一 K profiles | 28 x 8 = 224 |
| head dimension | 128 |
| SVD dimension | 32 |

Qwen3-0.6B 使用 GQA，同一层相邻两个 query heads 共享一个 KV head。实现没有为 448 个 query heads 重复保存 K，而是每层只保存 8 组共享 K，再在检索时将 16 个 Q 映射到对应 K。

所有 Q/K 均来自 Qwen3-0.6B 对真实 LongBench record 的 causal forward，记录位置在 RoPE 之前，没有合成向量。

每个 layer/KV-head 使用 8,192 个真实 K tokens 独立拟合 centered K-SVD：

```text
K_centered = K - mean(K)
Cov = K_centered^T K_centered
Cov V = V Lambda
k32 = normalize(k V32)
q32 = normalize(q V32)
```

协方差特征分解与求 K 矩阵的右奇异向量等价。Rank32 的能量保留率：

| 统计量 | 能量保留率 |
| --- | ---: |
| mean over 224 K profiles | 88.01% |
| minimum | 69.18% |
| maximum | 99.47% |

不同 head 的低秩程度差异较大。本文先固定使用用户提出的 32 维，后续可针对低能量 head 单独提高 rank。

## 3. 每个 Head 如何独立检索

每个问题从问题内容中选择最多 16 个字母数字 token Q，排除 `Question:` 和 `Answer:` 模板 token。

对每个 `(layer, query_head)` 独立计算：

```text
score_h(block)
  = mean over valid question Q
      max over K tokens in block cos(q32_h, k32_h)
```

每个 block 的前 16 tokens 不参与打分，以降低切块边界和通用前缀影响。每个 head 只保留自己的 Top-16，不与其他 head 比较原始分数。

全 head 共识实现了三种无监督融合：

| 方法 | 排序优先级 |
| --- | --- |
| `allhead_layer_consensus` | 覆盖层数 -> head 票数 -> RRF |
| `allhead_head_vote` | head 票数 -> 覆盖层数 -> RRF |
| `allhead_rrf` | head 内 reciprocal rank 总和 |

`selected4_independent_rrf` 使用同样的独立 head 排名逻辑，但只保留旧实验的 L3/H10、L21/H8、L6/H7、L16/H14，作为严格对照。

## 4. 全 Head 是否提供了更多答案候选

答案 block 只要出现在任意一个 head 的 Top-L 中，就计入候选并集 oracle。这里的 oracle 只用于判断 head 覆盖能力，不参与实际检索排序。

| 每个 head 的候选深度 | 全 448 heads 候选并集 | 固定 4 heads 候选并集 |
| --- | ---: | ---: |
| Top-1 | 35.94%（23/64） | 23.44%（15/64） |
| Top-2 | 43.75%（28/64） | 26.56%（17/64） |
| Top-4 | 54.69%（35/64） | 28.13%（18/64） |
| Top-8 | 62.50%（40/64） | 31.25%（20/64） |
| Top-16 | **71.88%（46/64）** | **35.94%（23/64）** |

这个结果支持用户提出的核心判断：固定 4 heads 明显漏掉了其他 head 能够检索到的答案。

不过，全 head Top-16 的平均候选并集有 4,299.8 个 blocks，必须继续压缩到 39 blocks。

## 5. 多数共识压缩到 39 Blocks 的结果

正式 Top-16/head 结果：

| 方法 | source-record coverage | record Top-1 | answer Recall@39 | answer MRR |
| --- | ---: | ---: | ---: | ---: |
| `allhead_layer_consensus` | 62.50% | 14.06% | 28.13% | 0.1490 |
| `allhead_head_vote` | 62.50% | 14.06% | 28.13% | 0.1557 |
| `allhead_rrf` | 65.63% | 14.06% | 28.13% | 0.1568 |
| `selected4_independent_rrf` | 81.25% | 39.06% | **34.38%** | 0.2288 |

改变每个 head 的 nomination depth 也没有修复多数共识：

| Head Top-L | layer consensus Recall@39 | head vote | RRF |
| --- | ---: | ---: | ---: |
| 1 | 29.69% | 29.69% | 29.69% |
| 2 | 28.13% | 29.69% | 29.69% |
| 4 | 29.69% | 29.69% | 29.69% |
| 8 | **31.25%** | 29.69% | 29.69% |
| 16 | 28.13% | 28.13% | 28.13% |

最佳全 head 多数共识只有 31.25%，仍低于固定 4 heads 独立 RRF 的 34.38%。

## 6. 为什么候选覆盖提高，但共识结果下降

对 Top-16/head 候选中的 gold blocks 统计 head 支持：

| Gold block 类别 | Blocks | 平均覆盖层数 | 平均支持 heads | 平均最佳 head rank |
| --- | ---: | ---: | ---: | ---: |
| 被共识 Top-39 保留 | 21 | 12.57 | 34.05 | 1.24 |
| 已被专业 head 找到但被 Top-39 丢弃 | 86 | **2.01** | **2.45** | 5.88 |
| 每个 query 的 Top-39 截止 block | 64 | 6.69 | 9.86 | 2.14 |

86 个被丢弃的 gold blocks 平均只得到约 2 个层、2.45 个 heads 支持，低于进入 Top-39 所需的约 6.69 层和 9.86 heads。也就是说，多数共识系统性删除了专业 head 的少数意见。

与此同时，共识选中的通用 blocks 会跨问题重复出现：

| Block | 进入多少个 query 的 Top-39 | 文本类型示例 |
| --- | ---: | --- |
| 1276 | 52/64 | 心脏病术语表和说明性开头 |
| 25076 | 42/64 | 体育招募文章开头 |
| 26906 | 38/64 | 游戏规则介绍和通用问句 |
| 27934 | 33/64 | 国会报告背景介绍 |

这些 block 通常含标题、定义、问句、背景介绍等多个 heads 都容易关注的结构，但不一定与当前问题相关。

## 7. Head 专门化证据

单 head Top-16 answer recall 最高的 heads：

| Layer / Head | Recall@16 |
| --- | ---: |
| L14 / H15 | **39.06%（25/64）** |
| L11 / H3 | 34.38%（22/64） |
| L3 / H10 | 29.69%（19/64） |
| L21 / H8 | 28.13%（18/64） |
| L14 / H14 | 28.13%（18/64） |

按数据集寻找同一样本内表现最好的 head，可观察到明显差异：

| 数据集 | 描述性最佳 Head | 命中 |
| --- | --- | ---: |
| HotpotQA | L11/H3 | 10/13 |
| MultiFieldQA | L8/H4 | 9/12 |
| Qasper | L14/H15 | 5/12 |
| 2WikiMQA | L3/H10 | 2/13 |
| MuSiQue | L14/H14 | 2/12 |

该表是在同一 64-query 样本上事后统计，只能作为 head 专门化诊断，不能直接当作已验证的 head router。下一轮必须使用独立 calibration/dev split 选 head，再在 test split 报告。

## 8. 模型 NLL

| 上下文 | Mean answer NLL | Delta vs original |
| --- | ---: | ---: |
| original source | 2.7536 | 0.0000 |
| source-oracle 10K | 2.7261 | -0.0275 |
| `allhead_layer_consensus` | 4.2961 | +1.5425 |
| `allhead_head_vote` | 4.2151 | +1.4616 |
| `allhead_rrf` | **4.1858** | **+1.4322** |
| `selected4_independent_rrf` | 4.3332 | +1.5796 |

全 head RRF 比固定 4 heads 的独立 RRF 有更低 NLL，但仍明显差于 Section 127 的 BM25 + question-likelihood + routed SVD32 深度方案（NLL delta +0.5247）。

## 9. 系统开销

| 项目 | 结果 |
| --- | ---: |
| SVD32 all-head index | 143,358,164,992 bytes（约 133.5 GiB） |
| 8 卡 profiling 总墙钟 | 299.5 s |
| 每 rank 索引阶段 | 236.6 至 250.2 s |
| 单卡峰值显存 | 2.77 GB |
| 8 卡全 head 检索 | 300.2 s |
| 平均每层扫描 | 10.72 s |
| 检索单卡显存 | 约 611 MB |

索引按层拆分并从 memmap 流式读取，因此显存很低；当前检索速度主要受 143 GB I/O、Python block/query 循环和大量小 einsum kernel 限制，尚未做 fused kernel 优化。

## 10. 对下一步的直接建议

全 head 实验已经把问题拆清楚：

```text
固定 4 heads：候选覆盖不足
全 head 多数共识：专业证据被多数票删除
```

下一步不应继续提高“所有 head 都同意”的权重，而应实现：

1. 按任务或 query 动态选择少量专业 heads；
2. 给不同层/head group 保留独立 block 配额，再做去重拼接；
3. 将多数共识 blocks 作为通用候选，将高置信专业 head blocks 作为保底候选；
4. 使用偶数 query 训练或选择 head router，奇数 query 做 held-out 验证；
5. 在 BM25/record router 先缩小范围后，再运行 all-head block 检索。

最合理的下一版 39-block 预算可以先设为：

```text
16 blocks：全 head 共识
16 blocks：专业 head/group 独立配额
 7 blocks：BM25 或风险回退
```

该分配需要在 dev split 上选择，并在 test split 固定后评测，不能直接用当前 64 个问题调到最优。

## 11. 复现

```bash
cd /home/fdong/ymluo/projects/parallel_block_retrieval
bash scripts/run_all_head_consensus_server.sh
```

主要代码：

```text
src/profile_all_head_qk.py
src/run_all_head_consensus_retrieval.py
src/analyze_all_head_consensus.py
scripts/run_all_head_consensus_server.sh
```

归档结果：

```text
outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile/
outputs/real_longbench_docqa_10m_allhead_consensus_20260711_v1/
```
