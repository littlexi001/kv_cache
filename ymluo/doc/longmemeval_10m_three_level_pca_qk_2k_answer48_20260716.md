# 10M 外部记忆三层检索实验：Selected-head PCA/QK 细排与 2K Reader

> **一句话结论：** 三层架构成功把每题的 10M 外部记忆压缩为最多 2K tokens，并在 8 张 3090 上完成 500 题；但当前 selected-head PCA/QK 细排没有改善质量，最好的仍是直接使用 BM25，Answer@48 为 **20.0%（94/470）**，平均 **1.40 秒/题**。

**日期：** 2026-07-16  
**模型：** Llama-3.1-8B-Instruct  
**状态：** 500 题全量实验完成；结论适用于本实验协议，不等价于直接在完整 10M KV cache 上执行稀疏 attention。

## 1. 实验回答什么问题

验证以下三层系统能否实际运行，并判断模型原生 Q/K 是否适合在 BM25 候选中继续细排：

```text
L2：每题 10M tokens 的外部记忆
    owner/tenant scope -> block BM25 -> 128 个候选 block

L1：模型原生细排
    selected layer/head 的 pre-RoPE Q/K
    PCA64 + 对称 INT4 模拟，或精确 128 维 QK
    -> 名义 Top-31

L0：GPU 工作上下文
    按 Llama tokenizer 严格打包到 <= 2,000 retrieved tokens
    -> 重新 prefill 文本 -> greedy 生成最多 48 tokens
```

主要指标 `Answer@48`：在最多 48 个生成 token 中，是否出现归一化后的标准答案字符串。

## 2. 数据与公平性

- 500 个 LongMemEval 问题分到 8 个独立分片，每个问题实际搜索其所在的 **10M-token** 记忆。
- 每个分片有 156,250 个 64-token block；8 卡各负责一个独立的 10M 分片。
- 500 题中有 470 个可回答问题和 30 个 abstention 问题。
- 答案标签不参与 scope、候选生成、head 选择、排序或上下文打包，仅用于事后评分。
- L2 使用已知 owner/tenant 元数据缩小 namespace。平均 scope 为 1,663.6 blocks，约占 10M 的 1.065%，再由 BM25 选 128 blocks。
- L0 的 2K 预算包含页面标题和日期，但不包含最终问题与系统指令；所有方法使用相同 reader prompt 和页面顺序。
- 每个方法最多先选 31 blocks，严格打包后实际平均约 24 blocks、1,961--1,979 tokens。

这里的 8 个分片用于并行评测，不表示每题只搜索 `10M / 8`；每道题的检索空间仍是完整 10M tokens。

## 3. 四种方法

| 方法 | 做法 |
|---|---|
| BM25 Top-31 | 保留 L2 的前 31 个 BM25 block，再按 2K 预算打包 |
| PCA64 selected-16 | 对 128 候选计算 8 个层、32 个 query head 的 layer-head 分数，从 256 个通道中按分数 margin 动态取 16 个；每通道取 Top-8 后用加权 RRF 融合 |
| Exact QK selected-16 | head 选择和融合与 PCA 相同，但直接使用 128 维 pre-RoPE QK 点积 |
| Hybrid BM25+PCA | 固定保留 BM25 Top-16，再用 PCA 排名补到 31 |

选择层为 `3, 7, 11, 15, 19, 23, 27, 31`。每个 block 被压成 4 个有序 segment mean；query 使用检索提示最后 8 个 token 的 Q。PCA basis 由每个 10M 分片随机抽取的 256 blocks 校准。

`PCA64 INT4` 在本轮中是逐向量对称量化再反量化的数值模拟，不是已打包的 INT4 自定义 kernel。候选 K 也是在线对 128 个 block 单独前向得到，尚未预计算成 SSD sidecar。

## 4. 主要结果

### 4.1 端到端质量与时间

| 方法 | Answer@48（470 可回答题） | 原始 500 题答案命中 | 严格 500 题准确率 | 精确 block 命中 | 覆盖全部证据 session | 平均 tokens | 平均时间/题 | P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BM25 Top-31** | **20.00%，94/470** | 18.80% | **24.80%，124/500** | **87.23%** | **69.57%** | 1960.6 | **1.399 s** | 1.171 / 1.969 s |
| Hybrid BM25+PCA | 18.09%，85/470 | 17.00% | 22.80%，114/500 | 85.32% | 66.17% | 1977.6 | 4.026 s | 3.871 / 4.720 s |
| PCA64 selected-16 | 8.72%，41/470 | 8.20% | 14.00%，70/500 | 59.57% | 50.21% | 1978.6 | 3.924 s | 3.799 / 4.707 s |
| Exact QK selected-16 | 8.72%，41/470 | 8.20% | 14.20%，71/500 | 58.30% | 55.11% | 1978.9 | 3.899 s | 3.771 / 4.697 s |

严格 500 题准确率把 30 个 abstention 的正确拒答也算作正确，因此不能替代主要的 Answer@48。BM25 对 30 个 abstention 全部拒答，Hybrid 和 PCA 各错 1 个，Exact QK 全部拒答。

相对 BM25 的配对结果：

- Hybrid：13 胜、22 负，`p=0.175`，没有显著差异，但均值更差。
- Exact QK：14 胜、67 负，`p=1.89e-9`，显著更差。
- PCA64：5 胜、58 负，`p=1.66e-12`，显著更差。

### 4.2 三层损失分解

L2 的 128 个候选已经很强：

| L2 指标 | 结果 |
|---|---:|
| 至少一个精确正 block 进入 Top-128 | **96.60%，454/470** |
| 覆盖全部证据 session | **96.17%** |
| 平均 BM25 查询时间 | **0.816 ms** |

L1 将 128 候选压到最终 2K 后：

| 方法 | 给定 L2 已命中，L1 仍保住精确 block | 被 L1 丢掉的 L2 命中题 | 精确 block 与全部证据 session 同时满足 | 在该联合条件下 Answer@48 |
|---|---:|---:|---:|---:|
| **BM25 Top-31** | **90.31%** | **44** | 308 | **27.60%** |
| Hybrid BM25+PCA | 88.33% | 53 | 292 | 27.05% |
| PCA64 selected-16 | 61.67% | 174 | 176 | 18.18% |
| Exact QK selected-16 | 60.35% | 180 | 186 | 18.28% |

因此本轮有两个明确损失点：

1. **L1 排序损失。** L2 已召回关键证据后，纯 Q/K 细排丢掉约 39%；BM25 只丢约 10%。
2. **L0 阅读和任务损失。** 即使至少一个精确 block 被命中并覆盖所有证据 session，最好的 Answer@48 也只有 27.6%。不过该联合条件仍不是“全部精确证据均已出现”的 oracle，不能把剩余损失全部归因于模型能力。

BM25 在不同题型上的 Answer@48 为：single-session-user 50.0%、single-session-assistant 37.5%、knowledge-update 29.2%、temporal-reasoning 10.2%、multi-session 5.8%、single-session-preference 0%。当前 2K 单次读取对多会话组合、时间推理和偏好类任务尤其不足。

## 5. 时间与 8 卡收益

模型原生路径的平均在线开销：

| 阶段 | 时间 |
|---|---:|
| L2 scope BM25 | 0.00082 s |
| query Q profiling | 0.0348 s |
| 128 候选 K profiling | **2.691 s** |
| PCA64 打分 | 均值 0.0053 s，P50 0.0034 s |
| Exact QK 打分 | 均值 0.0010 s |
| 2K prefill + 最多 48-token 生成 | 约 1.16--1.39 s |

矩阵打分本身已经很快，当前瓶颈是每题重新前向 128 个候选 block。若把 block 的分层 K 摘要离线预计算到 CPU/SSD，在线 fine-rank 可去掉大部分 2.691 秒；但在修复检索质量之前，做该工程优化没有意义。

8 卡完整运行：

| 指标 | 结果 |
|---|---:|
| 8 个进程耗时之和（串行参照） | 4,340.2 s，72.34 min |
| 实际墙钟时间 | 801.6 s，13.36 min |
| 实际加速 | **5.41x** |
| 排除一个分片首次加载失败后的同步稳态墙钟估计 | 556.2 s，9.27 min |
| 同步稳态加速 | **7.80x** |

实际墙钟时间包含第 6 个分片首次加载权重失败、延迟恢复的约 4 分钟，因此 5.41x 不是 GPU 计算本身的并行上限。各分片一题一 GPU 独立，稳态负载均衡接近线性。

## 6. 显存结论

本架构解决了此前“6.6% 的 10M token 仍然无法常驻 GPU”的问题：

- 10M 原文和 L2 索引留在 CPU/SSD；
- L1 只临时处理 128 个候选 block；
- L0 每题送入 GPU 的检索文本最多 2,000 tokens；
- 2,000 条 question-method 结果中，retrieval budget 违规数为 **0**，generation 超过 48 tokens 的违规数也为 **0**。

所以三层物理结构可以直接使用，并已在单张 24GB 3090 上容纳 Llama-3.1-8B 和该工作集。但本轮没有测量峰值显存，也没有实现“从 SSD 加载预先保存的精确 KV”，L0 采用的是重新 prefill 文本。

## 7. 为什么原生 Q/K 细排失败

本轮最重要的否定性结果是：**PCA 并不是主要问题，原始 QK 排序本身就不适合作为当前 block relevance。**

1. Exact QK 与 PCA 的 Answer@48 完全相同，精确 block 召回也都是约 59%；扩大 PCA 维度或去掉量化不会解决主问题。
2. PCA64 保留了平均 99.36% 的 K 二阶能量，但 PCA 与 Exact QK 的最终 block 集合 Jaccard 只有 26.4%。保留全局方差不等于保留近邻 Top-k，INT4、近似平分和 RRF 都会放大排序不稳定。
3. Q/K 是自回归 attention 的内部路由量，不是为跨文档语义相似度训练的 embedding。当前 query 和每个 block 独立前向，缺少它们在同一连续上下文中的隐状态条件。
4. 通过“哪个 head 的 Top-1 与 Top-2 margin 大”选择通道，只能找到区分度强的 head，不能保证该 head 的区分方向与答案相关。
5. 当前 Q 来自问题提示的最后 8 个 token，而不是模型已经生成若干 token 后的真实推理状态，因此还没有验证项目最核心的“随生成状态变化动态刷新证据”。
6. 此前 128K 实验验证的是连续原生上下文中的 attention-mass/PPL 保持；本轮验证的是独立 block 的外部记忆 relevance。两者不是同一个任务，前者的成功不能直接推出后者有效。

## 8. 当前结论与下一步

### 已被支持

- `10M -> scope -> 128 blocks -> <=2K tokens -> 48-token generation` 的三层系统可运行。
- 2K GPU 工作上下文足以让部分 LongMemEval 问题得到正确答案。
- L2 scope + BM25 可以低成本获得 96.6% 的 Top-128 精确 block 上限。
- 8 个独立 10M 分片适合多卡并行，稳态加速约 7.8x。

### 被当前结果否定

- 把独立 block 的 selected-head QK 或 PCA-QK 直接作为通用语义细排，当前明显不如 BM25。
- `PCA retained energy` 高不能作为 block retrieval 有效性的充分证据。
- 当前 Hybrid 不值得替换 BM25：它更慢、均值更差，也没有显著收益。

### 优先实验顺序

1. **做精确证据 Oracle-2K reader。** 将所有标注精确证据打包到相同 2K prompt，测 Answer@48，得到 L0 真正上限。
2. **保留 BM25，不让 QK 无条件替换。** 训练或校准一个只在预测有净收益时才启用的 risk gate，先证明 paired wins 大于 losses。
3. **改成真实生成状态 Q。** 每生成 8--16 tokens 更新一次 query，用当前 hidden state/Q 检索；比较静态问题 Q 与动态 Q 的证据增益。
4. **以证据效用蒸馏 head。** 用训练集上的 oracle evidence、teacher attention 或答案 NLL 改善量监督 layer-head gate，而不是按 score margin 无监督选 head。
5. **质量成立后再建 sidecar。** 将 block K 摘要预计算到 CPU/SSD，L1 只加载 128 候选的压缩记录，把 2.69 秒 profiling 从在线路径移除。

当前最可靠的系统版本是 **L2 owner-scope BM25 + L0 2K re-prefill**。模型原生 L1 仍是研究变量，不应作为已获得收益的主方法。

## 9. 产物

- 评测实现：`ymluo/projects/parallel_block_retrieval/src/evaluate_longmemeval_10m_per_head_pca_reader.py`
- 汇总实现：`ymluo/projects/parallel_block_retrieval/src/analyze_longmemeval_10m_per_head_pca_reader.py`
- 8 卡脚本：`ymluo/projects/parallel_block_retrieval/scripts/run_longmemeval_10m_per_head_pca_reader_8gpu.sh`
- 完整汇总证据：`ymluo/doc/1b_context_search_research_exploration/evidence/longmemeval_10m_selected_head_pca_qk_2k_answer48_20260716.json`
- 服务器原始结果：`/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/longmemeval_10m_selected_head_pca_reader_all500_v2`

