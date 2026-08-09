# Section 127：10M -> 10K 真实 KV 检索失败修复与混合方案（2026-07-10）

> 后续已完成 28 层 x 16 query heads 的独立检索与多数共识实验。全 head 候选覆盖显著提高，但多数共识会删除专业 head 的少数证据，详见 `section128_all_head_consensus_retrieval_20260711.md`。

## 1. 结论先行

旧版 `full128` 找不到答案 block，不是因为 32 维 SVD 太小，而是因为被近似的检索目标本身有问题：

1. 跨独立 record 直接比较 post-RoPE Q/K，位置旋转相位不一致；
2. 每个问题只取统一 `Answer:` 末尾 token 的一个 Q，问题内容几乎没有进入检索向量；
3. 在 256 个 token 和 4 个 layer/head 上取全局最大 raw logit，极易被冒号、引号、`answer is` 等高范数 token 劫持；
4. 不同 layer/head 的 raw logit 未校准，旧结果的 2,496 个返回槽位全部由同一个 L6/H7 profile 决定；
5. `full128` 只是 4 个抽样 profile 的 128 维 reference，不是完整模型 attention oracle。

因此 SVD32 即使 100% 恢复旧 `full128` 的 Top-39，也只是在准确复现一个错误目标。

本次已实现一个三阶段真实数据方案：

```text
问题字符串
  -> BM25 从 982 个 record 中产生 Top-5 候选
  -> Qwen3-0.6B 以 P(question | record) 重排 Top-5
  -> 在选中 record 内用真实 pre-RoPE multi-Q/K + SVD32 排序 block
  -> 不足 39 个 block 时用全局 BM25 补齐
```

最终 `deep_ql_record39_svd32` 在 64 个问题上的结果为：

| 指标 | 旧 post-RoPE `full128` | 最终混合方案 |
| --- | ---: | ---: |
| answer block Recall@39 | 0.00% | **82.81%（53/64）** |
| mean answer NLL | 4.6713 | **3.2783** |
| NLL delta vs original | +1.9178 | **+0.5247** |

这解决了“full128 完全找不到”的实现问题，但还没有达到“10M -> 10K 几乎无损”。

## 2. 实验对象

所有检索 K 和 query Q 都来自 Qwen3-0.6B 对真实 LongBench 文本的前向计算，没有合成高斯向量。

| 项目 | 数值 |
| --- | ---: |
| 文本 tokens | 9,999,872 |
| block 数 | 39,062 |
| block 大小 | 256 tokens |
| 独立 records | 982 |
| 评估问题 | 64 |
| 返回预算 | 39 blocks = 9,984 tokens |

语料已经移除格式不兼容的 TriviaQA few-shot demonstrations，并拒绝含独立 `Passage:/Question:/Answer:` 模板的记录。详细清洗过程见 Section 126。

需要特别说明：当前 10M 是 982 个独立 record 组成的全局 KV 索引，不是一条连续的 10M causal sequence。每个 record 最长 64 blocks，并独立 prefill。

## 3. 为什么旧 full128 失败

### 3.1 post-RoPE 不适合跨 record 路由

RoPE 后的点积可写成：

```text
q_m^T k_n = q^T R(n - m) k
```

这个分数依赖相对位置。不同 record 各自从位置 0 开始 prefill，把一个 record 的 query Q 与另一个 record 的 post-RoPE K 直接比较，相当于引入没有语义意义的位置相位。它可以用于同一 causal sequence 内的真实 attention，但不应直接充当跨独立 record 的向量数据库相似度。

### 3.2 query 取错了 token

旧版每个问题只保留 prompt 最后一个 token 的 Q。由于模板统一以 `Answer:` 结尾，64 个问题实际都在使用冒号附近的隐藏状态：

```text
64 个 query Q 的平均余弦相似度约为 0.670
```

检索器看到的主要是共享模板，不是问题中的实体和关系。

### 3.3 block-max 是海量检索中的极值放大器

旧分数为：

```text
score(block) = max over token, layer/head of raw(q^T k)
```

39,062 blocks 每个有 256 个 token。只要某个标点、模板词或高范数 K 产生一次异常大值，整个 block 就会被选中。旧版 2,496 个 Top-39 槽位只覆盖 254 个不同 block，平均 query 间 Top-39 Jaccard 达 0.320，说明结果被少量通用高分 block 劫持。

### 3.4 SVD 只能近似目标，不能修复目标

旧实验中 SVD32 从 782 个候选 block 做 full-QK rerank，可以 100% 恢复 `full128` Top-39。这证明低秩 proposal 有效，同时也证明 0% answer recall 的根因在 `full128` reference，而不是 SVD32。

## 4. 修复方案

### 4.1 第一阶段：BM25 表层路由

对 39,062 个真实 block 和 982 个 record 解码，建立英文 unigram + bigram BM25 索引。BM25 不读取参考答案，只使用问题字符串。

| 方法 | source record 覆盖率 | answer block Recall@39 | NLL delta |
| --- | ---: | ---: | ---: |
| `bm25_block` | 93.75% | 56.25% | +1.2589 |
| `bm25_record20` | 89.06% | 67.19% | +1.1206 |
| `bm25_record30` | 87.50% | 71.88% | +0.8732 |
| `bm25_record39` | 82.81% | 73.44% | +0.7965 |

这里的 `record39` 表示先选 BM25 Top-1 record，再从该 record 分配最多 39 个 block；若不足 39 个，则用全局 BM25 block 补齐。

BM25 已经远好于旧 `full128`，说明数据中确实存在可检索的词法证据。

### 4.2 第二阶段：问题似然 record 重排

只对 BM25 Top-5 records 运行 Qwen3-0.6B cross-encoder。对每个候选 record 计算：

```text
score(record) = - mean NLL(question | record + "\nQuestion:")
```

该分数只使用候选文档和问题，不使用参考答案。

| record 路由 | Top-1 source record recall |
| --- | ---: |
| BM25 Top-1 | 68.75% |
| Question likelihood Top-1 | **78.13%** |
| BM25 Top-5 oracle upper bound | 87.50% |

按 query id 奇偶拆成两个 32-query 子集后，question-likelihood recall 分别为 81.25% 和 75.00%，提升不是只来自一个子集。

8 卡并行完成 64 x 5 = 320 次候选 record 评分耗时 25.93 秒。这一阶段目前是深度方案的主要延迟来源。

### 4.3 第三阶段：真实 pre-RoPE multi-Q/K + SVD32

模型前向时在 RoPE 之前记录 Q/K，仍保留 causal prefill 产生的真实 hidden state，但移除跨 record 比较中无意义的位置旋转相位。

Query 不再只取 `Answer:` 末尾 token，而是从问题内容中选择最多 16 个含字母或数字的 token Q，排除 `Question:` 和 `Answer:` 分隔符。

对 8,192 个真实 calibration K tokens 拟合 centered SVD。Rank64 的四个 profile 能量保留率为：

```text
99.261%, 99.782%, 99.921%, 99.138%
```

实际深度检索使用前 32 维，并采用类似 ColBERT 的 late interaction：

```text
score(block)
  = mean over 4 profiles and valid question Q
      max over K tokens in block cos(q32, k32)
```

另外跳过每个 block 的前 16 个 token，减轻切块边界和通用前缀的影响。

在全 39,062 blocks 上单独使用该语义分数，answer recall 从 0% 提升到 34.38%，说明 query-content multi-Q 和 pre-RoPE 修复产生了真实信号，但单独承担全局路由仍然不够。

在已知正确 source record 的诊断中：

| source record 内预算 | answer block recall | 随机选择期望 |
| --- | ---: | ---: |
| Top-1 | 29.69% | 13.05% |
| Top-5 | 59.38% | 37.35% |
| Top-10 | 76.56% | 58.91% |
| Top-20 | 93.75% | 81.19% |

因此 pre-RoPE SVD32 更适合作为候选 record 内的细粒度 block 排序器，而不是全局 record 路由器。

### 4.4 风控与两种运行模式

实现了两种可选模式：

| 模式 | 路由策略 | answer Recall@39 | NLL delta |
| --- | --- | ---: | ---: |
| 快速 `risk_bm25_svd32` | BM25 margin 高时用 SVD32，低时回退 BM25 record30 | 75.00% | +0.5909 |
| 深度 `deep_ql_record39_svd32` | BM25 Top-5 全部做 question-likelihood 重排，再用 SVD32 | **82.81%** | **+0.5247** |

快速模式的 BM25 相对 margin 阈值 0.04 在偶数 query 上选择，再在奇数 query 上检查；全量 NLL delta 为 +0.5909。深度模式不使用该阈值，因此不会依赖这项调参。

## 5. 最终结果

| 上下文 | Answer Recall@39 | Mean answer NLL | Delta vs original | NLL 不升比例 |
| --- | ---: | ---: | ---: | ---: |
| original source | 不适用 | 2.7536 | 0.0000 | 100.00% |
| source-oracle 10K | 100.00% 定义下限 | 2.7261 | -0.0275 | 78.13% |
| 旧 post-RoPE `full128` | 0.00% | 4.6713 | +1.9178 | 未作为主结果 |
| `bm25_record39` | 73.44% | 3.5501 | +0.7965 | 40.63% |
| `risk_bm25_svd32` | 75.00% | 3.3445 | +0.5909 | 40.63% |
| `deep_ql_record39_svd32` | **82.81%** | **3.2783** | **+0.5247** | **42.19%** |

深度方案的 record Top-1 recall 为 78.13%，最终 context 的 source-record 覆盖率为 87.50%。两者不同，是因为未选中正确 record 时，全局 BM25 补位仍可能带回 source record 或答案 block。

## 6. 多卡并行收益

全局 pre-RoPE ColBERT 扫描对 64 个问题同时扫描 10M K tokens，索引预先驻留 GPU；计时不包含模型 profiling 和索引加载。

| 方法 | 1 卡 | 2 卡 speedup | 4 卡 speedup | 8 卡 | 8 卡 speedup | 8 卡效率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `colbert128` | 0.5827 s | 1.99x | 3.84x | 0.0786 s | 7.41x | 92.66% |
| `colbert32` | 0.5815 s | 1.97x | 3.82x | 0.0788 s | 7.38x | 92.20% |
| `colbert64` | 0.5821 s | 1.98x | 3.84x | 0.0789 s | 7.38x | 92.25% |
| `colbert32_rerank` | 0.8135 s | 1.81x | 3.09x | 0.1768 s | 4.60x | 57.51% |

结论是分片扫描、局部 Top-K、全局 merge 的数据并行框架扩展良好。SVD32 在当前实现中没有比 128 维显著更快，因为总时间主要受 block/token 遍历、归一化、Python/profile 循环和通信影响；要体现降维算力收益，需要继续融合 kernel 和预归一化索引。

8 卡真实 pre-RoPE Q/K profiling 中，各 shard 处理约 1.25M tokens，单卡峰值显存约 2.22 GB，最快 shard 70.31 秒，最慢 shard 104.94 秒。端到端延迟还必须把这项离线建库成本与 25.93 秒的在线 question-likelihood 路由成本分开报告。

## 7. 现在能够说明什么

可以支持的结论：

1. 真实 Qwen3 Q/K 可以按 block 分片并行建库和检索；
2. pre-RoPE K 的 32 维低秩空间保留了有用的局部语义排序信号；
3. 词法 record 路由与真实 KV block 排序具有互补性；
4. 8 卡全局扫描接近线性扩展，SVD32 达到 7.38x speedup；
5. 旧 `full128` 的 0% 不是数据完全不可检索，而是 reference 设计错误。

不能支持的结论：

1. 不能称为 10M -> 10K 近乎无损，当前 NLL 仍增加 0.5247；
2. 不能直接外推到 1B -> 1M，record 数、索引驻留、通信和候选路由都会变化；
3. 不能把 answer-string block recall 等同于最终生成准确率；
4. 不能把当前 982-record 索引称为单条 10M 长上下文；
5. 当前 query Q 来自其 source record 的真实 prefill，这是 KV-cache 检索设定，不是 query 从未看过语料的标准 RAG embedding 设定。

## 8. 下一轮应优先解决的问题

1. 用生成 EM/F1 或 LongBench 官方指标验证 53 个命中问题是否真的恢复答案；
2. 增加 supporting-fact 或 evidence-span 标注，替代仅检查答案字符串；
3. 训练或蒸馏轻量 record router，替代昂贵的 5-record Qwen question-likelihood；
4. 将 query Q 在候选 record 上重新计算，区分标准 RAG 与已有 KV-cache 两种部署设定；
5. 构造 source record 明显大于 39 blocks 的数据，避免当前平均 36 blocks 导致 source-oracle 过于容易；
6. 对 SVD32 索引预归一化并实现 fused late-interaction kernel，验证降维的真实吞吐收益；
7. 扩到 100M 后再评估 ANN 或分层 record/block 索引，暂不直接跳到 1B。

## 9. 一键复现

服务器入口：

```bash
cd /home/fdong/ymluo/projects/parallel_block_retrieval
bash scripts/run_hybrid_retrieval_solution_server.sh
```

脚本会自动选择显存占用不超过 1 GB 的空闲 GPU。8 卡不全空闲时会使用检测到的 6 卡或其他可用卡数，依次执行：

1. 清洗语料构造；
2. BM25 block/record 建库；
3. 真实 pre-RoPE Q/K profiling 与 SVD；
4. BM25 Top-5 question-likelihood 重排；
5. SVD32 block 检索；
6. 答案 NLL 评估。

主要代码：

```text
src/run_lexical_block_retrieval.py
src/profile_real_qk.py
src/rerank_records_by_question_nll.py
src/run_hybrid_block_retrieval.py
src/evaluate_retrieved_answer_nll.py
scripts/run_hybrid_retrieval_solution_server.sh
```

本地归档结果：

```text
outputs/real_longbench_docqa_10m_clean_bm25_20260710_v1/
outputs/real_longbench_docqa_10m_prerope_colbert_scaling_20260710_prerope_v1/
outputs/real_longbench_docqa_10m_record_question_nll_20260710_v1/
outputs/real_longbench_docqa_10m_hybrid_bm25_prerope_svd32_20260710_v3/
outputs/real_longbench_docqa_10m_hybrid_bm25_prerope_svd32_20260710_v4/
```
