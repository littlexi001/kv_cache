# 10M 逐步检索系统的 RAG Baseline

> **一句话结论：在同一份 10M tokens、同一批 500 条二跳问题和完全相同的 Qwen3-8B reader/verifier 下，BM25 + E5-base-v2 Hybrid-RAG 达到 43.6%（218/500）最终正确率，显著超过当前 SVD32 系统的 36.4%（182/500，配对 p=4.37e-5）。**

**日期：** 2026-07-13  
**服务器：** 8 x RTX 3090，本次生成和验证使用 GPU 0/1/2/4  
**语料：** 9,999,872 tokens，39,062 个 256-token blocks  
**测试集：** Official MuSiQue 2-hop test，500 条

## 1. 为什么要做这个 baseline

现有主线使用 BM25/词法信号和 Qwen3-0.6B 的四通道 SVD32 Q/K 特征检索证据。为了判断收益是否真正来自模型内生 Q/K 检索，需要与标准外部 embedding RAG 比较。

本实验遵循“只替换检索器”的控制变量协议：

- 数据、block 划分和 500 条测试问题不变；
- 第一跳仍读取 Top3 blocks，第二跳仍读取 Top16 blocks；
- 第一跳仍由 Qwen3-8B 生成 bridge，并无泄漏写回第二跳状态；
- 第二跳仍逐 block 生成 16 个候选，再由同一个冻结 Yes/No verifier 选择；
- 只把原检索器替换为标准外部 embedding 检索器；
- test gold 只用于最后统计，不参与检索、生成或候选选择。

因此，最终准确率差异可以主要归因于检索结果变化，而不是 reader、prompt 或读取预算变化。

## 2. RAG 方法

### 2.1 Dense retriever

使用未经本任务微调的 `intfloat/e5-base-v2`：

- passage：`passage: {block_text}`；
- query：`query: {current_step_state}`；
- mean pooling 后进行 L2 normalization；
- 使用 cosine/dot-product 检索；
- embedding 维度为 768，FP16 存储；
- 39,062 个真实文本 block 的索引大小约 58 MB；
- 没有合成文本、合成向量或高斯 target 注入。

动态 query 包含：当前 lookup entity、当前 atomic question、原始问题和已经生成的 compact state。第二跳使用 Qwen3-8B 实际生成的 bridge，不使用 gold bridge。

### 2.2 Hybrid retrieval

同时运行 BM25 和 E5 dense retrieval，各自取 Top512，然后使用 Reciprocal Rank Fusion：

```text
RRF(d) = 1 / (60 + rank_BM25(d)) + 1 / (60 + rank_E5(d))
```

在 dev 上比较纯 BM25、纯 E5 和 Hybrid-RAG。Hybrid 在第一跳 Top3 和第二跳 Top16 上均最好，因此冻结 Hybrid 配置并运行 test 严格链；没有使用 test 选择方法。

## 3. 检索结果

### 3.1 Dev 选型

| 步骤与预算 | BM25 | E5 Dense | Hybrid-RAG |
|---|---:|---:|---:|
| 第一跳 Recall@3 | 70.2% | 75.2% | **78.4%** |
| 第二跳 gold-state Recall@16 | 84.4% | 80.4% | **91.4%** |

### 3.2 Test 的 gold-state 检索能力

该表只用于诊断 retriever 上限；第二跳使用正确 bridge，因此不是最终严格链结果。

| 步骤与预算 | BM25 | E5 Dense | Hybrid-RAG |
|---|---:|---:|---:|
| 第一跳 Recall@3 | 71.6% | 79.0% | **85.2%** |
| 第一跳 Recall@16 | 86.6% | 91.0% | **95.2%** |
| 第二跳 Recall@3 | 60.2% | 63.4% | **72.0%** |
| 第二跳 Recall@16 | 89.8% | 78.6% | **91.4%** |

E5 在第一跳更强，BM25 在第二跳 Top16 更强；RRF 利用二者互补性，在两个步骤上都取得最高召回。

## 4. 500 条严格二跳链结果

| 指标 | 原 SVD32 系统 | Hybrid-RAG | 变化 |
|---|---:|---:|---:|
| 第一跳 Top3 block recall | 74.6% | **85.2%** | +10.6 pp |
| bridge 正确率 | 67.2% | **74.4%** | +7.2 pp |
| 动态第二跳 Top16 recall | 69.8% | **80.4%** | +10.6 pp |
| 16 个候选中至少一个正确 | 48.8% | **56.6%** | +7.8 pp |
| verifier 最终答案正确率 | 36.4% | **43.6%** | +7.2 pp |

最终正确率为 **218/500**。与原系统逐题比较：

- Hybrid-RAG 胜 56 条；
- 原系统胜 20 条；
- 424 条持平；
- McNemar exact `p=4.37e-5`。

因此 7.2 个百分点的提升具有统计显著性，不是 500 条样本上的随机波动。

## 5. 误差分解

| 条件 | 第二跳 Top16 recall / 最终正确率 |
|---|---:|
| bridge 正确时，第二跳 Top16 recall | **91.67%** |
| bridge 错误时，第二跳 Top16 recall | **47.66%** |
| 第二跳召回成功时，最终正确率 | **53.23%** |
| 第二跳召回失败时，最终正确率 | 4.08% |

主要结论：

1. RAG 首先改善第一跳召回，因此 bridge 正确率随之提升。
2. 正确 bridge 条件下，第二跳检索已经接近 92%；继续单纯提高检索召回的空间变小。
3. 16 个候选中有正确答案的比例是 56.6%，但 verifier 只保留到 43.6%，两者仍差 13.0 个百分点。
4. 下一阶段的主要瓶颈已经部分转移到候选抽取和 verifier，而不再只是 retriever。

## 6. 计算与时间

| 项目 | 结果 |
|---|---:|
| E5 索引构建 | 43.84 s，一次性 |
| E5 索引大小 | 58 MB |
| BM25 索引构建 | 约 19--20 s，一次性 |
| 2,000 条 gold-state query 批量检索 | 约 5.82 s，2.91 ms/query |
| 500 条动态第二跳批量检索 | 约 2.33 s，4.65 ms/query |
| 第一跳 8B bridge，4 卡整批 | 104.4 s |
| 第二跳 8B x 16 候选生成，4 卡整批 | 1,010.6 s |
| 16 候选 verifier，4 卡整批 | 400.0 s |

批量检索时间不包括一次性建库，也不等同于单请求端到端延迟。完整系统的大部分计算仍来自 8B 的 16 路候选生成和 verifier；更换成 E5+BM25 后，检索本身只占很小部分。

每题读取预算与原系统相同：第一跳 3 x 256 tokens，第二跳最多 16 x 256 tokens，合计最多 **4,864 retrieval tokens**，约占 10M 全文的 0.049%。

## 7. 对当前研究结论的影响

这次结果否定了一个可能的强结论：**当前四通道 SVD32 检索器并没有优于标准 RAG。** 在相同下游链路中，Hybrid-RAG 的最终正确率更高。

但它没有否定逐步动态检索的核心想法。相反，结果进一步支持：

- 每一步只读取少量证据并动态刷新 query 是可行的；
- 10M 全文不需要一次性交给模型；
- 第一跳状态质量会系统性影响第二跳；
- 强外部语义检索与词法检索的互补性非常重要。

下一步最有价值的实验不是继续把弱 BM25 入口与 SVD32 单独优化，而是：

1. 以 `BM25 + E5 RRF` 作为冻结的强全局入口；
2. 只在其候选集上加入四通道 SVD32 residual rerank；
3. 在 dev 上训练或校准融合权重，在 test 上冻结评估；
4. 要求新方法同时超过 43.6% 最终答案和 85.2%/80.4% 两步召回；
5. 单独优化 verifier，缩小 56.6% candidate oracle 与 43.6% 最终值之间的差距。

## 8. 复现文件

主要代码：

- `src/run_external_embedding_retrieval.py`
- `src/summarize_rag_baseline.py`
- `src/prepare_allhead_block_branches.py`
- `src/evaluate_global_step_branch_generation.py`
- `src/prepare_verified_chained_answer_steps.py`
- `src/score_candidate_support_distributed.py`

服务器主要输出：

- `outputs/rag_e5_base_index_v1/`
- `outputs/rag_e5_base_goldstate_retrieval_v1/`
- `outputs/rag_e5_hybrid_bridge8b_test500_v1/`
- `outputs/rag_e5_hybrid_strict_chain500_v1/`
- `outputs/rag_e5_hybrid_strict_second_retrieval_v1/`
- `outputs/rag_e5_hybrid_answer_direct_extract16_test500_v1/`
- `outputs/rag_e5_hybrid_answer_direct_extract16_support_scores_v1/`
- `outputs/rag_e5_hybrid_strict_chain500_summary_v1.json`
