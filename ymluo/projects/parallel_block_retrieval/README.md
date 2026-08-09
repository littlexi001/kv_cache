# 真实 Q/K 的 10M -> 10K 并行 Block 检索

本项目验证导师提出的 `1B -> 1M` 的缩小版问题：

```text
9,999,872 个真实文本 tokens
-> 39,062 个 256-token blocks
-> 检索 39 个 blocks（9,984 tokens）
```

当前主结果是 2026-07-10 的 BM25 + question-likelihood + pre-RoPE multi-Q SVD32 混合实验。所有 Q/K 均来自 Qwen3-0.6B 的真实前向，没有合成高斯向量。

| 指标 | 旧 post-RoPE `full128` | 当前深度方案 |
| --- | ---: | ---: |
| answer block Recall@39 | 0.00% | **82.81%** |
| mean answer NLL | 4.6713 | **3.2783** |
| NLL delta vs original | +1.9178 | **+0.5247** |

当前方案已经修复 `full128` 完全找不到答案 block 的问题，但 NLL 仍有明显损失，不能宣称实现了 10M -> 10K 近乎无损推理。完整修复分析见 `ymluo/doc/section127_parallel_block_retrieval_solution_20260710.md`。

## 数据清洗

第一版真实实验错误地加入了 LongBench `triviaqa`。该文件不是普通长文档 QA：`context` 是多组 `Passage/Question/Answer` few-shot demonstrations，真正待回答的 passage 和问题位于 `input`。这会让统一以 `Answer:` 结尾的 query Q 检索历史 `Answer:\n` 分隔符。

污染版的 2,496 个 full128 Top-39 槽位中：

```text
99.88% 来自 triviaqa
99.28% 含 Answer:\n
97.16% 的最高分 K token 是 Answer + :\n
```

因此污染版语义检索结果作废，只保留系统运行记录。

清洗版做了以下处理：

1. 移除 `triviaqa`，加入兼容的 `multifieldqa_en`；
2. 数据构造器默认拒绝 context 或 input 中独立成行的 `Passage:/Question:/Answer:` 模板；
3. 使用 QMSum 和 GovReport 的唯一真实文档补足干扰语料，它们不产生 QA query；
4. context 按 SHA-256 去重，不重复文档，不做合成填充；
5. 对服务器最终接受的 982 条记录重新扫描，模板记录数为 0。

## 清洗版数据

| 项目 | 数值 |
| --- | ---: |
| tokens | 9,999,872 |
| blocks | 39,062 |
| block size | 256 |
| unique source records | 982 |
| answer-aligned candidates | 535 |
| evaluation queries | 64 |
| output budget | 39 blocks / 9,984 tokens |

Block 构成：

| 数据集 | Blocks | 用途 |
| --- | ---: | --- |
| hotpotqa | 8,834 | QA + distractor |
| 2wikimqa | 4,823 | QA + distractor |
| musique | 10,528 | QA + distractor |
| qasper | 2,602 | QA + distractor |
| narrativeqa | 1,148 | QA + distractor |
| multifieldqa_en | 3,192 | QA + distractor |
| qmsum | 1,657 | distractor only |
| gov_report | 6,278 | distractor only |

10M 是多个独立 records 的全局索引，不是一条 10M causal sequence。每个 record 最多 16K tokens，并在自己的位置坐标内独立 prefill。

## 旧版 Q/K 与 K-SVD

实验使用四个 layer/query-head profile：

```text
L3/H10 -> KV H5
L21/H8 -> KV H4
L6/H7 -> KV H3
L16/H14 -> KV H7
```

对 8,192 个真实 calibration K tokens 拟合 centered SVD：

```text
K_centered = U Sigma V^T
q_r = q V_r
k_r = k V_r
score_r(q, k) = (q V_r) (k V_r)^T
```

Rank64 的能量保留率为 `96.93%, 98.74%, 99.59%, 97.05%`。检索仍投影 raw q/k，以保持与前期方法一致。

代码中的 `full128` 只表示上述四个 profile 的 128 维参考分数，不是全模型 attention oracle。当前 block score 在 block token 和四个 profile 上取 raw-logit 最大值。

## 旧版 post-RoPE 单 Q 基线

1 卡质量结果，多卡返回集合一致：

| 方法 | full128 Top-39 recall | full128 block mass | 相对 full128 mass | answer-block recall |
| --- | ---: | ---: | ---: | ---: |
| `full128` | 100.00% | 25.012% | 100.00% | 0.00% |
| `svd32` | 75.40% | 24.260% | 97.00% | 0.00% |
| `svd64` | 81.05% | 24.583% | 98.28% | 0.00% |
| `svd32_rerank` | 100.00% | 25.012% | 100.00% | 0.00% |
| `qabs8` | 1.56% | 0.730% | 2.92% | 0.00% |

Rank32 选 2%（782 blocks）后，full-QK rerank 完整恢复了 full128 Top-39。这证明 SVD32 是有效的 attention-score proposal，但 full128 本身没有找到语义证据。

## 失败诊断

清洗后不再有单一 TriviaQA 数据集劫持，但问题仍然存在：

```text
64 queries x 39 = 2,496 个返回槽位
只覆盖 254 个不同 blocks
不同 query Top-39 平均 Jaccard = 0.320
所属 source record 命中 = 2/64
answer block 命中 = 0/64
```

2,496 个最终 block 分数全部由 L6/H7 profile 决定。最高分 token 主要是自然文本中的 `answer is`、`Answer:`、引号和冒号。64 个 query Q 的平均余弦相似度约为 0.670，因为每个 query 都只使用同一个 `Answer:` 末尾 token 的 Q。

在正确 source record 内部，full128 对 answer block 的 Top-10 recall 为 60.94%，随机选择期望为 58.91%；当前一个 Q、四个 head 和 block-max 的语义排序信号仍然很弱。

## 旧版模型 NLL

| 上下文 | Mean answer NLL | Delta vs original |
| --- | ---: | ---: |
| original source | 2.7536 | 0.0000 |
| source-oracle 10K | 2.7261 | -0.0275 |
| `full128` 10K | 4.6713 | +1.9178 |
| `svd32` 10K | 4.8560 | +2.1025 |
| `svd32_rerank` 10K | 4.6713 | +1.9178 |

`source-oracle 10K` 不能被解释为强压缩证据：64 个 query 的 source 平均只有 36 blocks，37 条 source 本来就不超过 39-block 预算。全局检索 NLL 明显恶化，当前没有实现 10M -> 10K 近乎无损推理。

## 旧版多卡结果

| 方法 | 1 卡(s) | 2 卡 | 4 卡 | 8 卡 | 8 卡效率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full128` | 0.4899 | 1.97x | 3.85x | 7.39x | 92.4% |
| `svd32` | 0.4865 | 1.96x | 3.85x | 7.34x | 91.7% |
| `svd64` | 0.4871 | 1.96x | 3.84x | 7.26x | 90.8% |
| `svd32_rerank` | 1.1717 | 1.98x | 4.79x | 8.23x | 102.9% |
| `qabs8` | 0.8078 | 2.01x | 4.02x | 7.92x | 99.0% |

时间覆盖 64 个 query 的批量扫描，warmup 1 次、重复 3 次取中位数，不包含索引加载。`svd32_rerank` 的轻微超线性来自分片后 kernel/cache 形状变化，不能外推为稳定的硬件效率。

## 当前混合方案

旧 reference 已按以下方式修复：

1. BM25 对真实文本做 block 和 record 表层路由；
2. Qwen3-0.6B 用 `P(question | record)` 重排 BM25 Top-5 records，不读取参考答案；
3. 记录真实 causal prefill 中 RoPE 之前的 Q/K，避免跨独立 record 的位置相位污染；
4. 从问题内容选择最多 16 个 Q，而不是只取 `Answer:` 分隔符；
5. 在 32 维 K-SVD 空间计算归一化 multi-Q late interaction；
6. 语义排序只负责候选 record 内的 block 选择，全局词法结果负责补位和风控。

| 方法 | answer block Recall@39 | Mean answer NLL | Delta vs original |
| --- | ---: | ---: | ---: |
| `bm25_record39` | 73.44% | 3.5501 | +0.7965 |
| `risk_bm25_svd32` | 75.00% | 3.3445 | +0.5909 |
| `deep_ql_record39_svd32` | **82.81%** | **3.2783** | **+0.5247** |

Question-likelihood 将 record Top-1 recall 从 BM25 的 68.75% 提升到 78.13%。pre-RoPE SVD32 全局扫描的 1/2/4/8 卡耗时为 0.5815/0.2956/0.1523/0.0788 秒，8 卡加速 7.38x、并行效率 92.20%。该计时覆盖 64 个 query 同时扫描 10M K tokens，不包含离线 profiling 和索引加载。

## 全层全 Head 共识实验

进一步对 28 层 x 16 query heads 分别做独立 Top-16 检索。GQA 共享索引只存 28 层 x 8 KV heads，不复制 K，也不跨 head 比较 raw 分数。

| 候选或最终结果 | Answer recall |
| --- | ---: |
| 固定 4 heads Top-16 候选并集 | 35.94% |
| 全 448 heads Top-16 候选并集 | **71.88%** |
| 固定 4 heads 独立 RRF Top-39 | 34.38% |
| 全 head 最佳多数共识 Top-39 | 31.25% |

全 heads 确实发现了大量固定 4 heads 漏掉的证据，但多数共识会删除只有少数专业 heads 支持的答案：被召回后又丢弃的 gold blocks 平均只得到 2.01 层、2.45 heads 支持，而进入 Top-39 的平均门槛约为 6.69 层、9.86 heads。因此下一步应做专业 head 路由或 head-group 配额拼接，而不是继续增强多数票。

全 head SVD32 索引约 143.36 GB，8 卡 profiling 约 300 秒，8 卡流式检索约 300 秒。详细结果见 `ymluo/doc/section128_all_head_consensus_retrieval_20260711.md`。

## 运行

最终方案一键入口：

```bash
cd /home/fdong/ymluo/projects/parallel_block_retrieval
bash scripts/run_hybrid_retrieval_solution_server.sh
```

全 head 共识实验入口：

```bash
bash scripts/run_all_head_consensus_server.sh
```

脚本自动选择空闲 GPU。8 卡不全空闲时会使用检测到的可用卡，依次运行语料准备、BM25、真实 pre-RoPE Q/K profiling、question-likelihood record 重排、SVD32 block 检索和答案 NLL。

主要归档结果：

```text
outputs/real_longbench_docqa_10m_clean_bm25_20260710_v1/
outputs/real_longbench_docqa_10m_prerope_colbert_scaling_20260710_prerope_v1/
outputs/real_longbench_docqa_10m_record_question_nll_20260710_v1/
outputs/real_longbench_docqa_10m_hybrid_bm25_prerope_svd32_20260710_v4/
```

清洗和旧 reference 诊断见 `ymluo/doc/section126_parallel_block_retrieval_10m_20260710.md`；混合检索修复见 `ymluo/doc/section127_parallel_block_retrieval_solution_20260710.md`；全层全 head 实验见 `ymluo/doc/section128_all_head_consensus_retrieval_20260711.md`。
