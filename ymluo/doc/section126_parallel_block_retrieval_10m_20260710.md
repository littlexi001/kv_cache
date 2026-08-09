# Section 126：真实 Q/K 的 10M -> 10K 并行 Block 检索与数据清洗复验（2026-07-10）

> 本文记录清洗与旧 post-RoPE 单 Q reference 的失败诊断。后续修复方案已将 answer block Recall@39 从 0% 提升到 82.81%，详见 `section127_parallel_block_retrieval_solution_20260710.md`。

## 1. 当前结论

本项目验证导师提出的 `1B -> 1M` 的缩小版：

```text
10M real-text tokens -> 10K retrieved tokens
```

截至 `clean_v3`，可以确认三件事：

1. 8 卡分片扫描、局部 Top-K 和全局 merge 的系统框架有效；
2. SVD32 可以有效近似选定 Q/K reference，2% candidate 加 full-QK rerank 可恢复 100% reference Top-39；
3. 当前 Q/K reference 本身不是可靠的跨 record 语义路由器，answer block recall 为 0/64，不能宣称近乎无损推理成功。

第一版高斯向量实验只保留为系统 smoke。第一版真实 Q/K 实验又受到 LongBench `triviaqa` few-shot 模板污染，其语义结果也作废。本文以清洗版真实实验为主结果。

## 2. 为什么必须重新清洗

### 2.1 TriviaQA 的任务格式被误解

服务器上的 LongBench `triviaqa.jsonl` 不是普通文档 QA：

```text
context = 多组 Passage/Question/Answer demonstrations
input   = 当前待回答的 passage + question + 空 Answer
answers = 当前问题参考答案
```

旧构造器统一假设 `context` 是知识文档、`input` 是纯问题，因此把 demonstrations 建成了 K index，又把包含完整 passage 的 input 当成 question。

### 2.2 污染版的直接证据

污染版 full128 一共返回 `64 x 39 = 2,496` 个 block 槽位：

| 诊断 | 结果 |
| --- | ---: |
| 来自 TriviaQA 的槽位 | 2,493 / 2,496 |
| 含 `Answer:\n` 的槽位 | 99.28% |
| 最高分 K token 为 `:\n` | 2,476 / 2,496 |
| 最高分 pair 为 `Answer` + `:\n` | 2,425 / 2,496 |

Query 统一以 `Answer:` 结尾，并且只使用最后一个冒号 token 的 Q。因此旧 full128 主要在匹配历史答案分隔符，而不是当前问题证据。

只屏蔽 `Answer:\n` 后，所属 record 命中率从 1.56% 升到 7.81%，但 answer block recall 只从 1.56% 升到 3.13%。这说明模板是一个强伪特征，但不是唯一问题。

## 3. 清洗规则

清洗不是正则删除 TriviaQA demonstrations，因为那会改变任务语义。实际策略是：

1. 从 query 数据集中移除 `triviaqa`；
2. 加入 schema 兼容的 `multifieldqa_en`；
3. 构造器默认拒绝 context 或 input 中独立成行的 `Passage:/Question:/Answer:`；
4. context 使用 SHA-256 去重，避免同一文档因多个问题被重复写入；
5. 使用 QMSum 和 GovReport 的唯一真实文档补足 distractors；
6. QMSum/GovReport 不满足 exact-answer query 条件，因此不会产生评估问题；
7. 不使用合成文本、重复填充或合成向量。

最终又根据 `records.jsonl` 回读原始源文件，验证接受的 982 条记录中模板记录数为 0。

清洗代码：

```text
ymluo/projects/parallel_block_retrieval/src/prepare_real_longbench_corpus.py
```

服务器入口：

```text
ymluo/projects/parallel_block_retrieval/scripts/run_clean_real_qk_pipeline_server.sh
```

## 4. 清洗版 Corpus

| 项目 | 数值 |
| --- | ---: |
| requested tokens | 10,000,000 |
| actual tokens | 9,999,872 |
| block size | 256 |
| blocks | 39,062 |
| unique records | 982 |
| answer-aligned candidates | 535 |
| evaluation queries | 64 |
| output budget | 39 blocks = 9,984 tokens |

Block 构成：

| 数据集 | Records | Blocks | Query 角色 |
| --- | ---: | ---: | --- |
| hotpotqa | 171 | 8,834 | 是 |
| 2wikimqa | 170 | 4,823 | 是 |
| musique | 170 | 10,528 | 是 |
| qasper | 134 | 2,602 | 是 |
| narrativeqa | 20 | 1,148 | 是 |
| multifieldqa_en | 112 | 3,192 | 是 |
| qmsum | 35 | 1,657 | 否，仅 distractor |
| gov_report | 170 | 6,278 | 否，仅 distractor |

64 个 query 的分布为：2WikiMQA 13、HotpotQA 13、MuSiQue 12、Qasper 12、MultiFieldQA 12、NarrativeQA 2。

Gold answer block 仍定义为：block 与 reference answer 在 source context 中的大小写不敏感精确匹配位置重叠。这是弱标签，不等价于完整 supporting facts。

## 5. 10M 的含义

这不是一条 10M causal sequence：

1. 每个 record 最多保留 64 blocks，即 16K tokens；
2. 每个 record 独立执行 causal prefill，位置从 0 开始；
3. 抽取真实 post-RoPE K，再按 256 tokens 切 block；
4. 所有 records 的 K 合并成全局 10M index；
5. Query Q 由其 source record 加 question 的前向得到；
6. 使用该 Q 搜索全部 39,062 blocks。

因此当前实验更接近“多个独立 KV records 的全局检索”，不是单一 10M 长上下文压缩，也不是只给问题的外部 RAG。

## 6. 真实 Q/K Profile

模型为 Qwen3-0.6B，head dimension 128。沿用四个 profile：

| Layer | Query head | KV head |
| ---: | ---: | ---: |
| 3 | 10 | 5 |
| 21 | 8 | 4 |
| 6 | 7 | 3 |
| 16 | 14 | 7 |

每条 record 完整 prefill，在 q/k norm 后捕获 pre-RoPE 状态，再使用模型自己的 rotary embedding 得到真实 post-RoPE Q/K。

Query 输入为：

```text
[source record]
Question: [question]
Answer:
```

当前 `query_vector_tokens=1`，只使用最后一个 `Answer:` 冒号 token 的 Q。每个 block 前 16 tokens 不参与检索分数。

8 卡 profiling 最慢 shard 为 67.51 秒，单卡吞吐约 18.5K 至 19.5K tokens/s。整个 clean_v3 流水线耗时 263.4 秒。

## 7. K-SVD

用最前面的 8,192 个真实 K tokens 拟合 centered SVD：

```text
K_c = K - mean(K)
K_c = U Sigma V^T
```

检索时沿用前期方法，投影 raw q/k：

```text
q_r = q V_r
k_r = k V_r
score_r = (q V_r) (k V_r)^T
```

Rank64 能量保留：

| Profile | Energy |
| --- | ---: |
| L3/H10 | 96.93% |
| L21/H8 | 98.74% |
| L6/H7 | 99.59% |
| L16/H14 | 97.05% |

四 profile 的 FP16 索引理论大小为：raw K128 约 10.24 GB、KV64 约 5.12 GB、KV32 约 2.56 GB。

## 8. 检索定义

当前 block score 为：

```text
score(block) = max over selected profiles and tokens in block
```

方法：

| 名称 | 定义 |
| --- | --- |
| `full128` | 四个指定 profile 的 128 维 QK reference |
| `svd32` | Rank32 直接 Top-39 |
| `svd64` | Rank64 直接 Top-39 |
| `svd32_rerank` | Rank32 选 2%（782 blocks），候选内 full128 Top-39 |
| `qabs8` | 真实 Q 绝对值最大的 8 通道 partial QK |

需要纠正旧文档用词：`full128` 不是全模型 attention oracle。模型不会把不同层的 raw logits 直接取最大值；它只是在当前自定义 block score 下的全维 reference。

`exact block mass` 也是对 full128 block-max 分数做 softmax 的代理指标，不是真实全模型 token attention mass。

## 9. 清洗版检索质量

| 方法 | Reference Top-39 recall | Block mass | 相对 reference mass | Answer block recall |
| --- | ---: | ---: | ---: | ---: |
| `full128` | 100.00% | 25.012% | 100.00% | 0.00% |
| `svd32` | 75.40% | 24.260% | 97.00% | 0.00% |
| `svd64` | 81.05% | 24.583% | 98.28% | 0.00% |
| `svd32_rerank` | 100.00% | 25.012% | 100.00% | 0.00% |
| `qabs8` | 1.56% | 0.730% | 2.92% | 0.00% |

清洗后：

1. SVD32 direct recall 从污染版 69.43% 提高到 75.40%；
2. SVD32 2% candidate 完整包含 reference Top-39，rerank recall 从 98.12% 到 100%；
3. QABS8 仍然失败；
4. 所有方法 answer block recall 均为 0。

污染版和清洗版 query 集不同，因此 NLL 和 answer recall 不能作为严格配对提升；但模板吸引模式已经被明确移除。

## 10. 清洗后的剩余偏置

Full128 的 2,496 个返回槽位：

| 诊断 | 结果 |
| --- | ---: |
| unique selected blocks | 254 |
| 平均 query-pair Jaccard | 0.320 |
| 命中所属 source record | 2/64 |
| 命中 answer block | 0/64 |

不再有单一 TriviaQA 数据集控制结果，返回槽位分布在八个数据集。但若干 block 对几乎所有 query 都是高分，例如 block 14561 和 22954 出现在 64/64 个 Top-39 中。

新的最高分 token 主要是自然文本中的：

```text
answer is
Answer:
Source:
引号、冒号、右括号
```

64 个 query Q 的平均余弦相似度为 0.670。所有 2,496 个最终 block 分数仍由 L6/H7 profile 决定，说明跨层 raw-max 完全退化为单 profile。

## 11. 正确 Record 内的对照

只在每个问题自己的 source record 内使用相同 full128 排序：

| K | Full128 answer recall | 随机选择期望 |
| ---: | ---: | ---: |
| 1 | 10.94% | 13.05% |
| 5 | 42.19% | 37.35% |
| 10 | 60.94% | 58.91% |
| 20 | 78.13% | 81.19% |
| 39 | 98.44% | 93.47% |

Source 平均只有 36 blocks，Top-39 基本等于全选。小 K 下 full128 只略高于或低于随机，因此当前 query token/head/block score 对答案字符串 block 没有稳定语义排序能力。

这不否定之前的 SVD attention-recovery 结果。SVD 可以恢复 full-QK 高分 token，但“高 attention”不自动等于“回答问题所需 evidence”。

## 12. Answer NLL

| 模式 | Mean NLL | Median | Delta vs original | NLL 不上升比例 |
| --- | ---: | ---: | ---: | ---: |
| original source | 2.7536 | 2.2136 | 0.0000 | 100.0% |
| source-oracle 10K | 2.7261 | 2.3189 | -0.0275 | 78.1% |
| `full128` 10K | 4.6713 | 4.6975 | +1.9178 | 15.6% |
| `svd32` 10K | 4.8560 | 5.0039 | +2.1025 | 17.2% |
| `svd32_rerank` 10K | 4.6713 | 4.6975 | +1.9178 | 15.6% |

`source-oracle 10K` 不能证明 10M -> 10K 容量充分：

```text
source 平均 36 blocks
source 中位数 28 blocks
37/64 条 source <= 39 blocks
只有 27/64 条发生了 source 内压缩
```

Full128 和 rerank 上下文完全相同，因此 NLL 相同。实际检索比原始 source 增加约 1.92 NLL，近乎无损目标没有达到。

## 13. 多卡结果

| 方法 | 1 GPU(s) | 2 GPU speedup | 4 GPU speedup | 8 GPU speedup | 8 GPU efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full128` | 0.4899 | 1.97x | 3.85x | 7.39x | 92.4% |
| `svd32` | 0.4865 | 1.96x | 3.85x | 7.34x | 91.7% |
| `svd64` | 0.4871 | 1.96x | 3.84x | 7.26x | 90.8% |
| `svd32_rerank` | 1.1717 | 1.98x | 4.79x | 8.23x | 102.9% |
| `qabs8` | 0.8078 | 2.01x | 4.02x | 7.92x | 99.0% |

每个数字是 64 queries 批量扫描、warmup 1 次、重复 3 次的中位数，并取最慢 rank；不包含索引加载。Rerank 的轻微超线性来自分片后 kernel/cache 形状变化，不能解释为普遍硬件效率。

当前未融合实现中，SVD32 与 full128 直接扫描时间接近。低秩已确认的收益是索引存储和 candidate proposal，不是这版 Python kernel 的端到端算术加速。

## 14. 为什么跨 Record post-RoPE 有问题

Post-RoPE 分数可写成：

```text
(R_m q)^T (R_n k) = q^T R_(n-m) k
```

同一 causal record 中，`n-m` 是真实相对位置。不同 records 都从位置 0 开始，跨 record 的 `n-m` 没有共同坐标意义。39K blocks 中的偶然相位和极值 token 会淹没所属 record。

因此合理分工应是：

```text
跨 record：BM25 / embedding / pre-RoPE content router
同一 record 内：post-RoPE QK / SVD32 / full-QK rerank
```

## 15. 下一轮实验

在继续比较 rank32/rank64 前，应先让 full-dimension reference 对语义 evidence 有效：

1. 将 `query_vector_tokens` 改为问题最后 8、16、32 个内容 token；
2. 排除 `Question:/Answer:`、换行和标点 query token；
3. 比较 last-Q、multi-Q max、multi-Q mean 和 IDF-weighted Q；
4. 跨 record 建立 BM25、embedding 和 pre-RoPE 三个 router baseline；
5. post-RoPE QK 只对 router shortlist 使用；
6. 每个 layer/head 独立 Top-K 后 union，不跨层直接 max raw logits；
7. 对每个 head 做 score calibration，比较 z-score、percentile 和 per-head softmax；
8. 比较 block max、top-m mean 和 log-sum-exp；
9. 使用 supporting facts、answer generation 和 NLL 联合评估；
10. 构造 source 显著长于 10K 的样本，否则无法证明强压缩。

推荐最终架构：

```text
10M real blocks
-> lexical / semantic record router
-> candidate records or 0.5%~5% blocks
-> SVD32 QK proposal
-> full-QK rerank
-> risk gate / budget expansion
-> 39 final blocks
```

## 16. 文件与输出

主要代码：

```text
ymluo/projects/parallel_block_retrieval/src/prepare_real_longbench_corpus.py
ymluo/projects/parallel_block_retrieval/src/profile_real_qk.py
ymluo/projects/parallel_block_retrieval/src/run_real_qk_retrieval.py
ymluo/projects/parallel_block_retrieval/src/evaluate_retrieved_answer_nll.py
ymluo/projects/parallel_block_retrieval/src/summarize_real_scaling.py
ymluo/projects/parallel_block_retrieval/scripts/run_clean_real_qk_pipeline_server.sh
```

本地清洗版结果：

```text
ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_clean_record64/summary.json
ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_clean_postrope_qk64_profile/summary.json
ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_clean_scaling_20260710_clean_v3/
```

服务器完整 profile 保留在：

```text
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_clean_postrope_qk64_profile
```
