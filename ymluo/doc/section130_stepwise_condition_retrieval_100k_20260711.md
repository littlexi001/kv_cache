# Section 130：100K/500 可控合成数据上的逐步条件检索（2026-07-11）

## 1. 研究问题

本节验证一个比“一次从长上下文中找齐所有答案”更具体的假设：

> 模型在单个推理步骤中只需要少数当前前提；检索器应先恢复当前状态，再根据新状态检索下一条条件，而不是把完整问题一次性映射到固定 Top-K。

预算按以下公式限制：

```text
budget(full_tokens) = max(1000, 0.001 * full_tokens)
```

对 100K tokens，允许上限是 1000 tokens。本文 block 大小为 256 tokens；固定 Top-3 使用 768 tokens，动态停止版本的单跳和双跳分别只使用 256 和 512 tokens。

## 2. 数据不是合成向量

实验使用合成文本，但不使用高斯向量或预先构造的 query 相似向量：

1. 文本事实、干扰项和问题由程序控制生成；
2. BM25 直接读取 Qwen tokenizer 解码后的文本；
3. 模型判别由真实 `Qwen3-0.6B` 前向计算；
4. 之前的 all-head/SVD 实验使用真实模型 Q/K；
5. gold block 由生成器在写入事实时记录，不由检索器反推。

正式 held-out 数据目录：

```text
server: /home/fdong/ymluo/projects/parallel_block_retrieval/data/synthetic_controlled_100k_500_v2_heldout
```

| 项目 | 数值 |
| --- | ---: |
| 实际 tokens | 99,840 |
| blocks | 390 |
| block tokens | 256 |
| queries | 500 |
| train/dev/test | 300/100/100 |
| lexical/semantic/hard-negative/multihop | 各 125 |
| 单证据 query | 375 |
| 双证据 query | 125 |
| answer 缺失或 hard-negative 泄漏 | 0 |

v2 的 train、dev、test 使用互不重叠的问题、证据和否定状态措辞。test 特意使用 `superseded`、`annulled`、`expired`、`retired`、`simulation` 等 v1 规则表中没有的表达，用于检查模板过拟合。

## 3. 为什么必须使用 held-out 模板

v1 中，规则型逐步检索在 Top-3 下达到：

| 指标 | BM25 block | 规则逐步检索 |
| --- | ---: | ---: |
| 任一证据召回 | 89.0% | 100.0% |
| 全部证据召回 | 64.4% | 99.0% |
| hard-negative 命中 | 93.2% | 1.4% |

但 v1 的三种模板会同时出现在 train/dev/test。换成模板互斥的 v2 后，旧规则在 test 多跳题上的全部证据召回直接降为 0%。这证明 v1 的 99% 不能作为泛化证据。

v2 固定 Top-3 结果：

| 方法 | 全体全部证据召回 | test 全部证据召回 | test 多跳全部证据召回 |
| --- | ---: | ---: | ---: |
| block BM25 | 70.4% | 75.0% | 0.0% |
| 旧规则逐步检索 | 89.6% | 75.0% | 0.0% |

test 的 75% 正好来自 75 条单跳题；25 条多跳题没有一条找齐两块证据。因此 v2 是当前更可信的开发基准。

## 4. 失败的模型版本

### 4.1 一次性语义有效性精排

第一版使用句子 BM25 产生 16 个候选，再让 Qwen 判断候选是否“当前有效且有用”。它能把 test hard-negative 命中率从 BM25 的 35% 降到约 2%，但 test 多跳全部证据召回仍为 0%。

原因不是候选池完全没有证据，而是完整问题中的关系词（例如 `active wayfinding light`）召回大量“其他实体的直接答案”，淹没了只共享编号的 alias 映射。

### 4.2 稀有锚点候选并集

加入最高 IDF 锚点通道后，第一跳候选 oracle 从 80% 提升到 100%，但模型仍会把没有锚点的“直接答案形状”句排在映射句前。候选覆盖不等于最终检索成功。

### 4.3 自由文本下一跳生成

即使加入 few-shot，0.6B 模型仍经常把旧编号重新写进下一跳，而不是复制前提中新出现的实体。自由生成不是这里可靠的状态更新器，也明显增加延迟。

## 5. 当前方案：锚点状态机

当前方案把检索拆成两个有明确职责的状态。

### 5.1 第一状态：解析原问题锚点

1. 用 word unigram/bigram BM25 建立句子级倒排索引；
2. 从问题中选择最高 IDF n-gram；含数字的 identifier 优先；
3. 要求锚点的全部 token 在候选句中共现，避免 `Zephyr-0380` 命中只含 `Zephyr` 的句子；
4. 通用 invalid-status 风险层先惩罚作废、过期、模拟、缺失等高精度表层模式；
5. Qwen 再在剩余冲突候选中判断“当前、有效、正向”的映射或事实；风险词表描述状态类别，不包含任务实体、答案或关系模板。

### 5.2 第二状态：新实体反馈

1. 从第一条前提中提取相对原问题新增的多词专名；
2. 若没有专名，再回退到新增的高 IDF n-gram；
3. 新实体作为第二跳强制锚点，原问题只提供剩余关系词；
4. 再做一次候选生成和 Qwen 有效性精排；
5. 完成性 margin 高于阈值时动态停止，不补齐固定 K。

该更新本质上是受约束的 pseudo-relevance feedback，不依赖 `vessel`、`beacon` 或某种固定证据句式。

伪代码：

```text
state_0 = original_question
anchor_0 = rare_exact_anchor(state_0)
candidates_0 = exact_channel(anchor_0) union relation_channel(state_0)
premise_0 = validity_rerank(candidates_0, need=resolve(anchor_0))

if complete(question, premise_0):
    return [block(premise_0)]

anchor_1 = novel_entity(premise_0, exclude=question)
state_1 = anchor_1 + remaining_relation(question)
candidates_1 = exact_channel(anchor_1) union relation_channel(state_1)
premise_1 = validity_rerank(candidates_1, need=state_1)
return [block(premise_0), block(premise_1)]
```

## 6. 逐轮结果

held-out test 多跳题共 25 条：

| 版本 | 第一跳候选 oracle | 第一跳选中 gold | 第二跳候选 oracle | 第二跳选中剩余 gold | 两证据全召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整问题一次性精排 | 80% | 0% | 60% | 12% | 0% |
| 稀有锚点候选并集 | 100% | 0% | 100% | 8% | 0% |
| 锚点候选限定 | 100% | 64% | 48% | 12% | 4% |
| 精确锚点 + 新实体反馈 | **100%** | **100%** | **100%** | **96%** | **96%** |

最后一行来自 CPU 上的 25 条 held-out 多跳正式运行；它不依赖自由文本下一跳生成。动态停止后，多跳最多返回 2 blocks，即 512 tokens。

固定 Top-3 会在两条正确证据后机械补一个 block，因此 hard-negative 指标会被无意义地恶化。正式方法改为动态 1/2-block 返回，不再把“未使用预算”视为必须填满的容量。

完整 100 条 v2 test 的动态结果为：

| 指标 | 数值 |
| --- | ---: |
| 平均返回 blocks | 1.25 |
| 平均返回 tokens | 320 |
| 最大返回 tokens | 512 |
| 任一证据召回 | 100.0% |
| 全部证据召回 | 99.0% |
| 证据比例召回 | 99.5% |
| hard-negative 命中 | 0.0% |
| 多跳全部证据召回 | 96.0% |

结果目录：

```text
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v2_model_cpu_test_v7_dynamic
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v2_eval_model_test_v7_dynamic
```

### 6.1 后续 challenge 与盲测

由于 v2 test 已参与迭代，又构造了两套仅替换 test 措辞的 challenge：

1. v3 暴露了孤立 Yes/No 判别对新状态词不稳定；观察失败后加入通用 invalid-status 表层风险过滤，v3 从 80% 提升到 95%，因此 v3 只能视为开发集；
2. v4 使用风险词表未直接覆盖的 `illustrative`、`contemplated`、`ceased before ratification`、`imaginary`、`withholds` 等表达，评估后不再修改方法，作为最终 blind test。

v4 blind test 结果：

| 方法 | 平均 blocks | 全部证据召回 | hard-negative 命中 | 多跳全部证据召回 |
| --- | ---: | ---: | ---: | ---: |
| BM25 Top-3 | 3.00 | 77.0% | 79.0% | 8.0% |
| precision 锚点状态机 | 1.50 | 92.0% | **1.0%** | 96.0% |
| safety Top-2 分支 | **2.00** | **99.0%** | 37.0% | 96.0% |

precision 状态机平均使用 384 tokens，最大 512 tokens。分任务全部证据召回为 lexical 100%、hard-negative 100%、multihop 96%、semantic-paraphrase 72%。最后一个数字表明当前主要瓶颈已不是候选覆盖：四类任务的第一跳 candidate oracle 都是 100%，而是 0.6B validity scorer 对隐式权威性/未生效状态的判断。

safety 模式在 completion margin 低时保留第一跳 Top-2，并最多加入一个第二跳 block。它平均使用 512 tokens、最大 768 tokens，把全部证据召回提高到 99%，代价是 37% query 同时带回声明的 hard-negative。两种模式都低于 1000-token 上限，应按对漏召回和冲突项的容忍度选择，而不是只发布一个 Top-K。

v4 的单卡模型阶段为 28.46 秒/100 queries，峰值显存约 2.09 GiB；BM25 对 500 queries 的构建加打分为 0.21 秒。该时间是批量吞吐记录，不是单 token 在线延迟保证。

blind 结果目录：

```text
/home/fdong/ymluo/projects/parallel_block_retrieval/data/synthetic_controlled_100k_500_v4_blind
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v4_model_gpu_test_v1_blind
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v4_eval_model_test_v1_blind
```

### 6.2 答案 NLL

使用两张空闲 3090 对 v4 的 100 条 blind test 运行真实 Qwen3-0.6B answer-token NLL。原始 source context 每条约 25K tokens；BM25 和动态方法严格使用各自返回的最多 3 blocks，gold-block oracle 也使用相同上限。

| 上下文 | 平均 answer NLL | 相对原上下文 | NLL 不升高比例 |
| --- | ---: | ---: | ---: |
| 原 25K source | 3.6775 | 0.0000 | 100% |
| gold-block oracle | 0.8719 | -2.8056 | 99% |
| BM25 Top-3 | 4.8503 | +1.1728 | 31% |
| precision 状态机 | 1.5696 | -2.1079 | 90% |
| safety Top-2 分支 | **1.2142** | **-2.4633** | **97%** |

两种动态检索都显著接近 gold oracle，并优于原 25K source。safety 模式虽然带回更多冲突项，但更高的证据召回使 NLL 进一步改善。该现象来自受控语料中的大量冲突事实和干扰项，说明短上下文可被模型有效利用，但不能外推成所有真实长上下文都一定受干扰。结果目录：

```text
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v4_answer_nll_full_2gpu
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/synthetic_controlled_100k_500_v4_answer_nll_risk_full_4gpu
```

### 6.3 真实 10M LongBench 迁移

未修改状态机，直接迁移到现有 clean LongBench 语料：9,999,872 tokens、39,062 blocks、64 queries，并严格限制最多 3 blocks/768 tokens。另一个版本复用已有 question-likelihood `risk_record` 路由，只在路由 record 内做逐步检索。

| 方法 | 平均 blocks | 任一证据召回 | 全部证据召回 | source record 命中 |
| --- | ---: | ---: | ---: | ---: |
| BM25 Top-3 | 3.00 | 29.69% | 14.06% | 75.00% |
| 路由后 precision 状态机 | 1.86 | 29.69% | 12.50% | 73.44% |
| 路由后 safety 状态机 | 2.38 | 29.69% | 14.06% | 73.44% |
| 路由后固定 3-block validity 排序 | 3.00 | **32.81%** | **15.63%** | 73.44% |

真实迁移没有复现合成 blind test 的高召回。它说明当前状态机擅长“已找到明确实体/状态候选后的局部链展开”，不能替代 10M 全局 record routing，也不能可靠处理真实 QA 中没有显式字符串桥接的证据。

为区分“全局 record 路由错误”和“正确 record 内仍排不出证据”，又使用标注的 source record 做了 oracle record 内 BM25：

| source-record oracle 内预算 | 任一证据召回 | 全部证据召回 |
| ---: | ---: | ---: |
| 1 block | 31.25% | 18.75% |
| 2 blocks | 39.06% | 20.31% |
| 3 blocks | 40.63% | 20.31% |
| 8 blocks | 65.63% | 37.50% |
| 16 blocks | 79.69% | 56.25% |
| 39 blocks | 95.31% | 79.69% |

因此，即便 record 路由完全正确，3-block 预算下的句级 BM25 也只有 40.63% 任一证据召回。真实失败不只是全局路由问题，还包括答案跨度与问题词面不重合、证据跨多个 block，以及“答案 block”标注相对生成所需充分上下文并不完备。

现在已加入带语料形状、文件时间戳、模型名和 BM25 参数校验的 `joblib` sidecar。对 39,062 blocks 解码、切出 329,348 句并构建索引的纯 CPU 冷启动实测为 39.95 秒；后续运行直接命中缓存。此前约 550 秒是索引构建和完整模型检索的混合墙钟时间，不能作为纯索引开销。真实数据还暴露了“路由 record 内第二跳候选为空”的边界条件，现已改为空批次短路并加入回归测试。

进一步把非数字问题从“精确锚点硬过滤”改成 16 个开放候选，并使用命中率更高的 `likelihood_record` 路由：

| likelihood-record 路由方法 | 平均 blocks | 任一证据召回 | 全部证据召回 |
| --- | ---: | ---: | ---: |
| BM25 + validity 固定 Top-3 | 3.00 | 31.25% | 10.94% |
| validity-only 固定 Top-3 | 3.00 | **35.94%** | **12.50%** |
| precision 动态阈值 3 | 1.80 | 25.00% | 10.94% |
| safety 动态阈值 3 | 2.42 | 28.13% | 10.94% |

`likelihood_record` 本身命中 source record 的比例为 78.13%。开放候选使固定 validity 的任一证据召回比旧路由结果 32.81% 略高，但动态停止仍明显过早。

最后用标注 `source_record` 做完全正确的 record 路由，隔离局部排序能力：

| source-record oracle 路由方法 | 平均 blocks | 任一证据召回 | 全部证据召回 |
| --- | ---: | ---: | ---: |
| BM25 + validity 固定 Top-3 | 3.00 | 45.31% | 21.88% |
| validity-only 固定 Top-3 | 3.00 | **48.44%** | **21.88%** |
| precision 动态阈值 3 | 1.78 | 35.94% | 18.75% |
| safety 动态阈值 3 | 2.39 | 40.63% | 20.31% |

相比 source-record 内纯 BM25 Top-3 的 40.63%/20.31%，validity-only 对“至少找到一块证据”有 7.81 个百分点收益，但完整证据只提高 1.57 个百分点。阶段 oracle 进一步显示：第一跳 16-block 候选池为 any 79.69%/all 53.13%，两阶段候选并集达到 any 85.94%/all 57.81%；实际状态机远低于该上限。因此真实数据的当前首要瓶颈是候选 reranker 和停止决策，不是单纯继续扩大词法候选池。后续应在小候选集上训练可校准排序器，并把真实 Q/K 的分 head 特征作为输入，而不是直接把合成任务的规则继续外推。

结果目录：

```text
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_stepwise_3block_v1
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_stepwise_3block_routed_v2
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_stepwise_3block_routed_open_v3
/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_stepwise_3block_source_oracle_open_v4
```

## 7. 与 attention head 研究的关系

`section129_head_function_stability_20260711.md` 发现：

1. semantic-evidence heads 主要位于 L19-L24；
2. 最强的观测 head 包括 L21/H13、L22/H7、L21/H11、L24/H11、L19/H13；
3. head 有稳定 prior，但实际关注目标仍然 query/context dependent；
4. 这些是 attention pattern，不是已证明的因果功能。

本项目已经做过一个重要反例：把这些 post-RoPE attention 上发现的 semantic heads 直接用于全局 pre-RoPE cosine 融合，Top-3 任一证据召回只有约 5%-6%，没有改善。不能把“某 head 在完整前向中关注语义证据”误解成“该 head 的独立 K 向量适合跨文档全局近邻检索”。

正确的使用位置应是：

1. 词法/稀有锚点先产生小候选池；
2. 在候选上下文内使用真实 post-RoPE semantic-head attention 作为激活先验或风险信号；
3. 不同 head group 保留独立 quota，避免多数投票删除少数专业 head 找到的证据；
4. 必须再做 link/head ablation 和 answer NLL，不能只报告 attention recall。

## 8. 并行框架

海量上下文应按两个并行维度拆分。

### 8.1 单 query 的 block-shard 并行

用于 1B tokens 场景中的单步低延迟：

```text
query state
  -> CPU/inverted exact + lexical routing
  -> each GPU scans its local block/K-summary shard
  -> local Top-K
  -> global merge of only scores and block ids
  -> small candidate reranker
```

真实 10M pre-RoPE SVD32 扫描已有同机证据：

| GPU | 64 queries 扫描 10M K tokens | 加速比 |
| ---: | ---: | ---: |
| 1 | 0.5815 s | 1.00x |
| 2 | 0.2956 s | 1.97x |
| 4 | 0.1523 s | 3.82x |
| 8 | 0.0788 s | 7.38x |

8 卡并行效率约 92.2%。这部分证明 block-shard + local Top-K/global merge 可以近线性扩展，但不证明 SVD 分数本身足够准确。

### 8.2 多 query 的 query-shard 并行

Qwen 候选有效性判别是 query 独立的，按 `query_index % world_size` 分配。最终 v4 相同计算路径的单次结果为：

| GPU | 100 queries 模型阶段 | 加速比 |
| ---: | ---: | ---: |
| 1 | 28.46 s | 1.00x |
| 2 | 11.04 s | 2.58x |
| 3 | 8.52 s | 3.34x |

1/2 卡两次 query CSV 的 SHA-256 完全一致，3 卡的核心 precision 指标也一致。超线性部分可能来自 GPU 动态频率、缓存和服务器干扰，不能外推；应补充多次重复中位数。每张峰值显存约 2.09 GiB。

在生产实现中，还应跨 query 合并 Yes/No prompt batch，减少当前逐 query 小 batch 的 kernel 启动开销。

### 8.3 从 block ID 到 KV 工作集

当前代码输出 block ID；部署时这些 ID 应直接映射到离线或 prefill 阶段已经存储的 K/V block，不需要重新对完整原文前向。Qwen3-0.6B 有 28 层、8 个 KV heads、head_dim 128，FP16 K+V 的理论体积为：

```text
28 layers * 2(K,V) * 8 KV heads * 128 dims * 2 bytes
= 114,688 bytes/token
= 112 KiB/token
```

因此：

| 工作集 | KV 体积 |
| --- | ---: |
| 1 block / 256 tokens | 28 MiB |
| 2 blocks / 512 tokens | 56 MiB |
| 1000 tokens | 109.4 MiB |
| 1B tokens 原始 FP16 KV | 约 104.3 TiB |

检索索引必须是远小于原始 KV 的 sidecar。在线系统应把选中的 1/2-block KV 固定在当前 reasoning state 内，连续生成多个 token；只有 completion/risk 状态变化时才更新工作集。若每个 token 都重新跨卡或跨机搬运 56 MiB，检索计算本身再快也会被通信吞吐抵消。

## 9. 风控策略

最终检索器不应只返回一个不可解释分数。建议保留以下风险量：

| 风险量 | 处理 |
| --- | --- |
| exact anchor 无候选 | 回退 relation BM25 + SVD semantic candidates |
| anchor Top-1/Top-2 margin 低 | 保留两个映射分支并行展开 |
| validity Yes/No margin 低 | 增加一个 safety block，但总量不超过 3 blocks |
| completion margin 接近阈值 | 继续一跳，避免过早停止 |
| 新实体提取为空 | 使用新增高 IDF n-gram；仍为空再调用模型生成 |
| 多分支结果冲突 | 各 head/channel 独立 quota 后做全局去重，不做简单多数票 |

风险路径的最坏预算仍可限制为 3 blocks/768 tokens，低于 100K 场景的 1000-token 上限。

## 10. 当前结论与边界

可以确认：

1. 合成文本适合做机制研究，但必须使用 split-disjoint 模板；
2. 固定 Top-K embedding 检索不能自然完成链状态更新；
3. “精确字符串锚点 + 小候选语义有效性判断 + 新实体反馈”能在 held-out 多跳题上把全召回从 0% 提升到 96%；
4. 逐步检索可以把实际返回上下文控制在 256/512 tokens；
5. 多卡适合分别承担 block-shard 全局扫描和 query-shard 候选精排。
6. 在不再调参的 v4 blind test 上，整体全部证据召回为 92%，高于 BM25 的 77%，但仍有 8% 失败。

尚不能确认：

1. 合成题上的 96% 能迁移到 LongBench 或真实 1B-token 连续 KV cache；
2. 合成 blind test 上改善的 answer NLL 能否迁移到真实长上下文和自由生成准确率；
3. 0.6B validity scorer 对更隐式的时态、否定和冲突事实是否可靠；
4. semantic heads 是否具有足够因果贡献，可以据此永久删除其他 head 的 KV。

因此当前结果是一个成功的受控机制验证，不是“1B -> 1M 已解决”的结论。

## 11. 复现入口

一键脚本会自动检测空闲 GPU；没有空闲卡时直接退出，不抢占现有任务：

```bash
cd /home/fdong/ymluo/projects/parallel_block_retrieval
bash scripts/run_synthetic_stepwise_100k_server.sh
```

主要代码：

```text
src/prepare_synthetic_controlled_corpus.py
src/run_iterative_condition_retrieval.py
src/run_model_guided_condition_retrieval.py
src/analyze_model_guided_routes.py
src/evaluate_synthetic_benchmark.py
src/evaluate_retrieved_answer_nll.py
tests/test_stepwise_retrieval.py
```

下一阶段应在同一最终代码上完成：4/8 卡重复吞吐中位数、答案自由生成准确率，以及真实 LongBench 多跳数据迁移。
