# Section 174：从 Block Recall 到 Step-Sufficient Span 的受控实验

日期：2026-07-12

## 1. 本轮研究问题

此前 10M 全局实验已经证明分布式 SVD32 可以在约 39K blocks 中找到第一跳 block，但多跳生成仍不稳定。本轮不继续盲目调整全局分数，而是先回答一个更基础的问题：

> 如果检索器已经命中包含目标事实的 256-token block，这个 block 是否真的足以让 Qwen3-0.6B 完成当前一步？

如果答案是否定的，那么继续优化 gold-block Recall 不是正确目标。检索器必须进一步选择对当前 reasoning step 充分、同时不引入错误绑定的 token span。

## 2. 数据与评测协议

基础语料为：

- 99,840 tokens；
- 390 个 256-token blocks；
- 500 queries；
- 125 条 multihop queries；
- 每条 multihop query 有两个有序步骤：`resolve_bridge` 和 `resolve_answer_from_bridge`。

每一步显式记录：

```text
current step operator
current state
target fact
target output
minimal sufficient block
previous-step block
hard-negative block
```

query 375 已用于提示和单样本调试，因此正式 v4 test 统计排除该样本，只报告其余 24 条。Gold 只用于运行后判断事实是否被覆盖，不参与 span 排序、branch 选择或生成。

标签审计结果：

- 500 queries、625 steps；
- 目标事实缺失：0；
- 目标输出缺失：0；
- 第一跳 block 泄漏最终答案：0。

## 3. 关键发现一：Gold Block 不等于充分证据

### 3.1 Block 污染强度

生成器会把不同 query 的事实共同打包到同一个 256-token block：

| 范围 | 每 block 平均片段 | median | p95 |
|---|---:|---:|---:|
| 全部 390 blocks | 4.36 | 4 | 5 |
| multihop gold blocks | 4.72 | 5 | 6 |
| multihop hard-negative blocks | 4.56 | 5 | 5 |

因此 gold block 通常同时包含约 4～5 条其他事实。它只说明答案片段位于该 block，不能说明整个 block 对当前 step 有正效用。

### 3.2 第一跳读取上限

比较只给精确事实句、完整 gold block、以及 gold block 加一个 hard negative：

| Split | 精确事实句 | 完整 gold block | gold + negative |
|---|---:|---:|---:|
| dev，25 条 | 84.0% | 32.0% | 16.0% |
| test，24 条 | 75.0% | 20.8% | 4.2% |

配对结果：

- dev：精确事实到完整 block 下降 52 个百分点，McNemar `p=0.0002`；
- test：下降 54.2 个百分点，McNemar `p=0.0023`；
- test 目标 NLL 增加 `+0.688`，bootstrap 95% CI `[0.442, 0.940]`。

test 完整 gold block 的 19 个生成失败中：

- 68.4% 直接输出了同块其他 query 的实体；
- 不是模型没有看到目标事实，而是模型把当前 step 绑定到了另一个可读事实。

### 3.3 第二跳状态不是越多越好

test 第二步结果：

| 条件 | Step output hit | Mean NLL |
|---|---:|---:|
| 精确答案事实，不加旧状态 | 95.8% | 4.808 |
| 精确答案事实 + typed bridge | 87.5% | 约 4.05～4.31，依精度而异 |
| 完整答案 block，不加旧状态 | 62.5% | 6.319 |
| 完整答案 block + typed bridge | 45.8% | 约 5.03 |
| 只给 previous block + typed bridge | 0% | 12.725 |

加入 typed state 时，完整 block 的 teacher-forced NLL 下降，但 greedy hit 反而下降。这说明：

1. NLL 与实际解码轨迹不能互相替代；
2. 状态 token 会改变模型在污染 block 中的绑定目标；
3. set utility 是非加性的，不能把独立 block 分数直接相加。

## 4. 文本侧自动 Evidence Span

在不知道 `target_fact` 的情况下，先在已命中的 block 内按当前 step state 自动选择一个句子：

```text
step question + original question + typed bridge state
-> sentence lexical score
-> Top-1 complete sentence
```

该方法只把 gold block 当作候选输入，用于隔离“块内选择”环节；句子排序不读取 gold span。

### 4.1 Span 定位

| Step | dev 精确目标句 | test 精确目标句 |
|---|---:|---:|
| resolve_bridge | 24/25 | 23/24 |
| resolve_answer，typed state | 25/25 | 24/24 |

选中句平均只有约 16～20 tokens，相比原 256-token block 减少约 13～16 倍。

### 4.2 Qwen 单步生成

| Split / step | 完整 gold block | 自动 span | 精确事实 oracle |
|---|---:|---:|---:|
| dev bridge | 32.0% | **80.0%** | 84.0% |
| test bridge | 20.8% | **70.8%** | 75.0% |
| dev answer + state | 52.0% | 48.0% | 48.0% |
| test answer + state | 45.8% | **87.5%** | 87.5% |

配对检验：

- dev bridge：`32.0% -> 80.0%`，`p=0.0005`；
- test bridge：`20.8% -> 70.8%`，`p=0.0042`；
- test answer：`45.8% -> 87.5%`，`p=0.0129`。

dev answer 没有提高，且等于 fact oracle，说明该 split 的瓶颈是 0.6B 的步骤输出能力，不是 span 选择。

### 4.3 纯选择开销和风险

在 CPU 上重复 100 次测量：

| Step / 候选 | 目标事实选择率 | 平均输出 tokens | 平均选择耗时 |
|---|---:|---:|---:|
| bridge，1 个正确 block | 95.9% | 19.3 | 0.165 ms |
| answer，1 个正确 block | 100% | 16.3 | 0.174 ms |
| bridge，正确 + hard negative | **22.4%** | 19.4 | 0.332 ms |
| answer，正确 + hard negative | 93.9% | 16.3 | 0.342 ms |

第一跳 hard negative 与问题共享 lookup key，因此纯 lexical Top-1 会被无效映射吸引。正确设计不能把所有候选句混在一起做一次词面 Top-1，而应：

1. 每个候选 block 内先独立定位 span；
2. 再判断哪个 span 能产生有效状态转移；
3. margin 低时保留多个小分支，而不是拼成一个污染集合。

## 5. KV-Native Sentence Span

### 5.1 真实 Q/K Profile

新 profile 不再把完整 source 放在 query 前：

- K：Qwen3-0.6B 对真实 100K token corpus 前向得到的 pre-RoPE K；
- Q：只对当前 step state 前向，不含 source、gold block 或答案；
- layers：3、6、16、21；
- 每层保留全部 16 query heads 和 8 KV heads；
- SVD rank：32；
- K sidecar：204,472,320 bytes，约 195 MiB；
- 单卡 K build：7.91 s；
- 单卡峰值显存：2.21 GiB；
- rank32 平均能量保留率：93.0%。

这不是外部 embedding。每个 sentence span 的分数来自当前 step Q 与 span 内真实 K 的低秩 late interaction。

### 5.2 Operator-Specialist Heads

训练阶段只使用 train split：对每个 step operator，计算各 layer/head 对目标句和其他句的区分 margin，选择 8 个 specialist heads。dev/test 不重新选 head。

需要强调：选中 head 的平均 train margin 仍略小于 0，说明没有单个 head 能完美区分所有 hard negatives。当前方法选择的是相对更合适的少数 head，不是已经学到完美 utility function。

### 5.3 v4 句子检索

候选包含正确 block 和一个 hard-negative block：

| Split / step | All-head mean R@1 | Specialist R@1 | Specialist R@2 |
|---|---:|---:|---:|
| dev bridge | 28.0% | 28.0% | 72.0% |
| dev answer | 20.0% | **60.0%** | 88.0% |
| test bridge | 41.7% | **50.0%** | 87.5% |
| test answer | **0%** | **79.2%** | 87.5% |

test answer 中 specialist 相对 all-head mean：

- 正确 block 内：29.2% -> 95.8%，McNemar `p=3.05e-5`；
- 加 hard negative：0% -> 79.2%，McNemar `p=1.22e-4`。

这直接支持“不同 step 需要不同专业 heads”；把所有 heads 平均会淹没专门的下一跳信号。

句子级 KV 特征计算在 CPU 上平均约 11 ms/step-example，选中 span 约 17～21 tokens。

## 6. 并行小分支与 Transition Verifier

### 6.1 为什么不把 Top-2 句子拼起来

前面的实验已经证明额外 block/span 可能改变解码轨迹。风险路径不应把 Top-2 拼成一个 prompt，而应分别生成：

```text
candidate span 1 -> transition 1
candidate span 2 -> transition 2
...
state-anchor grounding verifier -> one accepted transition
```

分支可以跨 GPU 或在同一 GPU batch 中并行；每个分支只读取约 14～24 tokens。

### 6.2 Verifier

当前无训练 verifier 只使用以下运行时条件：

1. candidate sentence 是否包含当前 state anchor；
2. 生成结果是否错误地重复 anchor；
3. 生成的新 token 是否由 candidate sentence 支持；
4. 多个分支都合法时保留原 Q/K 排名。

选择阶段不读取 target answer 或 gold ID。

### 6.3 v4 开发结果

| Step | Top-1 generation | Any Top-2 oracle | Verifier |
|---|---:|---:|---:|
| bridge | 33.3% | 62.5% | **62.5%** |
| answer | 70.8% | 79.2% | **79.2%** |

该 verifier 是观察 v4 test 输出后形成的，因此 v4 只能视为开发结果。

## 7. 新 v5 模板盲测

为了评估 verifier 泛化，新增 v5 test-only 多跳模板并使用新 seed 重新排布全部 block：

```text
The communications ledger pairs CODE with UNIT.
The signal marker assigned to UNIT displays VALUE.
```

负例、问题和最终属性措辞也全部改变。Verifier 完全冻结；specialist heads 只用 v5 train split 选择。

### 7.1 Top-2 冻结结果

| Step | Span R@1 / R@2 | Top-1 generation | Top-2 oracle | Verifier |
|---|---:|---:|---:|---:|
| bridge | 83.3% / 100% | 66.7% | 83.3% | **83.3%** |
| answer | 33.3% / 66.7% | 33.3% | 62.5% | **62.5%** |

Verifier 在两类 step 都没有 loss：

- bridge：4 wins、0 losses；
- answer：7 wins、0 losses，McNemar `p=0.0156`。

这说明 state-anchor grounding 不是只对 v4 原句式有效。

### 7.2 Top-4 开发诊断

观察 v5 排名后，将 branch 上限放宽到 4，因此该结果不能再视为同一 blind protocol，只是下一版开发依据：

| Step | Span R@4 | Top-1 generation | Any Top-4 oracle | Verifier |
|---|---:|---:|---:|---:|
| bridge | 100% | 66.7% | 83.3% | **83.3%** |
| answer | 95.8% | 33.3% | 83.3% | **83.3%** |

answer verifier 为 12 wins、0 losses，`p=0.00049`。

4 分支当前仍是单进程内顺序调用。记录到的平均时间：

| Step | 单个 Top-1 generation | 4 分支串行总和 | 分支耗时最大值 |
|---|---:|---:|---:|
| bridge | 0.253 s | 1.075 s | 0.418 s |
| answer | 0.611 s | 2.357 s | 0.774 s |

最后一列只是各分支独立耗时的最大值，可作为理想并行临界路径估计；当前没有完成“同一 query 的 4 分支跨 4 GPU”严格 wall-clock scaling，不能把它直接当作实测并行延迟。

## 8. 基于现象形成的方法

当前有证据支持的系统结构为：

```text
typed step state: NEED(operator, anchor, relation)
        |
        v
operator-routed global candidate retrieval
  - identifier / exact lookup: lexical + validity expert
  - relation continuation: step-Q + specialist-head SVD32 expert
        |
        v
distributed block Top-L
        |
        v
sentence-boundary sidecar + KV-native span scoring
        |
        v
K=1 normally; low margin expands to 2..4 independent tiny branches
        |
        v
parallel step generation + state-anchor grounding verifier
        |
        v
verified transition -> update typed state -> next step
```

工作名可以暂称 **StepRoute-KV**，但在完成 10M 端到端和新 holdout 前不应把名称当作已定稿论文方法。

### 与 RAG 的关系

它有 RAG 式的“检索、更新状态、再次检索”控制流，但核心索引与加载路径不同：

- 不是外部 embedding 向量库；
- 使用模型自身 step Q 和历史 pre-RoPE K；
- sentence sidecar 只保存 token 边界；
- 最终加载的是原 KV span，而不是把检索文本重新拼入一个无限增长的 prompt；
- lexical 通道只负责适合精确标识符的 operator，不承担所有语义检索。

## 9. 与 10M 速度结果的连接

已有 10M 全局 SVD32 + raw128 检索稳态延迟：

| GPU | 每次全局检索 |
|---:|---:|
| 3 | 58.6 ms |
| 4 | 52.9 ms |
| 6 | 约 39～40 ms |
| 7 | 39.3 ms |

本轮局部 stage 的量级为：

- lexical sentence selection：每 block 约 0.17 ms；
- 两个 block 约 0.33～0.34 ms；
- 4-layer/all-head KV sentence scoring：CPU 约 11 ms/example；
- 主要在线成本仍是 Qwen step generation，而不是局部 span 排序。

但这两个数字尚未在同一 10M 端到端进程中连接，因此不能直接相加后宣称最终延迟。

## 10. 尚未完成的关键验证

1. 当前 KV sentence 实验的候选集合人为保证包含正确 block和一个 hard negative，用于隔离局部 stage；还不是从 10M 全局候选开始的端到端成绩。
2. 当前只 profile 4 个 layer 的全部 heads，不是 28 层全部 heads。
3. 合成文本仍比真实 LongBench 多跳关系更规则。
4. Top-4 是观察 v5 后确定的，需要新的 v6 或外部数据冻结验证。
5. 4 分支实际跨卡 wall-clock speedup 尚未测量。
6. 10M 持久索引仍未保存完整 V；命中 block 后暂时依赖 token 重前向重建 KV。

## 11. 下一步主线

按证据优先级，下一步应为：

1. 在 10M 的 39,062 blocks 上增加 sentence token-boundary sidecar；
2. 保留现有分布式 SVD32 block coarse retrieval；
3. 对 Top-L blocks 运行 operator-specialist sentence K scoring；
4. 在真实 10M 多跳 holdout 上比较 K=1 与风险自适应 K<=4；
5. 实现同一 query 分支跨卡或 batch 并行，严格测 1/2/4 GPU wall time；
6. 将 verifier 和 branch 上限冻结到新模板或外部多跳数据；
7. 最后再扩展到更长上下文和更多 relation/operator 类型。

## 12. 主要产物

代码：

```text
src/annotate_stepwise_sufficiency.py
src/evaluate_stepwise_set_utility.py
src/benchmark_stepwise_span_selector.py
src/profile_step_state_q.py
src/run_step_state_kv_span_retrieval.py
src/evaluate_kv_span_step_generation.py
src/analyze_stepwise_set_utility.py
src/analyze_stepwise_block_contamination.py
src/analyze_branch_transition_verifier.py
scripts/run_step_state_kv_span_server.sh
```

关键结果：

```text
outputs/stepwise_set_utility_v5_test24_gpu1_v1/
outputs/stepwise_set_utility_v5_devtest49_span_cpu4_v1/
outputs/stepwise_span_selector_benchmark_v1/
outputs/synthetic_controlled_v4_stepkv_sentence_span_v1/
outputs/synthetic_controlled_v4_stepkv_sentence_branch2_test24_v1/
outputs/synthetic_controlled_v5_stepkv_sentence_span_v1/
outputs/synthetic_controlled_v5_stepkv_sentence_branch2_test24_v1/
outputs/synthetic_controlled_v5_stepkv_sentence_branch4_test24_v1/
```

测试：`31 passed`。
