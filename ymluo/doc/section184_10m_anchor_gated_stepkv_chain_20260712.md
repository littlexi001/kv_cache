# 10M Anchor-Gated StepKV：从全局召回到逐步链式推理

## 1. 本轮要回答的问题

目标是在约 10M tokens 的历史信息中，每一步只找少量当前所需证据，并让
Qwen3-0.6B 先生成中间状态，再用该状态检索下一跳。评测必须同时满足：

1. 检索向量来自真实 Qwen 前向，不使用合成高斯向量。
2. 初始检索不知道 gold block 或答案。
3. 第二跳既要报告使用正确 bridge state 的逐 step 上限，也要报告使用模型生成
   bridge state 的真实链式结果。
4. 速度必须区分离线 profiling、在线全局检索、局部句子评分和生成。

## 2. 数据与索引

最新混合语料：

| 项目 | 数值 |
|---|---:|
| tokens | 9,999,872 |
| blocks | 39,062 |
| block tokens | 256 |
| 受控多跳 blocks | 390 |
| 真实长文本干扰 blocks | 38,672 |
| queries | 500 |

受控 block 只占总文本的 0.9984%，其位置经过随机一一映射。其余 block 来自
真实 LongBench 长文本记录。每条多跳问题有独立的第一跳事实、第二跳事实和
hard negative。

真实 K 索引保存：

- pre-RoPE K；
- raw128 K；
- SVD64 K，在线粗检索只使用前 32 维；
- sentence sidecar 只保存 token 边界，不保存 embedding。

三卡 profiling 10M tokens 用时约 171 秒，每卡约 19.5K tokens/s，峰值显存约
2.22GB。sentence sidecar 共 348,475 个句子，平均每 block 8.92 个句子，文件仅
1.71MB。

## 3. 固定四通道全局 Q/K 基线

固定四个通道为 L3/H10、L21/H8、L6/H7、L16/H14。无泄漏 typed step-Q 在
dev/test 共评测 98 个 steps。

| split | step | Top-512 coverage | raw128 Top-16 | 三卡延迟 |
|---|---|---:|---:|---:|
| dev | bridge | 36.0% | 8.0% | 74.6ms |
| dev | answer | 28.0% | 12.0% | 52.1ms |
| test | bridge | 41.7% | 29.2% | 52.0ms |
| test | answer | 25.0% | 16.7% | 52.1ms |

结论：局部 target-vs-negative 实验中的高召回不能替代 10M 全局 candidate
coverage。当前瓶颈首先是全局入口，而不是句子精排。

## 4. Operator specialist 的正负结果

训练 split 选出的四个 operator heads：

- bridge：L3/H10、L21/H9；
- answer：L6/H13、L6/H12。

每个 step 只扫描对应两条通道后：

| split | step | Top-512 coverage | raw128 Top-16 | 三卡延迟 |
|---|---|---:|---:|---:|
| dev | bridge | 48.0% | 4.0% | 62.3ms |
| dev | answer | 0.0% | 0.0% | 39.7ms |
| test | bridge | 62.5% | 29.2% | 39.6ms |
| test | answer | 0.0% | 0.0% | 39.6ms |

这不是简单的失败。它说明 specialist head 擅长在同锚点 hard negative 之间判别，
但不适合独立扫描 39K blocks。第二跳常见人物和关系文本会淹没组合语义。

## 5. 快速 anchor 路由

每个 step 已经具有当前锚点：

- 第一跳是原始 identifier；
- 第二跳是前一步生成并验证的 bridge entity。

因此建立 token 倒排表，先求锚点 term postings 的交集，再只对同锚点候选做
relation-sensitive 判别。

在使用正确 step state 的 dev/test 上：

| step | 平均查询延迟 | Top-3 coverage | Top-16 coverage |
|---|---:|---:|---:|
| bridge | 0.23ms | 100% | 100% |
| answer | 0.91ms | 100% | 100% |

索引构建用时 3.62 秒，共 134,221 个 term。相比完整 BM25 的约 345 至 362ms，
anchor index 快约三百倍。BM25 同样能达到 Top-16 100%，但不适合作为在线主路径。

## 6. 为什么不能直接做候选 union

将 anchor 命中的 2 至 12 个 blocks 与约 500 个全局 Q/K 候选合并，再用 raw K
整体重排，会让正确 block 掉到 200 至 500 名。union 提高 coverage，但不保证
后续排序稳定。

因此当前方法采用 gated composition：

```text
typed state
  -> anchor inverted lookup
  -> Top-3 same-anchor blocks
  -> operator-specialist sentence scoring inside each block
  -> span-major branches
  -> Qwen step generation
  -> frozen grounding verifier
  -> verified next state
```

## 7. 逐 step 的 block 和 sentence 结果

使用正确的 step state，test 上：

| step | anchor block Top-3 | 正确 block 内 span Top-1 | Top-2 | Top-3 |
|---|---:|---:|---:|---:|
| bridge | 100% | 100% | 100% | 100% |
| answer | 100% | 66.7% | 79.2% | 91.7% |

分支采用 span-major 顺序：先运行三个 blocks 各自的 Top-1 sentence，再运行各自
Top-2 和 Top-3。因此 test evidence coverage 为：

| step | K=3 | K=6 | K=9 |
|---|---:|---:|---:|
| bridge | 100% | 100% | 100% |
| answer | 66.7% | 79.2% | 91.7% |

## 8. 逐 step 的实际生成和 verifier

K=6 表示 Top-3 anchor blocks，每块最多两个 sentence branches。

| step | evidence recall | Top-1 生成正确 | 任一分支正确 | verifier 正确 |
|---|---:|---:|---:|---:|
| bridge | 100% | 25.0% | 83.3% | 75.0% |
| answer | 79.2% | 4.2% | 70.8% | 70.8% |

answer verifier 与 branch oracle 完全一致。bridge verifier 有 13 次净增益、1 次损失，
McNemar p=0.00183。answer 有 16 次净增益、0 次损失，p=3.05e-5。

注意：这一表中的 answer step 使用正确 bridge state，只是逐 step 结果，不是完整
两跳端到端成绩。

## 9. 使用模型生成 bridge 的真实链式状态

将第一跳 verifier 选择的原始生成文本写回 `BRIDGE_ENTITY`，不使用 gold 修正：

- bridge state 字符串正确率：75.0%；
- 新状态重新检索 10M 的 anchor Top-1：25.0%；
- Top-3：75.0%；
- Top-16：83.3%；
- Top-512：91.7%。

错误状态仍可能保留有用的人名片段，因此大预算召回可以高于实体字符串完全正确率。

在真实链式状态下，第二跳 evidence branch coverage：

| 分支预算 | coverage |
|---:|---:|
| K=3 | 41.7% |
| K=6 | 54.2% |
| K=9 | 66.7% |

### 9.1 无 gold 的 anchor 风险门控

只使用 dev 中有效 answer states 的候选数分布校准 95% 分位数，得到冻结阈值：

```text
accept anchor route iff 0 < candidate_count <= 18
```

应用于模型生成的 24 个 test states：

- 接受率：75.0%；
- 被接受状态的 bridge 正确率：100%；
- 被接受状态的目标 block Top-3：100%；
- 拒绝：6 条，恰好覆盖当前 6 个错误 bridge states。

该门控在选择时不使用 test gold。由于当前只有一个受控 test 模板族，完美分离仍需
新 v6 或外部数据验证。被拒绝状态应进入 semantic fallback、扩大生成草稿或重新生成
bridge，而不是继续输出最终答案。

### 9.2 完整两跳最终结果

K=9 answer generation 与同一个冻结 verifier 已完成：

| 指标 | 结果 |
|---|---:|
| 链式第二跳 evidence coverage | 66.7% |
| Top-1 最终答案正确 | 4.2%，1/24 |
| 任一分支最终答案正确 | 54.2%，13/24 |
| verifier 最终答案正确 | **54.2%，13/24** |
| verifier 胜/负/平 | 12/0/12 |
| McNemar p | 0.000488 |

verifier 再次与 branch oracle 完全一致。13 个最终正确样本的 bridge state 也全部正确，
因此“bridge 与最终答案都正确”的完整链式成功率同样为 54.2%。

按冻结 candidate-count 风险门控拆分：

| 子集 | 样本 | bridge 正确 | 最终答案正确 |
|---|---:|---:|---:|
| accepted | 18 | 100% | 72.2%，13/18 |
| rejected | 6 | 0% | 0% |
| all | 24 | 75.0% | 54.2%，13/24 |

这是当前第一个完成“第一跳生成状态写回、重新检索 10M、第二跳生成和无 gold
verifier”的完整结果。

## 10. 在线速度与存储路径

已测量的在线量级：

| 阶段 | 延迟 |
|---|---:|
| anchor lookup，正确 step state | 0.23 至 0.91ms |
| 两通道 10M SVD32/raw128 全局扫描，三卡 | 39.6ms |
| gated sentence scoring，三卡，正确 state | 1.8 至 3.4ms test 均值 |
| step-Q capture，GPU | 约 32ms |
| K=6 bridge 分支串行和 | 1.80s |
| K=6 bridge 理想临界分支 | 0.51s |
| K=6 answer 分支串行和 | 3.66s |
| K=6 answer 理想临界分支 | 0.84s |

真实链式 K=9 最终生成因运行时没有空闲 GPU，使用 CPU 完成：

- 9 分支串行和：17.77s/step；
- 单 step 最慢分支：2.77s；
- 该 CPU 数字用于完成功能与正确率验证，不代表计划中的 GPU 在线延迟。

新增 mmap backend 后，局部阶段不再要求把整库 K 常驻 GPU。在真实链式错误状态会
扩大候选的条件下：

- CPU step-Q：184ms；
- mmap raw K candidate sentence scoring：57.9ms；
- 平均候选 blocks：37；
- GPU 索引常驻显存：0。

## 11. 当前方法判断

本轮支持以下方法性结论：

1. 不同检索器不应无条件竞争同一个全局排名。identifier lookup、relation
   discrimination 和 branch verification 是不同 operator。
2. 稀有 anchor 是海量检索中高效且鲁棒的路由信号，Q/K 更适合作为局部判别器。
3. 候选 union 后全局重排会破坏强通道，必须使用 gated candidate set 或风险保留。
4. span-major 分支让预算具有明确含义，K=3/6/9 分别覆盖每块 1/2/3 个句子。
5. 逐 step 高分不等于链式高分，必须把生成状态真正写回下一跳。

当前最接近论文方法的名称可暂定为 `Anchor-Gated StepKV`。创新点不是简单复用 RAG，
而是将模型内部 Q/K 的职责限制在当前 operator 的局部证据判别，并通过 typed state、
小分支和 transition verifier 实现逐步 KV 激活。

## 12. 尚未完成

1. 同一 query 分支的严格 1/2/4 GPU wall-clock scaling。
2. 外部真实多跳数据的 ordered-step 标注与冻结验证。
3. 对非精确实体、别名和模糊状态增加 semantic anchor fallback。
4. 持久化完整 V 或实现 KV 重定位，当前命中 block 后仍需重建生成 KV。

## 13. v6 冻结验证状态

已新建 v6 phrasing family，并使用新 seed 20260715 重新生成 100K 控制集和 10M
混合语料。新模板将关系改为 routing token、fleet allocation、compliance pennant，
没有复用 v5 的 communications ledger 和 signal marker 表述。

v6 test 的 48 个 multihop steps 在正确当前状态下：

| step | 查询延迟 | anchor Top-1 | Top-3 | 平均候选 | 最大候选 |
|---|---:|---:|---:|---:|---:|
| bridge | 0.14ms | 62.5% | 100% | 2.0 | 2 |
| answer | 0.70ms | 4.2% | 100% | 11.6 | 17 |

v6 answer 的最大有效候选数 17，仍低于只用 v5 dev 冻结的风险阈值 18。该结果支持
anchor 路由和候选数置信度不依赖 v5 具体句式。

### 13.1 Sparse candidate K

v6 dev/test 的 anchor 候选全集只有 303 blocks，占 39,062 blocks 的 0.78%。因此不再
profile 全部 10M K，而是只对这 303 blocks 做 block-local Qwen 前向：

| 项目 | 全量 v5 record K | v6 sparse block-local K |
|---|---:|---:|
| profile tokens | 9,999,872 | 77,568 |
| raw+SVD K 存储 | 约 14.4GB | raw K 约 80MB |
| profiling | 三卡约 171s | CPU 90.0s |

存储缩小约 180 倍。block-local K 不等同于 record-context K，因此质量必须单独验证，
不能把它视为无损替代。

### 13.2 Anchor sentence gate

单独使用 sparse K 时，v6 answer 的正确句经常排在同 block 第 2 至 3 位。加入无 gold
约束“候选句必须包含当前 anchor”后：

| step | gate 前句子/step | gate 后句子/step | K=3 evidence |
|---|---:|---:|---:|
| bridge test | 20.7 | 2.0 | 100% |
| answer test | 123.5 | 3.0 | 100% |

所有 98 个 dev/test steps 均无需 fallback。此结果支持两级结构：字符串通道负责实体
一致性与快速剪枝，Q/K 只负责同锚点候选内的关系判别。

### 13.3 v6 逐 step 生成

使用 K=3、12-token decode：

| step | evidence | Top-1 | branch oracle | verifier |
|---|---:|---:|---:|---:|
| bridge | 100% | 41.7% | 62.5% | 62.5% |
| answer，正确 bridge state | 100% | 4.2% | 79.2% | 79.2% |

verifier 在两步都达到 branch oracle。bridge 的 McNemar p=0.0625，样本数仍不足；
answer 为 18 胜 0 负，p=7.63e-6。

### 13.4 v6 真实链式结果

将 bridge verifier 生成真实写回后，冻结 v5 dev 风险阈值 18 在 v6 test 上：

| 指标 | 结果 |
|---|---:|
| bridge state 正确 | 62.5%，15/24 |
| 风控接受 | 66.7%，16/24 |
| 接受状态 bridge precision | 93.75%，15/16 |
| 接受状态第二跳 evidence | 93.75%，15/16 |
| 接受子集最终答案 | 62.5%，10/16 |
| 全部问题最终正确 | 41.7%，10/24 |
| abstain | 33.3%，8/24 |

第二跳 verifier 再次达到 3 分支 oracle，10 胜 0 负，p=0.00195。v6 比 v5 的 54.2%
完整链式正确率更低，说明方法仍受第一步生成能力和模板变化影响，但 sparse K 路径已在
新 holdout 上完成了不依赖全量 KV 索引的真实两跳闭环。

## 14. 同一 query 的严格多卡分支并行

固定 8 个 v6 answer steps、每步 3 个生成分支、每分支最多 12 tokens。这里测的是同一个
query 的分支真正分发到不同 GPU 后的墙钟时间，不是把不同 query 混成吞吐量：

| GPU | 平均墙钟/step | 相对 1 卡加速 | 并行效率 | 生成吞吐 |
|---:|---:|---:|---:|---:|
| 1 | 1.243 s | 1.00x | 100.0% | 29.0 tok/s |
| 2 | 0.829 s | 1.50x | 75.0% | 43.4 tok/s |
| 3 | **0.453 s** | **2.74x** | **91.4%** | **79.4 tok/s** |
| 4 | 0.535 s | 2.32x | 58.1% | 67.3 tok/s |

只有 3 个分支时，3 卡已经是一卡一分支；第 4 卡没有工作可分，通信和调度反而使时间增加。
因此扩卡数量必须由同 query 的有效分支数决定，不能仅报告进程级吞吐。

## 15. 外部真实 2Wiki 小样本检查

在未参与合成模板设计的 LongBench 2Wiki 文本上，先导出了 3 条唯一标题链、6 个单步。
加入全名/姓氏 alias、相邻句窗口和关系明确的原子子问题后：

- 29 个 sparse candidate blocks，共 7,424 tokens；CPU K profile 10.5 s，raw K 约 7.6 MB；
- 257 个 sentence spans，sidecar 1,384 bytes，构建 0.21 s；
- 6/6 步的目标证据进入分支，6/6 的 branch oracle 和 verifier 正确；
- test 的真实 bridge 写回链 1/1 正确。

这只能作为接口 smoke test。样本太少，并且关系提示是在观察这 3 条数据后形成，不能作为泛化结论。

## 16. 31 条关系无关真实链

为避免只挑 3 个关系模板，从同一个真实 10M corpus 的 `records.jsonl` 恢复全部 2Wiki
候选：157 条 query 中，31 条能严格形成唯一、顺序明确且两个证据 block 分离的两跳链，
共 62 个 step；冻结划分为 train/dev/test = 13/7/11 条链。在线选择不读取答案，答案仅用于
实验后的 gold block 和生成命中统计。

### 16.1 全局 block 候选

| 方法 | test bridge Recall | test answer Recall | 在线时间 |
|---|---:|---:|---:|
| anchor Top16 | 90.9% | 81.8% | 1.7/6.1 ms |
| anchor Top512 | 90.9% | 90.9% | 同上 |
| BM25 Top3 | 72.7% | 45.5% | 约 0.30 s |
| BM25 Top64 | 90.9% | **100%** | 约 0.30 s |
| 固定 4-head SVD32/full-K Top512 | 36.4% | 27.3% | 77--79 ms，另加约 34 ms Q capture |

anchor 索引一次构建 7.60 s，62 步平均查询 3.79 ms。当前 BM25 每步重新执行 CPU sparse
矩阵评分，0.30 s 不是优化后的服务延迟；索引和 query batch 持久化后仍有下降空间。

### 16.2 all-head 负结果

对 BM25 Top64 中的 448 个 `(layer, query-head)` 独立评分，只用 train 标签分别为 bridge
和 answer 冻结选择 8 个通道，再评估 dev/test：

| test Top3 | BM25 原顺序 | 固定 4-head | train-selected 8-head | all-head mean/max/RRF |
|---|---:|---:|---:|---:|
| bridge | **72.7%** | 27.3% | 27.3% | 0--18.2% |
| answer | **45.5%** | 18.2% | 18.2% | 0% |

因此“所有 head 都关注的 block”不是稳定相关性信号。不同 head 的确有专门功能，但无选择地
平均、取最大值或投票会把位置、语法、复制和无关实体通道混在一起。当前 all-head 结果不应
作为主方法；旧固定 head 在新关系上同样不泛化。

### 16.3 Q/K 作为 block 内 token 压缩器

保持 BM25 block 顺序，仅在 Top64 blocks 内用固定 4-head full-K 选择每个 block 的 3 个句子：

- Q capture 约 35 ms/step；64-block sentence score 约 12--14 ms/step；
- 正确 block 已进入 BM25 Top3 时，保留 3 句后没有额外丢失：bridge 8/11，answer 5/11；
- 每个生成分支从固定 256 tokens 降到平均 94.4 tokens，减少 **63.1%**；
- 通用 prompt 的 branch oracle：full block 为 bridge 1/11、answer 2/11；压缩句为 2/11、2/11。

这支持新的职责分工：**lexical/anchor 决定 block，Q/K 负责同 block 内的 token 级压缩**。
Q/K 当前不能替代 BM25 做跨 block 语义排序，但可以减少传给下一步生成器的 KV 工作集。

## 17. 当前主要瓶颈：step controller

在正确证据已经存在时，Qwen3-0.6B 仍经常读取错误关系。例如电影句同时包含 director 和
主演时，模型会输出主演；父子句同时包含母亲时，会输出母亲。关系无关 prompt 在 test 上只有
bridge 1/11、answer 2/11 的 branch oracle。

只使用原问题、当前实体和 verified state 生成原子子问题的 0.6B controller 也没有改善：
bridge 变为 2/11，answer 变为 1/11，并出现把 `die` 改写成 `born` 的关系漂移。它没有读取
gold，但准确性不足。下一步应使用官方提供 supporting facts/evidence path/单跳 decomposition
的 2Wiki 或 MuSiQue 数据，把以下两项分开评估：

1. 给定正确原子 step query 时，10M block/span retrieval 的召回、KV 预算和延迟；
2. 不给 oracle decomposition 时，learned controller 的 step-query 正确率和错误传播。

继续在当前 31 条链上增加手写 relation pattern 会把 controller 过拟合误报成 retrieval 改进，
不适合作为论文主证据。

## 18. 官方 MuSiQue decomposition 的 10M 主评测

为解决 Section 17 中 controller 与 retrieval 混杂的问题，下载 MuSiQue 官方 answerable train/dev，
直接使用数据集提供的 `question_decomposition` 和 `paragraph_support_idx`。官方每一步包含原子
问题、step answer 和 supporting paragraph；在线检索不读取 step answer，answer 只用于评测
gold span。数据来自 MuSiQue 官方仓库发布的 v1.0 文件，而不是 LongBench 中已丢失 decomposition
字段的转换版。

旧版语料直接每 256 tokens 切块，会把 supporting paragraph 截断，导致“gold block”也不一定
包含足够证据。以下结果全部改用 paragraph-aligned v2：每个被选中的官方 supporting paragraph
完整放入一个 256-token block，其余空间和其余 block 都由真实 MuSiQue distractor paragraph
填充；support block 再随机散布到全库。没有合成文本、合成向量或 padding token。

| 项目 | 数值 |
|---|---:|
| requested tokens | 10,000,000 |
| actual tokens | 9,999,872 |
| blocks | 39,062 × 256 tokens |
| 两跳 queries / 原子 steps | 400 / 800 |
| train/dev/test queries | 200 / 100 / 100 |
| 唯一 support blocks | 725 |
| 使用的真实 filler paragraphs | 84,576 |
| support paragraph token mean / p95 / max | 116.5 / 215 / 253 |
| block-contained target 审计 | 800/800 通过 |
| 两跳落在同一 block | 0 |

### 18.1 10M 全局候选召回

| test step | 方法 | R@1 | R@3 | R@16 | R@512 |
|---|---|---:|---:|---:|---:|
| bridge | exact anchor | 62% | 67% | 76% | 94% |
| bridge | BM25 atomic query | **65%** | **76%** | **89%** | **97%** |
| answer | exact anchor | **14%** | 56% | 84% | 93% |
| answer | BM25 atomic query | 10% | **58%** | **92%** | **98%** |

BM25 全库解码约 4.23 s，索引构建约 19.53 s，二者应在服务启动时一次完成并常驻内存。
800-step 矩阵批处理为 0.476 s，约 0.595 ms/step。原实现对单查询仍有约 0.4--0.6 s
固定稀疏矩阵开销，因此新增完全等价的 feature-to-document postings 累加路径；8 条 query
逐元素最大分数误差为 0。

### 18.2 稳态在线检索速度

以下计时包含 query 向量化、39,062-block 打分和 Top-512 选择，不含一次性建库：

| backend | batch | mean batch | mean/query | p95 batch | throughput |
|---|---:|---:|---:|---:|---:|
| matrix | 1 | 587.91 ms | 587.91 ms | 673.53 ms | 1.70 q/s |
| postings | 1 | **0.688 ms** | **0.688 ms** | **0.859 ms** | **1,454 q/s** |
| postings | 8 | 4.249 ms | 0.531 ms | 4.660 ms | 1,883 q/s |
| postings | 32 | 16.860 ms | 0.527 ms | 17.215 ms | 1,898 q/s |

严格链的 10 个第二跳在实际候选生成器中使用 postings，总打分时间 2.86 ms；召回排序与
matrix backend 完全一致。exact anchor 的平均查询约 2--5 ms，但召回整体低于 BM25。

### 18.3 reader 上界与提示消融

先固定前 10 个 test query（20 个原子 step），令检索 100% 返回 gold block，测试
Qwen3-0.6B 是否能读取正确值：

| memory / prompt | bridge | answer |
|---|---:|---:|
| full block / legacy chain | 3/10 | 5/10 |
| full block / atomic | 9/10 | 2/10 |
| support paragraph / legacy | 5/10 | 4/10 |
| support paragraph / atomic | 8/10 | 3/10 |
| full block / step-type adaptive | **9/10** | **5/10** |

固定规则 `adaptive` 在 bridge step 使用 atomic prompt，在 answer step 使用 legacy chain
prompt，不读取 gold answer。结果说明第一跳更需要简洁的实体抽取指令，而第二跳需要保留原问题和
verified state；即使 gold block 正确，0.6B 的 answer reader 上界仍只有 50%，这是当前主要瓶颈。

### 18.4 BM25 Top3 与严格状态传播

同一 10-query 子集使用 BM25 Top3 full-block、adaptive prompt：

| step | Top3 block recall | 任一分支生成正确 | CPU 串行生成/step | 三分支理想关键路径 |
|---|---:|---:|---:|---:|
| bridge | 8/10 | 8/10 | 5.26 s | 2.26 s |
| gold-state answer 上界 | 5/10 | 2/10 | 7.26 s | 3.19 s |

随后使用冻结的无答案 verifier，从每条 bridge 的三个生成分支中选择一个实体。verifier 选对
6/10（Top1 为 5/10，oracle 任一分支为 8/10）。把所选生成实体真实写回第二跳问题后，再次
检索全部 39,062 blocks：Top3 为 5/10，Top16 为 9/10；最终三个 answer 分支中任一正确仅
1/10。错误 bridge、第二跳漏召回、reader 错误会串行级联，因此当前结果只证明了机制可运行，
还不能声称端到端近似无损。

### 18.5 当前方法到底是否是 RAG

控制流确实类似 multi-hop RAG：`当前状态 -> 找少量 block -> 生成新实体/值 -> 验证 -> 更新
状态 -> 再检索`。但需要明确当前实验的 controller 上界：关系和原子子问题来自官方 MuSiQue
decomposition，模型只生成 bridge value；它还不是对任意新问题自主抽取关系并规划下一跳。

本节当前的全局候选层是 BM25/exact anchor，还没有加入 SVD32 Q/K；后续正确比较应固定同一
官方 atomic query，在 BM25 Top-K 候选内测全维或低秩 Q/K 精排，再测 retrieval recall、reader
正确率和严格链最终答案。这样才能判断 KV 检索相对 RAG embedding/lexical 检索是否提供了额外收益。

## 19. BM25 Top16 内的真实 block-local K 精排

为隔离 KV 精排的增量价值，固定 Section 18 的 BM25 Top16，不改变 query、controller 或 reader。
800 个原子步骤共涉及 7,192 个唯一 block，测试集 Top16 可达召回为 90.5%。在 Qwen3-0.6B
上对这些 block 做真实前向，保存四个固定 layer/head 通道的 block-local pre-RoPE K；这里不是
整个 10M 序列的一次性全因果 KV，结论只适用于分块编码 KV memory。

### 19.1 K materialization 与 SVD32

| 项目 | 数值 |
|---|---:|
| profile blocks / tokens | 7,192 / 1,841,152 |
| raw K profile 时间 | 93.12 s |
| profile 吞吐 | 19,771 tokens/s |
| raw128 K 存储 | 1.76 GiB |
| train calibration blocks / tokens | 512 / 131,072 |
| SVD32 basis 拟合 | 0.92 s |
| 全部候选 K 投影 | 5.94 s |
| SVD32 K 存储 | 449 MiB |

四通道 rank-32 retained energy 为 96.82%、97.77%、99.52%、95.75%，平均 **97.47%**。
因此 `32 维是否足够表达 K 主子空间` 的答案是肯定的；后续失败不能简单归因于 rank 太小。

### 19.2 BM25、full128 与 SVD32 排序

测试集 Top3 block recall：

| step | BM25 | full128 max-QK | SVD32 max-QK | SVD/full Top1 agreement |
|---|---:|---:|---:|---:|
| bridge | **76%** | 53% | 51% | 92% |
| answer | **58%** | 22% | 24% | 83% |

开发集同样下降：bridge 70% -> 45%，answer 56% -> 29%/28%。SVD32 高度复现 full128 的
错误排序，说明低秩近似本身不是瓶颈；当前跨 block 的 raw max-QK score 与证据相关性不足。

Q 捕获约 35.4--35.5 ms/step，16-block mmap load 约 14.5 ms，full128 与 SVD32 GPU score
均约 1.85 ms。SVD32 在该小候选批次中没有 wall-clock 优势，但将 K 存储和传输体积降为四分之一；
总在线 KV 精排开销约 52 ms，明显高于常驻 postings BM25 的 0.69 ms。

### 19.3 融合与风险门控

QK 虽然整体更差，但与 BM25 存在少量互补：测试 bridge 的 `BM25 or full128 Top3 oracle`
为 79%，answer 为 68%，相对 BM25 分别有 3 和 10 条潜在 rescue。直接 z-score 融合只在 train
选择权重，冻结到 dev/test 后均退化为保留 BM25：bridge 选择 0.9 BM25 权重但 test 无变化，
answer 选择 1.0 BM25。

进一步使用 16 个无标签置信特征训练 OOF logistic rescue gate。bridge rescue AUC 在 test 可达
0.82--0.88，但正例只有 3/100，average precision 仅约 0.15--0.18；train 选择的安全阈值最终
不触发切换。answer gate 也没有 test 增益。当前互补 wins 不能被可靠地转化为无损路由策略。

### 19.4 单调追加与公平 block 预算

为了保证不破坏 BM25，测试保留 BM25 Top3，再追加 SVD 候选：

| test step | 策略 | 平均 blocks | recall | 同行数 lexical recall |
|---|---|---:|---:|---:|
| bridge | BM25 Top3 | 3.00 | 76% | 76% |
| bridge | + SVD Top2 | 4.22 | 79% | 79% |
| answer | BM25 Top3 | 3.00 | 58% | 58% |
| answer | + SVD Top1 | 3.39 | 61% | 59% |
| answer | + SVD Top2 | 4.10 | 66% | **68%** |

SVD Top1 在 answer 上有一个很小的低预算互补区间，但增加到约 4 blocks 后，直接多取 lexical
候选更好。固定 BM25 Top4 的测试 answer recall 为 68%，也高于 `Top3 + SVD Top2` 的 66%。
因此当前 QK 追加不具备有说服力的 cost-adjusted 增益。

### 19.5 排除 K 范数偏置

raw max-QK score 与候选最大 K norm 的平均 Pearson/Spearman 均在 -0.03 到 0.03 附近；QK
Top1 的 K-norm 百分位约 0.50--0.57，不集中在高范数极端值。精排失败并非简单的 K norm
或极值放大问题，因此没有依据直接改成 cosine normalization。

当前最合理的下一项诊断是拆开四个 layer/head 通道与获胜 query token：如果 train 选择的单通道
能在 dev/test 泛化，问题是跨 head max 聚合；如果获胜 token 主要是标点或功能词，问题是 query-token
max 聚合；如果两者都不是，则固定通道的 block-local Q/K 与跨块语义相关性本身不足，应停止该精排
方向，转向可训练的 passage head 或在真实同一推理轨迹中评测 causal KV。

## 20. 通道归因与轻量 passage head

### 20.1 layer/head 与 query token 归因

在相同 800 steps 上保存每个候选的四通道独立分数和获胜 query token。只使用 train 标签选择
单通道后，full128 均选择 `layer16/query_head14/kv_head7`：

| test step | 跨通道 max | train-selected 单通道 | 任意通道 oracle | BM25 |
|---|---:|---:|---:|---:|
| bridge | 53% | 58% | 77% | **76%** |
| answer | 22% | 36% | 52% | **58%** |

跨通道 max 聚合确实会丢失信息，单通道可恢复 5--14pp；但即使知道每条 query 应选哪个通道，
answer oracle 仍低于 BM25。QK Top1 的获胜 query token 主要是内容词：bridge 80--86%，answer
89--95%；reachable target 上内容词比例更高。因此失败既不是标点/stopword 主导，也不能只靠
删 query token 解决。

使用 train-selected full128 单通道做单调追加时，测试 answer 的 `BM25 Top3 + QK Top1`
以平均 3.68 blocks 得到 69%，同行数 lexical 为 68%；dev 持平，train 仅 +2.5pp，差异很小且
不稳定。bridge 在测试上持平。固定通道 QK 至多提供边缘互补，不能单独构成主方法。

### 20.2 pairwise linear passage head

为检验四通道是否包含可学习的互补信号，固定一个极小模型：每条 query 内分别标准化 BM25 和
每通道 full/SVD score，用 train gold block 构造正负候选差分，训练无截距 pairwise logistic
head。没有使用 dev/test 标签训练权重。

400-query 初步结果：

| features | step | train | dev | test | test BM25 |
|---|---|---:|---:|---:|---:|
| BM25 + full128 | bridge | 74.0% | 73.0% | 75.0% | **76.0%** |
| BM25 + full128 | answer | 51.5% | **58.0%** | 56.0% | **58.0%** |
| BM25 + SVD32 | answer | 52.5% | **60.0%** | **59.0%** | 58.0% |
| BM25 + full128 + SVD32 | bridge | **74.5%** | **76.0%** | 74.0% | **76.0%** |
| BM25 + full128 + SVD32 | answer | 51.5% | **60.0%** | **61.0%** | 58.0% |

answer 在 dev/test 有 +4/+3pp 的弱正信号，但 bridge 出现 train/dev 增益、test 下降。200/100/100
steps 的方差和 MuSiQue 官方 train/dev 来源差异使该结果不足以支持结论。

### 20.3 扩大到 2,000-query 的预注册复验

新建 paragraph-aligned v3：1,000 train、500 dev、500 test，共 2,000 个两跳问题和 4,000
原子步骤；仍为 9,999,872 tokens / 39,062 blocks。包含 2,816 个 support blocks，4,000/4,000
target spans 审计通过，无同 block 两跳。

BM25 Top3/Top16：

| split | bridge Top3/16 | answer Top3/16 |
|---|---:|---:|
| train | 69.9% / 83.2% | 47.8% / 77.4% |
| dev | 70.2% / 82.4% | 57.0% / 84.4% |
| test | 71.6% / 86.4% | 60.2% / 89.8% |

官方 train 与官方 dev 的 answer 分布差异稳定存在。扩大版固定同一四通道、SVD32 和 pairwise
结构，不再选择模型；同时报告两项：`train-only` 衡量跨来源泛化，`train+dev` 作为同域校准后
只评估冻结 test。Top16 涉及 20,187 个唯一 block，raw 四通道 K 约 4.93 GiB。GPU 流水线已
部署，完成后再判断可训练 passage head 是否值得保留。

### 20.4 2,000-query passage-head 正式结果

Top16 共涉及 20,187 个唯一 block。四通道 raw K materialization 用 244.04 s，约 21,177
tokens/s；train-only SVD32 平均保留 97.47% 能量，basis 拟合 1.16 s，投影全部候选 16.57 s。

预注册的 train-only SVD32 passage head 测试结果：

| step | BM25 Top3 | learned Top3 | wins/losses | McNemar p |
|---|---:|---:|---:|---:|
| bridge，500 steps | 71.6% | **74.6%** | 26 / 11 | **0.0201** |
| answer，500 steps | 60.2% | **62.8%** | 39 / 26 | 0.1360 |
| combined，1,000 steps | 65.9% | **68.7%** | 65 / 37 | **0.0072** |

这是真正的跨来源结果：head 只使用官方 train 的 gold block 学习，官方 dev 来源的 500 条 test
从未参与 basis、权重或方法选择。单独 SVD max-QK 仍很差（test bridge 38.6%，answer 25.0%），
有效的是 BM25 与四通道低秩内部状态的 pairwise 组合，而不是用 QK 替代 lexical retrieval。

固定结构后再加入 dev 做同域校准：

| features | bridge test | answer test | combined |
|---|---:|---:|---:|
| BM25 baseline | 71.6% | 60.2% | 65.9% |
| BM25 + SVD32 | 74.2% | 62.8% | 68.5% |
| BM25 + full128 + SVD32 | **75.4%** | 62.6% | **69.0%** |

`both` combined wins/losses 为 69/38，McNemar `p=0.0035`；但它需要同时读取 full128 和
SVD K。更适合作为主方法的是 train-only SVD32：仅需四个 32-d 通道、跨来源显著、存储约为
raw128 的四分之一。passage head 只有 5 个线性输入（BM25 + 4 个通道），额外计算可忽略。

当前 GPU 测得 16-candidate Q capture 约 34.6 ms，raw+SVD mmap load 约 14.3 ms，SVD score
约 1.68 ms。SVD-only runtime 不必加载 raw K，预计可删除约四分之三候选传输；需要在最终实现
中单独实测，不能直接把 raw+SVD 联合计时当成 SVD-only 延迟。

### 20.5 4,000-step 自适应 block 预算

使用完整 Top512 排名复验 3/16/512 风险门：

| 策略 | test recall | mean blocks |
|---|---:|---:|
| fixed K3 | 65.9% | 3.0 |
| fixed K8 | 80.2% | 8.0 |
| fixed K10 | 83.1% | 10.0 |
| fixed K16 | 88.2% | 16.0 |
| fixed K64 | 93.3% | 64.0 |
| fixed K512 | 97.6% | 512.0 |
| adaptive，dev target 0.80 | **84.1%** | **9.85** |
| adaptive，dev target 0.85 | 88.9% | 23.01 |
| adaptive，dev target 0.90 | 93.3% | 97.84 |

风险门在约 10-block 区间比固定 K10 高约 1pp，但高召回时仍频繁触发 512 fallback，平均预算
迅速上升。它可作为工程策略，但不是主要创新。当前主线应转为：BM25 postings 粗检索、SVD32
internal-state passage head 精排、Top3 reader、逐步状态更新。

### 20.6 paired Top3 reader

冻结前 100 个 test 两跳问题（200 atomic steps），BM25 与 train-only SVD32 passage head 使用
相同 Qwen3-0.6B、三个 full blocks、adaptive prompt、24-token decode 和同一张 GPU。只改变
Top16 到 Top3 的排序。

| metric | BM25 | SVD passage head | paired result |
|---|---:|---:|---:|
| overall retrieval Top3 | 65.5% | **69.5%** | +4.0pp，CI [0.5, 8.0] |
| overall Top1 generation | 23.0% | **27.0%** | wins/losses 10/2，`p=0.0386` |
| overall any-branch generation | 37.5% | 38.0% | 无显著变化 |
| bridge Top1 generation | 41% | **47%** | +6pp |
| answer Top1 generation | 5% | **7%** | +2pp |

overall Top1 generation paired bootstrap 95% CI 为 `[+1.0,+7.5]pp`。任一分支 oracle 基本不变，
说明 passage head 的作用是把已经存在的可读 block 排到第一，而不是单纯扩大候选覆盖。单卡两次
reader wall-clock 分别为 256.95 s 和 256.19 s；这些时间不含在线 SVD 检索，因为排序已离线冻结。

grounding verifier 在 bridge 上由 BM25 的 35% 提高到 SVD passage 的 42%，进一步说明第一跳
局部状态质量确实改善。

### 20.7 严格两跳状态传播

> **结果已撤回。** 本节第二跳检索和最终答案使用的状态残留 gold bridge；只有35%/42%的第一跳
> bridge正确率仍有效。无泄漏修正和替代结果见第22节。

将两组 verifier-selected bridge 分别真实写回第二跳问题，重新扫描全部 39,062 blocks，再用
BM25 Top3 生成最终 answer：

| strict-chain metric | BM25 first hop | SVD passage first hop |
|---|---:|---:|
| verified bridge correct | 35% | **42%** |
| second-hop retrieval Top3 | 53% | 53% |
| final Top1 answer | 3% | 4% |
| final any-branch answer | 18% | 19% |
| bridge 与 final Top1 同时正确 | 1% | 2% |
| bridge 与任一 final branch 同时正确 | 10% | 11% |

严格链最终 +1pp 不显著。第一跳 +7pp 被第二跳检索和极低的 answer reader 正确率吞没；当前
端到端瓶颈已经从 block retrieval 转移到 controller/verifier 和 0.6B relation reader。不能用
局部 reader 的显著结果宣称完整两跳推理已经解决。

### 20.8 SVD-only 在线延迟

最终部署路径只加载 SVD32，不读取 full128。GPU 上 200-step 实测：

| stage | median | p95 |
|---|---:|---:|
| 16-token Q capture | 35.88 ms | 36.19 ms |
| 16-block SVD K load + Q projection | 2.82 ms | 3.02 ms |
| 4-channel SVD QK | 1.80 ms | 1.93 ms |
| 5-input linear passage head | 0.19 ms | 0.24 ms |
| SVD passage total | **40.71 ms** | **41.00 ms** |

再加常驻 BM25 postings 的约 0.69 ms，稳态候选检索约 **41.4 ms/step**。三分支 reader 的平均
并行关键路径约 0.56 s，因此检索增加约 7%；相对单卡三分支串行生成约 1.2 s，增加约 3--4%。

到这里可以形成一个有实证支撑的核心方法：

1. postings/anchor 在 39k blocks 中亚毫秒产生 Top16；
2. 从当前 reasoning state 捕获四通道 Q；
3. 加载候选的 SVD32 block-local K；
4. 用 train-only 5-input pairwise passage head 排到 Top3；
5. 只把 Top3 block 交给并行 reader，生成并验证下一状态。

其创新点不是“低秩 QK 单独做语义检索”，因为实验已否定该说法；而是**把低秩内部注意力兼容性
作为 lexical coarse retrieval 的可学习残差信号**。扩大复验和 paired reader 已证明该残差信号
能显著改善局部读取，但严格链仍需要更强的 state verifier 和 relation reader。

### 20.9 answer reader 负结果

固定 100 个 test answer steps，直接提供 100% 正确的 gold block：

| reader condition | answer hit | mean F1 | mean generation |
|---|---:|---:|---:|
| full gold block + legacy | **30%** | **25.15** | 0.405 s |
| full gold block + extractive prompt | 14% | 17.41 | **0.299 s** |
| lexical Top2 sentences + legacy | **30%** | 25.12 | 0.442 s |

extractive prompt 的 paired wins/losses 为 5/21，`p=0.0025`，显著更差。无答案 lexical sentence
selector 的 Top1/Top2 answer-span recall 仅 57%/64%，平均读取 41/76 tokens；Top2 虽将上下文
减少约 70%，但 reader 准确率和 F1 均无提升，且 15 wins/15 losses。因此停止 prompt 微调和纯
lexical sentence 压缩。

失败样本主要包括：复述 subject/bridge、选择同段落中的错误关系值、只输出年份而漏月份、选择
相邻数值或人物。即使 gold block 正确，Qwen3-0.6B answer reader 上界也只有约 30%；严格链的
低最终正确率不能再归因于 block retrieval。更大 reader 的容量诊断尚未完成，服务器现有
Qwen3-8B 缓存不完整且外网 DNS 不可用。

## 21. 阶段性论文结论与边界

目前可由数据支持的核心结论：

1. 10M tokens 可表示为 39,062 个 256-token blocks；常驻 postings 在约 0.69 ms 内得到 Top512。
2. 四通道 K 的 rank-32 子空间平均保留约 97.5% 能量，K 存储和候选传输降为 raw128 的四分之一。
3. raw full128/SVD32 max-QK 单独精排显著弱于 BM25，因此“attention score 直接等于语义相关性”不成立。
4. train-only pairwise passage head 将 BM25 与四通道 SVD32 组合，在 1,000 个外部 test steps 上
   将 Top3 recall 从 65.9% 提到 68.7%，McNemar `p=0.0072`。
5. 同一 200-step paired reader 中，Top1 局部生成从 23% 提到 27%，wins/losses 10/2，
   `p=0.0386`；在线 SVD-only 精排约 41.4 ms/step。
6. 原 20.7 的严格链第二跳存在 gold bridge 残留，53% retrieval 和 3%/4% final Top1 均已失效；
   修复后第二跳 BM25 Top3 为 34%/38%，说明错误中间状态传播比旧结果显示的更严重。

适合继续发展的 ICLR 叙事是：**低秩内部状态不是独立的 RAG embedding，而是一个轻量、可并行、
模型内生的 residual ranker；它在 lexical coarse retrieval 后只检查少量 blocks，并改善下一步
局部生成。** 仍需补齐的关键证据是更强 reader 下的无泄漏严格链、更多模型/任务的跨域复验，
以及能把不确定状态池可靠压回 Top3 的关系支持验证器。当前不能声称模型每一步只读 Top3 已实现
近似无损推理。

## 22. 严格链泄漏修正与多状态检索

### 22.1 旧严格链结果为什么失效

`prepare_verified_chained_answer_steps.py` 原来只更新了 `compact_state_before`；仅 33/100 条样本有
`step_question_template`，而全部样本的 `lookup_key` 和 `retrieval_state` 都保留 gold bridge。
例如第一跳错误生成 `Scarlett O'Hara` 时，下一跳表面问题已变成配偶查询，但实际 BM25 状态仍为
`Vivien Leigh What is the spouse of Vivien Leigh?`。因此 20.7 中第二跳 Top3=53%、Top16=86%
以及基于这些 block 的最终答案生成均不能再作为严格链证据。35%/42% 的第一跳 bridge 正确率不受影响。

修复后统一用模型生成状态重写 `lookup_key`、`step_question`、`retrieval_state`、compact/full state；
优先把生成实体填入官方 `#1` 关系模板。100 条 test 的无泄漏结果：

| 第一跳来源 | bridge 正确 | 第二跳 BM25 Top1 | Top3 | Top16 |
|---|---:|---:|---:|---:|
| BM25 Top3 reader | 35% | 8% | 34% | 49% |
| SVD passage Top3 reader | **42%** | **10%** | **38%** | **51%** |

旧泄漏版本的 Top3/Top16 为 53%/86%；修复后大幅下降，证明主要瓶颈是错误中间状态导致的查询漂移，
而不是第二跳全库 BM25 本身。SVD 第一跳仍稳定带来约 +4pp，但不足以阻断误差传播。

### 22.2 不确定性保留：三个桥接假设并行检索

当前 verifier 过早把三个第一跳分支压成一个状态。test 上 verifier 单选 bridge 正确率为42%，但任一
分支包含正确 bridge 为51%；dev 对应33%和49%。因此保留三条状态，各自查询39,062个 block，再在
固定总 block 预算下去重融合。三倍 postings 查询仍只需约0.72 ms/原问题。

SVD 第一跳来源的结果：

| split | 方法 | Top3 | Top6 | Top9 | Top16 |
|---|---|---:|---:|---:|---:|
| dev | verifier 单状态 | **27%** | 34% | 37% | 42% |
| dev | 三状态轮转 | 22% | **38%** | **43%** | **53%** |
| test | verifier 单状态 | **38%** | **48%** | 49% | 51% |
| test | 三状态轮转 | 31% | 43% | **53%** | **58%** |

结论不是“始终分叉”：Top3/Top6 小预算时错误状态会挤掉正确 block；从 Top9 开始 dev/test 均有
+4--11pp。Top16 的 test 配对结果为 8胜1负，McNemar `p=0.0391`；BM25 第一跳来源为49%到57%，
9胜1负，`p=0.0215`。dev 上搜索 verifier 排名对应的单调配额后，冻结配额也没有超过近似均匀轮转，
说明高不确定性时应保留状态多样性，而不能继续重押 verifier Top1。

### 22.3 能否用 SVD32 把16候选压回Top3

为2745个动态候选 block 实际提取四通道 block-local K，共702,720 tokens；K profiling 22.27 s，
31,562 tokens/s。复用原 train-only SVD32 basis，投影仅2.44 s。每个问题对三个生成状态分别捕获 Q，
平均99.5--100.6 ms；共享16-block池的全部 SVD 打分和线性 head 约1.4 ms。

结果是否定的：

| 方法 | dev Top3 | test Top3 |
|---|---:|---:|
| 单状态 BM25 | **27%** | 38% |
| verifier-selected Q + passage head | 22% | **39%** |
| 三状态 passage score 最大值 | 21% | 31% |
| 三状态 passage score 平均值 | 16% | 20% |
| dev-trained 10-feature动态池校准器 | 29% | 38% |

动态校准器在test为3胜3负，完全无提升。说明在错误中间状态存在时，BM25共识、verifier分数和当前
四通道SVD32相似度仍不足以判断哪个候选真正支持“实体+关系”；错误 Q 会产生高分伪匹配。应停止继续
调聚合系数。下一项方法必须引入关系支持/状态转移验证信号，而不是继续叠加相似度分数。

## 23. 最终同查询并行测速

固定同一12个步骤、每步3个候选block、每分支最多24个生成token，真实把同一问题的分支分发到不同
RTX 3090：

| GPU | 平均墙钟 | 相对1卡 | 并行效率 | 聚合生成速度 |
|---:|---:|---:|---:|---:|
| 1 | 1.057 s | 1.000x | 100.0% | 28.45 tok/s |
| 2 | 0.715 s | 1.478x | 73.9% | 42.05 tok/s |
| 3 | **0.483 s** | **2.189x** | 73.0% | **62.29 tok/s** |

三分支在三卡上接近一分支一卡；效率损失来自模型复制、进程调度、不同生成长度和同步。结合单状态
约41.4 ms检索，传统单状态Top3路径中检索只占三卡reader墙钟的约8%。多状态SVD路径需要三次Q捕获，
约101 ms，当前尚未换来Top3召回提升，因此不能作为最终部署方案。

## 24. Train-only 状态转移验证器

### 24.1 冻结Qwen零样本支持判断

对每个第一跳分支构造三元组：原子问题、单个256-token证据block、模型生成的bridge。测试两种冻结模型
分数：`Yes/No`直接支持判断，以及候选答案的长度归一化条件对数似然。两种方法均不读取gold bridge。

| selector | dev bridge | test bridge | 在线额外耗时 |
|---|---:|---:|---:|
| 原grounding heuristic | 33% | **42%** | 近似0 |
| frozen Yes/No | **38%** | 37% | 42.5 ms/query |
| frozen answer logprob | 31% | 35% | 39.1 ms/query |
| dev选择融合 | **40%** | 39% | 约81.6 ms/query |

Yes/No在分支级仍有弱信号：test成对排序正确率77%、AUC约0.80；但旧heuristic成对正确率86%，且dev
融合无法迁移到test。因此冻结Qwen支持分数不能直接作为最终selector，继续调prompt没有依据。

### 24.2 500条train pairwise transition head

使用500条官方train问题；每条从BM25 Top3各读取一个block并生成bridge，共1500条分支。只在265个同时
含正负分支的train问题上构造1060个对称pairwise差分。部署版本只使用已有CPU可计算特征：grounding
score及其结构化分解、retrieval rank、生成长度；不使用Yes/No或答案似然，因此没有额外模型前向。

| selector | train | dev100 | test100 |
|---|---:|---:|---:|
| 原heuristic | 39.2% | 33% | 42% |
| structured transition head | **49.2%** | **44%** | **50%** |
| full head（含冻结Qwen分数） | 49.2% | 45% | 48% |
| any-branch oracle | 54.2% | 49% | 51% |

structured head在test100相对heuristic为9胜1负，`p=0.0215`，且只比三分支oracle少1条。虽然full head
在dev高1pp，但需约80 ms额外前向；按准确率/延迟Pareto选择structured版本。

扩大到冻结的完整500条test后：

| 第一跳选择 | bridge正确 |
|---|---:|
| 原始retrieval Top1分支 | 40.4%（202/500） |
| 原grounding heuristic | 35.4%（177/500） |
| train-only structured head | **43.2%（216/500）** |
| any-branch oracle | 50.6%（253/500） |

structured head相对旧heuristic为54胜15负，`p=2.61e-6`；相对原始Top1为14胜0负，`p=1.22e-4`。
这说明旧手写verifier在扩大样本后实际有系统性负作用，而train-only head稳定消除了错误切换。

### 24.3 接回无泄漏10M第二跳

完整500条中，transition head选出的bridge写回所有活动查询字段，再扫描39,062 blocks：

| metric | recall |
|---|---:|
| BM25 Top1 | 7.6% |
| BM25 Top3 | 37.0% |
| BM25 Top16 | 56.0% |
| Top3 given correct bridge | 65.3% |
| Top16 given correct bridge | 89.8% |

bridge修复并不会自动转化为固定Top3收益：正确bridge中约24.5%的target仍排在4--16位。当前下一瓶颈是
第二跳候选压缩，而不是第一跳状态选择。100条完整严格链中，bridge为50%、Top3 retrieval为39%；原始
最终Top1为3%、任一分支13%。

### 24.4 Answer-specific transition head

按相同协议生成500条train answer和100条dev answer，只用train标签训练answer selector。部署版继续采用
零额外模型前向的structured features：

| split | 原始Top1 | heuristic | structured head | any branch |
|---|---:|---:|---:|---:|
| train500 | 6.8% | 9.2% | **14.4%** | 17.8% |
| dev100 | 3% | 7% | **11%** | 17% |
| dynamic test100 | 3% | 5% | **9%** | 13% |
| strict test500 | 2.2% | 4.2% | **6.4%** | 9.4% |

dynamic test100中structured answer head相对原始Top1为7胜1负，`p=0.0703`；严格链最终9条正确中8条
同时bridge正确。扩大到冻结的500条严格链后，structured answer head相对原始Top1为23胜2负，
`p=1.94e-5`；相对旧heuristic为16胜5负，`p=0.0266`。最终32/500条正确，其中28条同时满足
bridge和最终答案正确，即严格联合正确率5.6%。

500条实际生成中，第一跳三分支总生成时间均值1.007 s、理想并行关键路径0.485 s；第二跳分别为
1.678 s和0.727 s。结合约41.4 ms第一跳SVD检索及0.34 ms第二跳BM25，三分支真实同查询并行时，
两步主路径估计约1.25 s；这是由分阶段实测相加的估计，不冒充统一端到端墙钟。

### 24.5 当前方法更新

当前最有证据的方法不再是“生成后用手写verifier选一个分支”，而是：

1. 10M postings得到Top16，train-only SVD32 passage head排Top3；
2. 三个reader各自只读一个block并生成候选状态；
3. train-only structured transition head用证据支持、检索rank和输出形态选择下一状态；
4. 把选中状态无泄漏写回，重新检索Top3；
5. 三个answer reader独立生成，answer-specific structured head选择最终输出。

两个transition head均为极小线性模型，不新增block读取或Qwen前向。第一跳选择在500条外部test上显著改善，
但完整推理仍受第二跳Top3候选压缩和0.6B reader能力限制，尚未实现近似无损推理。

## 25. 置信度门控的选择性深读

### 25.1 Head分数能否识别“全部分支都错”

冻结answer structured head后，只使用其三个线性分数，不重新训练风险模型。候选绝对最高分在dev100/test500
上预测所选答案正确的ROC-AUC分别为0.773/0.801，预测任一分支正确的AUC为0.688/0.784。Top1-Top2
margin在test仅0.551，因此风险门应使用绝对最高分，而不是常见的score gap。

使用dev分数的25%分位数`5.550365`作为固定阈值：

| test500分组 | queries | answer accuracy |
|---|---:|---:|
| 高置信度，不扩展 | 214 | **12.15%** |
| 低置信度，触发扩展 | 286 | 2.10% |

由于dev/test存在分数分布漂移，dev目标25%扩展在test实际触发57.2%。但低置信度集合捕获59.8%的当前
错误，以及61.3%的“当前答案错且目标block排在4--16位”机会。若提高到dev 50%分位阈值，test扩展
74.2%、捕获78.5%的机会，平均读取量会更高。因此先冻结较保守的25%分位阈值。

### 25.2 rank4--6第一轮扩展设计

不直接读取Top16。只对286条低置信度问题增加rank4--6三个block：

- 平均block预算：`3 + 0.572 * 3 = 4.716`；
- 高置信度214条仍严格读取3个block；
- 低置信度中有32条目标位于rank4--6；
- 完美reader上界为当前6.4%再加6.4pp，即12.8%，实际一定更低；
- 扩展分支与原Top3分开生成后合并，冻结answer head重新选择，不重跑或改变原Top3输出。

服务器当前8卡均被其他Llama-8B长基准占用。等待脚本`wait_run_confidence_extension_r4_6_v7.sh`已部署；
检测到任意1--6张真正空闲卡后自动按可用卡数运行、合并并输出paired结果，不与现有任务共卡。实验完成前
不能声称选择性深读已提高最终答案。

## 26. Top3 block 拼接 reader

为检验三个候选 block 是否应该联合读取，固定第一跳 train-only SVD32 passage head 的同一组 Top3，
不改变检索排序、模型、decode 上限或 atomic prompt。基线让三个 256-token block 分别生成后由 structured
transition head 选择；新方法按检索排名加入 `[Evidence 1..3]` 分隔符，拼成 768-token memory，只生成一次。

500 条 test 的第一跳结果：

| reader | first Top3 recall | bridge correct | paired wins/losses | p |
|---|---:|---:|---:|---:|
| 三分支 + structured head | 74.6% | 43.2%（216/500） | - | - |
| Top3 concat | 74.6% | **44.2%（221/500）** | 50 / 45 | 0.6817 |

拼接只增加 5 个净正确，但逐条变化很大且不显著。concat 的 221 条 target-hit 输出中，归一化后只有
89 条是纯 bridge 实体，132 条是包含正确实体的更长句子；命中输出平均 9.92 generated tokens。严格链
会把清洗后的完整生成文本写回第二跳，因此“提到正确实体”不等于“产生了干净的检索状态”。

无泄漏写回后重新检索 39,062 blocks：

| concat 状态 | queries | Top1 | Top3 | Top16 |
|---|---:|---:|---:|---:|
| bridge correct | 221 | 9.95% | 61.09% | 87.33% |
| bridge incorrect | 279 | 5.38% | 15.77% | 25.45% |
| overall | 500 | 7.4% | 35.8% | 52.8% |

与三分支 structured baseline 的严格配对：

| second-hop recall | baseline | concat | wins/losses | p |
|---|---:|---:|---:|---:|
| Top1 | 7.6% | 7.4% | 7 / 8 | 1.0000 |
| Top3 | **37.0%** | 35.8% | 27 / 33 | 0.5190 |
| Top16 | **56.0%** | 52.8% | 22 / 38 | 0.0519 |

因此 concat 没有改善当前 MuSiQue atomic-step 严格链。它略微增加 bridge 字符串命中，却使整体第二跳
Top3/Top16 分别下降 1.2/3.2pp。当前每个 atomic step 通常只需要一个支持 block，Top3 多数是一个
证据加两个干扰项；该实验不能否定真正多前提单步任务中的 set reader，但否定了在当前任务上无条件拼接。

速度方面，四卡 query-shard 的 500 条正式生成墙钟为 62.92 s；单条 concat generation 均值 0.327 s。
原三分支总生成均值为 1.007 s、理想并行关键路径 0.485 s，因此 concat 相对串行三分支约省 67.5%，
相对三卡同查询并行关键路径约省 32.6%。第二跳离线任务墙钟 39.70 s，其中 BM25 建库 23.89 s；常驻
索引下 500 条 BM25 matrix 查询仅 0.585 s。

## 27. Bridge reader 容量与显式推理提示消融

### 27.1 正确 block 单独读取与 Top3 拼接

在第一跳 Top3 召回成功的 373 条中，0.6B 把包含 target span 的正确 block 单独读取时，任一正确
block 分支答对 226 条（60.59%）；Top3 concat 答对 208 条（55.76%）。两者共同正确 174 条，
单 block 独有 52 条，concat 独有 34 条，均失败 113 条。按 gold block 在候选中的位置：

| gold block rank | queries | 单正确block reader hit |
|---:|---:|---:|
| 1 | 312 | 185（59.29%） |
| 2 | 41 | 27（65.85%） |
| 3 | 20 | 14（70.00%） |

因此约39.4%的失败在“证据完全正确且不含另外两个block”时已经发生，0.6B reader 是主瓶颈；
拼接两个额外干扰 block 再净损失4.83pp。失败并非主要由正确证据排在Top3后部引起。

### 27.2 0.6B 显式相关事实推理提示

固定同一 Top3 concat，新增严格结构：

```text
Relevant fact: <copy one supporting sentence>
Bridge entity: <only the shortest missing entity>
```

评测和状态传播只读取 `Bridge entity:` 字段，不能因 rationale 包含 gold 实体而命中。结果：

| 0.6B prompt | overall bridge | given Top3 recalled | given not recalled | wall / 500 |
|---|---:|---:|---:|---:|
| 直接最短答案 | **44.2%** | **55.76%** | 10.24% | **62.92 s** |
| Relevant fact -> entity | 37.8% | 46.92% | **11.02%** | 148.82 s |

显式推理降低8.84pp条件准确率，且36/500没有产生可解析的entity字段；单次生成均值从0.327 s增至
1.029 s。当前任务是局部关系读取，不需要自由推理链；0.6B会在复制事实、选择关系和遵循格式之间增加
错误。继续做普通prompt engineering没有依据。

### 27.3 Qwen3-8B 固定直接答案提示

保持相同Top3、相同顺序、相同atomic直接答案prompt和24-token decode，只把reader从Qwen3-0.6B换成
Qwen3-8B。检索仍由原冻结的0.6B SVD32 passage head产生，因此只隔离reader容量：

| reader | overall bridge | given Top3 recalled | given not recalled | 纯实体exact | wall / 4 GPU |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 44.2%（221/500） | 55.76%（208/373） | 10.24%（13/127） | 89/500 | 62.92 s |
| Qwen3-8B | **67.2%（336/500）** | **82.84%（309/373）** | **21.26%（27/127）** | **239/500** | 121.43 s |

8B相对0.6B为133胜18负，McNemar `p=7.32e-23`。单条8B generation均值0.490 s，约为0.6B
的1.50倍，而召回成功条件准确率增加27.08pp。模型容量是当前第一跳低正确率的主要原因。

将8B bridge无泄漏写回39,062 blocks：

| second-hop | 0.6B concat | 8B concat | wins/losses | p |
|---|---:|---:|---:|---:|
| Top1 | 7.4% | **11.0%** | 21 / 3 | 2.77e-4 |
| Top3 | 35.8% | **47.0%** | 73 / 17 | 1.95e-9 |
| Top16 | 52.8% | **69.8%** | 96 / 11 | 4.30e-18 |

8B正确bridge的336条中，第二跳Top3/Top16为61.01%/89.29%；错误bridge的164条中为
18.29%/29.88%。更大reader显著增加正确状态数量并改善综合召回，但给定正确状态后的Top3压缩仍约
61%，所以第二跳排序仍是独立瓶颈。合理优化顺序是：用8B作为reader上界和teacher，蒸馏/微调0.6B做
relation-conditioned entity extraction；不要让0.6B生成自由文本CoT；同时单独改进第二跳Top16到Top3。

## 28. Bridge来源 × 最终答案reader的2×2严格实验

为分离第一跳状态质量和最终answer reader容量，构造四组完整500-query链。第一次Top3始终来自冻结的
0.6B SVD32 passage head；0.6B/8B bridge分别生成自己的无泄漏第二跳状态并用同一BM25检索Top3；
最终三个256-token block按排名拼接，0.6B或8B使用同一adaptive answer prompt生成最多24 tokens。

总体最终答案命中：

| bridge reader | answer reader 0.6B | answer reader 8B |
|---|---:|---:|
| 0.6B | 7.0%（35/500） | **21.2%（106/500）** |
| 8B | 10.6%（53/500） | **31.6%（158/500）** |

固定0.6B bridge链，把answer reader换成8B为75胜4负，`p=5.24e-18`；固定8B bridge链为113胜8负，
`p=7.27e-25`。因此8B不仅改善bridge，也显著改善最终关系值读取。

只看正确答案block已进入第二跳Top3：

| bridge source | Top3 retrieved | 0.6B answer | 8B answer |
|---|---:|---:|---:|
| 0.6B bridge | 179 | 17.88%（32/179） | **49.72%（89/179）** |
| 8B bridge | 235 | 20.43%（48/235） | **55.32%（130/235）** |

即使证据已经在768-token拼接memory内，0.6B仍只答对约18%--20%，8B为约50%--55%；最终answer
reader与bridge reader一样存在显著容量瓶颈。正确block未进入Top3时，四组仍分别有0.93%、5.30%、
1.89%、10.57%命中，来自参数知识、替代证据或偶然命中，不能当作retrieval成功。

按第一跳bridge是否正确分组：

| chain | given bridge correct | given bridge incorrect | bridge+answer joint |
|---|---:|---:|---:|
| 0.6B bridge + 0.6B answer | 14.48% | 1.08% | 6.4% |
| 0.6B bridge + 8B answer | 40.27% | 6.09% | 17.8% |
| 8B bridge + 0.6B answer | 14.29% | 3.05% | 9.6% |
| 8B bridge + 8B answer | **43.45%** | 7.32% | **29.2%** |

固定0.6B answer，把bridge换成8B使总体7.0%升到10.6%（29胜11负，`p=0.00643`）；固定8B answer
使21.2%升到31.6%（65胜13负，`p=1.81e-9`）。两个阶段的容量收益可以叠加。

四卡500条answer生成墙钟：0.6B为90.56/91.00 s，8B为130.40/120.35 s；单条模型生成均值
分别约0.46 s和0.59--0.64 s。当前最强严格链为“0.6B SVD32检索 + 8B concat bridge + BM25二跳
+ 8B concat answer”，最终答案31.6%。它仍不是近似无损：第二跳Top3只有47%，且给定Top3正确证据时
8B reader也只有55.3%，所以检索排序和answer读取仍需分别优化。

## 29. 第二跳Top16拼接reader

固定当前最强8B bridge链、同一BM25第二跳排序、Qwen3-8B answer reader、adaptive prompt和24-token
decode，只把输入从Top3全block拼接（768 tokens）扩大到Top16（4,096 tokens）。严格链整体gold block
Recall@3/16为47.0%/69.8%；bridge正确子集Top16才是89.3%，不能把条件召回当作全体召回。

总体配对结果：

| answer memory | blocks/tokens | final answer | wins/losses vs Top3 | p |
|---|---:|---:|---:|---:|
| Top3 concat | 3 / 768 | 31.6%（158/500） | - | - |
| Top16 concat | 16 / 4,096 | **33.0%（165/500）** | 34 / 27 | 0.4426 |

Top16净增7条、+1.4pp，但不显著。按gold第二跳block的BM25排名拆分：

| gold rank | queries | Top3 reader | Top16 reader | wins/losses | p |
|---|---:|---:|---:|---:|---:|
| 1--3 | 235 | **55.32%** | 50.64% | 11 / 22 | 0.0801 |
| 4--16 | 114 | 18.42% | **35.09%** | 22 / 3 | 1.57e-4 |
| >16 / missing | 151 | 4.64% | 3.97% | 1 / 2 | 1.0000 |

Top16对rank4--16样本有真实且显著的补救，净救回19条；但在正确证据本已位于Top3的样本中，额外
13个干扰block净破坏11条。无条件扩展的收益和损失大部分抵消。

速度代价也很高：单条8B generation从0.587 s增至1.497 s（2.55倍），四卡500条墙钟从120.35 s
增至247.20 s（2.05倍），读取block预算从3增至16（5.33倍）。因此Top3不是绝对正确预算，但固定Top16
也不是好的主策略。数据支持置信度/风险门控：高置信度或Top3已充分支持时读3块；只有预测gold可能落在
4--16时才扩展，并在扩展前用关系感知reranker或sentence selector减少干扰。

## 30. 8B Top16失败是上下文容量还是证据干扰

第29节测试的是第二跳最终答案，不是第一跳bridge。为区分“8B不会读取正确证据”和“16块相似证据
造成干扰”，在原始gold第二跳状态上增加两个仅用于诊断的oracle reader：唯一完整256-token gold block，
以及只保留该block中的官方support paragraph。两者都没有检索错误或生成bridge错误。

| 8B oracle input | evidence recall | answer hit | mean generation |
|---|---:|---:|---:|
| 唯一gold block | 100% | 68.8%（344/500） | 0.357 s |
| 唯一gold support paragraph | 100% | **70.8%（354/500）** | 0.313 s |

因此8B有基本局部读取能力，但即使gold state和gold evidence都给定仍有约29%错误；把block内部无关文本
删掉只提升2pp，说明剩余错误主要是关系绑定、别名/值选择、生成和答案口径，而不是256-token块内噪声。

在8B bridge正确且gold block确实位于Top16的同一300条上严格比较：

| input | answer hit |
|---|---:|
| 唯一gold block + gold state oracle | 69.67%（209/300） |
| 唯一gold paragraph + gold state oracle | 72.00%（216/300） |
| 16 blocks + generated-correct state | **50.67%（152/300）** |

oracle full block相对Top16为75胜18负，`p=1.92e-9`。Top16中gold rank 1--3时答对116/205
（56.59%），rank 4--16时仅36/95（37.89%）。因此4,096 tokens没有超过模型可计算上下文，失败是
证据选择/位置/关系干扰：目标事实只占很少tokens，前排相似block提供竞争实体和值，模型对早期证据有偏置。

合理方案不是继续增加raw blocks，而是Top16召回后做relation-aware rerank、逐block候选抽取与支持验证、
sentence/span压缩，或只对低置信度样本扩展。4K context capacity只表示模型能执行前向，不保证在15个
hard negatives中稳定找到并使用唯一正确事实。

## 31. Top16逐block候选抽取与独立支持验证

固定最强8B bridge链和第二跳BM25 Top16，测试“Top16作为候选池而非原文拼接”。所有选择均不读取gold。

### 31.1 生成式 `SUPPORTED / NOT_SUPPORTED` 负结果

第一版让8B对每个block输出`SUPPORTED: answer`或`NOT_SUPPORTED`，再按BM25顺序选择第一个有效候选。
500×16次抽取用4卡耗时552.82 s。

| selector | final answer | vs Top16 concat wins/losses | p |
|---|---:|---:|---:|
| first supported | 25.6% | 38 / 75 | 6.41e-4 |
| grounded first | 25.6% | 38 / 75 | 6.41e-4 |
| answer consensus | 25.4% | 38 / 76 | 4.75e-4 |
| any-branch oracle | 32.6% | - | - |

203/500条没有任何SUPPORTED输出。349个gold-in-Top16问题中，gold block只被判SUPPORTED 209次，
真正抽对150次，另有59次在gold block上抽出错误值；7651个非gold分支有321次误报SUPPORTED。把支持
判断和答案抽取合并成一次生成产生高假阴性，oracle本身还低于Top16拼接33.0%，因此否定该实现。

### 31.2 直接候选抽取 + 独立冻结8B verifier

根据失败归因解耦两个操作：每个block无条件使用atomic prompt抽取最短候选；再对`question + block +
candidate`单独计算冻结Qwen3-8B Yes/No下一token margin和候选条件平均logprob。仍不使用train/dev/test
标签调阈值或融合权重。

逐block直接抽取结果：

| selection | final answer |
|---|---:|
| BM25 rank-1 candidate | 9.2% |
| 16 candidates任一正确oracle | **48.8%（244/500）** |
| answer-likelihood argmax | 29.4% |
| Yes/No support argmax | **36.4%（182/500）** |

Yes/No selector相对Top3 concat 31.6%为67胜43负，`p=0.0279`；相对Top16 concat 33.0%为56胜39负，
`p=0.1002`。因此它显著超过低预算Top3，且相对Top16有+3.4pp点估计但尚不显著。

按gold rank：

| gold rank | queries | Top16 concat | support selector | candidate oracle |
|---|---:|---:|---:|---:|
| 1--3 | 235 | 50.64% | **53.62%** | 68.09% |
| 4--16 | 114 | 35.09% | **44.74%** | 64.04% |
| >16 / missing | 151 | 3.97% | 3.31% | 7.28% |

候选化在rank4--16上改善最明显，但冻结verifier仍将244个oracle机会压成182个正确，剩余12.4pp是选择
损失。条件似然弱于直接支持判断，不能作为主selector。

计算成本不可部署：4卡逐block生成墙钟987.98 s，支持+似然打分394.24 s，合计约23.0分钟；每query
16分支生成总时间6.92 s，理想16卡关键路径仍约0.879 s，尚未计支持打分。当前结果只验证方法方向：
Top16候选化和独立关系验证优于直接拼接，但必须把8B teacher行为蒸馏成batch化小型extractor/verifier，
并只在低置信度请求触发。由于相对Top16尚不显著且成本极高，本轮不继续复制到第一跳。

## 32. 为什么局部support verifier仍损失12.4pp

在244个“16候选至少一个正确”的问题中，Yes/No verifier选对182个，损失62个。正确候选相对所有错误
候选的branch-pair排序准确率为74.32%（4095/5510），但每题需要从最多15个负候选中取最大值；只要一个
hard negative获得异常高分就会覆盖正确候选。正确候选的support-score最佳排名分布：Top1 182、Top2 36、
Top3 9，其余17条分布在4--16。62次选择损失中，错误获胜候选只有5次来自gold block，57次来自非gold
block；错误winner相对最佳正确候选的平均margin仍高达11.68，属于自信错误而非微小分差。

主要现象不是简单“证据不支持”：Top16中存在局部真实但benchmark不接受的替代事实。例如：

- `spouse of Vivien Leigh`：非gold passage明确支持第一任丈夫Herbert Leigh Holman，而gold为Laurence Olivier；
- `child of Chiang Ching-kuo`：多个子女均被各自passage直接支持，gold只指定Chiang Hsiao-wu；
- `first African-American student...`：候选`James Meredith`语义正确，但gold为完整名`James Howard Meredith`；
- `type of government`：候选`Communist`与gold`a communist government`存在答案粒度差异；
- 部分官方atomic question呈现`What is the located in...`等不自然模板，使局部关系方向更难校准。

当前verifier只看一个`question + block + candidate`并取冻结8B的Yes/No下一token margin。它能判断局部蕴含，
却没有候选间全局视角、关系基数/时间限定、benchmark gold偏好或校准后的拒答阈值；当所有score都低时仍被迫
argmax。下一代不能只做独立二元支持判断，应使用typed relation和qualifiers、候选间pairwise/listwise比较、
alias-aware目标，并在train/dev hard negatives上校准阈值；低置信度时回退到联合证据读取，而不是强选一个候选。
