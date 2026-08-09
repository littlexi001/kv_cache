# 10M 多级 Bridge-Search 控制器实验

## 1. 动机

此前逐 token 的 C1/C3 动态 Q 检索存在两个问题：第二跳信号不稳定，并且正确 block 常在模型已经生成错误答案后才进入 K3。新的方案不再让模型一边承诺答案一边盲目刷新，而是将生成拆成显式动作：

```text
SEARCH: <桥接实体 + 尚未解决的关系>
FINAL: <最终答案>
```

SEARCH 阶段只更新内部工作集，不输出最终答案。证据进入工作集后才允许 FINAL。

## 2. 实现流程

1. 问题 Q 在 10M 的 39,062 blocks 上执行分布式 SVD32 粗检索和 raw128 精排，得到初始 K3。
2. Qwen3-0.6B 根据问题和 K3 输出结构化 SEARCH 动作。
3. 控制器删除原问题已经出现的主体，只保留运行时发现的新专名。
4. 使用 SEARCH 文本在全部 39,062 blocks 上动态 BM25 检索。
5. 对歧义实体保留原问题中的路径上下文，例如 `Henry V + Agnes + husband + place of death`。
6. 如果模型重复旧实体，则使用关系类型和局部距离从当前 memory 抽取新实体；spouse、parent、sibling、author、director 等关系分别处理。
7. 在 BM25 Top-N 中定位 `Passage: <focus entity>` 标题，加载从实体标题开始的连续 3 blocks。
8. 最终生成显式绑定 focus entity，避免读取相邻人物的地点或属性。

对于更多级问题，第 2～7 步继续循环，直到输出 FINAL、达到最大 search hops，或置信度门控停止。

## 3. Lou Breslow 实验

问题：`Where was the wife of Lou Breslow born?`

| 阶段 | 结果 |
|---|---|
| 初始全局 SVD32/raw128 K3 | `[20088, 45, 73]` |
| 模型动作 | `SEARCH: Lou Breslow's spouse, Marion Byron.` |
| 新实体归一化 | `Marion Byron` |
| 答案 block BM25 排名 | block 20096，第 1 |
| 实体 passage K3 | `[20096, 20097, 20098]` |
| 最终输出 | `The wife of Lou Breslow was born in Dayton, Ohio.` |
| Answer hit | **true** |

选择逻辑未读取 `Dayton, Ohio` 或 gold block ID。gold 只用于运行结束后的排名和命中统计。

## 4. 更换问题：Agnes 两跳实验

问题：`Where was the place of death of Agnes Of Hohenstaufen's husband?`

答案标注：`Brunswick`

这条数据更难，因为初始 SVD32/raw128 K3 完全没有召回正确 source：

```text
[21348, 33246, 19372]
```

控制器执行了两轮 SEARCH：

1. `SEARCH: Agnes Of Hohenstaufen <spouse>`：全局 BM25 恢复 Agnes source。
2. 当前 memory 中出现 `wife of ... Henry V`。0.6B 模型没有正确重写 query，关系约束后备自动抽取 `Henry V`。
3. 使用 `Henry V + 原问题路径` 消歧，在 BM25 Top-N 中定位 `Henry V, Count Palatine of the Rhine` 标题。
4. 加载实体 passage `[13094, 13095, 13096]`。
5. block 13096 包含：`Henry died in 1227 and is entombed in Brunswick Cathedral.`

最终输出为：

```text
Brunswick Cathedral.
```

Answer hit 为 **true**。

这说明控制器能在初始语义检索失败时通过后续 lexical SEARCH 恢复，而不是只能处理 Lou 这一条固定样本。

## 5. 为什么不算答案作弊

当前实验没有：

- 把 `Marion Byron`、`Henry V`、`Dayton` 或 `Brunswick` 写进选择代码；
- 根据 gold block ID 修改候选顺序；
- 在已知 source 内直接扫描；
- 把参考答案提供给模型。

桥接实体来自模型动作或当前检索 memory，检索仍覆盖完整 10M 语料。gold ID 只在检索结束后评估。

这里确实加入了工程归纳偏置：关系类型表、新实体规则、路径消歧和 Passage 标题窗口。它们类似数据库查询规划器或知识图谱遍历规则，适用于一类问题，而不是某个答案。若代码中直接写 `if question contains Lou: search Marion Byron`，才属于答案泄漏。

不过，这两个成功样本仍不能证明总体泛化。规则是在观察失败后逐步形成的，需要在未参与开发的多跳 holdout 上冻结代码评估。

## 6. 多级问题如何扩展

将问题表示成尚未完成的关系路径：

```text
已知实体 --relation_1--> bridge_1
bridge_1 --relation_2--> bridge_2
...
bridge_n --final_relation--> answer
```

每轮只执行一次：

```text
当前 memory
-> 抽取新实体
-> 生成 SEARCH(entity + path context + unresolved relation)
-> 全局检索
-> 实体消歧
-> 更新 K3
```

停止条件不能只依赖固定 hop 数，应同时检查：

- 模型是否输出 FINAL；
- 新 SEARCH 是否重复旧实体；
- 检索结果是否包含 focus entity；
- Top-1/Top-2 分差和 lexical/semantic 一致性；
- 是否达到最大 hop，防止循环。

## 7. 性能和边界

- 2026-07-12 在 8 张空闲 RTX 3090 上重跑 Lou 样例，初始 10M SVD32/raw128 检索耗时 0.772 秒。该值是脚本启动后的第一次检索，包含首次 CUDA 同步，不能视为稳态吞吐。
- 第二跳在全部 39,062 blocks 上执行 CPU BM25，单次 query 耗时 0.496 秒。本次两次在线检索合计 1.268 秒。
- Bridge SEARCH 生成 13 tokens，CUDA 同步耗时 0.736 秒，即 17.66 token/s；最终答案生成 15 tokens，耗时 0.497 秒，即 30.16 token/s。两段生成合计 1.234 秒。
- 不计一次性建库和进程/模型加载，以上被单独计时的在线检索与生成阶段合计约 2.50 秒。这里仍是单样本延迟，不是并发吞吐测试。
- 当前每次运行重建 BM25 索引需要约 21～25 秒，本次为 21.90 秒。生产实现应持久化 CountVectorizer、稀疏权重和文档元数据，不能在线重建。
- 本次整条远程命令墙钟时间为 43.8 秒，其中还包括 torchrun 八进程启动、模型加载、Q/K 索引加载和 BM25 建库，不能直接当作常驻服务的每请求延迟。
- 当前专名抽取主要针对英文大写实体，跨语言需要 NER 或模型结构化输出。
- Passage 标题窗口利用了 LongBench 语料结构；无标题语料需要 passage segmentation 或实体边界索引。
- 0.6B 模型的 SEARCH rewriting 不可靠，必须保留确定性 parser 和风险回退。

## 8. 产物

- 控制器：`src/run_global_bridge_controller_single.py`
- 服务器启动器：`scripts/run_global_bridge_controller_single_server.sh`
- Lou 最终结果：`outputs/global_bridge_controller_q0_v4/result.json`
- Lou CUDA 同步计时结果：`outputs/global_bridge_controller_q0_timing_v1/result.json`
- Agnes 最终结果：`outputs/global_bridge_controller_q6_h2_v10/result.json`
