# 面向逐步 KV 检索的 Verified Fact State Controller

## 1. 设计来源

本方案不是从任意 prompt 变体开始，而是由 Section 170 的三个失败现象直接推出：

1. 全局 Q/K 经常先路由到错误 record，后续模型只能从错误 K3 抽取错误实体；
2. record-local Q/K 虽能提高部分 answer-block 排序，却会删除生成正确 bridge 所需的 block；
3. 固定执行 SEARCH 会覆盖已经足够的证据，已取到 gold block 也不保证 0.6B 正确读取关系。

因此每一步需要显式区分：

- 当前 K3 新验证了哪些事实；
- 现有事实链是否已经足够回答；
- 如果不足，下一条未解决关系是什么。

## 2. 状态表示

模型每次只读取当前 K3，并输出：

```text
FACT: <subject> | <relation> | <object> | <exact evidence quote>
FACT: <subject> | <relation> | <object> | <exact evidence quote>
FINAL: <answer>
```

或：

```text
FACT: <subject> | <relation> | <object> | <exact evidence quote>
SEARCH: <exact bridge entity> | <still-unresolved relation>
```

控制器维护一个 compact fact ledger。旧 K3 卸载后不保留其全部 token/KV，只保留已经验证的原子事实。例如：

```text
step 1 K3 -> FACT: Lou Breslow | spouse | Marion Byron
step 2 K3 -> FACT: Marion Byron | born in | Dayton, Ohio
ledger -> Lou Breslow --spouse--> Marion Byron --born in--> Dayton, Ohio
FINAL -> Dayton, Ohio
```

这比把所有历史 block 累积进上下文更接近目标假设：每一步只读取少量外部 KV，中间结论以很小的结构化状态继续参与推理。

## 3. 确定性 verifier

模型自报 FACT/FINAL 不能直接信任。当前原型执行三层检查。

### 3.1 Block-local evidence

一条 FACT 的 subject 和 object 必须在同一个当前已加载 block 内共同出现，标准化字符距离不超过 800。不能因为 subject 和 object 分别出现在两个无关 K3 block 中就拼成事实。

初版检查只验证实体共现，仍会接受 `Ben Affleck | actor | Clark Kent` 和 `Emergency Wedding | director | Willard Parker` 这类同块共现但关系错误的事实。当前代码已进一步要求模型给出同时包含 subject/object 的原文短句，短句必须能在该 block 中精确标准化匹配。它仍不能完整替代 relation entailment，但比纯共现更严格；后续仍应比较 relation alias、依存路径或 NLI verifier。

### 3.2 Immutable fact ledger

只有在该事实对应证据仍在当前 K3 时通过检查，才写入 ledger。后续步骤可以读取 ledger，但不能修改已验证事实，也不能把未支持的模型输出加入状态。

### 3.3 Connected FINAL

FINAL 只有在以下条件同时成立时通过：

1. 当前 action 中所有新 FACT 都通过 block-local evidence；
2. FINAL 候选是 fact graph 中的实体；
3. 从问题中显式出现的某个实体出发，可以沿已验证 fact graph 到达 FINAL 候选。

若模型输出 FINAL 但 verifier 拒绝，候选不会直接作为答案，而会降级为新的 SEARCH query 以寻找支持或反证。

## 4. 与固定检索的区别

旧控制器：

```text
固定 search_hops
-> 每轮强制 SEARCH
-> 最后一轮自由生成答案
```

新控制器：

```text
当前 K3 + compact verified ledger
-> 提取当前 FACT
-> verifier
-> chain complete: 提前 FINAL
-> chain incomplete: SEARCH 下一条关系
```

它允许不同问题使用不同检索步数，也允许在证据已充分时省掉后续 BM25 和第二次生成。代价是结构化 action 可能需要更多生成 tokens，并增加 verifier 开销。

## 5. 可证伪实验

第一阶段固定使用当前表现更好的 question-BM25 K3，只比较控制器：

| 组别 | 初始 K3 | 控制器 |
|---|---|---|
| baseline | question BM25 K3 | forced SEARCH, 1 hop |
| proposed | question BM25 K3 | verified FACT + FINAL/SEARCH |

必须报告：

- FACT parser success；
- 平均写入 ledger 的事实数；
- verifier rejection rate；
- verified early FINAL rate；
- early FINAL precision；
- bridge gold Recall@3；
- Answer hit 和 token F1；
- action generation tokens、在线 median/p95。

拒绝标准：

1. 如果 early FINAL 大量通过但 precision 低，说明共现 verifier 太弱；
2. 如果几乎没有 FACT 通过，说明 0.6B 的结构化抽取或 block K3 不足；
3. 如果准确率不升且生成延迟明显增加，不应把该控制器作为主方法；
4. 只有 6 条开发诊断成功不能作为结论，必须冻结后运行 36 条 holdout。

## 6. 当前运行状态

### 6.1 verified-v1 六条结果

两张 GPU 空闲后运行了 6 条集成测试。SSH 调用在 184 秒超时，服务器短暂不可连接；恢复后确认远端任务已完整结束，6 条结果和 summary 均存在，没有残留进程。

该版本是“无跨跳 ledger、实体共现 verifier”的初版：

| 控制器 | bridge gold Recall@3 | Answer hit | mean token F1 | mean action generation |
|---|---:|---:|---:|---:|
| forced SEARCH | 2/6 | 1/6 | 0.1863 | 0.613 s |
| verified-v1 | 3/6 | 1/6 | 0.2708 | 1.132 s |

verified-v1 的新增 bridge gold 是 query 7，Answer hit 没有新增。6 条中模型全部先尝试 FINAL；verifier 只接受 query 1，拒绝其余 5 条。被拒 FINAL 的文本被降级为检索 query，这种较完整的关系描述偶尔比旧 bridge query 更有效，但结构化生成耗时明显增加。

代表输出：

- query 1：`FACT: Arnold Richards | board member | YIVO; FINAL: YIVO`，提前 FINAL 正确；
- query 7：模型 FINAL 文本已经包含正确答案 `Bill Miner`，但 FACT graph 没有把 Bill Miner 连接进去，verifier 拒绝；降级查询命中 gold，最终生成却再次遗漏 Bill Miner；
- query 9/12：产生同块共现但关系错误的 FACT，证明 v1 verifier 精度不足。

因此 verified-v1 只支持一个较窄现象：**生成完整候选链再将未验证链用于 retrieval rewrite，可能提高下一跳召回**。它尚未提高最终 Answer hit，不能作为主结果。

### 6.2 当前 verified-v2

当前本地代码已经加入：

1. 跨跳 immutable fact ledger；
2. FACT 的同-block 约束；
3. subject/object 同时出现的 exact evidence quote；
4. 基于整个 ledger 的问题锚点到 FINAL 图连通验证。

服务器上的 v1 运行结束后，本地才完成这些修改。随后完成一跳和两跳 v2：

| 方法 | bridge gold Recall@3 | Answer hit | mean token F1 | median online | parser error |
|---|---:|---:|---:|---:|---:|
| v2，一跳 | 2/6 | 0/6 | 0.1042 | 1.620 s | 0 |
| v2，两跳 | 2/6 | 1/6 | 0.1875 | 2.581 s | 1/6 |

一跳中 6/6 FINAL 被拒，ledger 没有写入任何 FACT。主要原因是 0.6B 没有遵守 exact quote 格式：5 条不输出 quote，query 1 生成了 memory 中不存在的伪 quote。

两跳只恢复 query 3，但不是 fact ledger 生效：第二跳模型仍输出错误 `Anthony Longo`，现有确定性 fallback 从 gold K3 抽取 `Matthew Lawrence` 后才得到正确答案。query 7 的第二次 action 打满 64 tokens，只输出裸 `FINAL` 而无值，触发 parser error。

按预先拒绝标准，v2 被否定：没有 verified FACT、没有 early FINAL，准确率不优于 baseline，延迟约翻倍。当前不继续增加 action token 或调整 quote 格式。

## 7. 产物

- 状态与 verifier：`projects/parallel_block_retrieval/src/verified_step_state.py`
- 批量控制器：`projects/parallel_block_retrieval/src/run_global_bridge_controller_batch.py`
- 单元测试：`projects/parallel_block_retrieval/tests/test_global_bridge_batch.py`
- verified-v1 结果：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_bm25_verified_v1/`
- verified-v2 一跳：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_bm25_verified_v2/`
- verified-v2 两跳：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_bm25_verified_h2_v2/`
- 四模式配对结果：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_ablation_20260712_v2/result.json`
