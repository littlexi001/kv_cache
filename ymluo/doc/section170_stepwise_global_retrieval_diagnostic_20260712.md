# 10M 逐步检索的冻结策略诊断与层次化假设

## 1. 本轮目的

此前有两组尚未连接起来的结果：

1. 30 条 source 内动态 KV 实验表明，生成状态变化后能够召回静态问题 Q 没有持续保留的下一跳 block；
2. 10M 全局 Bridge-Search 只在 Lou Breslow 和 Agnes 两条开发样例上成功，不能证明泛化。

本轮不继续为个别样例增加规则，而是冻结现有 bridge 控制器，在未参与调试的问题上将误差拆成：

```text
原问题全局路由
-> K3 中间证据
-> bridge 实体/关系生成
-> 全局下一跳检索
-> K3 证据读取与答案生成
```

开发时使用过的 query 0 和 query 6 被显式排除。完整 holdout 包括其余 36 条 2WikiMQA、HotpotQA 和 MuSiQue 问题。

## 2. 新增批量评测基础设施

新增 `run_global_bridge_controller_batch.py`，一次启动后常驻：

- 10M pre-RoPE K 的 8 个 GPU shards；
- SVD32 basis 和 raw128 K；
- Qwen3-0.6B；
- 39,062-block BM25 稀疏索引。

每条结果分别记录：

- question-only BM25 的 source record/gold block 排名；
- 全局 SVD32 Top-512 和 raw128 K3；
- source record 是否进入粗排候选；
- bridge 输出、归一化实体和检索策略；
- 每跳 source/gold Recall@3；
- answer hit；
- Q capture、Q/K retrieval、BM25、bridge generation 和 answer generation 耗时。

Gold block ID 和参考答案只在检索及生成结束后用于统计，选择逻辑不读取 gold。

## 3. 6 条冻结 holdout 的首个观察

在 8 张 RTX 3090 上先运行 6 条 smoke test：query 1、3、7、9、12、13。

| 初始检索 | source record Recall@3 | 初始 gold Recall@3 | bridge 后 gold Recall@3 | Answer hit |
|---|---:|---:|---:|---:|
| 全局 SVD32/raw128 Q/K K3 | 1/6，16.7% | 0/6 | 0/6 | 0/6 |
| question BM25 block K3 | 6/6，100% | 2/6 | 2/6 | 1/6 |

这里只替换了初始 K3，bridge prompt、实体归一化、第二跳 BM25 和 final prompt 完全相同。因此可以得到两个受限结论：

1. 当前 4-channel Q/K 分数的主要问题首先出现在跨 record 全局路由，而不是 bridge 模型先生成错误；
2. 修复初始路由后只恢复一部分结果，bridge 实体/关系抽取和 K3 证据读取仍是独立瓶颈。

### 3.1 错误因果链实例

全局 Q/K 在 query 1 中选中无关 blocks `[2764, 31732, 33222]`，模型随后从错误 memory 抽取 `Lyon Cohen`，因此第二跳继续检索错误人物。模型是在错误证据上执行了合理但无用的动作。

使用 BM25 初始 K3 后，模型从正确 record 抽取 `YIVO`，下一跳选中 gold blocks，并正确回答 `YIVO`。

query 3 中，BM25 初始 K3 后 bridge 抽取 `Joey Lawrence`，下一跳也选中了 gold block，但模型最终回答了错误的兄弟。这证明 retrieval recall 与 reasoning correctness 必须分开报告，不能把 gold block 命中等同于推理成功。

## 4. 与已有 64-query 排序结果交叉验证

已有 `deep_ql_record39_svd32` 先做 record 路由，再在候选 record 内使用 pre-RoPE SVD32 排序。重新按更小预算统计其排序前缀：

| 预算 | answer block Recall | source record 覆盖 |
|---:|---:|---:|
| K1 | 21.88% | 78.12% |
| K3 | 35.94% | 78.12% |
| K5 | 40.62% | 78.12% |
| K10 | 59.38% | 78.12% |

这与 6 条在线诊断共同支持以下现象：

> pre-RoPE Q/K 在已路由到较小候选 record 后具有 block 细排信号，但不适合单独承担 982 records/39,062 blocks 的全局路由。

K3 不能一次覆盖多数最终答案 block，但逐步方法并不要求第一步读取最终答案，只要求读到足以确定 bridge 的少量证据。因此下一项实验必须直接评估“初始 K3 是否足以生成正确 bridge”，而不能只看最终 gold block Recall@3。

## 5. 基于观察提出的最小层次化消融

新增第三种初始检索模式：

```text
question BM25 Top-1 block
-> 定位该 block 所属 record
-> 只在该 record 内运行 SVD32 coarse retrieval
-> raw128 exact rerank
-> 返回 K3
-> 冻结的 bridge controller
```

该设计让两种信号分工：

- 字符串/稀疏信号负责海量语料中的 record routing；
- 真实 Q/K 负责候选 record 内与当前模型状态相关的 block ranking。

预先设定的判据是：只有当 `bm25_record_qk` 相比直接 BM25 K3 提高 bridge 正确率或最终答案率，才能证明 Q/K 细排在逐步推理中提供了额外价值；如果没有提升，应拒绝该组合，而不是继续调权重。

### 5.1 两卡 6-query 实测否定该假设

两张 GPU 短暂空闲后完成了相同 6 条的 `bm25_record_qk`：

| 初始检索 | 初始 record Recall@3 | 初始 gold Recall@3 | bridge gold Recall@3 | Answer hit | mean token F1 |
|---|---:|---:|---:|---:|---:|
| question BM25 K3 | 6/6 | 2/6 | 2/6 | 1/6 | 0.1863 |
| BM25 record + record-local Q/K K3 | 6/6 | 2/6 | 1/6 | 0/6 | 0.0196 |

配对结果中，record-local Q/K 对 bridge gold 和 Answer hit 都没有新增样例，反而分别丢失 query 3 和 query 1。按预先判据，该直接层次化组合被否定。

原因不是 record 路由失败，而是“对最终答案 block 有统计排序能力”不等于“能保留生成下一跳实体所需的桥接 block”：

- query 3 的 BM25 K3 让模型抽取 `Joey Lawrence`，下一跳命中 gold；Q/K K3 改成其他同-record blocks 后抽取 `Joseph Paul`，下一跳失败。
- query 12 的 BM25 K3 至少定位 `Emergency Wedding` passage；Q/K K3 加载了更早的 `Sophia Magdalena of Denmark`，模型错误抽取 `Adib Kheir`。

因此后续不能再把 answer-block Recall 当作 stepwise retrieval 的唯一训练目标。需要单独定义 bridge sufficiency：给定当前问题状态，K3 是否足以恢复正确的下一跳实体/关系。

### 5.2 连续窗口也不是通用修复

block 解码发现关键 passage 经常跨 256-token 边界，因此又在已保存的 64×39,062 BM25 分数上比较离散 Top3 和三种连续 K3 聚合：

| 方法 | source record Recall@3 | gold Recall@3 |
|---|---:|---:|
| 离散 BM25 Top3 | **75.00%** | **29.69%** |
| 连续窗口 score sum | 62.50% | 18.75% |
| 连续窗口 max + mean | 64.06% | 18.75% |
| 连续窗口 Top-2 score sum | 64.06% | 21.88% |

连续窗口会完整读取一个局部 passage，却失去离散 Top-K 对多个候选 passage 的覆盖，因此总体更差。合理方向不是无条件邻接扩展，而是 passage-aware boundary metadata 或在风险触发时只为高置信实体 passage 分配邻接预算。

### 5.3 固定 hop 会覆盖已经足够的证据

query 1 的 record-local Q/K K3 已包含 YIVO 和 Center for Jewish History 的成员关系，结合问题本身已经足以回答。但当前 prompt 强制输出一次 SEARCH，模型错误地把 focus 改回 `Arnold Richards`，最终丢失原本可以回答的样例。

这说明下一代控制器不能固定执行 N 次检索，而应允许：

```text
VERIFIED FACT -> FINAL
insufficient / unresolved relation -> SEARCH
```

FINAL 不能只由模型自报，必须验证候选答案和关系链确实被当前 K3 支持；否则 0.6B 模型可能过早停止。

## 6. 稳态速度观察

6 条 Q/K-init 批量运行中：

- 第一条 Q capture 为 481 ms，后续稳定为约 32～33 ms；
- 第一条全局 Q/K retrieval 为 635 ms，后续稳定为约 28.5～29.3 ms；
- 动态 bridge BM25 平均约 209 ms；
- 6 条批量 question BM25 总计 236 ms，即摊销约 39 ms/条；
- bridge generation 平均 433 ms；
- answer generation 平均 817 ms。

这证明常驻批处理消除了此前单条脚本中约 0.6～0.8 秒的 Q/K 冷启动假象。当前在线时间主要由两次 0.6B 生成和未优化的 CPU BM25 dense score materialization 主导，而不是 8 卡 SVD32 扫描。

BM25 建库仍需要约 21～29 秒，是离线一次性成本，生产系统必须持久化索引。

两卡 `bm25_record_qk` 的 Q/K retrieval 中位数为 79.2 ms，包含 record mask、分布式 merge 和 raw128 精排；首条冷启动使 p95 达到 533.9 ms。在线阶段中位数为 1.687 秒，主要仍由 bridge/final 两次生成和 CPU BM25 构成。由于 GPU 数不同，该速度不能与前述 8 卡结果直接作为 scaling 对比。

## 7. 当前状态和下一步判据

服务器 GPU 随后被其他 8 个实验占满。等待五分钟仍未出现两张空闲卡，因此没有抢占运行完整 36 条。代码和启动器已经同步，GPU 空闲后按以下顺序运行：

1. 36 条全局 Q/K-init，统计 source record 是否进入 SVD Top-512，以区分粗排丢失和 raw128 精排反转；
2. 36 条 question-BM25-init，作为正确 source 路由更强的控制变量；
3. 36 条 `bm25_record_qk`，验证层次化互补假设；
4. 冻结三种模式后，按数据集比较 bridge gold Recall、Answer hit 和稳态延迟。

6 条诊断已经同时观察到 bridge 抽错、固定 hop 覆盖足够证据、以及已取到 gold 仍读错。因此完整消融完成后，下一项有依据的 solution 是结构化事实状态和 verifier：`FACT(subject, relation, object, evidence) -> FINAL or NEXT(entity, relation)`。它必须与当前冻结控制器配对比较，而不是只展示成功案例。

## 8. 产物

- 批量评测器：`projects/parallel_block_retrieval/src/run_global_bridge_controller_batch.py`
- 单条控制器：`projects/parallel_block_retrieval/src/run_global_bridge_controller_single.py`
- 分布式受限范围检索：`projects/parallel_block_retrieval/src/run_global_dynamic_svd_kv_single.py`
- 服务器启动器：`projects/parallel_block_retrieval/scripts/run_global_bridge_controller_batch_server.sh`
- Q/K-init 6 条结果：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_h1_v1/`
- BM25-init 6 条结果：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_bm25init_h1_v1/`
- record-local Q/K 6 条结果：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_recordqk_h1_v1/`
- 三模式配对分析：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_ablation_20260712_v1/result.json`
- 连续窗口分析：`projects/parallel_block_retrieval/outputs/bm25_block_window_k3_20260712_v1/result.json`
- 配对分析器：`projects/parallel_block_retrieval/src/analyze_bridge_controller_ablation.py`
- 连续窗口分析器：`projects/parallel_block_retrieval/src/analyze_bm25_block_windows.py`
