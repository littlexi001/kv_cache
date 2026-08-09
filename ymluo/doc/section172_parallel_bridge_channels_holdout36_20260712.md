# 10M 并行 Bridge Channels：从 6 条正现象到 36 条否定结果

## 1. 动机

在 6 条冻结样例中，模型生成 bridge 与关系邻近实体抽取各自达到 2/6 gold Recall@3，但命中集合不同：

- 模型通道命中 query 1、3；
- 确定性通道命中 query 1、7。

因此提出一个不增加总 K3 的风险保留方案：

```text
current K3
-> channel M: Qwen SEARCH entity
-> channel D: relation-aware deterministic entity
-> BM25.score([query_M, query_D])  # 同一批稀疏矩阵乘法
-> channel M Top-2 + channel D Top-1
-> total K3
```

6 条离线配额中，M2+D1 的 bridge gold Recall 为 3/6，高于任一单通道的 2/6；D2+M1 仍为 2/6，K6 union 也只有 3/6。因此选择 M2+D1 做真实生成验证。

## 2. 六条生成闭环

| 方法 | bridge gold Recall@3 | Answer hit | mean token F1 | median online |
|---|---:|---:|---:|---:|
| model-only，一跳 | 2/6 | 1/6 | 0.1863 | 约 1.48 s |
| M2+D1，一跳 | 3/6 | 1/6 | 0.2333 | 1.25 s |
| M2+D1，两跳 | 3/6 | 1/6 | 0.1863 | 1.66 s |

不同运行的 GPU 数和服务器负载不同，不能用上表声称 M2+D1 更快。可比较的结论是：一跳 M2+D1 增加了 query 7 的 gold block，但没有增加 Answer hit；两跳只是把正确样例从 query 1 换成 query 3，没有净收益。

## 3. 完整 36 条 holdout

排除开发时使用的 query 0、6，在剩余 36 条 2WikiMQA、HotpotQA、MuSiQue 上冻结比较。两组都使用 3 张 GPU、question-BM25 initial K3、一轮 forced SEARCH 和总 K3。

| 方法 | initial record Recall@3 | bridge gold Recall@3 | Answer hit | mean token F1 | median online |
|---|---:|---:|---:|---:|---:|
| model-only | 91.67% | 30.56%，11/36 | **16.67%，6/36** | **0.1572** | 1.451 s |
| M2+D1 | 91.67% | 33.33%，12/36 | 11.11%，4/36 | 0.1192 | 1.106 s |

延迟差异主要来自同机负载和生成时间变化。新增一个 BM25 query row 后，bridge BM25 时间没有成为主瓶颈，但不能据此声称端到端加速。

## 4. 配对统计

### 4.1 Bridge gold

- wins：query 7、58；
- loss：query 18；
- 净增：1/36，+2.78 个百分点；
- McNemar exact p = 1.0。

该增益很小且不显著。

### 4.2 Answer hit 与 F1

- Answer wins：0；
- Answer losses：query 23、38；
- Answer hit 净变化：-2/36，-5.56 个百分点；
- McNemar exact p = 0.5；
- mean token F1 delta：-0.0380；
- paired bootstrap 95% CI：[-0.1061, 0.0087]；
- F1 wins/losses/ties：1/4/31。

M2+D1 没有带来任何新增正确答案，因此不能因为 retrieval recall 多一条就作为主方法。

### 4.3 Gold 到答案的转化率

| 方法 | bridge-gold queries | Answer hit given bridge gold | Answer hit without bridge gold |
|---|---:|---:|---:|
| model-only | 11 | **54.55%，6/11** | 0/25 |
| M2+D1 | 12 | **25.00%，3/12** | 1/24 |

无条件混合让 annotated gold 多一条，但将 gold 到正确答案的转化率降低一半以上。这说明新加入的 block 与 gold block 共同构成了更差的 K3 set，模型被冲突或无关证据误导。

### 4.4 Teacher-forced Answer NLL

单卡对相同 36 条和参考答案计算 K3 条件下的 token NLL：

| K3 | mean Answer NLL | median |
|---|---:|---:|
| source-oracle K3 | 2.2310 | 2.2120 |
| model-only | 4.5447 | **4.1744** |
| M2+D1 | **4.4403** | 4.4439 |

M2+D1 相对 model-only 的 paired mean delta 为 -0.1044，但：

- lower-NLL wins/losses/ties：13/18/5；
- median delta：+0.0317，反而略差；
- bootstrap 95% CI：[-0.6127, 0.2668]，跨 0。

平均值的改善主要由少数大幅下降样例驱动，不稳定且不显著。更重要的是，自由生成 Answer hit/F1 明确下降。因此不能只用 teacher-forced NLL 支持 M2+D1；不同 K3 会改变 greedy decoding trajectory，即使参考答案平均概率略升，也可能让实际输出更差。

## 5. 为什么 recall 上升但答案下降

固定占用一个 K3 槽位会破坏原模型通道的证据组合：

- query 18：model K3 `[28096,28098,28097]` 命中 gold；混合后第三块换成 28100，gold 丢失，但模型仍碰巧回答 `film director`。
- query 23：两组都命中 gold，但混合删除 model title window 的第三个连续 block，答案从包含 `British` 退化为只回答 `Cat Stevens`。
- query 38：两组都命中 gold，混合 memory 的组成变化使答案从正确 `2006` 变成 `2019`。
- query 7/58：混合新增 gold block，但 final generator 仍未回答正确。

这再次证明：

> answer-block Recall 不是 stepwise reasoning 的充分目标；K3 内 block 的组合和最终状态绑定同样重要。

更准确的新优化目标应是 **set-level evidence utility**：不是判断某个 gold block 是否进入 K3，而是判断整个 K3 集合能否让模型稳定完成当前一步，同时控制冲突证据和 distractor contamination。

## 6. 决策

无条件 M2+D1 被否定，不继续作为主线。虽然可以设想根据 title-window、通道一致性或 margin 设计风险门控，但 36 条中确定性通道没有产生任何 Answer win，当前证据不足以支持继续调门控。

更值得优先研究的是：

1. 未验证 bridge 不应被 final prompt 当作强制 focus；
2. gold K3 到正确答案之间需要更可靠的证据读取/状态更新；
3. 训练或评估目标应加入 bridge sufficiency 和 K3 set utility，而不只优化单 block gold recall。

### 6.1 Final focus prompt 消融

针对“错误 bridge 被 final prompt 强制绑定”的问题，在相同 6 条 M2+D1 K3 上比较：

| Final prompt | Answer hit | mean F1 | mean answer generation | mean online |
|---|---:|---:|---:|---:|
| bound focus | 1/6 | **0.2333** | 0.390 s | 1.204 s |
| evidence only | 1/6 | 0.1333 | **0.203 s** | **0.983 s** |

evidence-only 更短、更快，但正确样例从 query 1 换成 query 3，F1 明显下降。它不能作为默认 final prompt。该实验同时说明减少生成 token 是当前最直接的延迟优化，但必须与质量门槛联合优化。

### 6.2 Lexical-only 单卡路径

此前即使 initial retriever 为 question BM25，批量器仍要求至少 2 张 GPU、加载全部 Q/K shards，并安装 QKCapture hooks。现已改为：

- 只有 `qk` 和 `bm25_record_qk` 模式加载分布式索引并要求至少 2 卡；
- question-BM25 模式允许单卡 torchrun；
- lexical-only 生成不再安装 Q/K profiling hooks。

单卡路径已通过 6 条完整运行。这是确定的工程改进，减少了资源等待和无关索引加载，但不改变检索准确率。

## 7. 产物

- model-only 36 条：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout36_bm25_modelonly_v3/`
- M2+D1 36 条：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout36_bm25_model2det1_v3/`
- 配对统计：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout36_model2det1_ablation_v3/result.json`
- K3 Answer NLL：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout36_k3_nll_v1/`
- 实体通道分析器：`projects/parallel_block_retrieval/src/analyze_bridge_entity_extraction.py`
- 配对统计器：`projects/parallel_block_retrieval/src/analyze_bridge_controller_ablation.py`
- evidence-only 6 条：`projects/parallel_block_retrieval/outputs/global_bridge_controller_holdout6_bm25_model2det1_evidence_v1/`
