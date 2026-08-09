# Section 198：400+ 版本复盘与 ICLR 方法重置

更新时间：2026-07-12

## 1. 为什么版本很多，但论文方法推进有限

`v1-v451` 不是 451 个独立方法。大部分版本属于：

- budget、block size、recent/sink 等参数组合；
- 同一 scorer 的开关与 fallback 阈值；
- 针对单个 LongBench 任务的修补；
- 已有 policy 的 task-level source/overlay 组合；
- M10/M20 小样本上的候选筛选。

主要流程问题如下。

### 1.1 把 benchmark 优化当成方法创新

很多版本只提高了同一批 LongBench 样本上的 Pareto 点，没有引入新的可迁移学习机制。版本号增长不代表方法贡献增长。

### 1.2 目标函数不断变化

实验先后优化 Score、KV、attention 子系统速度、online speed、total speed、最差任务和 direct operator。没有从一开始固定一个统一的论文目标和停止门槛。

### 1.3 测试集参与了候选构造

v417-v428 的 source composition 使用 M100 任务结果选择来源。即使后续 router 不输入 task name，teacher policy 仍带有 benchmark-specific action semantics。

### 1.4 Pure KV 与系统 operator 混合

direct/extractive operator 带来很高质量和速度，但它不等价于 KV eviction。两类收益长期混在统一平均分中，推迟了方法边界的澄清。

### 1.5 缺少严格的负结果停止规则

当某个方向 oracle 上限已经不满足目标时，仍继续尝试 router、threshold 和局部配置。v441 首次明确建立 oracle gate，并停止了无效的 M50/router 扩展。

### 1.6 M10/M20 噪声被过度使用

小样本适合筛掉明显失败方案，不适合确认 1%-3% 的改进。部分版本差异来自样本方差，而不是稳定方法收益。

## 2. 400+ 版本仍然产生的有效研究结论

这些实验并非完全无效，已经确认：

1. block size 会改变检索与生成质量，不能只把它当系统参数；
2. 单一 query-local KV selector 无法统一支持 QA、summarization、few-shot 和 structured synthetic tasks；
3. source policy 的 fallback/reference 语义不能通过 naive overlay 组合；
4. v437 系统版本可真实达到 6.99% KV、104.06% Full quality 和 3.20x total speed；
5. v440 True-Pure 在 5.94% KV 下只有 69.87% Full；
6. v441 Pure oracle 恢复质量需要约 30% KV，说明原 selector 上限不足；
7. v444 统一三通道 selector 在约 10% KV 下最好只有 67.32% Full；
8. 静态 request-level risk router 和简单 post-probe router 均不能跨 family 泛化。

这些结论足以否定“继续微调现有 Pure selector/router”的路线。

## 3. v450 静态 Operator-Risk Router

候选动作：v437 system、v440 true-pure、Full。输入不含 task name 和固定 prompt template，只使用 query/context 与 sparse probe 的 gap、entropy、coverage 等特征。

Sample holdout：

| Score / Full | KV | Total speed | Full fallback | Unsafe rate |
|---:|---:|---:|---:|---:|
| 128.04% | 66.51% | 1.24x | 65.15% | 4.55% |

Family holdout 的主要失败：

| Family | Score / Full | KV | Full fallback | Unsafe rate |
|---|---:|---:|---:|---:|
| Single-doc QA | 78.99% | 22.52% | 16.67% | 35.00% |
| Multi-doc QA | 93.34% | 83.18% | 78.33% | 6.67% |
| Synthetic | 93.75% | 95.07% | 95.00% | 2.50% |
| Summarization | 99.68% | 92.00% | 91.67% | 5.00% |

结论：静态 router 只有通过大量 Full fallback 才能控制风险。

## 4. v451 Post-Probe Router

在 v450 基础上加入 sparse 输出的长度、拒答、不确定性、重复、grounding overlap 等特征。

Sample holdout：

| Score / Full | KV | Full fallback | Unsafe rate |
|---:|---:|---:|---:|
| 117.85% | 78.32% | 77.27% | 3.03% |

Family holdout 仍然失败：

- Single-doc QA：76.43% Full；
- Multi-doc QA：72.50% Full；
- Synthetic：93.75% Full，87.50% fallback；
- Few-shot：90.00% fallback。

结论：加入浅层输出特征不能解决 action semantics 与任务能力不匹配的问题，且 probe 执行成本尚未计入速度。

## 5. ICLR 方法重置

不再把主线定义为“一个 universal Pure KV selector”。推荐论文定位：

```text
Risk-Calibrated Operator Routing for Budgeted Long-Context Inference
```

方法只保留少量、定义稳定的通用 operator：

1. Retrieve：query-local evidence + sparse generation；
2. Aggregate：global coverage + generation；
3. Structured：数字、标签、列表与 exact operator；
4. Code：局部依赖与 recent span；
5. Full：风险 fallback。

router 先从 prompt 提取 task contract，再在 operator 内进行风险校准。论文不宣称这是纯 KV eviction；Pure KV 结果作为重要消融和失败边界。

## 6. 新版本规则

后续不再为普通参数实验递增“方法版本”。只有满足以下任一条件才创建新的方法版本：

- 引入新的算法机制；
- 改变方法的形式化目标；
- 通过此前未通过的 task/family holdout gate；
- 在冻结测试集上形成新的 Pareto frontier。

固定研究流程：

1. 先写假设和可能证伪结果；
2. 固定训练、校准、开发和冻结测试划分；
3. M10 只做 failure screening；
4. 先测 oracle 上限，再训练 router；
5. family holdout 不通过，不进入大规模表格；
6. System 与 Pure 指标始终分开报告；
7. 所有速度必须同时报告 online 和 total，并列出额外开销。
