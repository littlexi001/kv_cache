# Section 114: 当前迭代状态与判定规则

## 已确认的负结果

### 1. Longer consistency probe 不是主瓶颈

| Method | Score | KV ratio | Online |
|---|---:|---:|---:|
| v52 default probe | 0.373143 | 58.17% | 2.720s |
| v53 default probe | 0.375890 | 62.89% | 2.736s |
| v55 probe16 | 0.372339 | 58.19% | 2.778s |
| v56 probe16 | 0.375086 | 62.92% | 2.829s |

结论：

```text
Verifier 输出长度不是主要瓶颈。
继续加长 probe 只会变慢，不作为主线。
```

### 2. Global Coverage-MMR 不是直接突破点

| Method | Score | KV ratio | Online |
|---|---:|---:|---:|
| v53 consistency+qasper | 0.375890 | 62.89% | 2.736s |
| v65 global Coverage-MMR | 0.375736 | 60.89% | 2.792s |

结论：

```text
Coverage signal 本身没有破坏质量，但全局打开不够精准。
它更适合作为 task/risk-family action certificate，而不是全局 reranking bonus。
```

## 正在验证的主线

### v63/v64: benefit-calibrated conformal gate

目的：

```text
用 utility label 校准什么时候值得运行 counterfactual consistency verifier。
```

判定标准：

```text
如果 v64 >= v53 且 KV/online 不明显变差，
则把 utility-calibrated risk family gate 写成当前主方法。
```

### v66: task-scoped Coverage-MMR

目的：

```text
只在 QA/retrieval 任务打开 coverage novelty bonus，
避免 global Coverage-MMR 对 summarization/code 的无意义扰动。
```

判定标准：

```text
如果 v66 > v65 或接近 v53 且 KV 更低，
说明 coverage bonus 应该 task-scoped。
```

### v67/v68: pre-decode coverage-risk gate

目的：

```text
把 coverage 从 reranking bonus 变成 memory-action safety certificate。
如果 selected pages 覆盖 query anchors/entities/numbers 不足，
decode 前升级到更安全的 sparse action。
```

判定标准：

```text
v67 > v64: coverage-risk gate 本身有效。
v68 > v66: Coverage-MMR 与 coverage-risk gate 可以叠加。
trigger_rate 不应过高，否则只是粗暴加预算。
```

调度状态：

```text
v67 已用单卡提前启动，避免等待两张 GPU 同时空闲。
v68 由 watch_and_launch_v67_v68_after_v66_20260709.sh 在 v66 完成后自动补跑。
```

### v69: calibrated coverage-risk policy

目的：

```text
用 v66 作为 base，v64 作为 reference，
按 task 校准 coverage recall threshold tau_g，
自动生成 per-task calibrated coverage-risk policy。
```

自动流程：

```text
watch_calibrate_and_launch_v69_coverage_20260709.sh
```

判定标准：

```text
v69 >= max(v64, v66) 或在接近质量下明显降低 KV/online，
则 coverage-certified memory action 可以作为论文第二个核心创新点。
```

## 当前论文故事建议

如果 v64/v69 有正信号，论文主线可以写成：

```text
RiskKV-Block is not a retrieval method.
It is a risk-calibrated memory-action controller over materialized KV pages.
```

核心组件：

```text
1. Evidence-flow page action: relevance + local/coarse evidence support.
2. Coverage-certified memory action: selected KV pages must cover query evidence anchors.
3. Conformal memory-action risk gate: selectively runs counterfactual verifier.
4. Minimum-safe action fallback: sparse expansion or full KV only when certified unsafe.
```

这个叙事比“router 选择 block_size/top-k”更像顶会方法：

```text
从 compression heuristic 升级为 uncertainty-aware memory-action decision system。
```
