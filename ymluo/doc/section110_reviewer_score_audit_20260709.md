# Section 110: Reviewer Score Audit for RiskKV-Block

## 当前一句话贡献

建议论文最终收敛到下面这一句话：

```text
RiskKV-Block routes over complete KV memory actions and estimates action risk through label-free counterfactual consistency across sparse-memory executions.
```

这比“选择 top-k blocks”更强，因为它把贡献从 token selection 提升到 action-risk estimation。

## 可能的 reviewer 打分

在当前 m20 证据下，保守判断：

```text
Novelty:      3.0 / 5
Technical:    3.0 / 5
Empirical:    3.0 / 5
Clarity:      3.0 / 5
Overall:      weak 4 / borderline 5
```

如果 m50 稳定，并补上 baseline：

```text
Novelty:      3.5 / 5
Technical:    3.5 / 5
Empirical:    3.5 / 5
Clarity:      3.5 / 5
Overall:      5-level becomes plausible
```

如果 m100/full split 仍稳定，并且横向对比明显：

```text
Novelty:      4.0 / 5
Technical:    3.5-4.0 / 5
Empirical:    4.0 / 5
Overall:      5 to weak 6 possible
```

## 最强 selling points

1. `which memory action is safe` 是比 `which tokens to keep` 更高层的问题设定。

2. Memory-action consistency verifier 是 label-free：

```text
不看 gold answer
不看 full KV output
只比较同一模型在两个 sparse KV actions 下的输出一致性
```

3. v47 诊断显示 consistency disagreement 有风险识别能力：

```text
triggered subset:
  v35 base = 0.152604
  fallback = 0.224598
  full     = 0.224598

untriggered subset:
  v35/base kept = 0.405814
```

4. v52/v53 m20 actual 已超过 same-sample full KV：

```text
full KV: 0.372655, 100.00% KV
v52:     0.373143, 58.17% KV
v53:     0.375890, 62.89% KV
```

5. 负结果增强可信度：

```text
expanded sparse retry 不稳
code recent/hybrid 不稳
consistency-only 有效但不够，必须与 output/grounding verifier 组合
```

## 最可能被攻击的问题

### 1. Task-family policy 是否过拟合 m20

Reviewer 会问：

```text
这些 task-level fallback 是不是看了 test set 后手工挑出来的？
```

当前应对：

```text
m50 正在跑；m100 脚本已准备。
需要证明 v52/v53 不是 m20 accidental win。
```

### 2. 超过 full KV 是否可信

Reviewer 会问：

```text
为什么压缩后比 full 还高？是不是样本太少？
```

当前应对：

```text
只说 same-sample m20 observation；
解释为 compressed context can denoise distracting context；
必须用 m50/m100 验证，不作为唯一核心 claim。
```

### 3. Consistency verifier 的额外 decode 开销

Reviewer 会问：

```text
你为了省 KV 又多跑一次 decode，值得吗？
```

当前应对：

```text
v55/v56 probe16 已准备；
如果 probe16 保持质量并降低 online cost，这是关键补强。
```

### 4. 和 RAG 边界

Reviewer 会问：

```text
这是不是 query-aware retrieval/RAG？
```

当前应对：

```text
不是外部 retrieval；
输入是 already materialized KV cache；
动作是 active KV page selection / fallback；
risk signal 是同一模型在不同 KV action 下的 counterfactual consistency。
```

### 5. Baseline 不够

Reviewer 会问：

```text
为什么没有完整 AdaKV/SnapKV/PyramidKV/H2O 横向？
```

当前应对：

```text
内部 H2O/SnapKV eager attention 在 7.5k OOM；
StreamingLLM m50 正在跑；
最终仍需要官方或复现 baseline 表，尤其 question-aware LongBench。
```

## 下一步优先级

1. 等 m50 结果。

```text
如果 v52/v53 稳定，则更新论文主表，并启动 m100。
如果 v52/v53 不稳定，则回退到 v37/v52 的 safer story。
```

2. 跑 v55/v56 probe16。

```text
目标：证明 consistency verifier 可以低开销实现。
```

3. 做 consistency calibration 曲线。

```text
阈值 gamma = 0.5 / 0.67 / 0.8
观察 trigger rate, score, KV ratio, online。
```

4. 做 official-style LongBench Table 5 对比。

```text
至少报告：
Full KV
StreamingLLM
SnapKV / PyramidKV / AdaKV official numbers
RiskKV-Block compact / consistency-quality
```

5. 强化论文图：

```text
Figure 1: action-risk controller diagram
Figure 2: memory-action consistency verifier
Figure 3: score-KV Pareto
Figure 4: consistency triggered vs untriggered risk
```

## 当前建议

现在不要再把论文主线叫 router。

建议叫：

```text
RiskKV-Block: Counterfactual Risk Routing over KV Memory Actions
```

或者：

```text
RiskKV-Block: Label-Free Risk Verification for KV Cache Compression
```

如果 m50/m100 稳定，第二个标题更像顶会论文。
