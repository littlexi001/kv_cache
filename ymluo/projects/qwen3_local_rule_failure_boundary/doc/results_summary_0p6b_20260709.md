# 子问题 3：Qwen3-0.6B 局部规则长上下文失败边界结果

日期：2026-07-09

## 1. 实验目的

本实验测试：

```text
如果每一步推理只是简单短程规则，例如 if A then B，
小模型为什么放进长 context 后仍然失败？
```

重点不是证明模型不会做单步逻辑，而是区分失败来源：

1. 证据检索 / 筛选失败；
2. 相似干扰导致变量绑定错误；
3. 多条竞争链导致 start code 绑定到错误链；
4. 多步组合中的中间状态维护失败；
5. attention selectivity 是否随失败下降。

## 2. 样本格式

每个样本由三部分组成：

```text
[filler tokens + evidence rules + distractor rules]

Task: Follow symbolic rules exactly.
Use only lines beginning with VERIFIED RULE.
Ignore DECOY RULE and NOTE lines even if they look similar.
Start code: {start_code}
Apply exactly {chain_length} valid rule step(s), one step at a time.
What is the final active code?
Answer with the code only.
Answer:
```

### 2.1 证据信息

证据是目标链上的 `VERIFIED RULE` 行。例如：

```text
VERIFIED RULE T0: IF LA61-822 IS ACTIVE THEN LB71-739 BECOMES ACTIVE.
VERIFIED RULE T1: IF LB71-739 IS ACTIVE THEN LC38-299 BECOMES ACTIVE.
VERIFIED RULE T2: IF LC38-299 IS ACTIVE THEN LD38-299 BECOMES ACTIVE.
```

如果：

```text
Start code: LA61-822
Apply exactly 3 valid rule step(s)
```

则期望答案是：

```text
LD38-299
```

### 2.2 其它 token

其它 token 是自然语言 filler，用来填满指定 context length，不包含有效推理信息。例如：

```text
The archive describes schedules, room numbers, supply orders, weather notes, and maintenance logs.
A committee reviewed forms, labels, map references, and delivery times without changing any rule.
```

### 2.3 干扰条件

`distractor_count` 控制额外干扰行数量：

```text
0 / 16 / 64
```

`distractor_similarity` 有三类：

| 类型 | 形式 | 目的 |
|---|---|---|
| `low` | `NOTE` 行，不是规则 | 测试普通无关信息 |
| `high` | 相似 `VERIFIED RULE` | 测试相似有效规则干扰 |
| `conflict` | `DECOY RULE`，同 antecedent 错 consequent | 测试冲突规则抑制 |

示例：

```text
NOTE L3: The catalog mentions code LX12-345, but this note is not an active rule.
VERIFIED RULE H7: IF LA61-823 IS ACTIVE THEN LB71-749 BECOMES ACTIVE.
DECOY RULE X2: IF LA61-822 IS ACTIVE THEN LB99-111 BECOMES ACTIVE.
```

另外还有 `competitor_count`，表示额外完整竞争链数量：

```text
0 / 4
```

竞争链本身也是 `VERIFIED RULE`，但起点不是问题里的 `Start code`，所以不应使用。

## 3. 控制变量

Qwen3-0.6B phase1 使用 shuffled subset sweep。

| 控制变量 | 取值 |
|---|---|
| context length | 1k / 4k / 8k / 16k / 32k |
| relevant rule 位置 | 10% / 50% / 90% |
| distractor count | 0 / 16 / 64 |
| distractor similarity | low / high / conflict |
| requested rule gap | 0 / 512 / 2048 tokens |
| chain length | 1 / 2 / 4 |
| competitor count | 0 / 4 |
| seed | 0 / 1 |

完整全组合规模为：

```text
5 * 3 * 3 * 3 * 3 * 3 * 2 * 2 = 4860 cases
```

本轮先运行：

```text
420 valid cases
```

因此本轮适合分析主趋势和主要瓶颈，不是严格 full factorial 高阶交互实验。

注意：`rule_gap_tokens=2048` 是 requested gap。短 context 下若放不下，代码会裁剪成实际可放的 `actual_rule_gap_tokens`。

## 4. 指标

### 4.1 Candidate accuracy

每个样本构造一组候选答案 code：

1. 正确答案；
2. conflict / distractor / competitor 产生的错误 code；
3. 随机错误 code。

对每个候选只计算答案 code 部分的 conditional loss：

```text
NLL(candidate_code | context + question + "Answer: ")
```

如果正确 code 的 mean NLL 最低，则：

```text
candidate_correct = 1
```

最终：

```text
candidate accuracy = candidate_correct / total cases
```

### 4.2 Margin

`margin` 衡量正确候选相对最强错误候选的优势：

```text
margin = best_wrong_mean_nll - gold_mean_nll
```

解释：

| margin | 含义 |
|---:|---|
| > 0 | 正确答案优于所有错误候选 |
| < 0 | 至少一个错误候选优于正确答案 |
| 越大 | 正确越稳 |

### 4.3 Attention selectivity

在答案前一步，计算 attention mass 中 gold rules 相对所有 rule-like spans 的选择性：

```text
selectivity = gold_rule_mass / (gold_rule_mass + non_gold_rule_mass)
```

若失败时 selectivity 明显下降，说明模型没有稳定选中目标证据链。

## 5. 总体结果

| 模型 | cases | candidate accuracy | generation accuracy | attention samples |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 420 | 64.29% | 0.71% | 88 |

`generation accuracy` 很低，主要因为 base model 经常不遵守“只输出 code”的格式。本实验主要看 candidate accuracy。

## 6. 主效应结果

### 6.1 Context length

| length | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 1k | 53 | 62.26% | 0.888 | 0.632 |
| 4k | 91 | 59.34% | 0.675 | 0.189 |
| 8k | 94 | 67.02% | 0.752 | 0.320 |
| 16k | 87 | 65.52% | 0.995 | 0.440 |
| 32k | 95 | 66.32% | 0.713 | 0.390 |

结论：长度本身没有形成单调下降曲线。失败更像来自长 context 中的竞争证据和绑定压力，而不是 token 数本身。

### 6.2 Relevant rule 位置

| position | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 10% | 138 | 65.94% | 0.818 | 0.333 |
| 50% | 154 | 59.09% | 0.742 | 0.416 |
| 90% | 128 | 68.75% | 0.830 | 0.344 |

结论：位置效应弱于竞争链和相似干扰。

### 6.3 Distractor count

| distractor count | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 0 | 156 | 75.00% | 1.494 | 0.672 |
| 16 | 152 | 45.39% | 0.198 | 0.161 |
| 64 | 112 | 75.00% | 0.628 | 0.070 |

注意：64 个干扰的 accuracy 回升，说明这不是单纯“干扰越多越差”。64 组中部分样本没有竞争链或干扰类型较低风险；但 attention selectivity 已经很低，说明证据选择变差。

### 6.4 Distractor similarity

| similarity | cases | candidate acc | margin | attention selectivity |
|---|---:|---:|---:|---:|
| low | 138 | 76.09% | 1.380 | 0.448 |
| high | 131 | 54.20% | 0.338 | 0.243 |
| conflict | 151 | 62.25% | 0.655 | 0.401 |

结论：高相似 `VERIFIED RULE` 最危险。普通 NOTE 干扰影响较小。

### 6.5 Requested rule gap

| rule gap | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 0 | 135 | 58.52% | 0.487 | 0.253 |
| 512 | 160 | 65.62% | 0.926 | 0.379 |
| 2048 | 125 | 68.80% | 0.956 | 0.429 |

结论：本轮没有看到“规则距离越远越差”的简单趋势。可能是 gap 与其它变量混合后产生 confounding，需要单变量 ablation。

### 6.6 Chain length

| chain length | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 1 | 139 | 61.87% | 0.706 | 0.398 |
| 2 | 132 | 75.00% | 0.939 | 0.271 |
| 4 | 149 | 57.05% | 0.748 | 0.421 |

结论：chain=4 比 chain=2 明显更差，说明多步组合和中间状态维护有影响。但 chain=1 也不高，说明更大的瓶颈仍然是竞争链和变量绑定。

### 6.7 Competitor count

| competitor count | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|
| 0 | 224 | 79.46% | 1.574 | 0.547 |
| 4 | 196 | 46.94% | -0.098 | 0.126 |

结论：竞争链是最大失败源。

## 7. 关键交互

### 7.1 Competitor × similarity

| competitor | similarity | cases | candidate acc | margin | attention selectivity |
|---:|---|---:|---:|---:|---:|
| 0 | low | 75 | 100.00% | 2.623 | 0.577 |
| 0 | conflict | 77 | 75.32% | 1.386 | 0.673 |
| 0 | high | 72 | 62.50% | 0.683 | 0.341 |
| 4 | low | 63 | 47.62% | -0.100 | 0.133 |
| 4 | conflict | 74 | 48.65% | -0.106 | 0.115 |
| 4 | high | 59 | 44.07% | -0.084 | 0.138 |

结论：

```text
只要有 4 条竞争链，三种 similarity 都掉到约 44%-49%。
竞争链比单纯干扰相似度更致命。
```

### 7.2 Competitor × chain length

| competitor | chain length | cases | candidate acc | margin | attention selectivity |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 66 | 89.39% | 1.735 | 0.629 |
| 0 | 2 | 73 | 82.19% | 1.610 | 0.392 |
| 0 | 4 | 85 | 69.41% | 1.418 | 0.610 |
| 4 | 1 | 73 | 36.99% | -0.225 | 0.078 |
| 4 | 2 | 59 | 66.10% | 0.108 | 0.110 |
| 4 | 4 | 64 | 40.62% | -0.143 | 0.189 |

结论：

```text
无竞争链时，chain length 增加会降低准确率，但仍大多可做。
有竞争链时，chain=1 和 chain=4 都严重失败。
这说明失败不只是多步推理，而是 start-code 到正确链的绑定失败。
```

### 7.3 Correct vs wrong 的 attention 差异

| candidate correctness | cases | attention samples | margin | attention selectivity |
|---|---:|---:|---:|---:|
| correct | 270 | 55 | 1.428 | 0.511 |
| wrong | 150 | 33 | -0.347 | 0.122 |

结论：

```text
错误样本中 gold-rule attention selectivity 明显下降。
这支持“检索/筛选/绑定退化”解释。
```

## 8. 核心结论

### 结论 1：短程规则本身不是主要问题

在无竞争链、无干扰条件下：

```text
candidate accuracy = 100%
margin = 2.752
attention selectivity = 0.948
```

说明 0.6B 可以做简单局部规则。

### 结论 2：长 context 失败主要来自竞争链和变量绑定

最显著下降来自 `competitor_count`：

```text
0 competitor: 79.46%
4 competitors: 46.94%
```

而且 margin 从正值变成负值：

```text
1.574 -> -0.098
```

这说明模型经常更偏好错误链的答案。

### 结论 3：高相似 verified rules 比普通 filler 更危险

```text
low similarity: 76.09%
high similarity: 54.20%
```

模型不是简单被无关 token 干扰，而是被“形式上像有效规则”的信息干扰。

### 结论 4：长度本身不是单调失败边界

1k 到 32k 的 accuracy 没有单调下降：

```text
62.26%, 59.34%, 67.02%, 65.52%, 66.32%
```

这说明 Needle-in-the-haystack 难点不能简单归因于 context length，而应拆解为：

1. 检索相关证据；
2. 抑制相似干扰；
3. 把 start code 绑定到正确规则链；
4. 维护多步中间状态。

### 结论 5：attention selectivity 是失败诊断信号

正确样本：

```text
selectivity = 0.511
```

错误样本：

```text
selectivity = 0.122
```

失败时模型明显没有稳定选择 gold rules。

## 9. 当前实验限制

1. 本轮是 420-case shuffled subset，不是 4860-case full factorial。
2. `rule_gap=2048` 在短 context 中会被裁剪，严格分析距离效应需要看 `actual_rule_gap_tokens`。
3. generation accuracy 被输出格式严重污染，因此当前主指标应使用 candidate accuracy。
4. attention 只采样 88 条，用于诊断趋势，不是完整 attention sweep。

## 10. 下一步建议

为了写论文图，建议补三组严格单变量 ablation：

### A. 竞争链 ablation

固定：

```text
length=8192, position=50%, distractor_count=16,
similarity=high, gap=512, chain=2
```

只扫：

```text
competitor_count = 0 / 1 / 2 / 4 / 8
```

### B. 相似度 ablation

固定：

```text
length=8192, position=50%, distractor_count=16,
gap=512, chain=2, competitor=0
```

只扫：

```text
similarity = low / high / conflict
```

### C. 距离 ablation

固定：

```text
length=32768, position=50%, distractor_count=16,
similarity=high, chain=4, competitor=0
```

只扫：

```text
actual rule gap = 0 / 512 / 2048 / 4096 / 8192
```

这样可以把当前探索性结果变成更干净的论文图。
