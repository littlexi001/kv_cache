# Section 159: strict oracle action labels

日期: 2026-07-11

## 目的

为了避免继续盲调参数, 新增 `mine_oracle_action_labels_20260711.py`, 从已有候选方法的 per-sample 结果中挖掘监督标签:

- 当前样本是否存在安全稀疏动作;
- 最小安全动作是什么;
- 当前候选中分数最高的动作是什么;
- 如果没有安全稀疏动作, 标记为 `full_kv_required`。

安全动作定义:

```text
score(action) >= 0.95 * score(full KV same sample)
KV(action) <= 30%
```

缺少 full KV baseline 的样本默认跳过, 因此这份标签是严格 same-sample 对齐。

## 当前输入

候选目录来自当前已完成的 v300/v305/v306/v309/v310/v311/v312/v313/v315 等结果。

输出位置:

```text
outputs/riskkv_v19_oracle_action_labels_partial_strict_20260711/
```

主要文件:

```text
oracle_action_labels.csv
oracle_action_summary.csv
```

## 严格汇总

| Task | Samples | Safe sparse rate | Avg min-safe KV | Avg best score |
|---|---:|---:|---:|---:|
| 2wikimqa | 100 | 65.0% | 43.50% | 0.5301 |
| hotpotqa | 100 | 71.0% | 48.61% | 0.5696 |
| multifieldqa_en | 100 | 40.0% | 66.31% | 0.6489 |
| musique | 100 | 77.0% | 41.55% | 0.3054 |
| narrativeqa | 100 | 72.0% | 35.28% | 0.2583 |
| qasper | 100 | 42.0% | 64.91% | 0.5033 |
| qmsum | 100 | 51.0% | 56.40% | 0.1572 |
| repobench-p | 100 | 45.0% | 61.13% | 0.5648 |
| passage_retrieval_en | 100 | 100.0% | 2.06% | 1.0000 |
| passage_count | 100 | 99.0% | 3.41% | 0.3700 |
| trec | 100 | 89.0% | 13.60% | 0.7100 |
| samsum | 100 | 86.0% | 19.38% | 0.2713 |
| triviaqa | 100 | 74.0% | 32.85% | 0.5834 |

## 解释

这份标签说明: 固定预算压缩不可能同时覆盖所有任务和所有样本。部分任务有大量样本可以用 <=30% KV 安全完成, 但 qasper/multifieldqa_en/repobench-p/qmsum 等任务需要更谨慎的 fallback 或更强 evidence selector。

因此论文主线应该强调:

```text
sample-adaptive memory action routing
```

而不是宣称一个固定 block budget 适合所有 LongBench 样本。

## 下一步

1. 等 v316/v317 short-decode smoke 完成, 判断输出长度控制能否独立降低 online time。
2. 将 v316/v317 结果纳入 oracle-label 候选池, 重新生成 strict labels。
3. 用标签训练/蒸馏 risk-aware router:
   - input: score gap/top-k stability/task family/raw length/coverage recall/output risk flags;
   - output: `is_dangerous`, `min_safe_action`。
