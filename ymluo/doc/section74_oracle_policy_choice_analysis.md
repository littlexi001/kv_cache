# Section 74: Oracle Policy 选择机制分析

## 背景

本节分析已有 Qwen3-8B held-out trials 中 oracle policy 是如何选择 action 的。

使用的 benchmark 输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_adapter_router_20260705
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_extra_k348_20260705
```

候选 action：

```text
full_raw
recent_only
static_hier
summary1_8
summary1_4
summary1_2
retrieval_raw_k1
retrieval_raw_k2
retrieval_raw_k3
retrieval_raw_k4
retrieval_raw_k8
```

## 当前 Oracle 的定义

`oracle_match_full` 的逻辑是：

```text
对每个样例：
  full_score = full_raw score

  如果是 summary task:
    threshold = full_score - 0.03
  否则:
    threshold = full_score

  successful = 所有 score >= threshold 的 action
  选择 successful 中 token ratio 最低的 action
```

也就是说，这个 oracle 不是一定选择“答对”的 action，而是选择：

```text
达到 full_raw 表现的最便宜 action
```

## 当前 Oracle 结果

`oracle_match_full`：

```text
score = 0.8381
full_raw score = 0.7916
relative = 105.88%
token ratio = 17.85%
```

action 分布：

| action | count |
|---|---:|
| recent_only | 48 |
| summary1_8 | 29 |
| retrieval_raw_k1 | 21 |
| retrieval_raw_k2 | 14 |
| static_hier | 9 |
| summary1_4 | 2 |
| retrieval_raw_k3 | 2 |
| full_raw | 1 |
| retrieval_raw_k4 | 1 |
| retrieval_raw_k8 | 1 |

最重要的观察：

```text
oracle 几乎不选 full_raw。
128 个样例里只有 1 个样例选 full_raw。
```

这说明 action space 本身很强，绝大多数样例存在一个远小于 full context 的 action 可以达到或超过 full_raw。

## 按 Benchmark 看

| group | relative | token ratio | 主要 action |
|---|---:|---:|---|
| LongBench | 131.72% | 13.19% | recent_only, summary1_8, retrieval_raw_k1 |
| RULER 4k | 100.00% | 24.28% | summary1_8, recent_only, static_hier, retrieval_raw_k1/k2 |
| RULER 8k | 100.00% | 20.94% | recent_only, retrieval_raw_k1/k2, summary1_8 |
| RULER 16k | 110.71% | 13.01% | recent_only, retrieval_raw_k1/k2, summary1_8 |

特殊点：

```text
长上下文下 oracle 更激进。
RULER 16k 的 token ratio 只有 13.01%，反而比 RULER 4k 更低。
```

这和之前的直觉一致：

```text
短文本保守；
长文本可以更激进压缩。
```

## 一个很关键的特殊点：full_raw 失败时的 Oracle

当前 `oracle_match_full` 有一个需要注意的地方：

```text
exact task 如果 full_raw score = 0，
那么 threshold = 0。
此时所有 action 只要 score >= 0 都算 successful。
oracle 会直接选择 token 最低的 action。
```

这意味着：

```text
如果 full_raw 本身答错，
oracle_match_full 可能选择一个同样答错、但 token 很低的 action。
```

统计：

```text
full_raw score = 0 的样例: 20 / 128
```

这些样例的任务分布：

| task | count |
|---|---:|
| musique | 4 |
| passage_count | 4 |
| qasper | 4 |
| cwe | 4 |
| 2wikimqa | 3 |
| hotpotqa | 1 |

在这些 `full_raw = 0` 的样例上，当前 oracle 的 action：

| action | count |
|---|---:|
| recent_only | 16 |
| summary1_8 | 3 |
| static_hier | 1 |

这解释了为什么 `recent_only` 在 oracle 里很多。

## 更严格 Oracle

我额外重算了一个更严格版本：

```text
summary task:
  仍然使用 full_score - 0.03

exact task:
  如果任意 action 可以 score = 1，则 threshold = 1
  否则 threshold = max action score

然后选择满足 threshold 的最便宜 action。
```

结果：

```text
strict_exact_or_best:
  score = 0.8772
  full_raw score = 0.7916
  relative = 110.81%
  token ratio = 18.47%
```

对比当前 oracle：

| oracle | relative | token ratio |
|---|---:|---:|
| match_full_current | 105.88% | 17.85% |
| strict_exact_or_best | 110.81% | 18.47% |

结论：

```text
严格 oracle 的质量更高，token 只从 17.85% 增加到 18.47%。
```

所以 oracle 的强并不是完全来自“full_raw 失败时偷懒选便宜错答案”。即使严格要求 exact task 尽量答对，oracle 仍然很强。

## 更严格 Oracle 的 Action 分布

| action | count |
|---|---:|
| recent_only | 43 |
| summary1_8 | 30 |
| retrieval_raw_k1 | 23 |
| retrieval_raw_k2 | 14 |
| static_hier | 10 |
| summary1_4 | 3 |
| retrieval_raw_k3 | 2 |
| full_raw | 1 |
| retrieval_raw_k4 | 1 |
| retrieval_raw_k8 | 1 |

变化不大：

```text
recent_only 从 48 降到 43；
retrieval_raw_k1 从 21 增到 23；
summary1_8 从 29 增到 30。
```

这说明 oracle 的核心模式仍然是：

```text
大量样例只需要 recent / very small summary / k1-k2 retrieval。
极少数样例需要 full_raw 或 k8。
```

## Method 级别的特殊现象

每个方法相对 full_raw 的整体统计：

| method | relative | token ratio | score >= full count | score > full count |
|---|---:|---:|---:|---:|
| recent_only | 42.6% | 8.3% | 60 / 128 | 8 |
| static_hier | 52.5% | 13.7% | 70 / 128 | 8 |
| summary1_8 | 41.7% | 12.1% | 63 / 128 | 6 |
| summary1_4 | 41.6% | 22.9% | 61 / 128 | 2 |
| summary1_2 | 45.7% | 44.5% | 68 / 128 | 7 |
| retrieval_raw_k1 | 78.3% | 37.1% | 94 / 128 | 12 |
| retrieval_raw_k2 | 99.0% | 49.5% | 117 / 128 | 10 |
| retrieval_raw_k3 | 102.9% | 57.9% | 118 / 128 | 13 |
| retrieval_raw_k4 | 101.0% | 63.6% | 119 / 128 | 10 |
| retrieval_raw_k8 | 98.1% | 76.2% | 119 / 128 | 9 |

特殊点：

1. `retrieval_raw_k3/k4/k8` 覆盖率高，但 token cost 也高。
2. `retrieval_raw_k3` 平均 relative 最高，但 oracle 很少选它，因为很多样例有更便宜的方法已足够。
3. `recent_only` 平均质量低，但在约一半样例上达到 full_raw，因此 oracle 很喜欢它。
4. `summary1_8` 平均质量低，但在部分 RULER 和 summary task 上非常便宜且够用。

## Oracle 的真实规律

从 action 分布和任务分布看，oracle 不是简单的：

```text
长文本 -> 大 k
短文本 -> 小 k
```

更像是：

```text
1. 如果 recent_only 能达到 full_raw，就直接选 recent_only。
2. 如果 query 可以被粗 summary 支持，就选 summary1_8。
3. 如果需要 raw evidence，则优先选 retrieval_raw_k1。
4. 只有多证据或 k1 不够时，才选 k2/k3/k4/k8。
5. full_raw 几乎只是最后 fallback。
```

所以 router 要学的不是 “k 越大越稳”，而是：

```text
这个 query 是否真的需要 raw evidence？
如果需要 raw evidence，需要几个 evidence blocks？
如果不需要 raw evidence，recent 或 summary 是否足够？
```

## 对 Router 训练的启发

当前 pairwise ranker 失败的一个原因是 synthetic 数据把 retrieval 建模得太简单：

```text
证据被包含 -> 成功
包含更多 block -> 更稳
```

但真实 oracle 里不是这样：

```text
k8 并不常被选；
k3/k4/k8 虽然经常能达到 full_raw，但 oracle 更偏向便宜的 recent/summary/k1；
更多 raw block 可能带来干扰或更高成本，不一定值得。
```

下一版 router 更应该分成：

```text
head 1: need_raw_evidence
head 2: if raw, predict evidence block count / k
head 3: if not raw, choose recent vs summary
head 4: fallback risk
```

而不是直接在所有 action 上做一个 flat classifier。

## 简短结论

oracle 的特殊点是：

```text
1. 它主要靠 recent_only / summary1_8 / retrieval_raw_k1 省 token。
2. 它极少使用 full_raw。
3. 当前 match_full oracle 在 full_raw=0 时会选择很便宜的失败 action，需要注意解释口径。
4. 但即使用更严格 oracle，仍能做到 110.81% full_raw / 18.47% token。
5. 真正要学的是 need_raw + evidence_count + fallback，而不是直接学 action label。
```
