# Section 68: Router 蒸馏的 Oracle 上界分析

## 问题

如果 runtime router 能训练到最好，它的理论上界是什么？

这里的“理论上界”不是方法本身的数学极限，而是当前 action space 和当前 benchmark trials 下的 oracle 上界：

```text
给定每个样例上所有候选方法的真实结果，让 oracle router 事后选择最优 action。
```

当前 action space：

```text
full_raw
recent_only
static_hier
summary1_8
summary1_4
summary1_2
retrieval_raw_k1
retrieval_raw_k2
router
```

分析数据：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_adapter_router_20260705/trials.csv
```

## 三种 Oracle 定义

### 1. oracle_best_score

每个样例选择 score 最高的方法；如果多个方法 score 相同，选择 token ratio 最低的方法。

这个表示当前 action space 的质量上界。

### 2. oracle_match_full

每个样例选择不低于 adapted full_raw score 的最便宜方法。

这个表示“无质量损失压缩”的上界。

### 3. oracle_best_under_budget

给定 token budget，例如 20%、35%、50%，每个样例只在 token ratio 不超过 budget 的方法里选择 score 最高的方法。

这个表示固定压缩预算下的可达质量上界。

## Oracle 总体上界

| policy | group | score | full score | relative to full | token ratio |
|---|---|---:|---:|---:|---:|
| oracle_best_score | overall | 0.8784 | 0.7916 | 110.97% | 23.95% |
| oracle_best_score | LongBench | 0.5136 | 0.2913 | 176.32% | 31.79% |
| oracle_best_score | RULER | 1.0000 | 0.9583 | 104.35% | 21.33% |
| oracle_match_full | overall | 0.8391 | 0.7916 | 106.01% | 20.84% |
| oracle_match_full | LongBench | 0.3877 | 0.2913 | 133.09% | 19.58% |
| oracle_match_full | RULER | 0.9896 | 0.9583 | 103.26% | 21.26% |
| oracle_best_under_20pct | overall | 0.5879 | 0.7916 | 74.27% | 8.70% |
| oracle_best_under_35pct | overall | 0.7527 | 0.7916 | 95.09% | 12.43% |
| oracle_best_under_50pct | overall | 0.7996 | 0.7916 | 101.01% | 14.67% |

关键结论：

```text
如果 router 完美，在当前候选方法里可以达到：

overall:
  106.01% adapted full_raw performance
  20.84% active tokens

RULER:
  103.26% adapted full_raw performance
  21.26% active tokens

LongBench:
  133.09% adapted full_raw performance
  19.58% active tokens
```

这说明：当前方法失败的主要瓶颈不是 action space 没有潜力，而是 runtime router 没有学会 oracle policy。

## 固定 Token Budget 上界

| budget policy | overall score | relative to full | token ratio |
|---|---:|---:|---:|
| best under 20% | 0.5879 | 74.27% | 8.70% |
| best under 35% | 0.7527 | 95.09% | 12.43% |
| best under 50% | 0.7996 | 101.01% | 14.67% |

解释：

- 如果目标是 `95%+ full_raw performance`，当前 trials 中 35% 预算已经足够，oracle 只用了平均 `12.43%` tokens。
- 如果目标是超过 full_raw，50% 预算足够，oracle 实际平均只用了 `14.67%` tokens。
- 当前 runtime router 的实际结果是 `55.69% score / 28.34% tokens`，远低于 oracle。

## RULER 按长度上界

| policy | length | score | full score | relative to full | token ratio |
|---|---|---:|---:|---:|---:|
| oracle_match_full | 4096 | 1.0000 | 1.0000 | 100.00% | 24.28% |
| oracle_match_full | 8192 | 1.0000 | 1.0000 | 100.00% | 22.28% |
| oracle_match_full | 16384 | 0.9688 | 0.8750 | 110.71% | 17.23% |
| oracle_best_score | 4096 | 1.0000 | 1.0000 | 100.00% | 24.28% |
| oracle_best_score | 8192 | 1.0000 | 1.0000 | 100.00% | 22.28% |
| oracle_best_score | 16384 | 1.0000 | 0.8750 | 114.29% | 17.43% |

这个结果很重要：在 16k RULER 上，oracle 不但比 full_raw 更好，而且平均只需要约 17% active tokens。

## Oracle Action 分布

### oracle_best_score

| action | count |
|---|---:|
| recent_only | 43 |
| summary1_8 | 27 |
| retrieval_raw_k1 | 24 |
| retrieval_raw_k2 | 17 |
| static_hier | 9 |
| full_raw | 5 |
| summary1_4 | 2 |
| summary1_2 | 1 |

### oracle_match_full

| action | count |
|---|---:|
| recent_only | 48 |
| summary1_8 | 28 |
| retrieval_raw_k1 | 21 |
| retrieval_raw_k2 | 15 |
| static_hier | 9 |
| full_raw | 5 |
| summary1_2 | 1 |
| summary1_4 | 1 |

这说明一个理想 router 不是简单地总选 retrieval 或总选 summary，而是混合策略：

- 简单 / 局部任务选 `recent_only` 或 `summary1_8`。
- 精确回忆任务选 `retrieval_raw_k1/k2`。
- 少数情况仍需要 `full_raw`。

## 对 Router 蒸馏的启发

当前 runtime router 的 action 分布：

| action | count |
|---|---:|
| retrieval_raw_k1 | 89 |
| recent_only | 26 |
| full_raw | 9 |
| retrieval_raw_k2 | 4 |

问题：

- 过度偏向 `retrieval_raw_k1`。
- 很少选择 `retrieval_raw_k2`。
- 几乎没有学会在 summary / recent / retrieval 之间精细切换。

下一版 router 应该：

1. 使用更多非 benchmark synthetic exact/retrieval 任务生成 oracle label。
2. 把 `top-k` 从固定 `k1/k2` 改成动态 `k` 或 threshold selection。
3. 蒸馏目标不要只做 action classification，还要预测：
   - task 是否需要 exact raw evidence；
   - evidence 可能分布在几个 block；
   - summary-only 是否足够；
   - full_raw fallback 的置信度。
4. 训练时加入 cost-aware loss，让 router 学会在满足质量约束时选择最便宜 action。

## 当前判断

如果 router 接近 oracle，上界非常好：

```text
约 20.84% active tokens
不低于 full_raw 的 overall score
RULER 也能保持或超过 full_raw
```

因此目前最大的瓶颈不是方法没有上界，而是 router 蒸馏质量不足。

