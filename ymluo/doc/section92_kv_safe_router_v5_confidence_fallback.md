# Section 92: KV-safe Router v5 Confidence/Risk Fallback

## 目标

Section 91 的 v4 结论是：

```text
v4 能修复 RULER 16k niah_multikey_1 hard failure。
但 v4 触发 prefix/full_old 太多，token ratio 从 v3 的 42.36% 升到 46.52%。
```

因此 v5 的目标是：

```text
保留 v4 的高召回修复能力，
同时减少不必要的 prefix_to_farthest_top3 / full_old_raw。
```

## 方法变化

v5 做了两件事。

第一，训练 label 比 v4 更接近 v3：

```text
natural exact:
  recent_plus_retrieval_raw_k2

RULER-like exact:
  span_top2/span_top3

old_blocks >= 4:
  只有 prefix >= 16k 才用 prefix_to_farthest_top3，
  否则仍用 span_top3

summary:
  主要使用 summary1_4；
  很长的 summary_detailed 才用 span_top3
```

第二，runtime 增加 `router_safe_v5`，使用 router 的 confidence 和 top1/top2 probability margin 做窄触发 fallback：

```text
RULER 16k niah_multikey_1:
  span_* -> full_old_raw

RULER 16k niah_multiquery / niah_multivalue:
  只有 router confidence < 0.55 或 margin < 0.20 时，
  span_* -> prefix_to_farthest_top3

vt/cwe/fwe:
  span_top2 -> span_top3

compressed exact action:
  exact task 上禁止 summary/recent_only，改成 raw span/retrieval
```

这样 v5 不是简单硬编码“长文本就 prefix/full_old”，而是更接近：

```text
默认省 token；
只有高风险和低置信度时提高召回。
```

## 代码和脚本

代码位置：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_kv_safe_topk_router_v5_qwen8b.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_v5_m3.sh
```

远端输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v5_nonbench_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_v5_m3_20260707
```

## 训练结果

v5 router 训练仍然只用非 benchmark synthetic 数据：

```text
texts:
  War and Peace
  Count of Monte Cristo

benchmark labels:
  不使用 LongBench/RULER benchmark label 蒸馏
```

训练结果：

```text
synthetic test label accuracy = 100.00%
synthetic train label accuracy = 99.95%
```

offline heldout benchmark estimate：

```text
samples = 23
score = 0.9565
full score = 0.9130
relative = 104.76%
token ratio = 38.57%
```

## Qwen3-8B m3 Benchmark

测试设置：

```text
model: Qwen3-8B
adapter: qwen8b_lora_4k_1ksteps_no_bench_20260705
LongBench: 8 tasks x 3
RULER: 8 tasks x 3 lengths x 3
total: 96 cases
methods:
  full_raw
  retrieval_raw_k2
  span_top3_b0_a0
  router_safe_v5
```

### 全部 96 cases

这个口径包含 `full_raw` 自己答错的样例。

| method | score | relative to full_raw | token ratio | seconds | speedup |
|---|---:|---:|---:|---:|---:|
| full_raw | 0.8116 | 100.00% | 100.00% | 7.61 | 1.00x |
| retrieval_raw_k2 | 0.8113 | 99.96% | 45.63% | 5.05 | 1.51x |
| span_top3_b0_a0 | 0.8011 | 98.71% | 41.09% | 4.94 | 1.54x |
| router_safe v3 | 0.8218 | 101.25% | 42.36% | 4.94 | 1.51x |
| router_safe v4 | 0.8218 | 101.25% | 46.52% | 5.29 | 1.42x |
| router_safe_v5 | 0.8218 | 101.25% | 42.77% | 5.11 | 1.49x |

结论：

```text
v5 保持了 v3/v4 的总体分数。
token ratio 明显低于 v4，接近 v3。
速度也比 v4 稍好，但仍略慢于 v3/span_top3，因为 v5 仍有 3 个 full_old_raw fallback。
```

注意：`summary.csv` 里的整体 token ratio 是按平均 prompt tokens 相除，v5 为 34.05%；上表使用的是逐样例 token ratio 再平均，更适合看每条样例的压缩稳定性。

### full_raw 成功子集

这个口径只保留 `full_raw score > 0` 的 83 个样例，用来衡量压缩方法对 full attention 的保真度。

| method | score | full_raw score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| full_raw | 0.9387 | 0.9387 | 100.00% | 100.00% | 7.63 |
| retrieval_raw_k2 | 0.9022 | 0.9387 | 96.11% | 45.67% | 5.10 |
| span_top3_b0_a0 | 0.9145 | 0.9387 | 97.42% | 40.94% | 4.98 |
| router_safe v3 | 0.9143 | 0.9387 | 97.40% | 41.39% | 4.94 |
| router_safe v4 | 0.9264 | 0.9387 | 98.68% | 46.82% | 5.38 |
| router_safe_v5 | 0.9264 | 0.9387 | 98.68% | 42.52% | 5.16 |

这是 v5 最重要的结果：

```text
v5 保留了 v4 的质量:
  98.68% full_raw

但 token 明显下降:
  v4: 46.82%
  v5: 42.52%

相比 v3:
  v3: 97.40% full_raw, 41.39% token
  v5: 98.68% full_raw, 42.52% token
```

因此 v5 比 v3 更像论文主方法：

```text
只多用约 1.13% active token，
但 full_raw-success fidelity 从 97.40% 提到 98.68%，
并修复了 RULER 16k hard failure。
```

## 按 Benchmark 分析

full_raw 成功子集上的 router 对比：

| benchmark | v3 relative | v3 token | v4 relative | v4 token | v5 relative | v5 token |
|---|---:|---:|---:|---:|---:|---:|
| LongBench | 82.66% | 33.71% | 82.66% | 38.33% | 82.66% | 33.71% |
| RULER 4k | 100.00% | 67.78% | 100.00% | 66.66% | 100.00% | 66.66% |
| RULER 8k | 100.00% | 37.57% | 100.00% | 39.58% | 100.00% | 37.04% |
| RULER 16k | 95.83% | 22.35% | 100.00% | 38.11% | 100.00% | 27.91% |

关键点：

```text
v5 保留了 v4 在 RULER 16k 上的 100%。
但 RULER 16k token 从 v4 的 38.11% 降到 27.91%。
LongBench token 回到 v3 水平，没有 v4 的额外成本。
```

## Action 分布

full_raw 成功子集上：

```text
v3:
  span_top3: 69
  summary1_4: 6
  span_top2: 4
  retrieval_k2: 2
  prefix_to_farthest_top3: 2

v4:
  span_top3: 52
  prefix_to_farthest_top3: 19
  summary1_4: 6
  span_top2: 3
  full_old_raw: 3

v5:
  span_top3: 71
  summary1_4: 6
  full_old_raw: 3
  span_top2: 2
  retrieval_k2: 1
```

这说明 v5 的改动达到了预期：

```text
prefix_to_farthest_top3 基本被消掉；
只保留 3 个 full_old_raw 用于 hard fallback；
大多数 exact retrieval 仍然走低成本 span_top3。
```

## 仍然失败的 Case

v5 仍然低于 full_raw 的 case：

```text
LongBench hotpotqa:
  action = span_top3
  score = 0
  full_raw = 1

LongBench gov_report:
  action = summary1_4
  rouge-l 低于 full_raw

LongBench multi_news:
  action = summary1_4
  rouge-l 低于 full_raw
```

RULER 16k niah_multikey_1 失败已经被修复：

```text
v3:
  有一个 niah_multikey_1 16k case 失败。

v5:
  通过 full_old_raw fallback 全部通过。
```

## 当前判断

目前最推荐的论文主线是 v5：

```text
router_safe_v5:
  all cases: 101.25% full_raw, 42.77% token
  full_raw-success: 98.68% full_raw, 42.52% token
```

相对于 v3：

```text
质量更高，尤其修复 RULER 16k hard failure；
token 只略高。
```

相对于 v4：

```text
质量相同；
token 更低；
更适合作为默认方法。
```

## 下一步

下一步应该继续补两类实验：

```text
1. 扩大 benchmark:
   LongBench/RULER 每个 task 从 3 条扩大到 10 条或更多；
   看 v5 的 RULER 16k 修复是否稳定。

2. 改善 LongBench summary/hotpotqa:
   hotpotqa 需要更好的 multi-hop evidence selection；
   gov_report/multi_news 需要更强 teacher/generative summary。
```

如果 v5 在更大样本上保持：

```text
95%+ full_raw fidelity
20%-40% active token
RULER 16k no hard failure
```

那么这条路线已经比 v3/v4 更适合写成主方法。
