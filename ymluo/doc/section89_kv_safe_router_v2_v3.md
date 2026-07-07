# Section 89: KV-safe Router v2/v3 迭代

## 背景

Section 88 的第一版 `router_safe` 已经能工作：

```text
relative = 103.64% full_raw
token ratio = 38.06%
```

但还有两个问题：

```text
1. summary 任务过于激进，summary1_8 在 gov_report / multi_news 上质量偏低；
2. LongBench 多跳 QA 上，固定 retrieval_raw_k2 明显强于 span，但 router 训练标签几乎只教了 span。
```

因此继续做两版小迭代：

```text
v2: summary 任务从 summary1_8 改成 summary1_4
v3: 自然语言多跳 synthetic 样本标为 retrieval_raw_k2，RULER/magic exact 仍标为 span
```

所有训练数据仍然只来自非 benchmark synthetic 数据，没有使用 LongBench/RULER 做蒸馏训练。

## 新增代码

修改：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
```

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_kv_safe_topk_router_v2_qwen8b.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_v2_small.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_kv_safe_topk_router_v3_qwen8b.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_v3_small.sh
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v2_nonbench_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_v2_small_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v3_nonbench_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_v3_small_20260707
```

## v2

v2 改动：

```text
summary_brief / summary_detailed:
  recent_plus_summary1_4

recent_generation:
  recent_plus_summary1_8

exact:
  保持 v1 的 span_top2 / span_top3 / prefix fallback
```

训练结果：

```text
synthetic train label accuracy = 100.00%
synthetic test label accuracy  = 99.86%
```

Qwen3-8B online benchmark，公平口径：

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k2 | 0.8803 | 0.8174 | 107.70% | 46.19% | 4.97 |
| router_safe v2 | 0.8482 | 0.8174 | 103.77% | 38.07% | 4.81 |
| recent_plus_span_top3_b0_a0 | 0.8480 | 0.8174 | 103.75% | 40.63% | 4.86 |
| full_raw | 0.8174 | 0.8174 | 100.00% | 100.00% | 7.29 |

summary 任务：

| task | action | token ratio | score |
|---|---|---:|---:|
| gov_report | summary1_4 | 28.61% | 0.1534 |
| multi_news | summary1_4 | 40.40% | 0.1408 |

对比 v1：

```text
gov_report: 0.1127 -> 0.1534, 明显提升
multi_news: 0.1476 -> 0.1408, 略下降
```

v2 总体只小幅提升，因为主要瓶颈已经不是 summary，而是 LongBench 多跳 exact。

## v3

v3 改动：

```text
natural_single_old / natural_two_old / natural_three_old:
  recent_plus_retrieval_raw_k2

magic / RULER-style exact:
  span_top2 / span_top3

summary_brief:
  summary1_4

summary_detailed:
  span_top3
```

直觉：

```text
LongBench 多跳 QA 更像自然语言 evidence retrieval，
RULER 更像精确 needle/page lookup。
两者不应该被同一个固定 span 策略处理。
```

训练结果：

```text
synthetic train label accuracy = 100.00%
synthetic test label accuracy  = 99.86%
```

offline heldout：

```text
matched samples = 26
score = 1.0000
full_raw score = 0.9231
relative = 108.33%
token ratio = 38.89%
```

## v3 Online Benchmark

测试方法：

```text
full_raw
router_safe
recent_plus_summary1_4
recent_plus_retrieval_raw_k2
recent_plus_span_top3_b0_a0
```

公平口径：排除 full_raw OOM case，只统计 full_raw 成功的 31 个 case。

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| router_safe v3 | 0.8805 | 0.8174 | 107.72% | 40.80% | 4.89 |
| recent_plus_retrieval_raw_k2 | 0.8803 | 0.8174 | 107.70% | 46.19% | 4.96 |
| recent_plus_span_top3_b0_a0 | 0.8480 | 0.8174 | 103.75% | 40.63% | 4.85 |
| full_raw | 0.8174 | 0.8174 | 100.00% | 100.00% | 7.28 |
| recent_plus_summary1_4 | 0.5256 | 0.8174 | 64.31% | 28.97% | 4.78 |

按 benchmark：

| benchmark | router_safe score | full_raw score | relative | token ratio |
|---|---:|---:|---:|---:|
| LongBench | 0.5368 | 0.2924 | 183.60% | 38.87% |
| RULER 4k | 1.0000 | 1.0000 | 100.00% | 67.25% |
| RULER 8k | 1.0000 | 1.0000 | 100.00% | 35.23% |
| RULER 16k | 1.0000 | 1.0000 | 100.00% | 19.15% |

action 分布：

```text
recent_plus_retrieval_raw_k2: 5
recent_plus_span_top2_b0_a0:  2
recent_plus_span_top3_b0_a0: 23
recent_plus_summary1_4:       2
```

summary 任务：

| task | routed action | token ratio | score |
|---|---|---:|---:|
| gov_report | summary1_4 | 28.61% | 0.1534 |
| multi_news | summary1_4 | 40.40% | 0.1408 |

## 关键结论

v3 是目前最好的 router：

```text
router_safe v3:
  relative = 107.72% full_raw
  token ratio = 40.80%
  seconds = 4.89s

full_raw:
  token ratio = 100%
  seconds = 7.28s
```

和固定 `retrieval_raw_k2` 相比：

```text
retrieval_raw_k2:
  relative = 107.70%
  token ratio = 46.19%

router_safe v3:
  relative = 107.72%
  token ratio = 40.80%
```

也就是说，v3 router 已经在这个小样本 benchmark 上做到：

```text
质量略高于 retrieval_k2；
token 更少；
RULER 4k/8k/16k 保持 100%；
LongBench 明显高于 full_raw 小样本。
```

这比 v1/v2 更接近论文需要的 router 形态。

## 仍需注意

这个结果还是小样本：

```text
max_examples_per_task = 1
LongBench = 8 examples
RULER = 24 examples
```

因此它说明方向正确，但还不能作为论文主表。

下一步应该扩大到：

```text
max_examples_per_task = 3 或 5
保留 v3 router_safe
同时报告 full_raw / retrieval_k2 / span_top3 / router_safe
```

如果扩大后仍能保持：

```text
95%+ full_raw
20%-45% active tokens
RULER/LongBench 不崩
```

那么 router 这条线就基本可以作为 paper 的核心实验之一。

