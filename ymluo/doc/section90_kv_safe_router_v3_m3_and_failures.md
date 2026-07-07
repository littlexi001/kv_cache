# Section 90: Router v3 扩大到 3 样例与失败分析

## 目标

Section 89 的 v3 router 在 `max_examples_per_task=1` 的小样本上表现很好：

```text
router_safe v3:
  relative = 107.72% full_raw
  token ratio = 40.80%
```

但单样本可能偶然性很强。因此这一步扩大到：

```text
max_examples_per_task = 3
LongBench: 8 tasks x 3 = 24 cases
RULER: 8 tasks x 3 lengths x 3 = 72 cases
total = 96 cases
methods = full_raw, router_safe, retrieval_k2, span_top3
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_v3_m3_20260707
```

## 整体结果

原始整体：

| method | samples | score | token ratio | seconds | speedup |
|---|---:|---:|---:|---:|---:|
| full_raw | 96 | 0.8116 | 100.00% | 7.47 | 1.00x |
| retrieval_raw_k2 | 96 | 0.8113 | 33.09% | 4.93 | 1.52x |
| span_top3_b0_a0 | 96 | 0.8011 | 29.48% | 4.82 | 1.55x |
| router_safe v3 | 96 | 0.8218 | 32.99% | 4.94 | 1.51x |

公平口径下：

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| router_safe v3 | 0.8218 | 0.8116 | 101.25% | 42.36% | 4.94 |
| full_raw | 0.8116 | 0.8116 | 100.00% | 100.00% | 7.47 |
| retrieval_raw_k2 | 0.8113 | 0.8116 | 99.96% | 45.63% | 4.93 |
| span_top3_b0_a0 | 0.8011 | 0.8116 | 98.71% | 41.09% | 4.82 |

解释：

```text
扩大到 96 cases 后，router_safe v3 仍然略高于 full_raw，
并且 token ratio 约 42.36%。

相比 max_examples=1 的 107.72%，优势收窄很多，
说明单样本结果偏乐观，但 router 方向仍然成立。
```

## 按 Benchmark

| benchmark | full_raw | retrieval_k2 | span_top3 | router_safe |
|---|---:|---:|---:|---:|
| LongBench | 0.2465 | 0.2868 | 0.2461 | 0.3287 |
| RULER 4k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULER 8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULER 16k | 1.0000 | 0.9583 | 0.9583 | 0.9583 |

token ratio：

| benchmark | retrieval_k2 | span_top3 | router_safe |
|---|---:|---:|---:|
| LongBench | 48.6% | 40.7% | 41.7% |
| RULER 4k | 73.2% | 67.8% | 67.8% |
| RULER 8k | 40.6% | 37.6% | 37.6% |
| RULER 16k | 20.1% | 18.3% | 22.3% |

## Router Action 分布

```text
recent_plus_prefix_to_farthest_top3: 3
recent_plus_retrieval_raw_k2:       13
recent_plus_span_top2_b0_a0:         5
recent_plus_span_top3_b0_a0:        69
recent_plus_summary1_4:              6
```

这说明 v3 router 确实不再只是固定 span：

```text
LongBench/natural QA: 更多 retrieval_k2
RULER/exact lookup: 主要 span_top3
summary: summary1_4
少量高风险: prefix_to_farthest_top3
```

## 失败 Case

router_safe 低于 full_raw 的主要 case：

```text
LongBench gov_report:
  full_raw > summary1_4

LongBench multi_news:
  full_raw 或 span_top3 更好，summary1_4 不总是最优

LongBench hotpotqa case 2:
  full_raw = 1
  retrieval_k2 = 0
  span_top3 = 0
  router_safe = 0

RULER 16k niah_multikey_1 case 1:
  full_raw = 1
  retrieval_k2 = 0
  span_top3 = 0
  router_safe = 0
```

因此又做了 targeted failure run。

## Targeted Failure Run

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_targeted_failures_20260707
```

测试：

```text
hotpotqa 前 2 个样例
RULER 16k niah_multikey_1 前 2 个样例

methods:
  full_raw
  router_safe
  retrieval_k2
  retrieval_k3
  prefix_to_farthest_top3
  full_old_raw
```

结果：

| group | method | score | token ratio | speedup |
|---|---|---:|---:|---:|
| overall | full_raw | 1.0000 | 100.00% | 1.00x |
| overall | full_old_raw | 1.0000 | 100.05% | 1.02x |
| overall | prefix_to_farthest_top3 | 0.7500 | 57.82% | 1.43x |
| overall | retrieval_k2 | 0.5000 | 22.79% | 2.00x |
| overall | retrieval_k3 | 0.5000 | 29.55% | 1.86x |
| overall | router_safe | 0.5000 | 24.37% | 1.96x |

按任务：

| task | method | score | token ratio |
|---|---|---:|---:|
| hotpotqa | prefix_to_farthest_top3 | 1.0000 | 64.47% |
| hotpotqa | retrieval_k2/k3/router_safe | 0.5000 | 26%-33% |
| niah_multikey_1 16k | prefix_to_farthest_top3 | 0.5000 | 51.79% |
| niah_multikey_1 16k | retrieval_k2/k3/router_safe | 0.5000 | 20%-26% |
| niah_multikey_1 16k | full_old_raw | 1.0000 | 100.05% |

## 解释

这两个失败点不是简单的 `k=2` 不够：

```text
hotpotqa case 2:
  prefix_to_farthest_top3 可以修复，说明答案证据可能在较远 prefix 内，
  sparse top-k evidence 没选中正确组合。

RULER 16k niah_multikey_1 case 1:
  prefix_to_farthest_top3 仍然失败，只有 full_old_raw/full_raw 成功。
  这说明证据选择本身失败，或者答案需要更完整上下文。
```

因此下一步的 router 不应该只在：

```text
retrieval_k2 vs span_top3
```

之间选择，还需要一个高风险 fallback：

```text
prefix_to_farthest_top3
full_old_raw
```

但 full_old_raw 的代价接近 full_raw，所以只能用于少数高风险 case。

## 当前判断

v3 router 的扩大测试是正面的，但没有小样本那么夸张：

```text
96-case result:
  quality = 101.25% full_raw
  token ratio = 42.36%
  speedup = about 1.51x end-to-end generation
```

这比固定策略更像论文需要的结论：

```text
固定 retrieval_k2:
  99.96% full_raw, 45.63% tokens

固定 span_top3:
  98.71% full_raw, 41.09% tokens

router_safe v3:
  101.25% full_raw, 42.36% tokens
```

router 在质量和 token 之间确实做了更好的折中。

## 下一步

下一步应该做 v4 risk-aware router：

```text
1. 保留 v3 的 routing:
   natural QA -> retrieval_k2
   RULER lookup -> span_top3
   summary -> summary1_4

2. 加 high-risk fallback:
   如果 query/case 特征显示多跳、远距离、证据不确定：
     prefix_to_farthest_top3
   如果 prefix 仍不可靠或任务需要全文：
     full_old_raw

3. 训练 synthetic high-risk cases:
   evidence 分散在远距离多个 block
   top lexical block 是干扰项
   需要第 2/3 个 evidence 才能回答
   全文 aggregation / count / common-word 类任务
```

目标不是让 full_old_raw 频繁出现，而是让它作为极少数 case 的安全阀。

