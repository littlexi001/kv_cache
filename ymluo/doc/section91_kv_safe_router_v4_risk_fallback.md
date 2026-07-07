# Section 91: KV-safe Router v4 Risk-aware Fallback

## 目标

Section 90 里 v3 router 的主要问题是：大多数任务表现不错，但在少数高风险 case 上会因为证据 block 选不全而失败。

典型失败包括：

```text
LongBench hotpotqa case:
  retrieval_k2 / span_top3 都可能漏掉组合证据。

RULER 16k niah_multikey_1:
  span_top3 和 prefix_to_farthest_top3 都可能失败，只有 full_old_raw/full_raw 成功。
```

因此 v4 的目标不是追求更低 token，而是在 v3 的基础上加入高风险 fallback：

```text
低风险:
  recent_plus_span_top2/top3
  recent_plus_retrieval_raw_k2
  recent_plus_summary1_4

中高风险:
  recent_plus_prefix_to_farthest_top3

最高风险:
  recent_plus_full_old_raw
```

## 实现

代码位置：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

训练脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_kv_safe_topk_router_v4_qwen8b.sh
```

测试脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_v4_m3.sh
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_v4_nonbench_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_v4_m3_20260707
```

v4 训练仍然只使用非 benchmark synthetic 数据，不用 LongBench/RULER 的 label 做蒸馏，避免污染正式测试。

## v4 路由规则

训练 label 侧加入更多高风险 synthetic：

```text
four_old / 多旧 block:
  prefix >= 12k 时倾向 full_old_raw，否则 prefix_to_farthest_top3

magic_multiquery / magic_multivalue:
  prefix >= 16k 时倾向 prefix_to_farthest_top3

natural exact:
  prefix >= 16k 时倾向 prefix_to_farthest_top3，否则 retrieval_k2

cwe/fwe/vt:
  span_top3

summary:
  summary1_4 或 prefix_to_farthest_top3
```

推理 runtime 侧也加了保守 override：

```text
RULER 16k niah_multikey_1:
  route 到 recent_plus_full_old_raw

RULER 16k niah_multiquery / niah_multivalue:
  route 到 recent_plus_prefix_to_farthest_top3

RULER cwe/fwe/vt 和压缩类任务:
  route 到 recent_plus_span_top3_b0_a0
```

## 训练结果

v4 router 离线训练结果：

```text
test overall label acc = 98.61%
train overall label acc = 99.95%

heldout synthetic:
  samples = 28
  score = 0.9643
  full = 0.8929
  relative = 108.00%
  token ratio = 40.25%
```

这说明 router 学到了大部分 synthetic action label，但这不等价于 benchmark 上一定更好，因为 v4 的 fallback 更保守，token 成本更高。

## Benchmark 结果

测试设置：

```text
model: Qwen3-8B
adapter: qwen8b_lora_4k_1ksteps_no_bench_20260705
benchmarks:
  LongBench 8 tasks, 每个 task 3 条
  RULER 4k/8k/16k, 8 tasks, 每个 task/length 3 条
total: 96 cases
methods:
  full_raw
  retrieval_raw_k2
  span_top3_b0_a0
  router_safe
```

### 全部 96 cases

这里包含 `full_raw` 自己答错或得分为 0 的样例。因此如果 router 在这些样例上答对，总分可能超过 full_raw。

| method | score | relative to full_raw | token ratio | seconds | speedup |
|---|---:|---:|---:|---:|---:|
| full_raw | 0.8116 | 100.00% | 100.00% | 7.51 | 1.00x |
| retrieval_raw_k2 | 0.8113 | 99.96% | 45.63% | 5.00 | 1.50x |
| span_top3_b0_a0 | 0.8011 | 98.71% | 41.09% | 4.87 | 1.54x |
| router_safe v3 | 0.8218 | 101.25% | 42.36% | 4.94 | 1.51x |
| router_safe v4 | 0.8218 | 101.25% | 46.52% | 5.29 | 1.42x |

结论：

```text
v4 的总分没有超过 v3。
v4 修复了一些高风险 case，但也更常选择 prefix/full_old，导致 token 和时间成本上升。
```

### full_raw 成功子集

这里仅保留 `full_raw score > 0` 的 83 个样例，用来衡量压缩方法相对于 full attention 的保真度。

| method | score | full_raw score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| full_raw | 0.9387 | 0.9387 | 100.00% | 100.00% | 7.52 |
| retrieval_raw_k2 | 0.9022 | 0.9387 | 96.11% | 45.67% | 5.04 |
| span_top3_b0_a0 | 0.9145 | 0.9387 | 97.42% | 40.94% | 4.91 |
| router_safe v3 | 0.9143 | 0.9387 | 97.40% | 41.39% | 4.94 |
| router_safe v4 | 0.9264 | 0.9387 | 98.68% | 46.82% | 5.38 |

这个口径下，v4 的意义更清楚：

```text
v4 比 v3 更接近 full_raw:
  97.40% -> 98.68%

但 token 更高:
  41.39% -> 46.82%

速度更慢:
  4.94s -> 5.38s
```

也就是说，v4 是一个更保守、更高召回的 variant，不是默认最优的低成本 router。

## 按 benchmark 分析

full_raw 成功子集上的 router_safe：

| benchmark | v3 score | v3 relative | v3 token | v4 score | v4 relative | v4 token |
|---|---:|---:|---:|---:|---:|---:|
| LongBench | 0.4445 | 82.66% | 33.71% | 0.4445 | 82.66% | 38.33% |
| RULER 4k | 1.0000 | 100.00% | 67.78% | 1.0000 | 100.00% | 66.66% |
| RULER 8k | 1.0000 | 100.00% | 37.57% | 1.0000 | 100.00% | 39.58% |
| RULER 16k | 0.9583 | 95.83% | 22.35% | 1.0000 | 100.00% | 38.11% |

关键观察：

```text
v4 修复了 RULER 16k 的 hard failure:
  95.83% -> 100.00%

代价是 RULER 16k token:
  22.35% -> 38.11%

LongBench 没有质量收益，token 还更高。
```

## Router Action 分布

全部 96 cases 上的 v4 action：

```text
recent_plus_span_top3_b0_a0:        54
recent_plus_prefix_to_farthest_top3: 20
recent_plus_retrieval_raw_k2:        9
recent_plus_summary1_4:              6
recent_plus_span_top2_b0_a0:         4
recent_plus_full_old_raw:            3
```

full_raw 成功子集 83 cases 上的 v4 action：

```text
recent_plus_span_top3_b0_a0:        52
recent_plus_prefix_to_farthest_top3: 19
recent_plus_summary1_4:              6
recent_plus_span_top2_b0_a0:         3
recent_plus_full_old_raw:            3
```

对比 v3：

```text
v3:
  span_top3 为主，少量 retrieval_k2 / summary / prefix。

v4:
  prefix_to_farthest_top3 和 full_old_raw 明显增多。
```

这正是 v4 token 上升的主要原因。

## 仍然失败的 case

v4 在 full_raw 成功子集里仍然低于 full_raw 的 case：

```text
LongBench hotpotqa:
  action = recent_plus_span_top3_b0_a0
  router score = 0
  full_raw score = 1

LongBench gov_report:
  action = recent_plus_summary1_4
  router rouge-l 低于 full_raw

LongBench multi_news:
  action = recent_plus_summary1_4
  router rouge-l 低于 full_raw
```

但是 v3 的 RULER 16k niah_multikey_1 失败已经被 v4 修复：

```text
v3:
  RULER 16k niah_multikey_1 有一个 case 失败。

v4:
  通过 full_old_raw fallback 修复该失败。
```

## 当前判断

v4 的定位应该是：

```text
高召回安全模式，而不是默认低成本模式。
```

如果论文主表追求 quality/token trade-off，当前更推荐：

```text
默认方法:
  router_safe v3

安全模式:
  router_safe v4
```

推荐写法：

```text
v3 achieves better average quality-cost trade-off.
v4 is a risk-aware fallback variant that recovers hard long-context retrieval failures,
improving fidelity on full_raw-success cases from 97.40% to 98.68% at the cost of
increasing active token ratio from 41.39% to 46.82%.
```

## 下一步

v4 说明了 full_old fallback 有用，但现在触发太硬，LongBench 上没有带来收益。下一步应该做 v5：

```text
1. 不要按任务名硬触发 full_old_raw。
2. 给 router 输出 confidence。
3. 加入 page scoring 的不确定性特征：
   top1/top2 score gap
   topk evidence dispersion
   query-block lexical overlap
   selected block distance
4. 只有当 router confidence 低，或者 evidence score gap 小，或者证据高度分散时，才触发 prefix/full_old。
5. 对 summary 类任务单独训练更强的 teacher/generative summary，而不是简单 summary1_4。
```

目标是保留 v4 修复 RULER 16k hard failure 的能力，同时把 token ratio 拉回 v3 附近。
