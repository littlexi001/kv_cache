# Section 191: Constrained Source Composer for Low-KV Frontier Routing (2026-07-12)

## 动机

v427/v428 说明一个关键现象：不同任务的最优 KV frontier 不同，而且不能简单 overlay 参数，否则会破坏 `reference` / fallback 语义。正确做法是保留每个源 policy 的完整任务片段，也就是 source-preserving composition。

v429 是第一版手工 source-preserving 组合。为了让这个思路更像论文方法，而不是人工挑任务，我新增了 constrained source composer。

## 方法

给定一组已经验证可运行的 frontier policy，每个任务选择一个 source policy。目标是最大化平均 score，同时满足：

1. 平均 KV ratio 不超过给定阈值；
2. 平均 online latency 对应的 speed 不低于给定阈值；
3. 单任务 KV ratio 不超过可选上限，避免个别任务退化成高 KV。

形式上，对任务集合 `T` 和候选源集合 `S`，选择：

```text
max    (1 / |T|) sum_t score(t, s_t)
s.t.   (1 / |T|) sum_t kv(t, s_t) <= B_kv
       (1 / |T|) sum_t online(t, s_t) <= B_time
       kv(t, s_t) <= B_task, for all t
```

实现上已经从 Lagrange penalty grid 升级到离散 DP exact composer。脚本：

```bash
scripts/build_source_composer_policy_v430_20260712.py
```

这个脚本输出的是普通 `__extends + __task_sources` policy，因此仍然走已有 RiskKV runtime，不需要 oracle。

## 已生成候选

| 方法 | Composer | 预测 score | 预测 KV | 单任务 KV 上限 | 预测 speed |
|---|---|---:|---:|---:|---:|
| v430 | Lagrange, KV<=6%, speed>=6x | 0.3862 | 5.53% | 14.18% | 6.82x |
| v431 | Lagrange, KV<=8%, speed>=5x | 0.3887 | 6.23% | 14.18% | 5.22x |
| v433 | DP exact, KV<=6%, speed>=6x | 0.3864 | 5.54% | 14.18% | 6.47x |
| v434 | DP exact, KV<=8%, speed>=5x | 0.3887 | 6.23% | 14.18% | 5.21x |
| v435 | DP exact, KV<=10%, speed>=3.5x | 0.3920 | 8.71% | 31.51% | 4.39x |

v430 和 v431 已经在服务器后台运行 M100。v433/v434 是更论文级的 exact composer 版本；v430/v431 仍然有用，因为它们已经在跑，可以直接给我们验证 composer 预测是否兑现。

## 当前后台任务

| 实验 | 状态 |
|---|---|
| v427 LongBench M200 | running |
| v428 LongBench M200 | running |
| full_kv LongBench M200 | running |
| v427 RULER M50 4k/8k/16k | running |
| full_kv RULER M50 4k/8k/16k | running |
| v429 LongBench M100 | running |
| v430 LongBench M100 | running |
| v431 LongBench M100 | running |

## RULER 风险和补充实验

检查 policy 解析后发现，LongBench source-composer policy 在 RULER 任务名上不会命中 `narrativeqa/qasper/...` 这些任务 key，而是回落到 parent wildcard。当前 RULER v427 验证实际接近 512-token wildcard，因此 4k setting 的 KV ratio 可能超过 10%，短上下文下 online overhead 也可能吃掉速度收益。

这不是盲扫参数，而是由 task-name fallback 机制暴露出的评测风险。已经准备了一个 RULER-specific 低预算 policy：

```text
configs/riskkv_task_policy_v436_ruler_lowkv_b224_20260712.json
```

核心设置：

```text
budget_tokens=224, sink_tokens=32, recent_tokens=32, page_tokens=64
```

预期作用是在 RULER 4k 下把 KV ratio 压到约 10% 附近，同时保留 query-aware lexical/multiscale 检索能力。启动脚本：

```bash
scripts/launch_v436_ruler_lowkv_20260712.sh
```

等 SSH 恢复或有空卡后可以直接运行。RULER 汇总脚本已经加入 v436：

```bash
scripts/summarize_v427_ruler_validation_20260712.py
```

## 下一步判断

如果 v430/v431 真实 M100 接近预测，那么当前最有希望的主线应从 v427 升级为：

**Source-Preserving Frontier Routing + Constrained Source Composer**

论文故事会更强：

1. 首先证明单一 KV budget/frontier 不是最优；
2. 观察到不同任务需要不同安全 frontier；
3. naive parameter overlay 会破坏 fallback/reference 语义；
4. source-preserving composition 保留源 policy 的完整动作语义；
5. constrained composer 自动选择 source，在 KV/latency 约束下最大化质量。

如果 v430/v431 没有兑现，则优先检查哪些任务的 source transfer 失效，再决定是用 M20/M100 split 重新估计 source，还是引入 sample-level composer。
