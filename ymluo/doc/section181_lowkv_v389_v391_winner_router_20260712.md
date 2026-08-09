# section181：v389/v391 winner-router 进展

日期：2026-07-12

## 新完成结果

这轮先读取 overnight 的 M100 结果。关键新增结果：

| 方法 | samples | score | KV keep | speed/full | 结论 |
|---|---:|---:|---:|---:|---|
| v384 task-gated | 1600 | 0.3877 | 10.34% | 4.65x | 质量高于 v377，但 KV 略超 10% |
| v383 conf040 | 1600 | 0.3704 | 6.87% | 5.38x | 极低 KV Pareto 点 |
| v386 task-knapsack | 320 | 0.4008 | 9.36% | 5.66x | M20 通过，已经进入 M100 |
| v387 M100 planner | 320 | 0.3939 | 6.20% | 5.80x | M20 通过，已经进入 M100 |

当前已完成 M100 主线仍是 v377：

| 方法 | score | KV keep | speed/full |
|---|---:|---:|---:|
| v377 | 0.3811 | 9.78% | 4.52x |
| v384 | 0.3877 | 10.34% | 4.65x |

v384 说明 learned/task-gated 方向能提高质量，但直接作为主结果会卡在 KV 10% 线外。

## v389：completed-M100 task knapsack v2

把新完成的 v380/v381/v382/v383/v384 加入任务级 Pareto/knapsack 后，10% 预算下最优任务组合是：

| 目标 KV | expected score | expected KV | expected speed/full |
|---:|---:|---:|---:|
| 10% | 0.3906 | 9.93% | 4.95x |

v389 已启动：

- 配置：`configs/riskkv_task_policy_v389_m100_task_knapsack_v2_20260712.json`
- watcher：`scripts/watch_v389_m100_task_knapsack_v2_20260712.sh`
- 状态：正在跑 m20，过 gate 后自动跑 m100。

它的意义是：只用 completed M100 evidence，不依赖 v385 未完成结果，给出一个比 v377 更强、仍在 10% 内的稳态候选。

## 样本级 oracle gap

用已完成 M100 candidate 做逐样本 oracle，发现：

| 策略 | score | KV keep | speed/full |
|---|---:|---:|---:|
| v389 static task policy | 0.3906 | 9.93% | 4.91x |
| sample-level max-score oracle | 0.4290 | 8.46% | 5.49x |

这个 gap 很关键：更高分不是来自更大预算，而是来自同一任务内部的样本级 action 切换。也就是说，继续堆静态 task table 的收益有限；真正的论文方法应该是“risk-aware / winner-aware operator routing”。

## v390：M100-only winner router

v390 训练目标：

- 输入：v389 static policy 的 runtime features。
- label：同一 sample 上 completed-M100 candidates 中分数最高的 action。
- fallback：低置信度退回 v389 static。

离线结果：

| split | base score | learned score | KV keep | speed/full | 结论 |
|---|---:|---:|---:|---:|---|
| all | 0.3906 | 0.4081 | 8.93% | 4.88x | 大幅提升 |
| calibration | 0.3425 | 0.3477 | 7.50% | 4.92x | 正收益 |
| test | 0.4119 | 0.4103 | 8.91% | 4.97x | 略低于 base |

结论：winner router 确实抓到了 oracle gap，但全局启用仍有泛化风险，主要掉在 Hotpot/LCC。

## v391：task-gated winner router

根据 split 诊断，只在更稳的任务上启用 v390：

- `qasper`
- `2wikimqa`
- `narrativeqa`
- `repobench-p`

离线估计：

| score | KV keep | speed/full | calibration gain | test gain |
|---:|---:|---:|---:|---:|
| 0.4010 | 9.87% | 4.98x | +0.0000 | +0.0031 |

v391 已启动：

- 配置：`configs/riskkv_task_policy_v391_task_gated_winner_router_20260712.json`
- watcher：`scripts/watch_v391_task_gated_winner_router_20260712.sh`
- 状态：正在跑 m20，过 gate 后自动跑 m100。

注意：v391 的 task set 是探索性选择，用到了 split 诊断；后续如果要写论文，需要重新设计严格 train/validation/test 划分，不能把这个当最终 protocol。

## 当前判断

现在最值得关注的顺序：

1. v385 M100：如果最终仍保持 10% KV 内且接近 v300 的 95%，它是质量主线。
2. v391 M100：如果真实 M100 接近离线 0.401/9.87%，它会成为新的低 KV 主线。
3. v389 M100：如果 v391 不稳，v389 是 completed-M100 evidence 的稳态后备。
4. v387 M100：如果质量不掉太多，它是 6% KV 左右的 speed/Pareto variant。

方法故事也更清楚了：

- Block/operator 本身提供多个低 KV 候选。
- Task-level knapsack 保证全局 KV 预算。
- Winner-aware router 利用样本级可判别特征，在不增加 KV 的情况下提高质量。

这比单纯“固定预算 block retrieval”更像一个完整 ICLR 方法。
