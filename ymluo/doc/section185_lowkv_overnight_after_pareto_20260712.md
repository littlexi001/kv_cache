# section185：Low-KV overnight after-pareto 结果跟踪

更新时间：2026-07-12 07:16:57

目标：KV ratio 1%-10%，端到端/online speed 2.5x+，LongBench 分数达到 full baseline 的 95%+。

Full baseline：score=0.3658，online=3.0988s，speed=1.00x。

## 当前结论

- v393/v394 已经把 10% 档推到 score=0.3911，约为 full 的 106.9%，速度约 4.8x。
- v395 在 7.5% 档达到 score=0.3849，约为 full 的 105.2%，速度约 5.25x。
- v396 在 5.0% 档达到 score=0.3752，约为 full 的 102.6%，速度约 7.01x，是目前最强的极低 KV 可用点。
- 新增 v402-v405 是 pairwise safety planner + task gate，用于验证是否能在 v396 强基底上继续提升；M20 高分要和同 sample-id 的 v396 对比后再判断，避免样本波动误判。
- 同 sample-id M20 诊断显示，v404/v405 的高 M20 分数主要来自 v396 强基底，本身还没有证明稳定超过 v396；真正需要看它们的 M100 结果。
- v398/v401-v406 的同 sample-id M20 复核整体没有稳定超过 v396；因此当前主线转向 expanded frontier，而不是继续堆 sample-level router/planner。
- v400 M100 已完成：score=0.3682、KV=7.09%、speed=6.25x，质量和 KV 都不如 v396，因此 cost-router 不是当前主线。
- v407 是新增的 exact 6% task frontier，离线 M100 聚合预估 score=0.3804、KV=5.85%，用于填补 v396(5%) 和 v395(7.5%) 中间档。
- v408/v410 是 4.5%/6.5% exact frontier，用来验证 v396 以下和 v395 以下的稳定可用区间。
- v413/v414/v415 是 expanded frontier：加入 legacy question-aware b128/b256/b512 作为候选后重新做 knapsack。离线预估显示 3.5%-4.5% KV 也可能达到 95%+ full，但 legacy 样本池不同，必须以重新跑出的 M20/M100 为准。
- same-sample 离线复核显示，expanded frontier 在 v396 M100 同一批 sample 上仍然有希望：v413≈3.49% KV/98.5% full，v414≈4.00% KV/100.4% full，v415≈4.50% KV/102.8% full。
- v417 是 3.0% expanded frontier；same-sample 离线复核约为 2.99% KV/96.2% full/7.32x，是当前最激进的过线候选。
- v419 是新增 macro-mode router：不使用 task one-hot，只用 family + 检索置信度/coverage/长度等运行时特征，在 operator/direct、low-KV frontier、legacy sparse retrieval 之间选择。离线 cal/test 均过 95% full，已排 M20 gate。
- v412 是 v408 的 no-direct fairness ablation，用于区分纯 KV 选择收益和 direct structured operator 带来的任务捷径收益。

## M100 结果

| 方法 | 说明 | 状态 | samples/progress | Score | vs full | KV ratio | speed |
|---|---|---:|---:|---:|---:|---:|---:|
| v381 | completed 5.75% planner | done | 1600 | 0.3662 | 100.11% | 5.75% | 5.99x |
| v382 | completed 5.43% planner | done | 1600 | 0.3641 | 99.54% | 5.43% | 6.14x |
| v389 | completed 9.93% task knapsack | done | 1600 | 0.3906 | 106.78% | 9.93% | 4.36x |
| v393 | after-pareto 10% task knapsack | done | 1600 | 0.3911 | 106.91% | 9.99% | 4.79x |
| v394 | exact 10% task knapsack | done | 1600 | 0.3911 | 106.91% | 9.99% | 4.84x |
| v395 | exact 7.5% task knapsack | done | 1600 | 0.3849 | 105.22% | 7.49% | 5.25x |
| v396 | exact 5% task knapsack | done | 1600 | 0.3752 | 102.57% | 5.00% | 7.01x |
| v397 | after-pareto cost router 10% | running/missing | 0 | - | - | - | - |
| v398 | after-pareto cost router 7.5% | running/missing | 462 | - | - | - | - |
| v399 | after-pareto cost router 5.5% | running/missing | 0 | - | - | - | - |
| v400 | completed-anchor cost router 10% | done | 1600 | 0.3682 | 100.67% | 7.09% | 6.25x |
| v401 | completed-anchor cost router 5.5% | running/missing | 1133 | - | - | - | - |
| v402 | task-gated pair planner 10% | running/missing | 1078 | - | - | - | - |
| v403 | task-gated pair planner 7.5% | running/missing | 995 | - | - | - | - |
| v404 | after-pareto pair planner 10% | running/missing | 1067 | - | - | - | - |
| v405 | after-pareto pair planner 7.5% | running/missing | 1028 | - | - | - | - |
| v406 | after-pareto pair planner 10%, base v381 | running/missing | 556 | - | - | - | - |
| v407 | exact task knapsack 6% | running/missing | 0 | - | - | - | - |
| v408 | exact task knapsack 4.5% | running/missing | 0 | - | - | - | - |
| v410 | exact task knapsack 6.5% | running/missing | 0 | - | - | - | - |
| v412 | v408 no-direct fairness ablation | running/missing | 0 | - | - | - | - |
| v413 | expanded frontier 3.5% | running/missing | 0 | - | - | - | - |
| v414 | expanded frontier 4.0% | running/missing | 0 | - | - | - | - |
| v415 | expanded frontier 4.5% | running/missing | 0 | - | - | - | - |
| v417 | expanded frontier 3.0% | running/missing | 0 | - | - | - | - |
| v419 | macro-mode router 4.5%, no task one-hot | running/missing | 0 | - | - | - | - |

## M20 闸门/运行中结果

| 方法 | 说明 | 状态 | samples/progress | Score | vs full | KV ratio | speed |
|---|---|---:|---:|---:|---:|---:|---:|
| v397_m20 | after-pareto cost router 10% M20 | running/missing | 0 | - | - | - | - |
| v398_m20 | after-pareto cost router 7.5% M20 | done | 320 | 0.3979 | 108.78% | 6.79% | 5.66x |
| v399_m20 | after-pareto cost router 5.5% M20 | running/missing | 0 | - | - | - | - |
| v401_m20 | completed-anchor cost router 5.5% M20 | done | 320 | 0.3883 | 106.16% | 5.98% | 6.52x |
| v402_m20 | task-gated pair planner 10% M20 | done | 320 | 0.3936 | 107.61% | 5.76% | 5.96x |
| v403_m20 | task-gated pair planner 7.5% M20 | done | 320 | 0.3879 | 106.05% | 5.03% | 6.82x |
| v404_m20 | after-pareto pair planner 10% M20 | done | 320 | 0.3976 | 108.70% | 5.12% | 8.21x |
| v405_m20 | after-pareto pair planner 7.5% M20 | done | 320 | 0.3956 | 108.14% | 5.05% | 8.07x |
| v406_m20 | after-pareto pair planner 10%, base v381 M20 | done | 320 | 0.3958 | 108.21% | 5.80% | 6.13x |
| v407_m20 | exact task knapsack 6% M20 | running/missing | 0 | - | - | - | - |
| v408_m20 | exact task knapsack 4.5% M20 | running/missing | 0 | - | - | - | - |
| v410_m20 | exact task knapsack 6.5% M20 | running/missing | 0 | - | - | - | - |
| v412_m20 | v408 no-direct fairness ablation M20 | running/missing | 0 | - | - | - | - |
| v413_m20 | expanded frontier 3.5% M20 | running/missing | 0 | - | - | - | - |
| v414_m20 | expanded frontier 4.0% M20 | running/missing | 0 | - | - | - | - |
| v415_m20 | expanded frontier 4.5% M20 | running/missing | 72 | - | - | - | - |
| v417_m20 | expanded frontier 3.0% M20 | running/missing | 0 | - | - | - | - |
| v419_m20 | macro-mode router 4.5%, no task one-hot M20 | running/missing | 0 | - | - | - | - |

## Top completed M100

| rank | 方法 | 说明 | Score | vs full | KV ratio | speed |
|---:|---|---|---:|---:|---:|---:|
| 1 | v394 | exact 10% task knapsack | 0.3911 | 106.91% | 9.99% | 4.84x |
| 2 | v393 | after-pareto 10% task knapsack | 0.3911 | 106.91% | 9.99% | 4.79x |
| 3 | v389 | completed 9.93% task knapsack | 0.3906 | 106.78% | 9.93% | 4.36x |
| 4 | v395 | exact 7.5% task knapsack | 0.3849 | 105.22% | 7.49% | 5.25x |
| 5 | v396 | exact 5% task knapsack | 0.3752 | 102.57% | 5.00% | 7.01x |

## Top <=6% KV completed M100

| rank | 方法 | 说明 | Score | vs full | KV ratio | speed |
|---:|---|---|---:|---:|---:|---:|
| 1 | v396 | exact 5% task knapsack | 0.3752 | 102.57% | 5.00% | 7.01x |
| 2 | v381 | completed 5.75% planner | 0.3662 | 100.11% | 5.75% | 5.99x |
| 3 | v382 | completed 5.43% planner | 0.3641 | 99.54% | 5.43% | 6.14x |

## 方法解释

v393-v396 是 task-level Pareto/knapsack：先把多个候选 policy 的 M100 结果按任务聚合，再在全局平均 KV 约束下选择每个任务的最优 policy。

v402-v405 是新增的 task-gated pair planner：先训练二分类 safety planner，按成本从低到高判断候选动作是否安全；再只在 cal/test fold 稳定的任务上启用 planner，最后用全局 KV knapsack 选择启用任务。这个设计是为了减少样本级 router 的过拟合，同时保留动态预算思想。

