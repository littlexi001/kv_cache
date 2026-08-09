# section182：v389 gate 通过后的 live status

日期：2026-07-12

## 当前最强短样本结果

v389 的 m20 已完成并通过 gate：

| 方法 | samples | score | KV keep | speed/full | 状态 |
|---|---:|---:|---:|---:|---|
| v389 completed-M100 task knapsack v2 | 320 | 0.4085 | 8.95% | 5.70x | 已自动进入 M100 |

这个结果比 v386 m20 更强：

| 方法 | score | KV keep | speed/full |
|---|---:|---:|---:|
| v386 | 0.4008 | 9.36% | 5.66x |
| v389 | 0.4085 | 8.95% | 5.70x |

v389 是目前最值得等待完整 M100 的低 KV 主候选。

## v390/v391 校验

已检查 v390 的 `action_policy.json`，确认它在选择 v380/v381/v382/v378 这类 nested learned-router candidate 时，不会丢失以下字段：

- `ours_learned_router_model_path`
- `ours_learned_router_action_policy_json`
- `ours_learned_router_confidence_threshold`
- `ours_learned_router_default_action`
- `ours_learned_router_base_action_router_mode`

这意味着 v391 的 task-gated winner router 在真实 benchmark 中能正确复现被选中的内部 router，而不是退化成不完整配置。

## 仍在运行的重点实验

| 实验 | 当前作用 | 状态 |
|---|---|---|
| v385 M100 | 检查 M20 最高质量候选能否泛化 | 运行中 |
| v386 M100 | 第一版 M100 task-knapsack | 运行中 |
| v387 M100 | 6% KV 左右激进压缩 Pareto | 运行中 |
| v389 M100 | 当前最强 task-knapsack v2 | 运行中 |
| v391 M20/M100 | task-gated winner-router，验证 sample-level routing | m20 运行中 |
| v388 watcher | 等 v385 M100 完成后自动训练后续 planner | 等待中 |

## 方法判断

目前证据支持的路线是：

1. 用 completed-M100 candidates 构造 task-level Pareto/knapsack，得到 v389 这种稳定低 KV 主线。
2. 再用 winner-aware router 尝试吃掉 sample-level oracle gap。
3. winner router 不能全局开，必须 task-gated 或 risk-gated；否则 Hotpot/LCC 这类任务会在 holdout 上掉分。

下一步应优先等待 v389/v391 的 M100 结果。如果 v391 M100 接近离线估计，它会是比 v389 更像论文主方法的版本；如果 v391 不稳，v389 仍然是强且干净的主线。
