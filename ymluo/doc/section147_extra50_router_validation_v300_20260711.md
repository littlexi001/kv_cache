# Section 147: extra-50 validation and v300 robust router

日期：2026-07-11

## 背景

v293/v294 说明 sample-level action router 是有效方向，但 M150 overall 仍然包含 M100 的前 100 个样本。为了避免被 M100 阈值过拟合误导，本节把 M150 拆成两部分：

- `overlap100`: 每个 task 中与 M100 相同的 100 条样本。
- `extra50`: 每个 task 中 M150 额外新增的 50 条样本。

这个拆分更接近 validation set，用于判断 router 是否真泛化。

## Overall extra-50

| Method | Split | Samples | Score | KV keep | Online |
|---|---|---:|---:|---:|---:|
| v286 | overlap100 | 1600 | 0.437803 | 28.62% | 0.5757s |
| v286 | extra50 | 800 | 0.425551 | 28.76% | 0.5868s |
| v294 | overlap100 | 1600 | 0.440786 | 27.25% | 0.5715s |
| v294 | extra50 | 800 | 0.425244 | 28.10% | 0.5875s |
| v300 | all150 | 2400 | 0.434884 | 27.74% | 0.5753s |
| v300 | extra50 | 800 | 0.426183 | 28.40% | 0.5834s |

解释：

- v294 在 M150 overall 上最高，但 extra-50 上比 v286 低 0.0003，说明 2Wiki router 有轻微过拟合。
- v300 关闭 2Wiki router 后，M150 overall 仍高于 v286，同时 extra-50 也高于 v286。
- 因此 v300 是更保守的 validation-robust 主线；v294 是 score-oriented Pareto 点。

## Key-task extra-50

| Task | v286 score | v286 KV | v294 score | v294 KV | v300 score | v300 KV | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| narrativeqa | 0.171836 | 31.54% | 0.175354 | 33.39% | 0.175354 | 33.39% | 分数泛化，KV 在 extra-50 上略高；可作为质量 action。 |
| hotpotqa | 0.432238 | 62.38% | 0.438832 | 54.86% | 0.438832 | 54.86% | 同时提分降 KV，强泛化，保留。 |
| 2wikimqa | 0.482358 | 48.10% | 0.467333 | 43.31% | 0.482358 | 48.10% | v294 规则在 extra-50 掉分，v300 关闭该规则。 |

## Candidate-action follow-up

补跑了三个 M150 candidate action：

- `v295_v275_narrative_m150`
- `v296_v275_2wikimqa_m150`
- `v297_v287_hotpot_m150`

结果：

| Task | v286 | v294 | Candidate | Min-safe oracle vs v286 |
|---|---:|---:|---:|---:|
| narrativeqa | 0.184924 / 39.55% | 0.189091 / 35.24% | v275 all: 0.170906 / 29.07% | 0.192332 / 25.98% |
| 2wikimqa | 0.457066 / 42.00% | 0.468603 / 38.63% | v275 all: 0.422159 / 28.51% | 0.463963 / 30.45% |
| hotpotqa | 0.494778 / 65.31% | 0.509257 / 54.89% | v287 all: 0.464773 / 45.68% | 0.521111 / 51.56% |

这说明仍然存在更细粒度的 oracle 空间，但全任务 candidate action 本身并不可靠。下一步应该学习更精细的 sample-level gating，而不是把 candidate action 全局打开。

## v300 Configuration

配置文件：

- `configs/riskkv_task_policy_v300_action_router_extra50_robust_20260711.json`

和 v294 的区别：

- 保留 `narrativeqa` action router。
- 保留 `hotpotqa` action router。
- 关闭 `2wikimqa` action router。
- qasper / multifield 继续关闭。

## 当前主线建议

论文主线建议报告两个点：

| Name | Use | Result |
|---|---|---|
| v294 score-oriented router | 强调 M150 overall 最优 | 0.435605 / 27.53% KV |
| v300 validation-robust router | 强调 extra-50 泛化 | 0.434884 / 27.74% KV，extra50 0.426183 |

如果只选一个作为主方法，建议用 v300；如果画 Pareto 曲线，v294 和 v300 都保留。

## 下一步

1. 用 M100 训练、extra-50 验证一个 learned action router v3。
2. 目标不是追求更多规则，而是让 router 输出置信度：当 validation confidence 不够时回到 v286 safe action。
3. 对 hotpot 可以进一步细化，因为 M150 candidate oracle 还有 51.56% KV / 0.5211 score 的空间。

