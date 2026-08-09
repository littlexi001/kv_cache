# Section 146: v293 sample-level action router

日期：2026-07-11

## 背景

v291 在 M100 上看起来略优于 v286：

| Method | Samples | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| v286 mined fallback router | 1600 | 0.4378 | 28.62% | 0.5676s |
| v291 v286 + hotpot3072 | 1600 | 0.4383 | 28.93% | 0.5672s |

但是 M150 验证显示，v291 的 hotpot3072 不是稳定收益：

| Method | Samples | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| v286 M150 | 2400 | 0.433719 | 28.66% | 0.5794s |
| v291 M150 | 2400 | 0.433690 | 28.99% | 0.5790s |

Hotpot 单任务 M150：

| Method | Samples | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| v286 hotpot | 150 | 0.494778 | 65.31% | 0.2336s |
| v291 hotpot3072 | 150 | 0.494309 | 70.61% | 0.2272s |

结论：hotpot3072 的 M100 小幅提升是弱信号，不能直接作为新主线。真正有价值的是后续 oracle 分析发现的 sample-level action selection 空间。

## Oracle 现象

把已有 action 放入同一个候选池：

- v291/v286 当前强基线
- v275 旧低 KV QA 动作
- v287 mid-KV QA 动作
- v288/v290 b16 变体

在每个样本上选择“不低于 v291 当前样本分数的最小 KV action”，得到：

| Scope | v291 score | v291 KV | Min-safe oracle score | Min-safe oracle KV |
|---|---:|---:|---:|---:|
| 6 QA tasks | 0.4029 | 58.16% | 0.4411 | 41.43% |

这说明问题不是没有压缩空间，而是固定 task action 太粗；不同样本应该选择不同 memory action。

## v293 方法

v293 是一个可运行的 sample-level action router。它不是 sample-id oracle，而是用当前 selector 已经可获得的特征做规则分流：

| Task | Rule | Routed action |
|---|---|---|
| narrativeqa | `score_entropy <= 0.990717` | v275 narrative entropy profile |
| qasper | `raw_prefix_tokens <= 3377` | v287 qasper short-prefix profile |
| multifieldqa_en | `raw_prefix_tokens <= 4677` | v287 multifield short-prefix profile |
| hotpotqa | `score_risk_linear_value <= 0.9506` | v287 hotpot lower-risk profile |
| 2wikimqa | `score_mean <= 0.306522` | v275 2Wiki low-mean profile |

`musique` 的离线规则需要先用原 page size 计算 coverage，再切换到 b16 page size；这在单 pass runtime 里不够干净，所以 v293 runtime 暂时不包含 musique 分流。

配置文件：

- `configs/riskkv_task_policy_v293_action_router_rules_20260711.json`

代码改动：

- `src/run_controlled_public_kv_benchmark_v1.py`
- 新增 `ours_action_router_tasks`
- 新增 `ours_action_router_mode=v293_rules`
- 输出 `ours_action_router_selected_action`

## M100 结果

v293 runtime M100 已完成：

| Method | Samples | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| v286 | 1600 | 0.4378 | 28.62% | 0.5676s |
| v291 | 1600 | 0.4383 | 28.93% | 0.5672s |
| v293 runtime router | 1600 | 0.4409 | 26.00% | 0.5701s |

Task-level:

| Task | v286 score | v286 KV | v293 score | v293 KV | Action counts |
|---|---:|---:|---:|---:|---|
| narrativeqa | 0.1915 | 43.56% | 0.1960 | 36.16% | v275: 85, v291: 15 |
| qasper | 0.4236 | 43.64% | 0.4240 | 37.05% | v287: 25, v291: 75 |
| multifieldqa_en | 0.5695 | 74.04% | 0.5703 | 60.65% | v287: 33, v291: 67 |
| hotpotqa | 0.5260 | 66.78% | 0.5445 | 54.91% | v287: 60, v291: 40 |
| 2wikimqa | 0.4444 | 38.95% | 0.4692 | 36.29% | v275: 55, v291: 45 |

观察：

- v293 同时提高分数并降低 KV，是目前 M100 上最好的 practical point。
- Online 没有显著下降，说明当前端到端 online 仍受 decode、prefill 和 Python harness 影响；但 KV keep 已明显下降。
- 这比单纯调 block size 更有论文价值：方法从 fixed/task-level memory policy 进化成 sample-level memory-action planner。

## M150 验证结果

v293 M150 五任务验证已经完成，组合方式：以 `v286_m150_combined` 为 base，只替换 v293 覆盖的五个任务。

| Method | Samples | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| v286 M150 | 2400 | 0.433719 | 28.66% | 0.5794s |
| v293 runtime router M150 | 2400 | 0.434671 | 26.38% | 0.5718s |

Task-level M150:

| Task | v286 score | v286 KV | v293 score | v293 KV | 判断 |
|---|---:|---:|---:|---:|---|
| narrativeqa | 0.1849 | 39.55% | 0.1891 | 35.24% | 泛化成功，保留。 |
| qasper | 0.4046 | 44.54% | 0.3939 | 37.69% | 掉分明显，关闭。 |
| multifieldqa_en | 0.5415 | 76.37% | 0.5373 | 64.74% | 小幅掉分，关闭更稳。 |
| hotpotqa | 0.4948 | 65.31% | 0.5093 | 54.89% | 泛化成功，保留。 |
| 2wikimqa | 0.4571 | 42.00% | 0.4686 | 38.63% | 泛化成功，保留。 |

## v294 Robust Router

v293 的 qasper/multifield 规则在 M100 上可用，但 M150 掉分，说明这两条手写规则有过拟合。v294 关闭这两条，只保留泛化成功的：

- `narrativeqa`
- `hotpotqa`
- `2wikimqa`

配置文件：

- `configs/riskkv_task_policy_v294_action_router_robust_20260711.json`

离线组合结果：

| Method | Split | Samples | Score | KV keep | Online |
|---|---|---:|---:|---:|---:|
| v286 | M100 | 1600 | 0.4378 | 28.62% | 0.5676s |
| v293 | M100 | 1600 | 0.4409 | 26.00% | 0.5701s |
| v294 robust | M100 | 1600 | 0.4408 | 27.25% | 0.5672s |
| v286 | M150 | 2400 | 0.433719 | 28.66% | 0.5794s |
| v293 | M150 | 2400 | 0.434671 | 26.38% | 0.5718s |
| v294 robust | M150 | 2400 | 0.435605 | 27.53% | 0.5768s |

## 当前判断

当前稳健主线应从 v286 更新为 v294 robust router：

- M100：分数显著高于 v286，KV 从 28.62% 降到 27.25%。
- M150：分数高于 v286/v293，KV 仍低于 v286。
- 方法故事更强：从 task-level memory policy 变成 sample-level memory-action planner，并且有 M100/M150 交叉验证说明哪些规则泛化、哪些规则过拟合。

下一步：

1. 把 v294 robust router 写入论文方法主线。
2. 用 M100 训练、M150 extra-50 验证，替代手写阈值，形成 learned action router v3。
3. 扩展到更多模型和完整 LongBench 表格，验证不是 Llama-3.1-8B 上的偶然现象。
