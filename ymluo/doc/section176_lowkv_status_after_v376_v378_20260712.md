# section176：v376/v378 后的低 KV 状态更新

日期：2026-07-12

## 当前最强完成结果

截至目前，最强的已完成 all-task M20 低 KV 方法是 v376。

| 方法 | samples | score | KV keep | speed/full | vs full | vs v300 |
|---|---:|---:|---:|---:|---:|---:|
| v376 strict10 Pareto fused | 320 | 0.3969 | 6.03% | 5.81x | 108.49% | 90.35% |
| v370 uncertainty ladder | 320 | 0.3844 | 6.14% | 9.58x | 105.09% | 87.52% |
| v365 ultra skeleton | 320 | 0.3776 | 8.24% | 5.79x | 103.23% | 85.97% |
| v368 direct operator extreme mix | 320 | 0.3754 | 7.99% | 8.18x | 102.64% | 85.48% |

这说明：相对 full KV baseline，目标已经在 M20 上成立，而且 v376 把 KV 从 v365 的 8.24% 进一步压到 6.03%，分数还更高。

## v376 的任务级结构

v376 主要强在极低 KV 的 easy/structured tasks，把全局 KV 压得很低；主要质量缺口仍在 `hotpotqa`、`narrativeqa`、`2wikimqa`、`repobench-p`。

| task | v376 score | v376 KV |
|---|---:|---:|
| hotpotqa | 0.2292 | 9.75% |
| narrativeqa | 0.1435 | 5.45% |
| 2wikimqa | 0.2883 | 14.63% |
| repobench-p | 0.4112 | 9.94% |
| multifieldqa_en | 0.4397 | 5.60% |
| qasper | 0.3380 | 10.34% |
| triviaqa | 0.6186 | 2.84% |
| passage_retrieval_en | 1.0000 | 1.21% |

## v378 离线训练结果

v378 是 sample-level policy-action planner，训练已经跑通并生成：

- `outputs/riskkv_v19_policy_action_planner_v378_20260712/model.pkl`
- `outputs/riskkv_v19_policy_action_planner_v378_20260712/action_policy.json`
- `configs/riskkv_task_policy_v378_policy_action_planner_20260712.json`

离线 all-sample 估计：

| method | score | KV keep | speed/full |
|---|---:|---:|---:|
| base v365 | 0.3776 | 8.24% | 5.79x |
| learned v378 | 0.3809 | 7.37% | 6.14x |
| oracle policy selection | 0.4320 | 7.29% | 6.84x |

解释：

- v378 learned router 目前只比 v365 小幅提升，还没有超过 v376。
- 但 oracle policy selection 很强，说明“样本级选择完整 policy”这条线有真实上界，不是死路。
- 下一步不是继续加预算，而是改进 policy router 的学习方式，例如从 safety classifier 改成 oracle-action distillation / cost-sensitive multiclass router。

## 新增 v379

M100 partial 显示：hotpotqa 在 v375 的中高预算证据图上质量明显更好，但 v376 的其它任务更省 KV。因此新增 v379：

- 继承 v376。
- 只把 `hotpotqa` 切回 v375 风格的 mid-KV graph bridge/certificate。
- 目标：在全局 KV 仍接近 10% 的情况下，修复 v376 的 hotpotqa 质量缺口。

配置：

`configs/riskkv_task_policy_v379_hotpot_midkv_global10_20260712.json`

已提交 v379 M20，等待 GPU 或运行中。

## 运行中任务

- v368 M100 confirm：运行中。
- v375 M100：运行中。
- v376 M100：运行中，是当前最重要的主候选验证。
- v377 M20：已从 OOM 后重启，运行中。
- v378 M20：运行中。
- v379 M20：已提交，等待/运行中。
