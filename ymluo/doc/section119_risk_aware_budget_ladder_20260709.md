# Section 119: Risk-aware budget ladder

## 为什么需要 v91

v81 已经证明 qasper 可以从 full fallback 释放出来，但整体 KV keep 仍在 58.31%。原因是 `hotpotqa`、`musique`、`trec`、`passage_count`、`repobench-p` 仍然依赖 full fallback。固定预算 release 的问题是：同一个任务内部既有容易样本，也有需要多跳证据或完整 recent context 的困难样本。

v91 的目标是把任务级 fallback 改成样本级最小安全动作：

```text
query/context -> page scores -> risk signals -> budget ladder -> block selection
```

## 方法定义

对于每个开启 ladder 的任务，预定义一个预算动作集合：

```text
B = {b1, b2, ..., bm}
```

其中 `b1` 是低成本动作，`bm` 可以是高预算或近似 full action。对每个样本，先计算 block scorer 的不确定性：

- `gap2 = score(top1) - score(top2)`
- `entropy = normalized_entropy({score_i})`
- `top = score(top1)`

然后根据阈值选择风险等级：

```text
r_gap = max{k | gap2 <= tau_gap[k]}
r_ent = max{k | entropy >= tau_ent[k]}
r_top = max{k | top <= tau_top[k]}
r = max(r_gap, r_ent, r_top)
action = B[min(r, |B|-1)]
```

直觉：

- top block 明显领先时，用低预算。
- 多个 block 分数接近、熵高、top score 低时，说明证据不集中，升级预算。
- 只有高风险样本才接近 full KV。

## 和旧策略的区别

| 策略 | 决策粒度 | 问题 |
| --- | --- | --- |
| v72/v81 full fallback | task-level | 稳但 KV keep 高 |
| v86-v90 fixed release | task-level/action-level | 能诊断任务，但不能区分同任务难易样本 |
| v91 budget ladder | sample-level | 更像 router，可解释，也更适合蒸馏 |

## 当前实现

代码新增：

- `--ours_budget_ladder_tasks`
- `--ours_budget_ladder_tokens`
- `--ours_budget_ladder_gap2_thresholds`
- `--ours_budget_ladder_entropy_thresholds`
- `--ours_budget_ladder_top_score_thresholds`

输出新增：

- `ours_budget_ladder_active`
- `ours_budget_ladder_selected_budget`
- `ours_budget_ladder_level`
- `ours_budget_ladder_reasons`

配置：

```text
configs/riskkv_task_policy_v91_risk_aware_budget_ladder_20260709.json
```

启动：

```bash
SAMPLES=20 GPUS=5,6,7 \
  nohup bash scripts/run_riskkv_v91_budget_ladder_20260709.sh \
  > outputs/logs/run_riskkv_v91_budget_ladder_20260709.nohup.log 2>&1 &
```

## 预期用途

v91 不只是为了提高当前分数，还服务于论文创新点：

1. 从 fixed compression ratio 变成 risk-conditioned memory action。
2. 把 KV 压缩表述为最小安全动作选择，而不是简单 top-k pruning。
3. 可以从 v86-v90 和 v91 的结果蒸馏 supervised router：输入 score gap、entropy、coverage recall、task family、block size，输出最小安全预算。

## 当前运行状态

截至 2026-07-09 13:08：

- v86-v88 的 fixed release 已经显示：HotpotQA/MuSiQue 固定 2048 会明显掉分。
- v91 已同步到服务器，并在 GPU 7 启动 targeted m20。
- v89 static release 首次 OOM，已用 retry label 在 GPU 5 重跑。
- v90 full release adaptive 正在 GPU 6 继续跑。

下一步等 v91 结果出来后，重点看：

- 是否能在 HotpotQA/MuSiQue 上恢复接近 v81 的分数。
- 是否比 v81 显著降低 100% keep 的样本比例。
- `ours_budget_ladder_selected_budget` 的分布是否合理；如果大量样本被升到最高预算，说明阈值过保守或 scorer 不足。

## 完整结果

v91 targeted m20 已完成：

| 任务 | Score | KV keep | 结论 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.4008 | 100.00% | 恢复 v81 质量，但全部升到 full |
| MuSiQue | 0.3000 | 100.00% | 恢复 v81 质量，但全部升到 full |
| TREC | 0.7000 | 40.34% | 低于 v81 的 0.7500 |
| PassageCount | 0.1500 | 100.00% | 质量恢复，但全部升到 full |
| RepoBench-P | 0.5167 | 100.00% | 质量恢复，但全部升到 full |
| Qasper | 0.4733 | 67.81% | 低于 v81 的 0.5332，且比 v81 更耗 KV |

Overall：Score 0.4235，KV keep 84.69%。

结论：v91 的风险信号能识别困难样本，但阈值过于保守；它不是当前主线。它的价值在于证明了 hard tasks 的 risk signal 很强，后续可以作为 router 的 high-risk detector，而不是直接作为 compression policy。
