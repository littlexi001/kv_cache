# Section 111: Pre-decode Score-Risk Planner

## 动机

v52/v53 的核心优点是 memory-action consistency verifier 能识别危险样本：

```text
先跑 primary sparse KV action
再跑 expanded sparse KV action
若两个 sparse 输出不一致，则 fallback full KV
```

这个信号很强，但代价是多一次 decode。下一步需要验证一个更便宜的版本：不等输出生成，在 block selection 之后、decode 之前，用 evidence score 的不确定性判断当前 memory action 是否危险。

## 新模块

新增模块叫 `score_risk`：

```text
score_risk_active(task) = task in configured score-risk tasks

score_risk_trigger =
  gap2 <= threshold_gap2
  and entropy >= threshold_entropy
  and optional gap3 / top-score conditions

if score_risk_trigger:
  replace current memory action budget with score_risk_budget_tokens
```

其中：

```text
gap2    = top1 page score - top2 page score
entropy = normalized entropy over positive page scores
```

含义是：如果最相关 block 和第二相关 block 分不开，同时整体 page-score 分布很平，说明检索证据不稳定；这时 512-token action 风险高，应当扩预算或 full fallback。

## 和 v52/v53 的区别

v52/v53 是 post-decode verifier：

```text
看两个 sparse-memory outputs 是否一致
```

v57/v58 是 pre-decode planner：

```text
只看 block-score landscape
不多生成一次答案
```

因此 v57/v58 如果效果接近 v52/v53，论文故事会更强：RiskKV-Block 同时有 cheap pre-decode confidence gate 和 stronger post-decode counterfactual verifier。

## 旧 m20 离线校准

用 v35 和 full KV 的同样本 m20 结果，在 `narrativeqa / 2wikimqa / qasper` 上做初步阈值扫描：

```text
gap2 <= 0.15 and entropy >= 0.93
```

校准输出已保存到服务器：

```text
outputs/score_risk_calibration_v35_vs_full_m20_20260709.csv
```

如果触发后直接用 full KV，上界估计如下：

```text
tasks: narrativeqa + 2wikimqa + qasper
base score: 0.3295
full score: 0.3999
triggered: 49 / 60
estimated score: 0.3988
estimated KV ratio on these tasks: about 85.0%
```

更激进的阈值示例：

```text
gap2 <= 0.10 and entropy >= 0.93
estimated score: 0.3682
estimated KV ratio on these tasks: about 73.6%
```

这个阈值比较保守，触发率高。它适合验证 risk signal 是否可靠，不一定是最终 Pareto 最优点。

## 当前新增实验

新增两个 policy：

```text
configs/riskkv_task_policy_v57_predecode_score_risk_full_20260709.json
configs/riskkv_task_policy_v58_predecode_score_risk_2048_20260709.json
```

v57：

```text
触发 score_risk 后直接使用 full-cache action
目的：验证 pre-decode risk signal 本身能否恢复质量
```

v58：

```text
触发 score_risk 后只扩到 2048 tokens
目的：寻找比 v57 更好的 quality / KV Pareto
```

进一步新增 selective counterfactual verification：

```text
configs/riskkv_task_policy_v59_selective_counterfactual_20260709.json
configs/riskkv_task_policy_v60_selective_counterfactual_qasper_full_20260709.json
```

v59：

```text
先用 score_risk 判断 action 是否危险
只有危险样本才运行 consistency verifier
qasper 仍使用 sparse bridge action
```

v60：

```text
v59 + qasper full fallback
用于和 v53 做更公平的质量优先对比
```

这两个变体的目的不是单纯提高分数，而是验证一个更强的系统故事：

```text
cheap pre-decode risk gate
  -> selective counterfactual verification
  -> minimum-safe memory action
```

如果 v59/v60 能保持接近 v52/v53 的质量，同时显著降低 consistency_check_rate 或 online time，这会比单独 v52/v53 更适合作为论文主方法。

新增脚本：

```text
scripts/run_riskkv_v57_v58_score_risk_m20_20260709.sh
scripts/watch_and_launch_v57_v58_after_v55_v56_20260709.sh
scripts/run_riskkv_v59_v60_selective_counterfactual_m20_20260709.sh
scripts/watch_and_launch_v59_v60_after_v57_v58_20260709.sh
```

服务器状态：

```text
v57/v58 watcher 已启动。
它会先等待 v55/v56 probe16 完成，再等待两张空 GPU，然后自动启动 v57/v58 m20。

v59/v60 watcher 已启动。
它会等待 v55/v56 完成，再等待两张空 GPU，然后自动启动 selective counterfactual m20。
```

## v59/v60 离线 stitched 估计

用已有 `v35 / v52 / v53 / full` m20 同样本结果做 corrected stitched 估计：

```text
outputs/selective_counterfactual_offline_sweep_corrected_20260709.csv
outputs/selective_counterfactual_v59_estimate_from_v52_m20_20260709.csv
outputs/selective_counterfactual_v60_estimate_from_v53_m20_20260709.csv
```

关键估计：

```text
v52 actual:
  score = 0.373143
  KV    = 58.17%
  online= 2.720s
  consistency_check_rate = 16.56%

v59 selective estimate:
  score = 0.373143
  KV    = 57.03%
  online= 2.708s
  consistency_check_rate = 13.13%

v53 actual:
  score = 0.375890
  KV    = 62.89%
  online= 2.736s
  consistency_check_rate = 16.56%

v60 selective estimate:
  score = 0.375890
  KV    = 61.75%
  online= 2.723s
  consistency_check_rate = 13.13%
```

这个估计说明 v59/v60 不一定显著提升分数，但可能让方法故事更干净：用 cheap score-risk gate 只在必要时运行 expensive counterfactual verifier。

## 论文价值

如果 v58 能接近 v52/v53，同时 online speed 明显更好，则主故事可以升级为：

```text
RiskKV-Block first estimates memory-action risk from the evidence-score geometry.
Only unresolved high-risk cases require post-decode counterfactual consistency.
```

这比单纯 router 更像一个完整系统：

```text
cheap pre-decode planner
strong post-decode verifier
minimum-safe memory action
```

## 下一步

1. 等 m50 主实验完成，确认 v52/v53 是否稳定。
2. 等 v55/v56，确认 consistency verifier 的 probe16 版本是否能显著降开销。
3. 等 v57/v58，判断 pre-decode risk 是否能替代一部分 post-decode verifier。
4. 等 v59/v60，判断 pre-decode risk 是否能筛掉低风险样本上的 consistency verifier。
5. 如果 v58/v59/v60 质量不足，继续扫：

```text
gap2 threshold: 0.08 / 0.10 / 0.12 / 0.15
entropy threshold: 0.93 / 0.95 / 0.97
expanded budget: 1024 / 1536 / 2048 / full
```
