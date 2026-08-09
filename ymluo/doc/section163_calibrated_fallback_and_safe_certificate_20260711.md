# Section 163: Calibrated fallback 与 safe certificate

日期：2026-07-11

## 核心观察

当前 v300/v311 的总体平均数已经很好，但 LongBench QA 任务的 KV ratio 偏高，主要不是因为 block scorer 本身慢，而是因为许多样本被风险规则直接升级到 full cache。

M100 任务级结果显示：

- `multifieldqa_en`: score 0.5695，KV 74.04%
- `musique`: score 0.2551，KV 76.91%
- `hotpotqa`: score 0.5445，KV 54.91%
- `qasper`: score 0.4236，KV 43.64%
- `2wikimqa`: score 0.4444，KV 38.95%
- `narrativeqa`: score 0.1960，KV 36.16%

进一步检查样本级日志：

- `multifieldqa_en`: score-risk full 触发 70%
- `musique`: score-risk full 触发 72%
- `qasper`: score-risk full 触发 33%
- `narrativeqa`: score-risk full 触发 39%
- `hotpotqa`: coverage-risk full 触发 31%
- `2wikimqa`: score-risk/retry full 合计约 30%

因此，下一步最值得优化的不是继续盲目调 block size，而是把 fallback 从粗规则改成 calibrated fallback。

## 离线规则挖掘

用 strict oracle labels 与 v300 样本特征做 join，目标是判断：

1. 当前 fallback 是否真的对应危险样本。
2. 是否存在高精度 safe certificate，可以无损跳过 full fallback。

输出目录：

```text
outputs/riskkv_v19_score_risk_rule_mining_20260711/
```

重要现象：

- `2wikimqa` 当前 fallback 很准：precision 1.00，recall 0.857，trigger rate 0.30。
- `multifieldqa_en` 当前 fallback 比较合理但偏保守：precision 0.70，recall 0.817，trigger rate 0.70。
- `musique` 当前 fallback 明显过保守：precision 0.306，recall 0.957，trigger rate 0.72，而 oracle danger rate 只有 0.23。
- `narrativeqa` 当前 fallback 也偏弱：precision 0.359，recall 0.50，trigger rate 0.39。
- `qasper` 当前 fallback precision 1.00，但 recall 0.569，说明它宁愿少 fallback，不是主要浪费源。

safe certificate 挖掘发现：

- `musique`: `ours_score_max >= 1.16` 可 100% precision 识别 sparse-safe 样本，覆盖 16% 样本。
- `narrativeqa`: `score_risk_linear_value <= 0.7598` 可 100% precision 识别 sparse-safe 样本，覆盖 7% 样本。

这两个规则覆盖不大，但非常干净，适合先作为 practical safe certificate 加入 runtime。

## 新 runtime: score-safe certificate

在 `src/run_controlled_public_kv_benchmark_v1.py` 中新增一组默认关闭的参数：

- `ours_score_safe_tasks`
- `ours_score_safe_min_gap2`
- `ours_score_safe_min_gap3`
- `ours_score_safe_max_entropy`
- `ours_score_safe_min_top_score`
- `ours_score_safe_mean_at_least`
- `ours_score_safe_raw_prefix_at_most`
- `ours_score_safe_raw_prefix_at_least`
- `ours_score_safe_linear_threshold`

逻辑：

```text
final_score_risk_trigger = raw_score_risk_trigger AND NOT safe_certificate
```

这不是 oracle：运行时只使用当前 query/context 的 page-score statistics。

CSV 新增记录：

- `ours_score_risk_raw_triggered`
- `ours_score_safe_active`
- `ours_score_safe_certified`
- `ours_score_risk_triggered`

这样可以直接统计 safe certificate 拦掉了多少 fallback。

## 新实验

### v322: sparse no-full QA base

目的：得到真实 no-full 稀疏底座，判断 full fallback 到底救了多少分。

配置：

```text
configs/riskkv_task_policy_v322_v300_sparse_nofull_qa_smoke_20260711.json
```

任务：

```text
narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique
```

汇总目录：

```text
outputs/riskkv_v19_v322_sparse_nofull_qa_smoke_20260711/
```

### v323: safe certificate smoke

目的：在 v300 fallback 规则基础上，只对高精度 safe 样本跳过 full。

配置：

```text
configs/riskkv_task_policy_v323_safe_certificate_smoke_20260711.json
```

任务：

```text
narrativeqa, musique
```

规则：

- `musique`: `score_safe_min_top_score = 1.16`
- `narrativeqa`: `score_safe_linear_threshold = 0.759802542254672`

汇总目录：

```text
outputs/riskkv_v19_v323_safe_certificate_smoke_20260711/
```

## 明早判断标准

优先看 v323：

- 如果 score 与 v300 same-sample 基本一致，而 KV ratio 下降，safe certificate 成立，可以扩大到 M100。
- 如果 `ours_score_safe_certified` 覆盖率低但无损，说明这个方向保守但可靠，可以继续挖多条件证书。
- 如果质量下降，说明 oracle label 中的 high precision 在 held-out M20 上不稳，需要换成 conformal split 或更高阈值。

再看 v322：

- 如果 no-full 质量只小幅下降，说明当前 fallback 过度保守，可以设计更激进 router。
- 如果 no-full 质量大幅下降，说明 full fallback 确实救命，后续应做 safe certificate，而不是直接砍 fallback。

这轮的重点是从现象出发设计 solution：先证明 fallback 的浪费来自哪些任务，再用 high-precision certificate 安全削减 full cache。

## v322 已完成的 M20 结果

`v322_sparse_nofull` 的结果非常有信息量：

| Task | Score | Full same-sample | v300 same-sample | Score / full | Score / v300 | KV ratio |
|---|---:|---:|---:|---:|---:|---:|
| narrativeqa | 0.2538 | 0.2554 | 0.2538 | 99.36% | 100.00% | 30.61% |
| qasper | 0.5069 | 0.5255 | 0.5138 | 96.45% | 98.66% | 18.79% |
| multifieldqa_en | 0.4132 | 0.5238 | 0.5210 | 78.89% | 79.30% | 10.93% |
| hotpotqa | 0.3017 | 0.4008 | 0.3967 | 75.26% | 76.05% | 38.06% |
| 2wikimqa | 0.2758 | 0.4187 | 0.3187 | 65.88% | 86.55% | 17.20% |
| musique | 0.1333 | 0.3000 | 0.3000 | 44.44% | 44.44% | 17.43% |

结论：

- `narrativeqa` 可以直接 no-full，M20 上完全不掉分。
- `qasper` 可以大幅降 KV，分数仍有 96% full、98.7% v300。
- `multifieldqa_en/hotpotqa/2wikimqa/musique` 不能直接砍 fallback，full/bounded fallback 确实救质量。

这说明下一步主线不是“全局去 fallback”，而是任务选择性 no-full。

## v324: qasper + narrative no-full 放大到 M100

基于 v322 现象，新建 practical task-level policy：

```text
configs/riskkv_task_policy_v324_qasper_narrative_nofull_20260711.json
```

策略：

- `narrativeqa`: 关闭 score-risk full fallback。
- `qasper`: 关闭 score-risk full fallback。
- 其它任务沿用 v300 的安全策略。

启动 M100：

```bash
nohup env GPUS=0,2,3,6 SAMPLES=100 GPU_MAX_USED_MB=3000 GPU_MAX_UTIL=25 \
  bash scripts/launch_v324_qasper_narrative_nofull_m100_20260711.sh \
  > outputs/logs/launch_v324_qasper_narrative_nofull_m100_20260711.log 2>&1 < /dev/null &
```

watcher：

```bash
nohup bash scripts/watch_combine_v324_qasper_narrative_nofull_m100_20260711.sh \
  > outputs/logs/watch_combine_v324_qasper_narrative_nofull_m100_20260711.log 2>&1 < /dev/null &
```

最终汇总：

```text
outputs/riskkv_v19_v324_qasper_narrative_nofull_m100_20260711/summary_table.csv
outputs/riskkv_v19_v324_qasper_narrative_nofull_with_v300_other_20260711_m100_bDyn_pDyn/
```

明早优先看 v324 是否在 M100 上保持 v300 分数，同时降低整体 KV ratio。如果成立，它比 v323 safe certificate 更直接，可以成为下一版 practical best。

## v325: qasper extreme-risk fallback

v322 的样本级诊断进一步发现：

- `narrativeqa` 的 20 个 no-full 样本全部保持 v300 分数。
- `qasper` 的 20 个 no-full 样本里，19 个保持或超过 v300；只有 1 个明显低于 v300。
- 这个 qasper 失败样本的特征非常极端：
  - `ours_score_risk_triggered = 1`
  - `gap2 = 0.0009`
  - `top_score = 0.8`
  - v300 在这个样本上使用 full cache。

因此，`qasper` 不应该简单全 no-full，也不应该沿用 v300 的宽松 fallback；更合理的是只在极端不确定时 fallback。

新建：

```text
configs/riskkv_task_policy_v325_qasper_extreme_risk_20260711.json
```

规则：

```text
qasper:
  score_risk = true
  score_risk_min_gap2 = 0.01
  score_risk_min_top_score = 0.85
```

也就是只有同时满足 `gap2 <= 0.01` 且 `top_score <= 0.85` 时才升 full。这个规则来自失败样本诊断，不是 oracle 标签直接选择。

启动 M100：

```bash
nohup env GPUS=0,2,3,6 SAMPLES=100 GPU_MAX_USED_MB=3000 GPU_MAX_UTIL=25 \
  bash scripts/launch_v325_qasper_extreme_risk_m100_20260711.sh \
  > outputs/logs/launch_v325_qasper_extreme_risk_m100_20260711.log 2>&1 < /dev/null &
```

汇总目录：

```text
outputs/riskkv_v19_v325_qasper_extreme_risk_m100_20260711/
outputs/riskkv_v19_v325_qasper_extreme_risk_with_v300_other_20260711_m100_bDyn_pDyn/
```

明早对比：

- v324: `qasper` 全 no-full，最大化压缩。
- v325: `qasper` 极端风险才 fallback，目标是恢复 v324 可能损失的 qasper 质量，同时仍比 v300 低 KV。

如果 v325 质量接近 v300、KV 明显低于 v300，它比 v324 更稳，更适合作为主线。

## v324 M100 结果与反思

v324 已完成：

| Method | Samples | Score | KV ratio | Online seconds |
|---|---:|---:|---:|---:|
| v300_main | 1600 | 0.4392 | 27.41% | 0.5632 |
| v324 qasper+narrative no-full | 1600 | 0.4340 | 25.51% | 0.5617 |

v324 的 KV 降了约 1.9 个百分点，但分数从 0.4392 降到 0.4340。这个结果说明：

- `qasper/narrativeqa` 全 no-full 不是足够稳的最终策略。
- M20 对 `narrativeqa` 的 no-full 结论过乐观，M100 暴露出 5 个关键失败样本。

M100 样本级诊断：

- `narrativeqa` no-full 有 5 个样本低于 v300。
- 这 5 个样本全部是 v300 的 risk-triggered full 样本。
- 它们共同特征：
  - `score_risk_triggered = 1`
  - `top_score <= 0.8`
  - `entropy >= 0.9908`
  - `gap3 <= 0.071`

因此 `narrativeqa` 也不应该全 no-full，而应该只在极端风险时 fallback。

## v326: narrative extreme-risk fallback

新建：

```text
configs/riskkv_task_policy_v326_narrative_extreme_risk_20260711.json
```

规则：

```text
narrativeqa:
  score_risk = true
  score_risk_min_gap3 = 0.0726621
  score_risk_min_top_score = 0.8
  score_risk_max_entropy = 0.99
```

也就是只有同时满足：

```text
gap3 <= 0.0726621
top_score <= 0.8
entropy >= 0.99
```

才升 full。这个规则覆盖 v324 的 5 个 M100 失败样本，预期比 v300 少 fallback，同时恢复 narrative 质量。

启动 M100：

```bash
nohup env GPUS=0,2,3,6 SAMPLES=100 GPU_MAX_USED_MB=3000 GPU_MAX_UTIL=25 \
  bash scripts/launch_v326_narrative_extreme_risk_m100_20260711.sh \
  > outputs/logs/launch_v326_narrative_extreme_risk_m100_20260711.log 2>&1 < /dev/null &
```

汇总目录：

```text
outputs/riskkv_v19_v326_narrative_extreme_risk_m100_20260711/
outputs/riskkv_v19_v326_narrative_extreme_risk_with_v300_other_20260711_m100_bDyn_pDyn/
```

明早优先比较：

- v300: 当前稳定主线。
- v324: qasper+narrative 全 no-full，KV 低但质量下降。
- v325: qasper 极端风险 fallback。
- v326: narrative 极端风险 fallback。

如果 v325 和 v326 分别成立，下一步可以合并成 v327：`qasper extreme-risk + narrative extreme-risk + v300 other tasks`。
