# Section 112: Conformal Risk Controller

## 为什么需要这一版

v52/v53 的 memory-action consistency verifier 有强风险识别能力，但它对所有指定任务都运行第二个 sparse action：

```text
primary sparse action
expanded sparse action
if answers disagree: fallback full KV
```

这给论文带来一个自然攻击点：

```text
你为了压缩 KV，又多跑一次 decode，开销是否值得？
```

v59/v60 用 heuristic score-risk gate 只在危险样本上运行 consistency verifier。v61/v62 进一步把这个 gate 变成 conformal risk controller：阈值不再手调，而是由校准集上的危险动作覆盖率决定。

## 风险分数

对一次 sparse memory action，先得到 block score landscape：

```text
gap2    = top1 block score - top2 block score
gap3    = top1 block score - top3 block score
entropy = normalized entropy over positive block scores
top     = top1 block score
```

当前 conformal risk score 定义为：

```text
r(x, a) = entropy - w2 * gap2 - w3 * gap3 - wt * top
```

默认校准使用：

```text
w2 = 1.0
w3 = 0.0
wt = 0.0
```

直觉：

```text
entropy 越高，说明证据分布越平，风险越高
gap2 越小，说明 top evidence 不稳定，风险越高
```

## Conformal 校准

用已有 m20 同样本结果校准：

```text
base:      v35 selective grounding
reference: v52 / v53 consistency-quality
tasks:     narrativeqa, multifieldqa_en, 2wikimqa
label:     consistency disagreement or consistency full fallback
```

危险标签不看 gold answer，也不看 full KV output：

```text
danger = 1 if reference consistency verifier disagrees or falls back
```

给定目标覆盖率 `1-alpha`，从危险样本的 risk score 分布中选择阈值 `tau`，使得：

```text
Pr[r(x, a) >= tau | danger] >= 1-alpha
```

运行时：

```text
if r(x, sparse_action) >= tau:
    run counterfactual consistency verifier
else:
    accept primary sparse action
```

## m20 校准结果

校准 sweep 输出：

```text
outputs/conformal_risk_controller_sweep_m20_20260709.csv
```

质量优先点：

```text
target recall: 0.80
threshold tau: 0.8441593191391078
actual danger recall: 0.8462
precision: 0.4889
triggered dangerous-task samples: 45 / 60
```

基于 stitched 估计：

```text
v52 actual:
  score = 0.373143
  KV    = 58.17%
  online= 2.720s
  consistency_check_rate = 16.56%

v61 conformal estimate:
  score = 0.373143
  KV    = 57.03%
  online= 2.707s
  consistency_check_rate = 12.50%

v53 actual:
  score = 0.375890
  KV    = 62.89%
  online= 2.736s
  consistency_check_rate = 16.56%

v62 conformal estimate:
  score = 0.375890
  KV    = 61.75%
  online= 2.723s
  consistency_check_rate = 12.50%
```

速度优先点：

```text
target recall: 0.70
threshold tau: 0.8715246550830982
v53-based estimate:
  score = 0.373126
  KV    = 60.88%
  online= 2.718s
  consistency_check_rate = 11.25%
```

当前先跑质量优先点 v61/v62。速度优先点可作为后续 v63/v64。

## Probe-length ablation: v55/v56

为了确认 consistency verifier 的瓶颈是不是 probe 太短，补跑了 probe16：

```text
v55 = v52 + consistency probe max tokens 16
v56 = v53 + consistency probe max tokens 16
```

结果：

| Setting | Score | KV ratio | Online | Consistency check rate | Disagreement rate |
|---|---:|---:|---:|---:|---:|
| v52 probe default | 0.373143 | 58.17% | 2.720s | 16.56% | 8.13% |
| v53 probe default | 0.375890 | 62.89% | 2.736s | 16.56% | 8.13% |
| v55 probe16 | 0.372339 | 58.19% | 2.778s | 16.56% | 8.13% |
| v56 probe16 | 0.375086 | 62.92% | 2.829s | 16.56% | 8.13% |

结论：

```text
Longer consistency probe does not improve quality and only increases online time.
```

这说明主瓶颈不是 verifier answer length，而是：

```text
1. primary sparse memory action 是否选到了覆盖充分的证据；
2. 哪些样本值得触发 verifier / fallback；
3. fallback 应该是 sparse expansion 还是 full memory。
```

因此后续主线转向 coverage-certified memory action，而不是继续拉长 verifier。

## Leave-one-task-out 泛化检查

为了检查 conformal gate 是否只是 m20 上手调，新增 leave-one-task-out 评估：

```text
scripts/evaluate_conformal_risk_loto_20260709.py
outputs/conformal_risk_loto_v61_from_v52_m20_20260709.csv
outputs/conformal_risk_loto_v62_from_v53_m20_20260709.csv
outputs/conformal_risk_loto_recall_sweep_m20_20260709.csv
outputs/conformal_risk_loto_weight_sweep_v53_m20_20260709.csv
```

设置：

```text
held out one task from {narrativeqa, multifieldqa_en, 2wikimqa}
calibrate threshold on the other two tasks
evaluate danger recall / precision / quality on held-out task
```

target recall 0.80, v53-based 结果：

```text
heldout narrativeqa:
  heldout danger recall = 1.000
  heldout trigger rate  = 95.0%

heldout multifieldqa_en:
  heldout danger recall = 1.000
  heldout trigger rate  = 70.0%

heldout 2wikimqa:
  heldout danger recall = 0.636
  heldout trigger rate  = 60.0%
```

扫 `target_recall in {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}` 和多组 `gap2/gap3/top` 权重后，最小 held-out recall 仍然卡在 `0.636`。

结论：

```text
global conformal threshold is not robust enough across heterogeneous QA families.
```

这不是失败，而是给论文方法一个更合理的边界：

```text
RiskKV should calibrate conformal risk by task/risk family,
not use a single global threshold for all QA tasks.
```

因此当前 v61/v62 可以作为 proof-of-concept；最终主方法应写成 risk-family conformal calibration：

```text
tau_g = Quantile({r(x_i, a_i): danger_i=1, g(x_i)=g}, 1-alpha)
```

其中 `g` 可以是 task family、benchmark family、context length bucket 或 verifier family。

## Benefit-calibrated conformal gate

仅用 consistency danger 做标签仍然不完全等价于“运行 verifier 会提升最终任务分数”。因此新增一个 utility-oriented 校准标签：

```text
benefit = 1 if score(reference_policy) - score(base_policy) > 0.01
```

这个标签只用于离线校准，不在推理时使用 gold answer。它回答的问题是：

```text
Which primary sparse actions are worth paying for counterfactual verification?
```

按 task family 分别校准，target recall = 0.80：

```text
narrativeqa:
  tau = 0.8441593191391078
  benefit recall = 1.00
  trigger rate = 90%

multifieldqa_en:
  tau = 0.8763454987254873
  benefit recall = 1.00
  trigger rate = 65%

2wikimqa:
  tau = 0.946880252247314
  benefit recall = 1.00
  trigger rate = 40%
```

组合 stitched 估计：

```text
v52 actual:
  score = 0.373143
  KV    = 58.17%
  online= 2.720s
  check = 16.56%

v63 benefit-conformal estimate:
  score = 0.374025
  KV    = 56.17%
  online= 2.701s
  check = 10.63%

v53 actual:
  score = 0.375890
  KV    = 62.89%
  online= 2.736s
  check = 16.56%

v64 benefit-conformal estimate:
  score = 0.376772
  KV    = 60.89%
  online= 2.717s
  check = 10.63%
```

这比 v61/v62 更适合作为下一版主方法候选：

```text
v61/v62: danger-recall conformal gate
v63/v64: benefit-recall conformal gate
```

论文里可以把它表述为 utility-calibrated risk control over memory actions：

```text
calibrate not only which actions are unsafe,
but which actions are worth paying an additional counterfactual check for.
```

## 当前实验队列

新增配置：

```text
configs/riskkv_task_policy_v61_conformal_counterfactual_20260709.json
configs/riskkv_task_policy_v62_conformal_counterfactual_qasper_full_20260709.json
configs/riskkv_task_policy_v63_benefit_conformal_counterfactual_20260709.json
configs/riskkv_task_policy_v64_benefit_conformal_counterfactual_qasper_full_20260709.json
```

新增脚本：

```text
scripts/run_riskkv_v61_v62_conformal_counterfactual_m20_20260709.sh
scripts/watch_and_launch_v61_v62_after_v59_v60_20260709.sh
scripts/run_riskkv_v63_v64_benefit_conformal_m20_20260709.sh
scripts/watch_and_launch_v63_v64_after_v55_v56_20260709.sh
```

服务器状态：

```text
v61/v62 watcher 已启动。
它会等待 v59/v60 完成，再等待两张空 GPU，然后自动启动。

v63/v64 watcher 已启动。
它会等待 v55/v56 完成，再等待两张空 GPU，然后自动启动。v63/v64 优先级更高，因为 stitched 估计同时改善质量、KV、online 和 verifier check rate。

m100 stable watcher 已启动：

```text
scripts/watch_and_launch_m100_if_m50_stable_20260709.sh
```

它会等待 m50 的 full/v37/v52/v53 全部完成，然后读取 m50 分数；只有 consistency policy 接近或优于 full/v37 时才启动 m100，避免浪费 4 张 GPU 复验已失败的策略。
```

## 论文意义

这一版能把方法主线从经验 router 提升为：

```text
conformal risk control over KV memory actions
```

相比“选择哪些 blocks”，这个表述更强：

```text
1. action 是完整 KV 行为，包括 sparse, expanded sparse, full fallback
2. risk 是从 memory-action score geometry 中校准出来的
3. counterfactual verification 只在风险证书不足时触发
4. 校准目标可以报告为 danger recall / trigger rate / quality-cost Pareto
```

如果 v61/v62 真实跑分接近 stitched 估计，这会比 v52/v53 更适合作为论文主方法。
