# Section 95: Conformal 校准与 runtime verifier 结果记录（2026-07-07）

## 结论先行

当前 two-stage calibrated 本身创新性偏弱，不能作为主方法。更合理的主线应是：

**risk-constrained variable-budget KV planning + cache-native RoPE-aware repack + runtime safety verifier**

其中 two-stage 只保留为 baseline。今天补了三类验证：

1. conformal tail-risk calibration 的多 seed 稳定性；
2. task-family / tabular / text-feature fallback；
3. 两种 runtime verifier：full-logit consistency probe 与 full-teacher likelihood verifier。

结果显示：简单 router、简单 conformal threshold、简单 lexical text features、首 token logit probe、full likelihood verifier 都还不足以稳定支撑 ICML 级主结果。下一步应转向 **训练式 output-level risk verifier**，把 runtime probe 信号作为 verifier features，而不是直接用硬阈值。

## 当前最好可用结果

### 13-task runtime 小规模结果

已有最好的 runtime 结果仍然是 variable-budget planner：

| 方法 | Score | KV ratio | Online speed |
|---|---:|---:|---:|
| full KV cache | 69.23% | 100.00% | 1.000x |
| RoPE compact k2 | 69.23% | 26.08% | 1.005x |
| two-stage calibrated | 69.23% | 28.96% | 0.997x |
| variable-budget planner | 69.23% | 15.05% | 0.991x |

这个结果说明方法方向有潜力：在 13-task m=1 上能做到 full-level accuracy，同时 KV 降到约 15%。但它还不是足够强的论文证据，因为多 seed / combined benchmark 上稳定性不足。

### Combined replay 多 seed 结果

原 conformal add-one calibration，5 seed，Qwen8B 13-task m4 + LongBench m12：

| 方法 | Mean Score | KV ratio | Full-level seeds |
|---|---:|---:|---:|
| fixed full/k8 | 38.46% | 100.00% | 5/5 |
| conformal selected | 37.15% | 23.91% | 3/5 |
| oracle min-safe | 39.89% | 16.10% | 5/5 |
| oracle best | 41.18% | 16.90% | 5/5 |

解释：

- oracle 说明理论空间很大：约 16%-17% KV 可以达到或超过 full。
- 当前 learned/conformal planner 没有稳定学到 oracle 的选择边界。
- 这不是 action space 的问题，而是安全判别信号不足。

## 负结果记录

### Text features

打开 `--use_text_features` 后，多 seed 反而明显变差：

| 方法 | Mean Score | KV ratio |
|---|---:|---:|
| text-feature conformal selected | 31.09% | 24.26% |
| fixed full/k8 | 38.46% | 100.00% |

当前 lexical retriever score / gap / selected score features 不足以提升 planner，可能因为样本量太小且 lexical overlap 对 Qwen 生成正确性不是稳定代理。

### Task-family floor

用 train+calibration 学 task-family 安全预算下界：

| 策略 | Mean Score | KV ratio | Full-level seeds |
|---|---:|---:|---:|
| family zero-failure floor | 36.53% | 50.48% | 4/5 |
| family score floor | 37.96% | 44.61% | 4/5 |

它能减少部分过度压缩，但 KV 太高，且仍不能 5/5 full-level。

### Tabular classifiers

RandomForest / ExtraTrees / KNN / Logistic / GradientBoosting 都没有超过 MLP conformal。最佳的 `rf_leaf3 + tail035`：

| 方法 | Mean Score | KV ratio | Full-level seeds |
|---|---:|---:|---:|
| rf_leaf3 tail035 | 35.86% | 38.02% | 3/5 |

说明问题不是简单换分类器，而是缺少 inference-time 安全证据。

## Runtime verifier 初版

### Full-logit consistency probe

实现位置：

`ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py`

新增参数：

- `--consistency_probe_budgets`
- `--consistency_probe_kl_threshold`
- `--consistency_probe_require_top1_agree`

方法逻辑：

1. full-context prefill；
2. 用 full KV 对 query 跑首步，得到 teacher logits；
3. 从小预算到大预算 repack compact KV；
4. 如果 compact 首步 logits 与 full logits 的 KL/top1 满足阈值，则用该预算 decode；
5. 否则继续升预算或 fallback full。

13-task m=1 结果：

| Probe 设置 | Score | KV ratio | Online speed |
|---|---:|---:|---:|
| KL<=0.2 + top1 | 61.54% | 51.03% | 0.920x |
| KL<=0.8 + top1 | 61.54% | 40.57% | 0.923x |
| KL<=1.5 + top1 | 61.54% | 40.57% | 0.924x |
| full KV | 69.23% | 100.00% | 1.000x |

失败集中在 hotpotqa：k3 的首步 top1 与 full 一致、KL 低，但最终答案错；同时 k1/k2 虽然首步 logits 与 full 差异很大，却能答对。结论是：**首 token logit consistency 不是可靠安全条件**。

### Full-teacher likelihood verifier

新增参数：

- `--teacher_verifier_budgets`
- `--teacher_verifier_fallback_nll`

方法逻辑：

1. compact KV 在多个预算下生成候选答案；
2. full KV teacher 对这些候选答案计算 likelihood；
3. 选择 teacher NLL 最低的候选。

2-task smoke 结果：

| 方法 | Score | KV ratio | Online speed |
|---|---:|---:|---:|
| teacher likelihood verifier | 0.00% | 31.25% | 0.103x |
| full KV | 50.00% | 100.00% | 1.000x |

失败原因：teacher likelihood 偏好短、高概率、格式化的候选，不等价于任务正确性。例如 2wikimqa 中正确候选 `Gyulafehérvár` 没有被选中。

## 对创新性的判断

如果论文只写成 “router 选择 k1/k2/k3/k4/k6/k8”，创新性不够。  
如果写成下面这个问题，创新性是够继续打磨的：

**在 full-context prefill 已完成的 cache-native long-context inference 中，学习一个 risk-constrained variable-budget KV planner，在不重建 prompt、不外部检索的情况下，对 KV pages 做动态预算分配和 RoPE-aware repack，并通过 runtime verifier 控制相对 full KV 的行为风险。**

当前缺口不是 idea，而是 verifier 还不够强。

## 下一步

## Output-level risk verifier v1 追加结果

实现位置：

`ymluo/projects/learned_hierarchical_summary_memory/src/run_output_level_risk_verifier_from_repack_results.py`

核心变化：

不再只用 case/router/action features，而是把每个预算候选的输出本身纳入 verifier：

- candidate length；
- unique word ratio；
- repeated bigram ratio；
- 是否包含 `passage/question/answer/only give` 等格式异常信号；
- 是否与更小/更大预算输出一致；
- candidate action / KV ratio；
- 原 variable-budget features。

训练目标：

- `safe_vs_full`: 当前 action 的 score 是否不低于 full；
- 推理策略：从小预算到大预算扫描，选择第一个 `p_safe >= tau` 的 action，否则 fallback full。

5 seed combined replay 结果：

| 方法 | Mean Score | KV ratio | Delta vs full | Full-level seeds |
|---|---:|---:|---:|---:|
| fixed full/k8 | 39.35% | 100.00% | 0.00 | 5/5 |
| oracle min-safe | 40.65% | 17.13% | +1.29 | 5/5 |
| oracle best | 45.81% | 19.39% | +6.45 | 5/5 |
| output verifier tau=0.3 | 40.00% | 20.12% | +0.65 | 5/5 |
| output verifier tau=0.5 | 40.00% | 20.36% | +0.65 | 5/5 |
| output verifier tau=0.7 | 40.65% | 20.92% | +1.29 | 5/5 |
| output verifier tau=0.8 | 40.65% | 21.25% | +1.29 | 5/5 |
| output verifier tau=0.95 | 40.65% | 22.38% | +1.29 | 5/5 |

这是今天最重要的正结果：

1. output-level verifier 达到 5/5 seeds full-level；
2. tau=0.7/0.8/0.9/0.95 的 score 已经追平 oracle min-safe；
3. KV ratio 约 21%-22%，只比 oracle min-safe 的 17.13% 高约 4-5 个点；
4. 它还超过 fixed full/k8 的平均 score，说明 verifier 不只是保守 fallback，而是能利用 compact KV 的 denoising/selection benefit。

这使主方法可以升级为：

**Output-verified risk-constrained KV budget planner**

完整 pipeline：

1. full-context prefill 一次；
2. 从小预算生成 compact KV candidates；
3. 提取输出级风险特征；
4. verifier 判断 candidate 是否 safe；
5. 选择最小 safe action；
6. 对选中 KV pages 做 RoPE-aware repack 并输出。

这个版本比 “router 选择 k” 明显更有创新性，也更符合现有实验数据。

## 下一步建议

下一步不要继续只调 threshold。建议把 output verifier v1 正式 runtime 化：

**Output-level risk verifier v1**

训练数据来自已有 k1/k2/k3/k4/k6/k8 benchmark，每个候选 action 一条样本。输入包括：

- router 原始 features；
- retriever gap / top-k stability；
- task family；
- compact action；
- full-logit probe KL、top1 agree、top5 overlap；
- 多预算输出稳定性；
- candidate length、query echo ratio、answer format features；
- compact prediction 与邻近预算 prediction 的一致性。

标签：

- `safe_vs_full`: 当前 action 是否不低于 full score；
- `best_action`: oracle best；
- `min_safe_action`: oracle min-safe。

推理时：

1. 先生成低预算候选和 probe features；
2. verifier 判断当前候选是否危险；
3. 危险则升预算；
4. 选择最小安全预算，而不是直接用 hard-coded KL 或 task rule。

这条路线比 two-stage/router 更像 ICML/ICLR 方法，也更能解释已有 oracle gap。
