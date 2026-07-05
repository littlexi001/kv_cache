# Oracle Regret Memory Planner V7 结果（2026-07-03）

## 做了什么

这轮实现了一个面向论文主线的 V7 闭环：

```text
oracle regret labels
+ causal page influence labels
+ held-out mixed benchmark
+ query-side KV gather speed
+ range-SDPA speed closure
```

新增代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_oracle_regret_memory_planner_v7.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_oracle_regret_memory_planner_v7_server.sh
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_range_sdpa_speed_closure_v7_server.sh
```

V7 的输出文件包括：

```text
causal_page_influence_labels.csv
  每个 page 的 ablation delta-NLL、causal label、page features。

oracle_regret_labels.csv
  每个 task / budget / SLA / method 的 oracle mode、objective regret、correct/loss/token/speed regret、Pareto 标记。

strategy_results.csv
  full / recent / lexical / learned causal / set utility / oracle causal 的真实 KV gather 结果。

planner_results.csv
  risk_calibrated_progressive_planner_v7 的 held-out 结果。

learned_page_model.json
  用 train split 的 causal page labels 训练出来的小 page scorer。

learned_plans.json
  根据 train split oracle regret 学到的每个 variant / budget / SLA 的 expert 顺序。
```

## V7 mixed benchmark

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_memory_planner_v7_4x_b135_20260703_v7_regretfb
```

配置：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
variants = casual_recent, temporal_fact, multihop_bridge, summary_theme, compare_score
tasks_per_variant = 4
train/test = 2/2 per variant
distractor_pages = 16
budgets = 1,3,5 pages
max_ablate_pages = 12
```

规模：

```text
tasks = 20
page_rows = 404
strategy_rows = 360
oracle_regret_rows = 1080
planner_rows = 180
positive_page_rate = 19.3%
model_valid = 1
```

## Held-out test 结果

下面是 test split 的 ALL 汇总。

| method | budget | Acc | PPL | online sec | total sec | keep frac | evidence hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_kv_cache | 1/3/5 | 60.0% | 11.54 | 0.295 | 0.354 | 100.0% | 100.0% |
| set_utility_kv_gather_v7 | 1 | 80.0% | 8.86 | 0.251 | 0.310 | 10.3% | 60.0% |
| set_utility_kv_gather_v7 | 3 | 80.0% | 7.49 | 0.258 | 0.317 | 18.0% | 90.0% |
| set_utility_kv_gather_v7 | 5 | 80.0% | 7.66 | 0.270 | 0.329 | 26.5% | 100.0% |
| learned_causal_kv_gather_topk | 3 | 70.0% | 7.49 | 0.268 | 0.328 | 18.4% | 90.0% |
| lexical_kv_gather_topk | 1 | 80.0% | 8.86 | 0.260 | 0.320 | 10.3% | 60.0% |
| recent_kv_gather_topk | 1 | 80.0% | 11.20 | 0.268 | 0.328 | 11.4% | 40.0% |

当前最好的单步 deployable 方法是：

```text
set_utility_kv_gather_v7, budget=3 或 5
```

它相比 full：

```text
accuracy: 60% -> 80%
PPL:      11.54 -> 7.49 / 7.66
online:   0.295s -> 0.258s / 0.270s
KV kept:  100% -> 18.0% / 26.5%
```

这说明在这个 mixed synthetic suite 上，固定 full context 不是最优策略；page-level sparse memory 既能减少干扰，也能降低 query-side latency。

## Oracle regret label 分布

test split 上，deployable oracle mode 不是单一方法。

| SLA | full | recent | lexical | learned causal | set utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality | 66 | 48 | 24 | 30 | 12 |
| balanced | 66 | 24 | 30 | 42 | 18 |
| speed | 36 | 36 | 30 | 54 | 24 |

这个分布是关键证据：

```text
No single KV strategy is Pareto-optimal.
```

同一个 mixed workload 里，full、recent、lexical、learned causal、set utility 都在某些样本/SLA 下成为 oracle。这个结果支持把论文主线写成 memory planning，而不是单个 KV compression expert。

## 当前 planner 的问题

`risk_calibrated_progressive_planner_v7` 目前还没有达到理想效果：

| budget | SLA | Acc | PPL | online sec | keep frac |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | quality | 60.0% | 12.05 | 0.399 | 91.2% |
| 3 | quality | 60.0% | 11.88 | 0.322 | 91.4% |
| 5 | quality | 60.0% | 11.55 | 0.321 | 92.4% |

失败主要来自 `compare_score` 和 `multihop_bridge` 的 held-out 样本。planner 在这些样本上接受了 full context，因为 full 的模型 margin 很高；但 full 实际会被 decoy/conflict page 干扰。

这说明：

```text
模型自身 margin 不是可靠风险校准信号。
full context 也不能被当作永远安全的 fallback。
```

下一版 planner 需要学习：

```text
full-context decoy risk
typed coverage risk
multi-hop coverage risk
oracle regret under SLA
```

也就是说，planner 不能只问“模型是否自信”，还要问“这个任务类型下 full 是否可能因为 decoy 过度自信”。

## Range-SDPA 速度闭环

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/range_sdpa_speed_closure_v7_20260703_v7_smoke
```

配置：

```text
task_variant = chain_story_conflict
contexts = 10k, 20k
tasks_per_length = 4
sparse_attention_impl = range_sdpa
mode = full vs chain_typedhier_role_auto_p1
typed_record_format = answerline_summary
typed_record_answer_override = true
skip_lm_answer_when_override = true
```

结果：

| Context | Mode | Acc | Query PPL | Eval sec | Kept frac | Kept tokens | Evidence hit | Decoy hit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 100% | 16.22 | 10.09 | 100.0% | 10060.6 | 100% | 100% |
| 10k | chain_typedhier_role_auto_p1 | 100% | 17.11 | 4.70 | 7.48% | 752.3 | 100% | 0% |
| 20k | full | 100% | 17.17 | 11.79 | 100.0% | 20060.3 | 100% | 100% |
| 20k | chain_typedhier_role_auto_p1 | 100% | 17.76 | 4.80 | 3.80% | 763.3 | 100% | 0% |

速度：

```text
10k: 10.09 / 4.70 = 2.15x
20k: 11.79 / 4.80 = 2.46x
```

这个结果是当前最干净的 range-SDPA speed closure：

```text
typed route 保持 100% accuracy；
decoy hit 从 100% 降到 0%；
只保留 3.8%-7.5% tokens；
range_sdpa eval latency 比 full 快 2.1x-2.5x。
```

需要注意：这里 `typed_record_answer_override=true`，所以它验证的是 typed sidecar reader + range-SDPA sparse memory 的系统路径，不是纯 LM 自己从 sparse KV 中生成答案。

## 当前判断

这轮已经把四件事打通：

```text
1. causal page influence labels 可以自动生成。
2. oracle regret labels 可以按 SLA 生成。
3. held-out mixed benchmark 可以跑，并能显示 no single strategy is best。
4. range-SDPA 可以在 typed long-range task 上给出真实 wall-clock speedup。
```

最强正向结果：

```text
set_utility_kv_gather_v7:
  held-out mixed test 上 80% acc、PPL 7.49、18% KV、online 0.258s。

chain_typedhier_role_auto_p1 + answerline_summary + range_sdpa:
  10k/20k 上 100% acc、0% decoy hit、2.1x-2.5x eval speedup。
```

当前最弱环节：

```text
risk_calibrated_progressive_planner_v7 的 risk gate 还不够好。
```

下一步建议：

```text
1. 用 oracle_regret_labels.csv 训练一个真正的 regret predictor，而不是只按 variant 平均排序。
2. 加入 full-context decoy-risk 特征，避免把 full 当默认安全 fallback。
3. 把 margin calibration 换成 held-out reliability calibration，例如 temperature / isotonic / conformal abstention。
4. 扩大 mixed benchmark，每类至少 20-50 个 held-out 样本。
5. 把 V7 set_utility / learned causal scorer 接到 range_sdpa，而不是只接 KV gather。
```

## 扩展实验：mixed 5x20

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_memory_planner_v7_20x_b1358_20260703_v7_expanded_5x20
```

配置：

```text
variants = 5
tasks_per_variant = 20
train/test = 10/10 per variant
distractor_pages = 32
budgets = 1,3,5,8
tasks = 100
page_rows = 3620
oracle_regret_rows = 7200
positive_page_rate = 19.1%
```

test ALL：

| method | budget | Acc | PPL | online sec | keep frac | evidence hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_kv_cache | all | 50.0% | 13.14 | 0.267 | 100.0% | 100.0% |
| set_utility_kv_gather_v7 | 3 | 40.0% | 17.78 | 0.173 | 9.9% | 94.0% |
| set_utility_kv_gather_v7 | 5 | 36.0% | 17.55 | 0.173 | 14.6% | 100.0% |
| set_utility_kv_gather_v7 | 8 | 36.0% | 18.10 | 0.173 | 22.2% | 100.0% |
| learned_causal_kv_gather_topk | 3 | 36.0% | 21.19 | 0.173 | 10.4% | 84.0% |
| lexical_kv_gather_topk | 3 | 38.0% | 20.88 | 0.173 | 10.1% | 80.0% |

这个扩展结果没有复现 4-shot smoke 里 `set_utility` 全面超过 full 的结论。主要变化是：样本数和 distractor 增大后，单步 sparse expert 虽然能保住 evidence hit，但 answer accuracy 和 PPL 不稳定。

更重要的是 oracle 上限仍然明显好于 full：

| SLA | budget | Oracle Acc | Oracle PPL | keep frac | online sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality | 1 | 72.0% | 10.23 | 64.3% | 0.231 |
| quality | 3 | 68.0% | 8.88 | 58.5% | 0.224 |
| speed | 1 | 72.0% | 10.94 | 47.3% | 0.214 |
| speed | 3 | 68.0% | 9.16 | 45.8% | 0.210 |

这说明：

```text
单个 set_utility expert 不够稳；
但按样本选择 expert 的 oracle 仍然同时超过 full 的 Acc/PPL/online。
```

因此 mixed 5x20 支持的不是“set_utility 已经解决问题”，而是：

```text
oracle regret planner 有真实上限；
现在缺的是把 oracle choice 学出来。
```

## 扩展实验：hard-noise 5x10

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_memory_planner_v7_10x_b358_20260703_v7_hardnoise_5x10
```

配置：

```text
variants = 5
tasks_per_variant = 10
train/test = 5/5 per variant
distractor_pages = 64
budgets = 3,5,8
tasks = 50
page_rows = 3410
oracle_regret_rows = 2700
positive_page_rate = 8.8%
```

test ALL：

| method | budget | Acc | PPL | online sec | keep frac | evidence hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_kv_cache | all | 60.0% | 12.45 | 0.425 | 100.0% | 100.0% |
| set_utility_kv_gather_v7 | 3 | 48.0% | 14.23 | 0.173 | 5.1% | 96.0% |
| set_utility_kv_gather_v7 | 5 | 48.0% | 14.09 | 0.173 | 7.7% | 100.0% |
| set_utility_kv_gather_v7 | 8 | 40.0% | 14.35 | 0.174 | 11.7% | 100.0% |
| learned_causal_kv_gather_topk | 5 | 44.0% | 15.50 | 0.173 | 8.3% | 96.0% |
| lexical_kv_gather_topk | 5 | 48.0% | 15.27 | 0.173 | 7.8% | 88.0% |

hard-noise 下，单步 sparse expert 的质量低于 full，但速度和 KV 成本显著更低：

```text
full online = 0.425s
set_utility online = 0.173s
speedup = 2.45x
KV kept = 5.1%-11.7%
```

oracle 上限：

| SLA | budget | Oracle Acc | Oracle PPL | keep frac | online sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| quality | 3 | 76.0% | 7.45 | 46.8% | 0.285 |
| quality | 5 | 76.0% | 7.49 | 51.9% | 0.296 |
| speed | 3 | 76.0% | 7.73 | 39.3% | 0.265 |
| speed | 5 | 76.0% | 7.75 | 40.8% | 0.265 |

这是更强的 planner evidence：

```text
full = 60% acc / PPL 12.45 / online 0.425s
oracle speed planner = 76% acc / PPL 7.73 / online 0.265s / 39.3% KV
```

也就是说，在 hard-noise 分布上，单个 sparse expert 不够，但“正确选择 expert”的空间很大。

## 扩展实验：range-SDPA 10k/20k/39k

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/range_sdpa_speed_closure_v7_20260703_v7_expanded_range
```

配置：

```text
task_variant = chain_story_conflict
contexts = 10k,20k,39k
layouts = e05_d90,e20_d80,e35_d70
tasks_per_layout = 4
total = 36 tasks per mode
sparse_attention_impl = range_sdpa
mode = full vs chain_typedhier_role_auto_p1
typed_record_format = answerline_summary
typed_record_answer_override = true
skip_lm_answer_when_override = true
```

结果：

| Context | Mode | Acc | Cal acc | Query PPL | Eval sec | Kept frac | Kept tokens | Evidence hit | Decoy hit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 100% | 100% | 16.66 | 5.08 | 100.0% | 10060 | 100% | 100% |
| 10k | typed range-SDPA | 100% | 100% | 18.30 | 4.31 | 7.60% | 765 | 100% | 0% |
| 20k | full | 100% | 100% | 16.34 | 6.54 | 100.0% | 20060 | 100% | 100% |
| 20k | typed range-SDPA | 100% | 100% | 17.79 | 4.29 | 3.76% | 755 | 100% | 0% |
| 39k | full | 100% | 100% | 14.34 | 9.15 | 100.0% | 39061 | 100% | 100% |
| 39k | typed range-SDPA | 100% | 100% | 14.04 | 4.34 | 1.94% | 758 | 100% | 0% |

速度：

```text
10k: 5.08 / 4.31 = 1.18x
20k: 6.54 / 4.29 = 1.53x
39k: 9.15 / 4.34 = 2.11x
```

这个扩展结果比 mixed V7 更稳定：

```text
typed range-SDPA 在 10k/20k/39k 全部保持 100% accuracy；
decoy hit 全部从 100% 降到 0%；
KV kept 随长度从 7.6% 降到 1.94%；
速度收益随上下文长度变大，在 39k 达到 2.11x；
39k 上 PPL 还略好于 full。
```

## 扩展后的总判断

扩展实验把结论分得更清楚：

```text
1. typed range-SDPA 路径是目前最稳定的正结果。
   它在 chain/conflict 长程任务上跨 10k/20k/39k 都稳定。

2. V7 set_utility 单步 expert 在小样本上好，但扩大后不稳定。
   它能省 KV、提速，但 Acc/PPL 不能稳定超过 full。

3. oracle regret 上限很强。
   mixed 5x20 和 hard-noise 5x10 都显示：如果能选对 expert，
   可以同时超过 full 的 Acc、PPL、online latency。

4. 当前最大问题不是没有好 expert，而是 planner 没学会何时用哪个 expert。
   特别是 full context 有 decoy-risk，不能作为默认安全 fallback。
```

下一步应该从“继续调单个 set_utility”转向：

```text
train regret predictor:
  input = query features + page-score stats + causal coverage estimate + full-decoy-risk features
  target = oracle_regret_labels.csv 中的 deployable_oracle_mode / objective_regret

then:
  planner 直接预测每个 expert 的 regret，
  选择最低 expected regret 的 plan，
  low confidence 才 fallback 到 full 或 typed verifier。
```
