# Calibrated Extension Skip-Gate 设计草案（2026-06-29）

## 背景

当前主方法：

```text
Pairwise-CIC
+ online blockwise selection
+ conditional validation-prior horizon-anchor rescue gate
```

质量证据已经覆盖 hard-topic、War/Monte、Needle-style、RULER-style synthetic smoke。主要短板仍是 gate 成本。

`docs/pcic_extension_waste_2026_06_29.md` 的后验分析显示，许多 extension 最终没有改变 early initial combo：

| case | avoidable extension fraction |
| --- | ---: |
| hard | 0.437 |
| monte | 0.400 |
| needle | 0.462 |
| ruler multi-needle | 0.375 |
| ruler topic-switch | 0.669 |
| ruler variable | 0.429 |

这说明速度提升不只能靠 fused kernel，还可以减少不必要的 extension 次数。

## 目标

设计一个 online skip-gate，在只看 early sentinel / calibration / memory 特征时判断：

```text
Should we extend this block from early sentinel to full horizon?
```

目标不是盲目 early-exit，而是预测：

```text
extension 后最终 selected combo 是否仍等于 early selected combo
```

## 候选特征

只使用 extension 前已经可见的字段：

```text
sentinel_cascade_initial_selected_combo
sentinel_cascade_initial_route
sentinel_cascade_initial_best_margin
sentinel_cascade_initial_pairwise_delta
memory_combo
min_loss_combo
memory_delta_loss
min_loss_delta_loss
lazy_pairwise_calib_gap
memory_risk_max_loss_gap
min_loss_risk_max_loss_gap
sentinel_cascade_anchor_combos
```

派生特征：

```text
initial_selected_is_anchor
initial_selected_is_memory
initial_selected_is_min_loss
abs(initial_pairwise_delta)
initial_margin / uncertainty_floor
memory_vs_min_loss_disagree
```

## 规则方向

### Rule A：conservative no-change predictor

只在多条件同时满足时 skip extension：

```text
if initial_selected == memory_combo == validation_anchor
and abs(initial_pairwise_delta) <= eps_pairwise
and memory_risk_max_loss_gap <= tau_risk:
    skip extension
```

适合 Needle / RULER easy-regime，但可能错过 hard-topic delayed-win block。

### Rule B：learned shallow gate

用已有 block 级 CSV 做二分类：

```text
label = 1 if final_combo == initial_combo else 0
```

训练一个极小 logistic / decision tree，输出是否 skip。为了论文可解释性，优先 decision tree，再手工转成规则。

### Rule C：risk-aware budget allocator

不是二分类，而是动态分配 horizon：

```text
easy block: initial only
medium block: top2 extension
hard block: top2 + validation anchor extension
```

这更符合 paper 主线：rescue gate 不是固定长度，而是 risk-calibrated horizon budget allocation。

## 必须避免

前面已经有负结果：

- anchor-match early-exit；
- low-spread early-exit；
- dense mixed path。

所以新 skip-gate 不能只是单一 heuristic，必须用跨任务后验分析校准，并报告 false-skip 风险。

## 下一步实验

1. 用 `docs/pcic_extension_waste_blocks_2026_06_29.csv` 训练/搜索简单规则。
2. 在 hard / RULER-style / Needle 上离线评估 false skip。
3. 只把 zero-false-skip 或极低 false-skip 的规则接入 online runner。
4. 再跑主线实验，报告：
   - ΔPPL；
   - gate_s；
   - skipped extension blocks；
   - false skip count。

## Rule search 初步结果

搜索脚本：`scripts/search_pcic_skip_gate_rules.py`

结果文档：`docs/pcic_skip_gate_rule_search_2026_06_29.md`

当前样本上最简洁的 zero false-skip 规则是：

```text
if initial_selected_combo in validation_prior_anchors
and sentinel_horizon_gain_ratio <= 0:
    skip extension
```

后验结果：

| rule | selected blocks | saved seconds | saved fraction | selected cases |
| --- | ---: | ---: | ---: | --- |
| `anchor_hit_and_ratio_le_0` | 6 | 29.856 | 0.285 | `needle:1;needle:3;ruler_multineedle:0;ruler_variable:2;ruler_topicswitch:1;ruler_topicswitch:2` |

解释：

- 该规则没有覆盖 Hard-topic 的 delayed-win blocks，因此当前样本上没有 false skip；
- 它主要覆盖 Needle / RULER-style easy-regime；
- 它节省的是 extension 部分的后验上界，不是端到端真实 speedup；
- 下一步可以把该规则作为保守 online skip-gate 接入 runner，然后实测 ΔPPL / gate_s。

## Online 接入结果

接入参数：

```text
--sentinel_cascade_skip_anchor_nonpositive_gain true
```

汇总文档：`docs/pcic_skip_gate_anchor_gain_online_2026_06_29.md`

结果：

| task | ΔPPL change | gate_s change | skipped | same combos |
| --- | ---: | ---: | ---: | --- |
| hard | 0.000000 | -0.952 | 0 | True |
| needle | 0.000137 | +18.868 | 3 | False |
| ruler multi-needle | 0.000000 | +4.726 | 1 | True |
| ruler variable | 0.000026 | +12.672 | 2 | False |
| ruler topic-switch | 0.000163 | +24.987 | 3 | False |

结论：

- 后验 zero-false-skip 规则在线后在 corrected gate 口径下能省 probe；
- 但多个 easy-regime 任务出现 combo 改变；
- 部分任务 combo 被提前固定为 anchor，质量有小幅退化；
- 该规则不进入主方法，默认保持关闭；
- 下一步 skip-gate 必须优化质量稳定性，改成更强的 learned/calibrated gate，而不能只用单条后验规则。

Corrected gate 补充：

`docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md` 修正了旧 `gate_s` 漏算 initial candidates 的问题。修正后该规则在 Needle / RULER-style 上确实降低 corrected gate，但质量不够稳。
