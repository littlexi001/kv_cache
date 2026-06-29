# Anchor Nonpositive-Gain Skip-Gate Online Eval（2026-06-29）

## 规则

```text
if initial_selected_combo in validation_prior_anchors
and sentinel_horizon_gain_ratio <= 0:
    skip extension
```

原始 CSV：`docs/pcic_skip_gate_anchor_gain_online_2026_06_29.csv`

## 结果表

| case | task | mode | avg_delta_ppl | method/base | gate_s | extended | early | skipped | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hard_base | hard | base | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard_skip | hard | skip | -0.049633 | 4.151 | 52.696 | 4 | 0 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| needle_base | needle | base | -0.000166 | 3.401 | 39.909 | 4 | 0 | 0 | `2,0/2,0/2,0,7,12/2,0` |
| needle_skip | needle | skip | -0.000029 | 4.559 | 58.777 | 1 | 3 | 3 | `2,0/2,0/2,0/2,0` |
| ruler_multineedle_base | ruler_multineedle | base | 0.000074 | 2.781 | 16.650 | 3 | 0 | 0 | `2,0/2,0/2,0` |
| ruler_multineedle_skip | ruler_multineedle | skip | 0.000074 | 3.301 | 21.376 | 2 | 1 | 1 | `2,0/2,0/2,0` |
| ruler_variable_base | ruler_variable | base | -0.000564 | 2.441 | 13.249 | 3 | 0 | 0 | `0,13/2,0/2,0` |
| ruler_variable_skip | ruler_variable | skip | -0.000538 | 3.801 | 25.921 | 1 | 2 | 2 | `2,0/2,0/2,0` |
| ruler_topicswitch_base | ruler_topicswitch | base | 0.000139 | 2.086 | 9.931 | 3 | 0 | 0 | `0,13/2,0/2,0` |
| ruler_topicswitch_skip | ruler_topicswitch | skip | 0.000302 | 4.690 | 34.918 | 0 | 3 | 3 | `2,0/2,0/2,0` |

## Base vs Skip

| task | ΔPPL change | gate_s change | method/base change | skipped | same combos |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 0.000000 | -0.952 | 0.007 | 0 | True |
| needle | 0.000137 | 18.868 | 1.158 | 3 | False |
| ruler_multineedle | 0.000000 | 4.726 | 0.520 | 1 | True |
| ruler_variable | 0.000026 | 12.672 | 1.360 | 2 | False |
| ruler_topicswitch | 0.000163 | 24.987 | 2.604 | 3 | False |

## 解释

- 该规则是保守 skip-gate 候选，目标是减少 easy-regime 中无效 extension。
- 如果 ΔPPL 不退化且 gate_s 下降，可进入下一轮更大样本验证。
- 如果质量退化或 gate 不降，则说明后验规则没有在线泛化，应退回设计阶段。

## Corrected gate 口径补充

后续发现旧 `gate_s` 对 extended cascade 漏算了未进入 extension 的 initial candidates，因此 base run 的 gate 被低估。修正口径见：`docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md`

corrected 结果：

| task | ΔPPL change | corrected gate_s change | corrected method/base change | skipped | same combos |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 0.000000 | -1.591 | +0.009 | 0 | True |
| needle | +0.000137 | -14.282 | -0.792 | 3 | False |
| ruler multi-needle | 0.000000 | -3.647 | -0.341 | 1 | True |
| ruler variable | +0.000026 | -5.510 | -0.554 | 2 | False |
| ruler topic-switch | +0.000162 | -4.625 | -0.519 | 3 | False |

修正后结论更精确：

- skip-gate 的确能降低 corrected gate 成本；
- 但它会改变 Needle / RULER variable / RULER topic-switch 的 combo，并带来小幅质量退化；
- 因此该规则仍不进入主方法，但它不再是“速度也失败”，而是“速度有效、质量不够稳”。
