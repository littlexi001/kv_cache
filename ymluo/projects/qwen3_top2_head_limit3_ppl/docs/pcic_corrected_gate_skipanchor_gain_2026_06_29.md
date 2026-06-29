# Corrected Cascade Gate Seconds（2026-06-29）

## 背景

旧 `gate_seconds` 在 extended cascade 中只统计 extended candidate set，漏掉了未进入 extension 的初始候选 probe。因此 extended runs 的 gate 成本被低估。本分析只读已有 CSV，从 `rescue_rule.sentinel_cascade_initial_seconds` 和 `sentinel_cascade_extension_seconds` 重新计算 corrected gate。

原始 CSV：`docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.csv`

## 结果表

| case | task | mode | avg_delta_ppl | old gate_s | corrected gate_s | corrected method/base | extended | skipped | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hard_base | hard | base | -0.049633 | 53.648 | 89.624 | 6.227 | 4 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard_skip | hard | skip | -0.049633 | 52.696 | 88.033 | 6.236 | 4 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| needle_base | needle | base | -0.000166 | 39.909 | 81.705 | 5.869 | 4 | 0 | `2,0/2,0/2,0,7,12/2,0` |
| needle_skip | needle | skip | -0.000029 | 58.777 | 67.423 | 5.077 | 1 | 3 | `2,0/2,0/2,0/2,0` |
| ruler_multineedle_base | ruler_multineedle | base | 0.000074 | 16.650 | 42.868 | 5.534 | 3 | 0 | `2,0/2,0/2,0` |
| ruler_multineedle_skip | ruler_multineedle | skip | 0.000074 | 21.376 | 39.222 | 5.193 | 2 | 1 | `2,0/2,0/2,0` |
| ruler_variable_base | ruler_variable | base | -0.000564 | 13.249 | 41.103 | 5.388 | 3 | 0 | `0,13/2,0/2,0` |
| ruler_variable_skip | ruler_variable | skip | -0.000538 | 25.921 | 35.593 | 4.834 | 1 | 2 | `2,0/2,0/2,0` |
| ruler_topicswitch_base | ruler_topicswitch | base | 0.000139 | 9.931 | 39.543 | 5.209 | 3 | 0 | `0,13/2,0/2,0` |
| ruler_topicswitch_skip | ruler_topicswitch | skip | 0.000302 | 34.918 | 34.918 | 4.690 | 0 | 3 | `2,0/2,0/2,0` |

## Base vs Skip corrected

| task | ΔPPL change | corrected gate_s change | corrected method/base change | skipped | same combos |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 0.000000 | -1.591 | 0.009 | 0 | True |
| needle | 0.000137 | -14.282 | -0.792 | 3 | False |
| ruler_multineedle | 0.000000 | -3.647 | -0.341 | 1 | True |
| ruler_variable | 0.000026 | -5.510 | -0.554 | 2 | False |
| ruler_topicswitch | 0.000162 | -4.625 | -0.519 | 3 | False |

## 解释

- corrected gate 更接近真实候选 probe 成本。
- 旧文档里的 extended-run gate_s 需要谨慎解释；质量结论不受影响。
- 后续所有 gate/speed claim 应优先使用 corrected gate 或修正后的 runner 输出。
