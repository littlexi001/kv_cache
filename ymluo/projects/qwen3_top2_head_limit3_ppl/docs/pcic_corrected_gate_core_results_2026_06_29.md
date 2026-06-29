# Corrected Cascade Gate Seconds（2026-06-29）

## 背景

旧 `gate_seconds` 在 extended cascade 中只统计 extended candidate set，漏掉了未进入 extension 的初始候选 probe。因此 extended runs 的 gate 成本被低估。本分析只读已有 CSV，从 `rescue_rule.sentinel_cascade_initial_seconds` 和 `sentinel_cascade_extension_seconds` 重新计算 corrected gate。

原始 CSV：`docs/pcic_corrected_gate_core_results_2026_06_29.csv`

## 结果表

| case | task | mode | avg_delta_ppl | old gate_s | corrected gate_s | corrected method/base | extended | skipped | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hard_top2 | hard | top2 | 0.004371 | 27.085 | 32.530 | 2.972 | 1 | 0 | `0,7/2,0,7,12/0,6/0,13` |
| hard_cond | hard | cond | -0.049633 | 53.648 | 89.624 | 6.227 | 4 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard_b8_top2 | hard_b8 | top2 | 0.006074 | 49.737 | 75.197 | 3.263 | 5 | 0 | `0,7/2,0,7,12/0,6/0,13/2,0,7,12/0,13/7,13/0,13` |
| hard_b8_cond | hard_b8 | cond | -0.040598 | 107.643 | 179.677 | 6.248 | 8 | 0 | `0,6/2,0,7,12/0,6/0,6/7,13/0,6/7,13/2,0` |
| war_top2 | war | top2 | -2.135311 | 6.544 | 6.544 | 2.596 | 0 | 0 | `0,7/0,7` |
| war_cond | war | cond | -2.135311 | 6.659 | 6.659 | 2.603 | 0 | 0 | `0,7/0,7` |
| monte_top2 | monte | top2 | -0.219215 | 6.573 | 9.913 | 3.388 | 2 | 0 | `2,7/2,0` |
| monte_cond | monte | cond | -0.219215 | 6.641 | 10.011 | 3.392 | 2 | 0 | `2,7/2,0` |
| needle_top2 | needle | top2 | 0.000118 | 13.206 | 37.263 | 3.262 | 4 | 0 | `2,0,7,12/2,0/2,0/2,0` |
| needle_cond | needle | cond | -0.000166 | 39.909 | 81.705 | 5.869 | 4 | 0 | `2,0/2,0/2,0,7,12/2,0` |
| ruler_multineedle_top2 | ruler_multineedle | top2 | 0.000074 | 11.081 | 28.577 | 4.051 | 3 | 0 | `2,0/2,0/2,0` |
| ruler_multineedle_cond | ruler_multineedle | cond | 0.000074 | 16.650 | 42.868 | 5.534 | 3 | 0 | `2,0/2,0/2,0` |
| ruler_variable_top2 | ruler_variable | top2 | 0.017397 | 11.857 | 24.768 | 3.699 | 2 | 0 | `2,0/2,0,7,12/2,0` |
| ruler_variable_cond | ruler_variable | cond | -0.000564 | 13.249 | 41.103 | 5.388 | 3 | 0 | `0,13/2,0/2,0` |
| ruler_topicswitch_top2 | ruler_topicswitch | top2 | 0.000302 | 6.530 | 26.031 | 3.809 | 3 | 0 | `2,0/2,0/2,0` |
| ruler_topicswitch_cond | ruler_topicswitch | cond | 0.000139 | 9.931 | 39.543 | 5.209 | 3 | 0 | `0,13/2,0/2,0` |

## Base vs Skip corrected

| task | ΔPPL change | corrected gate_s change | corrected method/base change | skipped | same combos |
| --- | ---: | ---: | ---: | ---: | --- |

## 解释

- corrected gate 更接近真实候选 probe 成本。
- 旧文档里的 extended-run gate_s 需要谨慎解释；质量结论不受影响。
- 后续所有 gate/speed claim 应优先使用 corrected gate 或修正后的 runner 输出。
