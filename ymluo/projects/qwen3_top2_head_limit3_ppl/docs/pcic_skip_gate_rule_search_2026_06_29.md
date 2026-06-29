# PCIC Skip-Gate Rule Search（2026-06-29）

## 目的

在 `pcic_extension_waste_blocks_2026_06_29.csv` 上搜索简单可解释规则。规则只使用 extension 前可见特征，目标是 zero false-skip 下最大化可省 extension 秒数。

完整规则 CSV：`ymluo\projects\qwen3_top2_head_limit3_ppl\docs\pcic_skip_gate_rule_search_2026_06_29.csv`

## Zero False-Skip Top Rules

| rule | selected | saved_s | saved_frac | selected_cases |
| --- | ---: | ---: | ---: | --- |
| `anchor_hit_and_margin_le_0.00671903_and_ratio_le_0` | 6 | 29.856 | 0.285 | `needle:1;needle:3;ruler_multineedle:0;ruler_variable:2;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.0100904_and_ratio_le_0` | 6 | 29.856 | 0.285 | `needle:1;needle:3;ruler_multineedle:0;ruler_variable:2;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_ratio_le_0` | 6 | 29.856 | 0.285 | `needle:1;needle:3;ruler_multineedle:0;ruler_variable:2;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.000914671_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.00225478_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.00327363_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.00527303_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.00581508_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.0058938_and_ratio_le_0` | 5 | 24.906 | 0.238 | `needle:1;needle:3;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_0.000119314_and_ratio_le_0` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_6.00495e-05_and_ratio_le_0` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_6.00495e-05_and_ratio_le_0.0155119` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_6.00495e-05_and_ratio_le_0.0971145` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_7.61193e-05_and_ratio_le_0` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_7.61193e-05_and_ratio_le_0.0155119` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_8.51464e-05_and_ratio_le_0` | 4 | 18.247 | 0.174 | `needle:1;ruler_multineedle:0;ruler_topicswitch:1;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_5.22137e-05_and_ratio_le_0` | 3 | 14.924 | 0.142 | `needle:1;ruler_multineedle:0;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_5.22137e-05_and_ratio_le_0.0155119` | 3 | 14.924 | 0.142 | `needle:1;ruler_multineedle:0;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_5.22137e-05_and_ratio_le_0.0971145` | 3 | 14.924 | 0.142 | `needle:1;ruler_multineedle:0;ruler_topicswitch:2` |
| `anchor_hit_and_margin_le_4.79652e-05_and_ratio_le_0` | 2 | 11.627 | 0.111 | `needle:1;ruler_multineedle:0` |

## 解释

- `false-skip` 表示规则选择跳过 extension，但后验显示 extension 会改变最终 combo。
- zero false-skip 规则只是当前样本上的候选规则；上线前必须在新样本上验证。
- 如果最优规则只覆盖很少 block，说明 skip-gate 需要更多训练数据或更强特征。
