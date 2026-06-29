# RULER-style Offline Smoke（2026-06-29）

## 目的

不下载外部数据，生成三类 RULER-style 长上下文文本，检查 paper 主线是否只在 hard-topic/小说文本上成立。

任务：multi-needle、variable binding、topic switch。

validation-prior anchor 见：`docs/pcic_ruler_style_validation_prior_2026_06_29.md`
原始 CSV：`docs/pcic_ruler_style_smoke_2026_06_29.csv`

## 结果表

| case | task | mode | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| multineedle_top2 | multineedle | top2 | 0.000074 | 2.206 | 11.081 | 3 | 0 | `` | `2,0/2,0/2,0` |
| multineedle_cond | multineedle | cond_auto_anchor | 0.000074 | 2.781 | 16.650 | 3 | 0 | `2,0` | `2,0/2,0/2,0` |
| variable_top2 | variable | top2 | 0.017397 | 2.315 | 11.857 | 2 | 1 | `` | `2,0/2,0,7,12/2,0` |
| variable_cond | variable | cond_auto_anchor | -0.000564 | 2.441 | 13.249 | 3 | 0 | `2,0` | `0,13/2,0/2,0` |
| topicswitch_top2 | topicswitch | top2 | 0.000302 | 1.731 | 6.530 | 3 | 0 | `` | `2,0/2,0/2,0` |
| topicswitch_cond | topicswitch | cond_auto_anchor | 0.000139 | 2.086 | 9.931 | 3 | 0 | `2,0` | `0,13/2,0/2,0` |

## 解释边界

- 这是无下载 synthetic smoke，不等同正式 RULER / LongBench。
- 若 cond_auto_anchor 在多个任务上保持或改善 PPL drift，说明主线有跨模式迹象。
- 若某些任务退化，应作为 rescue gate 的失败模式，用于后续 adaptive trigger / benchmark 设计。
