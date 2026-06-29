# PCIC Extension Waste 后验分析（2026-06-29）

## 目的

该分析只解析已有 `pcic_r_blockwise_results.csv`，不跑模型。目标是量化 cascade extension 中有多少 block 最终选择没有变化，从而判断下一步是否值得设计更强 skip gate。

block 级 CSV：`docs/pcic_extension_waste_blocks_2026_06_29.csv`
summary CSV：`docs/pcic_extension_waste_summary_2026_06_29.csv`

## 汇总

| case | task | blocks | extended | no-change ext | avoidable s | avoidable frac | anchor-hit avoidable s | avg ΔPPL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard | hard | 4 | 4 | 2 | 15.622 | 0.437 | 4.446 | -0.049633 |
| monte | monte | 2 | 2 | 1 | 2.211 | 0.400 | 0.000 | -0.219215 |
| needle | needle | 4 | 4 | 2 | 13.314 | 0.462 | 13.314 | -0.000166 |
| ruler_multineedle | ruler_multineedle | 3 | 3 | 1 | 4.972 | 0.375 | 4.972 | 0.000074 |
| ruler_topicswitch | ruler_topicswitch | 3 | 3 | 2 | 6.620 | 0.669 | 6.620 | 0.000139 |
| ruler_variable | ruler_variable | 3 | 3 | 1 | 4.950 | 0.429 | 4.950 | -0.000564 |
| war | war | 2 | 0 | 0 | 0.000 | 0.000 | 0.000 | -2.135311 |

## 解释

- `no-change ext`：已经运行 extension，但最终 combo 与 early initial combo 相同。
- `avoidable s`：如果能提前识别这些 no-change block，理论上可省的 extension 秒数。
- 这是后验上界，不是可直接写进主方法的规则；它用于判断下一步 skip-gate 设计是否有空间。
