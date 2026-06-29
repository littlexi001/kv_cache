# Fixed / Online PCIC / Blockwise Oracle 对比（2026-06-29）

目的：验证 Horizon-PCIC 不是一个固定 combo 可以替代的小改，而是接近 blockwise oracle 的在线策略选择器。

## 已完成结果

| dataset | blocks | best fixed | fixed ΔPPL | online ΔPPL | oracle ΔPPL | online-fixed | online-oracle gap | online/base | gate_s | online combos | oracle combos |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Hard-topic eval64 | 4 | `0,6` | -0.012719 | -0.076940 | -0.076940 | -0.064221 | 0.000000 | 4.028 | 25.392 | `0,7;0,6;0,13;2,0,7,12` | `0,7;0,6;0,13;2,0,7,12` |
| Hard-topic eval128 | 4 | `0,6` | -0.020744 | 0.004371 | -0.049633 | 0.025115 | 0.054004 | 2.648 | 27.085 | `0,7;2,0,7,12;0,6;0,13` | `0,6;2,0,7,12;0,6;0,6` |
| War and Peace | 2 | `0,7` | -2.135311 | -2.135311 | -2.135311 | 0.000000 | 0.000000 | 2.596 | 6.544 | `0,7;0,7` | `0,7;0,7` |
| Count of Monte Cristo | 2 | `2,0` | -0.009005 | -0.219215 | -0.219215 | -0.210210 | 0.000000 | 2.596 | 6.573 | `2,7;2,0` | `2,7;2,0` |

## 缺失结果

- 无。

## 判据

- 若 online 明显优于 best fixed，说明动态选择不是可有可无。
- 若 online 接近 blockwise oracle，说明 Pairwise-CIC + rescue gate 的选择信号有效。
- 若 online 与 best fixed 接近，需要增加非平稳文本、更多 blocks 或更强候选集来证明策略切换价值。
