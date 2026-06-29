# Conditional Auto-Anchor Batched Gate 对比（2026-06-29）

## 目的

本实验检验当前 paper 主线的系统侧实现：`Pairwise-CIC + online blockwise selection + conditional validation-prior horizon-anchor rescue gate` 在开启 `--sentinel_batched_candidates true` 后，是否保持选择语义，并观察真实 wall-clock gate 变化。

该实验不下载任何外部数据，只使用服务器本地已有模型、Hard-topic、War、Monte 文本和 validation-prior anchor 输出。

原始 CSV：`docs/pcic_condautoanchor_batched_gate_2026_06_29.csv`

## 结果表

| run | dataset | mode | avg_delta_ppl | method/base | gate_s | extended | early | batched | anchors | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hard_serial | hard | serial | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| hard_batched | hard | batched | -0.049778 | 6.254 | 79.524 | 4 | 0 | 1 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| war_serial | war | serial | -2.135311 | 2.603 | 6.659 | 0 | 2 | 0 | `0,7` | `0,7/0,7` |
| war_batched | war | batched | -2.133787 | 3.304 | 8.689 | 0 | 2 | 1 | `0,7` | `0,7/0,7` |
| monte_serial | monte | serial | -0.219215 | 2.600 | 6.641 | 2 | 0 | 0 | `2,0` | `2,7/2,0` |
| monte_batched | monte | batched | -0.251805 | 3.376 | 8.466 | 2 | 0 | 1 | `2,0` | `2,7/2,0` |

## Serial vs Batched 差异

| dataset | ΔPPL diff | method/base diff | gate_s diff | same combos |
| --- | ---: | ---: | ---: | --- |
| hard | -0.000145 | 2.110 | 25.876 | True |
| war | 0.001524 | 0.701 | 2.030 | True |
| monte | -0.032590 | 0.776 | 1.825 | True |

## 解释原则

- 如果 `same combos=True` 且 `ΔPPL diff≈0`，说明 batch-row budget map 没有改变 selector 语义。
- 如果 `gate_s` 下降，说明当前 eager batched path 已有真实收益。
- 如果 `gate_s` 不降或上升，说明 batch-row 表达已跑通，但仍需要 fused candidate probe / tensorized mask path 才能支撑速度 claim。

## 后续 dispatch 优化

后续补充见：`docs/pcic_condautoanchor_batched_gate_optdispatch_2026_06_29.md`

优化后 gate：

- Hard：`79.524s -> 61.139s`，仍高于 serial `53.648s`。
- War：`8.689s -> 7.462s`，仍高于 serial `6.659s`。
- Monte：`8.466s -> 7.467s`，仍高于 serial `6.641s`。

结论不变但更精确：batch-row 语义和 dispatch 优化都成立；真实 speed claim 仍需要 fused/tensorized candidate probe。
