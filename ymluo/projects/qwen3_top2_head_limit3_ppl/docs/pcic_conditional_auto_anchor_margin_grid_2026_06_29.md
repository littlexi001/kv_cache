# Conditional Auto-Anchor Margin Grid（2026-06-29）

## 目的

上一轮 conditional auto-anchor 使用经验阈值：

```text
sentinel_cascade_accept_margin = 0.012
```

本轮做小网格，确认该阈值不是偶然，并观察质量 / gate 代价 trade-off。

网格：

```text
0.008, 0.010, 0.012, 0.015
```

原始 CSV：`docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.csv`

## 结果表

| margin | dataset | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.008 | Hard-topic eval128 | -0.023677 | 4.039 | 50.159 | 3 | 1 | `0,6` | `0,6;2,0,7,12;0,6;0,13` |
| 0.008 | War | -2.135311 | 2.601 | 6.691 | 0 | 2 | `0,7` | `0,7;0,7` |
| 0.008 | Monte | -0.219215 | 2.605 | 6.688 | 2 | 0 | `2,0` | `2,7;2,0` |
| 0.010 | Hard-topic eval128 | -0.023677 | 4.033 | 51.132 | 3 | 1 | `0,6` | `0,6;2,0,7,12;0,6;0,13` |
| 0.010 | War | -2.135311 | 2.604 | 6.631 | 0 | 2 | `0,7` | `0,7;0,7` |
| 0.010 | Monte | -0.219215 | 2.599 | 6.629 | 2 | 0 | `2,0` | `2,7;2,0` |
| 0.012 | Hard-topic eval128 | -0.049633 | 4.144 | 53.648 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| 0.012 | War | -2.135311 | 2.603 | 6.659 | 0 | 2 | `0,7` | `0,7;0,7` |
| 0.012 | Monte | -0.219215 | 2.600 | 6.641 | 2 | 0 | `2,0` | `2,7;2,0` |
| 0.015 | Hard-topic eval128 | -0.049633 | 4.165 | 53.439 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| 0.015 | War | -2.135311 | 2.859 | 7.699 | 1 | 1 | `0,7` | `0,7;0,7` |
| 0.015 | Monte | -0.219215 | 2.599 | 6.566 | 2 | 0 | `2,0` | `2,7;2,0` |

## 结论

1. `0.008` 和 `0.010` 会保留 Hard-topic block3 的错误选择 `0,13`，只能把 ΔPPL 改到 `-0.023677`，没有达到 oracle。
2. `0.012` 和 `0.015` 都能修复 Hard-topic eval128，达到 ΔPPL `-0.049633`。
3. `0.015` 会让 War 进入一次不必要 extension，gate 从 `6.659s` 增到 `7.699s`。
4. `0.012` 是当前最均衡阈值：Hard-topic 达到 oracle，同时 War/Monte gate 接近 top2 baseline。

## 推荐配置

当前推荐：

```text
sentinel_cascade_accept_margin = 0.012
```

配套方法名：

```text
Conditional Validation-Prior Horizon-Anchor Rescue
```

## Paper 表述

这组 grid 支持一个可解释的 trade-off：

- 阈值太低：过早 early accept，会漏掉 delayed-win policy；
- 阈值太高：过度 extension，会增加稳定文本上的 gate 成本；
- 中间阈值：既修复 delayed-win，又避免稳定文本过度扩展。

这比单点调参更适合写进论文，因为它展示了 rescue gate 的机制边界。

## 下一步

1. 把固定阈值 `0.012` 改为 adaptive margin：例如按候选 loss 方差、best-runner-up gap 或历史 delayed-win rate 自适应。
2. 跑更长 blocks，确认 `0.012` 不只是 4-block Hard-topic 上的偶然。
3. 开始标准 benchmark smoke，证明方法不只是 synthetic / book text 上成立。
