# Conditional Auto-Anchor Rescue 实验记录（2026-06-29）

## 目的

上一轮 auto-anchor 证明 validation-prior anchor 可以修复 Hard-topic eval128，但 War 上出现不必要 gate 开销：

- War top2 baseline gate = `6.544s`
- War unconditional auto-anchor gate = `8.863s`

本轮测试 conditional auto-anchor：

```text
如果 early horizon margin 足够大，直接 early accept；
如果 early horizon margin 低于阈值，才进入 cascade extension；
extension 阶段加入 validation-prior anchor。
```

本次阈值：

```text
sentinel_cascade_accept_margin = 0.012
```

## 实验入口

脚本：`scripts/run_pcic_conditional_auto_anchor_suite.sh`

自动 anchor：

- Hard-topic：`0,6`
- War：`0,7`
- Monte：`2,0`

## 结果表

| run | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hard top2 32→64 | 0.004371 | 2.648 | 27.085 | 1 | 3 | `` | `0,7;2,0,7,12;0,6;0,13` |
| hard auto-anchor | -0.049633 | 4.164 | 51.997 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| hard conditional auto-anchor | -0.049633 | 4.144 | 53.648 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| war top2 | -2.135311 | 2.596 | 6.544 | 0 | 2 | `` | `0,7;0,7` |
| war auto-anchor | -2.135311 | 3.119 | 8.863 | 2 | 0 | `0,7` | `0,7;0,7` |
| war conditional auto-anchor | -2.135311 | 2.603 | 6.659 | 0 | 2 | `0,7` | `0,7;0,7` |
| monte top2 | -0.219215 | 2.596 | 6.573 | 2 | 0 | `` | `2,7;2,0` |
| monte auto-anchor | -0.219215 | 2.602 | 6.670 | 2 | 0 | `2,0` | `2,7;2,0` |
| monte conditional auto-anchor | -0.219215 | 2.600 | 6.641 | 2 | 0 | `2,0` | `2,7;2,0` |

## 结论

1. Hard-topic eval128：conditional auto-anchor 保持 oracle 质量，ΔPPL = `-0.049633`。
2. War：conditional auto-anchor 保持质量，同时 gate 从 unconditional 的 `8.863s` 降到 `6.659s`，接近 top2 baseline `6.544s`。
3. Monte：conditional auto-anchor 保持质量，gate 与 top2 baseline 基本一致。
4. 这说明 validation-prior anchor 应作为 conditional rescue，而不是无条件加入。

## 对 paper 方法的影响

当前最强主线应更新为：

```text
Pairwise-CIC
+ online blockwise sparse-attention policy selection
+ conditional validation-prior horizon-anchor rescue gate
```

该版本的创新点更清晰：

- Pairwise-CIC 负责候选 policy 的 block-local counterfactual calibration；
- online selection 负责每个 block 动态选择；
- conditional rescue gate 只在短 horizon 不确定时引入 validation-prior anchor；
- 避免了“固定 sparse rule”或“手写 anchor”的创新性风险。

## 仍需补强

- 当前 conditional threshold `0.012` 是经验值，需要做小网格或自适应阈值；
- 需要更长 blocks 和标准 benchmark；
- 需要 batched/fused probe 降低真实 wall-clock。

## Margin grid 后续结果

小网格已补，见：`docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.md`

结论：

- `0.008/0.010` 仍会漏掉 Hard-topic eval128 的 delayed-win block3；
- `0.012/0.015` 可以达到 Hard-topic oracle；
- `0.015` 会让 War 多一次不必要 extension；
- 当前推荐继续使用 `0.012`，后续再发展 adaptive margin。
