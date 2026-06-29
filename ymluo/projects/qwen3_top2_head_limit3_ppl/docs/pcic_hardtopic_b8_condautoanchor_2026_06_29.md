# Hard-topic b8 Conditional Auto-Anchor 验证（2026-06-29）

## 目的

前面 Hard-topic eval128 只跑了 4 blocks。为了确认 `conditional validation-prior horizon-anchor rescue gate` 不是 4-block 偶然，本轮扩展到：

```text
num_blocks = 8
eval_tokens_per_block = 128
sentinel_cascade_accept_margin = 0.012
validation-prior anchor = 0,6
```

实验入口：`scripts/run_pcic_hardtopic_b8_conditional_auto_anchor.sh`

逐块 CSV：`docs/pcic_hardtopic_b8_condautoanchor_blocks_2026_06_29.csv`

## 结果表

| run | blocks | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| b4 top2 | 4 | 0.004371 | 2.648 | 27.085 | 1 | 3 | `` | `0,7/2,0,7,12/0,6/0,13` |
| b4 cond auto-anchor | 4 | -0.049633 | 4.144 | 53.648 | 4 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| b8 top2 | 8 | 0.006074 | 2.512 | 49.737 | 5 | 3 | `` | `0,7/2,0,7,12/0,6/0,13/2,0,7,12/0,13/7,13/0,13` |
| b8 cond auto-anchor | 8 | -0.040598 | 4.161 | 107.643 | 8 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6/7,13/0,6/7,13/2,0` |

## 逐块观察

关键变化：

| block | top2 combo | top2 ΔPPL | cond combo | cond ΔPPL | 变化 |
| ---: | --- | ---: | --- | ---: | --- |
| 0 | `0,7` | 0.032644 | `0,6` | -0.079549 | 修复 delayed-win |
| 3 | `0,13` | 0.067980 | `0,6` | -0.035842 | 修复 delayed-win |
| 4 | `2,0,7,12` | 0.063619 | `7,13` | -0.055333 | 新增修复 |
| 5 | `0,13` | 0.011271 | `0,6` | -0.009794 | 新增修复 |
| 7 | `0,13` | 0.013730 | `2,0` | -0.003614 | 新增修复 |

## 结论

1. b8 top2 仍然出现多个 delayed-win / short-horizon miss，平均 ΔPPL = `0.006074`。
2. b8 conditional auto-anchor 把平均 ΔPPL 改善到 `-0.040598`。
3. 这说明 rescue gate 的收益不只出现在前 4 个 block，而是在更长 block 序列中仍然存在。
4. 代价是 gate 从 `49.737s` 增到 `107.643s`，仍需 batched/fused probe 优化。

## Paper 意义

这组结果强化了 paper 主线：

```text
Pairwise-CIC + online blockwise selection + conditional validation-prior horizon-anchor rescue gate
```

更重要的是，它说明方法不是只修补一个单独 block，而是在长序列上持续修正短 horizon 选择错误。

## 限制

- b8 还没有 blockwise oracle，因此不能声称 b8 conditional 已达到 oracle。
- b8 conditional 所有 8 个 block 都进入 extension，说明 Hard-topic eval128 的 margin 分布很困难。
- 下一步需要 standard benchmark smoke，以及 probe batching / fusion 来降低 gate 开销。
