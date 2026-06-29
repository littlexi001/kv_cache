# Anchor-Match Early Exit 负面消融（2026-06-29）

## 目的

Needle-style smoke 显示 easy retrieval 场景下 conditional auto-anchor 仍然过度 extension。一个自然想法是：

```text
如果 early selected combo 已经等于 validation-prior anchor，
则直接 early-accept，不再扩展到长 horizon。
```

本轮实现参数：

```text
--sentinel_cascade_anchor_accept_on_match true
```

并在 Hard-topic / War / Monte / Needle 四类场景上测试。

原始 CSV：`docs/pcic_anchor_match_early_exit_ablation_2026_06_29.csv`

## 结果表

| run | avg_delta_ppl | method/base | gate_s | extended | early | anchor_match_early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hard cond | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| hard anchor-match early | -0.049633 | 4.809 | 62.742 | 3 | 1 | 1 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| war cond | -2.135311 | 2.603 | 6.659 | 0 | 2 | 0 | `0,7` | `0,7/0,7` |
| war anchor-match early | -2.135311 | 2.599 | 6.621 | 0 | 2 | 0 | `0,7` | `0,7/0,7` |
| monte cond | -0.219215 | 2.600 | 6.641 | 2 | 0 | 0 | `2,0` | `2,7/2,0` |
| monte anchor-match early | -0.219215 | 2.603 | 6.574 | 2 | 0 | 0 | `2,0` | `2,7/2,0` |
| needle cond | -0.000166 | 3.401 | 39.909 | 4 | 0 | 0 | `2,0` | `2,0/2,0/2,0,7,12/2,0` |
| needle anchor-match early | -0.000029 | 4.566 | 59.459 | 1 | 3 | 3 | `2,0` | `2,0/2,0/2,0/2,0` |

## 结论

该 trigger 不应作为默认方法。

原因：

1. Hard-topic 质量保持，但 `method/base` 从 `4.144` 升到 `4.809`，gate 从 `53.648s` 升到 `62.742s`。
2. War / Monte 基本不变，只能说明没有破坏质量。
3. Needle 的 extension 从 `4` 降到 `1`，但 wall-clock 反而变差，且 ΔPPL 从 `-0.000166` 变成 `-0.000029`。
4. 说明当前计时路径中，减少 extension 次数并不必然降低实际 `method/base`；真正需要的是 batched/fused probe，而不是简单 early-exit heuristic。

## 对方法的影响

保留代码参数用于消融：

```text
--sentinel_cascade_anchor_accept_on_match
```

但 paper 主方法不采用该 trigger。当前默认仍应是：

```text
sentinel_cascade_accept_margin = 0.012
sentinel_cascade_anchor_accept_on_match = false
```

## 下一步

更有价值的方向不是继续加 heuristic early-exit，而是：

1. batched/fused candidate probe，降低所有 sentinel gate 的实际代价；
2. adaptive margin，用统计量调节是否进入长 horizon，而不是直接按 anchor match early-accept；
3. 正式 LongBench / RULER 子集验证质量。
