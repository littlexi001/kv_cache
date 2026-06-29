# Low-Spread Early Exit 负面消融（2026-06-29）

## 目的

Needle-style smoke 中多个候选策略的 early sentinel loss 很接近，说明 easy retrieval regime 里多个 policy 近似等价。为减少这类场景下的过度 extension，本轮测试一个低分歧提前接受规则：

```text
if early best-vs-runner-up loss spread <= 0.001:
    accept early
```

实现参数：

```text
--sentinel_cascade_accept_low_spread 0.001
```

原始 CSV：`docs/pcic_low_spread_early_exit_ablation_2026_06_29.csv`

## 结果表

| run | avg_delta_ppl | method/base | gate_s | extended | early | low_spread_early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hard cond | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| hard low-spread | -0.049633 | 4.161 | 52.526 | 4 | 0 | 0 | `0,6` | `0,6/2,0,7,12/0,6/0,6` |
| war cond | -2.135311 | 2.603 | 6.659 | 0 | 2 | 0 | `0,7` | `0,7/0,7` |
| war low-spread | -2.135311 | 2.602 | 6.530 | 0 | 2 | 0 | `0,7` | `0,7/0,7` |
| monte cond | -0.219215 | 2.600 | 6.641 | 2 | 0 | 0 | `2,0` | `2,7/2,0` |
| monte low-spread | -0.219215 | 2.598 | 6.483 | 2 | 0 | 0 | `2,0` | `2,7/2,0` |
| needle cond | -0.000166 | 3.401 | 39.909 | 4 | 0 | 0 | `2,0` | `2,0/2,0/2,0,7,12/2,0` |
| needle low-spread | 0.000003 | 4.692 | 61.147 | 0 | 4 | 4 | `2,0` | `0,13/2,0/2,0/2,0` |

## 结论

该 heuristic 不应进入主方法：

1. Hard-topic / War / Monte 质量保持，代价基本持平。
2. Needle 上 extension 从 `4` 降到 `0`，说明 low-spread trigger 确实命中了 easy-regime。
3. 但 Needle wall-clock 反而变差：`method/base` 从 `3.401` 升到 `4.692`，`gate_s` 从 `39.909s` 升到 `61.147s`。
4. Needle 质量也从 `-0.000166` 退到 `0.000003`，虽然绝对幅度很小。

因此，减少 extension 次数并不等价于降低当前实现的 wall-clock。瓶颈更可能来自：

- 每个候选 sentinel 的串行前缀计算；
- repeated Python / model forward overhead；
- 没有真正 fused/tensorized 的 probe path；当前 batch-row eager path 仍不能消除主要开销；
- 小样本 wall-clock 方差。

## 当前默认

继续保持：

```text
sentinel_cascade_accept_margin = 0.012
sentinel_cascade_accept_low_spread = 0.0
sentinel_cascade_anchor_accept_on_match = false
```

## 方法判断

两组 early-exit 负面消融（anchor-match / low-spread）共同说明：

> 继续堆 heuristic early-exit 不能解决速度问题；下一步必须做系统层 fused/tensorized probe，或者先补正式 benchmark 质量证据。

## 下一步

建议优先级：

1. 如果目标是 speed claim：实现 fused/tensorized sentinel probe。
2. 如果目标是 paper innovation / quality claim：准备正式 LongBench / RULER 子集。
3. 不建议继续尝试简单 early-exit heuristic。
