# War / Monte Auto-Anchor 负面检查（2026-06-29）

## 目的

Hard-topic eval128 上，validation-prior auto-anchor 修复了 delayed-win failure。接下来需要确认：

1. auto-anchor 不会破坏原本已经接近 oracle 的 War / Monte；
2. validation-prior anchor 可以在不同文本上自动选择不同 combo；
3. rescue gate 的 anchor 不是固定写死为 `0,6`。

## Anchor 自动选择

脚本：`scripts/run_pcic_warmonte_auto_anchor_check.sh`

选择器：`scripts/select_pcic_validation_anchor.py`

War validation prior：

- 排序文档：`docs/pcic_war_auto_anchor_validation_prior_2026_06_29.md`
- 自动 anchor：`0,7`
- 依据：`0,7` fixed validation 平均 ΔPPL = `-2.135311`

Monte validation prior：

- 排序文档：`docs/pcic_monte_auto_anchor_validation_prior_2026_06_29.md`
- 自动 anchor：`2,0`
- 依据：`2,0` fixed validation 平均 ΔPPL = `-0.009005`

这说明 auto-anchor 不是固定选 `0,6`，而是随 validation prior 自动变化。

## 结果表

| run | avg_delta_ppl | gate_s | anchors | combos | 结论 |
| --- | ---: | ---: | --- | --- | --- |
| War top2 baseline | -2.135311 | 6.544 | `` | `0,7;0,7` | 已达到 oracle |
| War auto-anchor | -2.135311 | 8.863 | `0,7` | `0,7;0,7` | 质量不变，代价略升 |
| Monte top2 baseline | -0.219215 | 6.573 | `` | `2,7;2,0` | 已达到 oracle |
| Monte auto-anchor | -0.219215 | 6.670 | `2,0` | `2,7;2,0` | 质量不变，代价基本不变 |

## 结论

1. auto-anchor 在 War 上自动选 `0,7`，在 Monte 上自动选 `2,0`，不是 hard-coded anchor。
2. auto-anchor 没有破坏 War / Monte 已有 oracle 质量。
3. War 的 gate 从 `6.544s` 增到 `8.863s`，说明当 baseline 已经稳定时，anchor 可能带来不必要开销。
4. Monte 的 gate 从 `6.573s` 增到 `6.670s`，几乎无额外代价。

## 对方法的启发

最终 paper 方法不应“总是加入 anchor”，而应使用条件触发：

```text
if short-horizon confidence is low
or early selection conflicts with validation prior / risk memory
or delayed-win detector fires:
    include validation-prior anchor in cascade extension
else:
    keep cheap top2 cascade
```

这样可以保留 Hard-topic eval128 的质量修复，同时避免 War 上不必要的 gate 开销。

## 下一步

1. 实现 conditional auto-anchor：只在 early margin 低或 prior conflict 时加入 anchor。
2. 在 Hard-topic eval128 / War / Monte 三组上复测。
3. 目标：Hard-topic 保持 `-0.049633`，War/Monte 保持 oracle，同时把 War gate 从 `8.863s` 拉回接近 `6.544s`。
