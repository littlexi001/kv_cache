# Result Summary: slot_context_bridge_abcd_context_length

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

Story:

```text
This bridge experiment aligns the old r-B / AB / CB / DB setting with the current full-NTP slot-context protocol, to decide whether the old NMI decay contradicts the new fixed-B slot-context result.
```

## 0. Closure Summary

目的：检验 0524 的 AB/CB semantic-prior decay 和 0526 slot-context dominance 是否矛盾，并隔离 short slot cue 与 long slot context 对 B-position routing 的影响。

假设 / 问题：在同样使用 normal full-sequence causal NTP、B-position target metrics、top-1 selected-gate MoE 的前提下，`r-B / AB / CB / DB` 中更长的 role context 是否比短 role cue 更能保持或提高 route-role NMI？

结论：不矛盾。bridge 说明旧实验的 NMI 下降不是“只要有 context semantic init 就必然衰减”的普遍定律；在当前 full-NTP + B-position pressure 对齐后，long role context 明显增强 route-role alignment，尤其 fixed-B + semantic init 达到 final NMI `1.000` across all seeds。multi-B 仍然 seed-unstable，因此它支持“context signal helps routing”，但不支持“普通 top-1 NTP 自然稳健形成 feature-specialized experts”。

关键证据：fixed-B 下 `long_role_semantic_init` final route NMI 为 `1.000`，`short_role_semantic_init` 只有 `0.467`，`distributed_role_semantic_init` 为 `0.893`。multi-B 下 semantic init 均优于对应 random init：short `0.338 > 0.047`，long `0.593 > 0.373`，distributed `0.297 > 0.010`。所有条件 target accuracy 都是 `1.000`，所以差异不是任务没学会。

结论边界：Assign-Utility 不能单独判定 specialization，因为 multi-B `distributed_role_random_init` 的 Assign-Utility 为 `0.998` 但 route NMI 只有 `0.010`。bridge 只能说明 context length/strength 改善 routing alignment，不能单独证明 causal expert utility 已经稳定按 slot 分化。

下一步决策：fixed-B 已足够支持 weaker claim；不要把 fixed-B 扩成主结论。后续主战场仍是 multi-B 下显式 route-function binding signal。

## Observation

主指标是 B-position `NMI(route, role)`，因为 bridge 要回答的是 role/context 是否进入 router partition。`assignment_utility_agreement` 只作为功能一致性检查，不能单独盖过 route heatmap。

| Mode | Condition | Init | Final NMI | Final Assign-Utility | Target Acc |
|---|---|---|---:|---:|---:|
| fixed-B | short role | random | 0.000 | 1.000 | 1.000 |
| fixed-B | short role | semantic | 0.467 | 1.000 | 1.000 |
| fixed-B | long role | random | 0.159 | 1.000 | 1.000 |
| fixed-B | long role | semantic | 1.000 | 1.000 | 1.000 |
| fixed-B | distributed role | random | 0.000 | 1.000 | 1.000 |
| fixed-B | distributed role | semantic | 0.893 | 1.000 | 1.000 |
| multi-B | short role | random | 0.047 | 0.934 | 1.000 |
| multi-B | short role | semantic | 0.338 | 0.759 | 1.000 |
| multi-B | long role | random | 0.373 | 0.912 | 1.000 |
| multi-B | long role | semantic | 0.593 | 0.921 | 1.000 |
| multi-B | distributed role | random | 0.010 | 0.998 | 1.000 |
| multi-B | distributed role | semantic | 0.297 | 0.733 | 1.000 |

## Key Figures

![bridge final NMI by seed](figures/bridge_abcd_final_nmi_by_seed_heatmap.png)

This figure is the main collaborator-facing bridge plot: rows preserve seed-level results rather than seed averaging. It shows that fixed-B long semantic is stable across seeds, while multi-B semantic improves alignment but remains seed-dependent.

![bridge short vs long init-final route heatmaps](figures/bridge_abcd_short_long_init_final_route_heatmaps.png)

This figure compares random vs semantic initialization and step-0 vs final routing for short and long role contexts. It supports the interpretation that long context makes the semantic prior much more stable in fixed-B and partially helpful in multi-B.

## Comparison To 0524

0524 tested whether semantic/context initialization alone could create stable functional specialization in an AB/CB synthetic line. Its key result was that semantic init improves early B-context routing NMI but final NMI decays, while hidden states still contain recoverable semantic boundaries.

The bridge differs in the decisive ways:

| Aspect | 0524 AB/CB line | 0526/0527 bridge |
|---|---|---|
| Training objective | NTP, but metric tied to AB/CB context groups and class-specific utility | normal full-sequence causal NTP, primary metric only at B position |
| Context set | `r-B / AB / CB` style groups | `r-B / AB / CB / DB`, four roles matched to four experts |
| Target pressure | semantic grouping not always forced to be the only B-position disambiguator | same B under different roles predicts different target |
| Main question | does semantic init create stable functional expert specialization? | does stronger role context increase B-position route-role alignment? |
| Conclusion | semantic separability is not sufficient for stable functional specialization | context strength helps routing alignment, especially fixed-B |

因此二者没有直接矛盾：0524 反驳的是“semantic init alone is sufficient for functional specialization”；bridge 支持的是更弱的“when target pressure and B-position metric are aligned, stronger context can improve routing alignment”。

## Interpretation

fixed-B 是 sanity/control：同一个 B hidden state 只随 role context 变化。如果 long semantic 在这里达到 diagonal routing，说明 context signal 确实能控制 router partition。这个结果强支持 weaker hypothesis。

multi-B 是主难点：不同 `B_i` identity 同时存在时，long semantic 仍然最好，但不是全 seed 对角。它说明 role signal 增强了 routing alignment，却没有消除 identity variation 和 top-1 path dependence。

distributed role 的结果要谨慎读：semantic init 比 random 明显提高 NMI，但 Assign-Utility 反而低于 random。这不是反例，而是再次说明 Assign-Utility 可能被 common utility collapse 扭曲，必须和 route NMI/heatmap 一起读。

## Claim Update

- Supported: stronger role/slot context helps B-position routing align with slot/role, fixed-B long semantic 是稳定正例。
- Weakened: context strength alone is sufficient for robust multi-B functional specialization。
- Clarified: 0524 和当前结果不矛盾；它们测试的强度不同，主指标和 target pressure 也不同。

## What Cannot Be Claimed

不能声称 ordinary top-1 MoE 在多 B identity 下自然形成稳定 slot-specific causal experts。也不能用 fixed-B 的完美 diagonal routing 直接推出 multi-B 或 real-corpus specialization。

## Next Decision

保留 bridge 作为可比性审计和 fixed-B positive control。下一步固定 `multi_b_primary / long_role_semantic_init` 或原 `A_long_repeated_slot_init`，加入显式 route-function binding signal，判断 route NMI、forced utility heatmap、Assign-Utility 是否同步改善。
