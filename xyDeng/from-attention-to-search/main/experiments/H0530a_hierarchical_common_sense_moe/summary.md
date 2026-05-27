# Result Summary: H0530a Hierarchical Common/Sense MoE

Hypothesis purpose card:

```text
Projects/from-attention-to-search/main/hypotheses/H0530a_hierarchical_common_sense_moe/purpose_card.md
```

## Observation

主指标是 tail bucket 的 ablation-based $S_{\mathrm{sense}}$ 和 tail route-function alignment。它决定判断，因为 H0530a 问的是 Hierarchical MoE 是否改善 functional specialization，而不是 routing pattern 是否更可解释。

结果不支持 hierarchy 改善：

| Condition | Head $S_{\mathrm{sense}}$ | Middle $S_{\mathrm{sense}}$ | Tail $S_{\mathrm{sense}}$ |
|---|---:|---:|---:|
| `flat_zipfian` | `0.0595` | `0.0528` | `0.0612` |
| `hierarchical_zipfian` | `0.0358` | `0.0176` | `0.0465` |

Tail route-function alignment 明显变差：

| Condition | Head alignment | Middle alignment | Tail alignment |
|---|---:|---:|---:|
| `flat_zipfian` | `0.8542` | `0.8128` | `0.7930` |
| `hierarchical_zipfian` | `0.8432` | `0.3480` | `-0.0693` |

Uniform-eval CE / accuracy 也显示 hierarchy 的 tail 表现更差：

| Condition | Head CE | Middle CE | Tail CE | Tail acc |
|---|---:|---:|---:|---:|
| `flat_zipfian` | `0.0005` | `0.0219` | `0.0055` | `0.9978` |
| `hierarchical_zipfian` | `0.0011` | `0.0182` | `0.0920` | `0.9735` |

Routing diagnostics 中 hierarchy 的 route-token / route-family normalized mutual information 更高：

| Condition | NMI(route, token) | NMI(route, family) | NMI(route, context) |
|---|---:|---:|---:|
| `flat_zipfian` | `0.1102` | `0.0609` | `0.1002` |
| `hierarchical_zipfian` | `0.1461` | `0.1031` | `0.1002` |

但这没有转化为 better functional specialization。

## Interpretation

H0530a 的结果支持 Rival 1 / Rival 2：

```text
Hierarchy changed routing structure, but did not improve ablation-based expert function.
```

更具体地说，当前 minimal common/sense hierarchy 让 fine routing 更关联 token/family，但 tail route-function alignment 反而崩掉。这说明仅靠 two-level routing reference 还不能解决 route assignment、expert utility、training objective 之间的不对齐。

## Limitation

这是第一版 minimal hierarchy：

- hierarchy 有额外路径和容量，不能作为公平架构优劣结论；
- fine reference 使用 family-reference residual，不代表所有 possible hierarchical MoE；
- no load balance / no route-function objective，不能否定带显式 utility-aware objective 的 hierarchy。

因此不能 claim Hierarchical MoE 一般无效，只能说当前 common/sense hierarchy 没有在这个 setup 中改善 functional specialization。

## Key Figures

![H0530a bucketed sense specialization](figures/h0530a_bucketed_sense_specialization.png)

这张图说明：`hierarchical_zipfian` 的 $S_{\mathrm{sense}}$ 没有超过 `flat_zipfian`，middle/tail 尤其弱。

![H0530a route NMI diagnostics](figures/h0530a_route_nmi_diagnostics.png)

这张图说明：hierarchy 的 routing 更关联 token/family，但这不是 functional evidence。

## Claim Update

Supported:

- routing structure 可以变得更 token/family-associated，而不带来 better expert causal function。

Weakened:

- 当前 two-level common/sense hierarchy 可以直接改善 tail context-sense functional specialization。

Still unclear:

- 加 load balance、exploration、route-function consistency regularizer 或 expert utility-aware objective 后，hierarchy 是否变得有效。

## What Cannot Be Claimed

不能 claim：

- Hierarchical MoE 一般失败；
- common/sense decomposition 没有研究价值；
- 额外容量一定有害；
- real-language MoE 会同样表现。

## Next Decision

下一步不应继续只改 routing geometry 或 hierarchy form。更合理的分支是把 objective 和 functional utility 绑定起来，例如 route-function consistency / utility-aware routing / exploration-load control。

## Result Documents

Detailed:

```text
Projects/from-attention-to-search/main/experiments/H0530a_hierarchical_common_sense_moe/detailed.md
```

Curated tables:

```text
Projects/from-attention-to-search/main/experiments/H0530a_hierarchical_common_sense_moe/tables/
```

Runner:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/scripts/run_h0529a_zipfian_frequency_shortcut.py
```

Config:

```text
XingyuD/Attention_Search_Experiments/active/synthetic_data_understanding/configs/h0529a_zipfian_frequency_shortcut.json
```

ACP job:

```text
pt-1d236anf
```
