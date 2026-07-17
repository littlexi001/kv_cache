# Qwen3-0.6B 各 head 对真实证据的关注：冲突配对实验

日期：2026-07-14

## 结论摘要

Qwen3-0.6B 对真实规则的注意力高度集中在中后层少数 head，而不是均匀分布。最稳定的高证据 head 是：

```text
L26H02, L17H08, L21H13, L11H11, L24H06, L26H00
```

其中存在明显分工：

- `L26H02` 是最强的 gold-vs-DECOY 判别 head；无竞争链且含冲突时，13.54% 注意力落在两条 gold rules，gold token 的注意力密度是 DECOY token 的约 7.3 倍。
- `L17H08` 是高覆盖检索 head；无冲突时，其 top 2% attention logits 覆盖 43.5% 的 gold-rule tokens。但它最容易被真正冲突劫持，gold mass 从 14.61% 降至 11.15%，DECOY mass 同时增加 9.29 个百分点。
- `L21H13` 也是高质量检索 head，但对冲突和竞争链均有中等脆弱性。
- `L19H12` 在存在 4 条竞争链时仍能同时保持略高于 DECOY 和 competitor 的 gold-token 密度，但 gold mass 只有 1.66%，属于较弱但相对平衡的 selector。

冲突并没有重新组织“哪些 head 负责证据”的整体格局：含冲突与不含冲突的 448-head gold-mass map 相关系数约为 0.994，top-10 head 重合 8--9 个。它主要降低已有证据 head 的强度，并把部分注意力转向同前件、错误后件的 DECOY rules。

竞争链的影响明显大于 DECOY 冲突：增加 4 条有效但起点错误的竞争链后，平均 gold mass 下降约 57%，gold-rule selectivity 下降约 61%。这与此前“竞争链 / start-code 绑定是主要失败源”的结论一致。

## 1. 严格配对设计

固定条件：

| 变量 | 设置 |
|---|---:|
| model | Qwen3-0.6B |
| context body | 8192 tokens |
| actual prompt | 8265 tokens |
| gold-rule depth | 50% |
| gold chain length | 2 |
| requested rule gap | 512 tokens |
| DECOY count | 16 |
| competitor chains | 0 / 4 |
| seeds | 0--7 |
| condition | conflict / nonconflict |
| variants | 32（16 对） |

每个 conflict 样本包含 16 条：

```text
DECOY RULE: gold antecedent -> wrong consequent
```

它的 nonconflict 配对样本保留：

- 8265-token 总长度；
- 所有 rule span 的位置和 token 数；
- `DECOY RULE` 标签；
- wrong consequent；
- filler、问题、gold chain 和候选答案集合。

唯一变化是把每条 DECOY 的 antecedent 改为等 token 长度、且不在 gold chain 中的近似 code。每对样本总共只改变 16 个 prompt tokens。因此，本实验中的“无冲突”准确含义是：仍有等量 DECOY 数据，但它们不再与 gold rule 共享前件。

## 2. 指标

指标在 `Answer:` 后预测第一个答案 token 的 query 上逐层逐 head 计算：

- `gold_rule_mass`：两条真实 `VERIFIED RULE` span 的 attention mass；
- `gold_rule_selectivity = gold / (gold + decoy + competitor)`；
- `gold_uniform_enrichment`：gold mass 除以 gold span 占全部 KV tokens 的比例；
- `gold_vs_decoy_log2_density_ratio`：按 span token 数校正后的 gold/DECOY 密度比；
- `gold_top2_token_recall`：该 head 的 top 2% logits 覆盖了多少 gold-rule tokens；
- candidate accuracy 和 gold-vs-best-wrong margin。

每个 prompt 有 58 个 gold-rule tokens、486--487 个 DECOY-rule tokens；有竞争链时另有 252 个 competitor-rule tokens。top 2% 对应 166 / 8265 个 KV tokens。

## 3. 整体结果

| 条件 | competitors | n | candidate acc | margin | gold mass | selectivity | uniform enrichment | top-2% gold recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nonconflict | 0 | 8 | 25.0% | -0.261 | 0.861% | 0.0709 | 1.227x | 0.0779 |
| conflict | 0 | 8 | 0.0% | -0.336 | 0.829% | 0.0683 | 1.181x | 0.0759 |
| nonconflict | 4 | 8 | 0.0% | -0.278 | 0.369% | 0.0275 | 0.525x | 0.0332 |
| conflict | 4 | 8 | 0.0% | -0.343 | 0.352% | 0.0267 | 0.502x | 0.0328 |

这里的 gold mass 是对全部 448 个 head 的平均，因此会稀释少量高证据 head。head-level 结果更有解释力。

## 4. 哪些 head 最关注真实证据

### 4.1 无竞争链

| 条件 | head | gold mass | selectivity | enrichment | top-2% gold recall | log2 gold/DECOY density |
|---|---|---:|---:|---:|---:|---:|
| nonconflict | L17H08 | 14.61% | 0.190 | 20.82x | 0.435 | +0.92 |
| nonconflict | L26H02 | 14.30% | 0.480 | 20.37x | 0.207 | +2.95 |
| nonconflict | L21H13 | 10.63% | 0.157 | 15.15x | 0.371 | +0.60 |
| nonconflict | L11H11 | 9.40% | 0.163 | 13.39x | 0.293 | +0.35 |
| conflict | L26H02 | 13.54% | 0.466 | 19.30x | 0.181 | +2.87 |
| conflict | L17H08 | 11.15% | 0.136 | 15.88x | 0.403 | +0.36 |
| conflict | L21H13 | 9.37% | 0.138 | 13.35x | 0.360 | +0.38 |
| conflict | L11H11 | 9.15% | 0.154 | 13.03x | 0.282 | +0.30 |

`L17H08` 的 gold-token 覆盖最高，但 `L26H02` 的真假规则判别最干净。两者并不是同一种功能。

### 4.2 四条竞争链

| 条件 | head | gold mass | selectivity | enrichment | log2 gold/DECOY density |
|---|---|---:|---:|---:|---:|
| nonconflict | L21H13 | 6.01% | 0.088 | 8.57x | +0.18 |
| nonconflict | L17H08 | 5.23% | 0.065 | 7.46x | -0.80 |
| nonconflict | L26H02 | 4.05% | 0.089 | 5.77x | +1.34 |
| nonconflict | L24H14 | 4.02% | 0.067 | 5.73x | -0.71 |
| conflict | L21H13 | 5.22% | 0.073 | 7.43x | -0.11 |
| conflict | L17H08 | 4.18% | 0.049 | 5.96x | -1.28 |
| conflict | L26H02 | 3.88% | 0.084 | 5.53x | +1.18 |
| conflict | L26H00 | 3.69% | 0.048 | 5.26x | -0.52 |

`L26H02` 仍能区分 gold 与 DECOY，但含冲突、4 competitors 时它有 27.41% attention mass 落到 competitor rules，gold 只有 3.88%。因此它是“真假标签/前件冲突判别”较强的 head，但不是 start-code-to-chain 绑定的完整解决方案。

## 5. 冲突的配对效应

下表均为 `conflict - nonconflict`，括号内为跨 seed 标准误：

| competitors | Δ accuracy | Δ margin | Δ gold mass | Δ selectivity | gold mass 下降的 seed |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.250 (0.164) | -0.0748 (0.0363) | -0.0321 pp (0.0123) | -0.00257 (0.00066) | 7 / 8 |
| 4 | 0.000 (0.000) | -0.0648 (0.0283) | -0.0162 pp (0.0078) | -0.00077 (0.00036) | 6 / 8 |
| pooled | -0.125 (0.085) | -0.0698 (0.0223) | -0.0242 pp (0.0073) | -0.00167 (0.00043) | 13 / 16 |

平均效应看起来不大，是因为大多数 head 本来几乎不看 gold rules。对证据 head 的影响更明显：

| head | competitors | Δ gold mass | SEM | Δ selectivity | Δ DECOY mass |
|---|---:|---:|---:|---:|---:|
| L17H08 | 0 | -3.465 pp | 1.160 pp | -0.054 | +9.293 pp |
| L21H13 | 0 | -1.261 pp | 0.447 pp | -0.019 | +1.443 pp |
| L21H01 | 0 | -0.814 pp | 0.283 pp | -0.016 | +6.263 pp |
| L26H02 | 0 | -0.750 pp | 0.339 pp | -0.013 | +0.164 pp |
| L17H11 | 0 | -0.680 pp | 0.142 pp | -0.019 | +11.265 pp |
| L17H08 | 4 | -1.051 pp | 0.589 pp | -0.016 | +5.510 pp |
| L21H13 | 4 | -0.797 pp | 0.237 pp | -0.014 | +2.528 pp |

这说明冲突的主要机制不是让所有 head 轻微变差，而是对少数证据检索 head 产生集中劫持。

## 6. 分层结构与两步证据

平均 gold mass 最高的层主要是：

```text
L26, L24, L17, L21, L25, L19
```

高证据 head 通常对第一条规则 `T0` 的 attention 高于第二条 `T1`。例如无竞争、无冲突时：

| head | T0 mass | T1 mass |
|---|---:|---:|
| L17H08 | 10.41% | 4.20% |
| L26H02 | 8.22% | 6.07% |
| L21H13 | 5.67% | 4.96% |
| L11H11 | 7.36% | 2.04% |

这表明 answer query 上的注意力不是只检索最后一步 consequent；部分 head 更强地绑定 start code 与第一步规则，另一些 head（如 `L26H02`、`L21H13`）对两步更均衡。

## 7. 解释

1. **证据选择是稀疏的 head 功能。** 多数 head 几乎不读 gold rules，少数中后层 head 的 enrichment 达到 10--20 倍。
2. **高覆盖不等于高判别。** `L17H08` 覆盖 gold tokens 最广，但同时大量注意 DECOY；`L26H02` 覆盖较低，却能更好地区分 gold 与 DECOY。
3. **冲突主要劫持既有检索 head。** head map 几乎不变，但高证据 head 的 gold mass 下降、DECOY mass 上升。
4. **竞争链是更强的绑定难题。** 有效但起点错误的 `VERIFIED RULE` 会同时压低 gold mass 和 selectivity；没有单个 head 能稳定解决“DECOY 抑制 + start-code 绑定”两件事。
5. **attention 是诊断，不是因果证明。** 当前结果说明信息流位置和错误相关机制；要证明这些 head 对答案是必要/充分的，还需要 head ablation、attention patching 或 KV 保留干预。

## 8. 限制与下一步

- 每个 cell 只有 8 个 seeds，head 排名和 paired effect 仍是探索性结果；未做 448-head 多重比较校正。
- 只分析答案前第一个 query；没有跟踪生成多个答案 token 时的变化。
- candidate set 很难，本设置的绝对 accuracy 较低，因此 margin 比 accuracy 更稳定。
- 结论来自人造 code/rule 数据，迁移到自然语言证据需要再验证。

优先的因果实验：

1. 单独消融/保留 `L26H02`、`L17H08`、`L21H13`，比较 candidate margin；
2. 在 conflict/nonconflict 配对间 patch 这些 head 的 attention output；
3. 只保留这些 head 的 top-2% KV tokens，检查是否能复现全模型效果；
4. 在真实多跳 QA 上标注 gold evidence span，复用同一套 per-head 指标验证 head 身份是否稳定。

## 9. 产物

- 原始逐 head 数据：`../outputs/head_evidence_attention_8k_20260714/head_attention.csv`
- 原始逐事件数据（gzip）：`../outputs/head_evidence_attention_8k_20260714/head_event_attention.csv.gz`
- 条件汇总：`../outputs/head_evidence_attention_8k_20260714/condition_summary.csv`
- 逐 head 汇总：`../outputs/head_evidence_attention_8k_20260714/head_summary_by_condition.csv`
- 配对冲突效应：`../outputs/head_evidence_attention_8k_20260714/paired_conflict_effect_by_head.csv`
- top heads：`../outputs/head_evidence_attention_8k_20260714/top_heads.csv`

![conflict gold mass heatmap](../outputs/head_evidence_attention_8k_20260714/heatmap_conflict_gold_mass.png)

![paired conflict effect heatmap](../outputs/head_evidence_attention_8k_20260714/heatmap_conflict_delta_gold_mass.png)
