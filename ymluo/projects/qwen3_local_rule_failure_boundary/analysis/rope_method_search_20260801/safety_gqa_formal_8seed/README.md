# Suppression-certificate safety：8 seeds 正式结果

**模型：** Qwen3-8B，NF4/BF16  
**长度：** 8K、32K、64K  
**样本：** 每个长度 8 seeds；每层每个 Query head 在 gold、conflict、lexical-format、filler 四类中各取 1 个匹配 token。  
**实现约束：** 32 个 Query heads / 8 个 KV heads 按真实 GQA 映射；prefix KV 只读；8K/32K 另测 untouched native，64K 以逐元素 no-op 的 instrumented path 为主基线。

## 结论

**RoPE suppression 不是“真实证据证书”，只凭 suppression 大小进行修复应当停止。**

原因有两条，并且方向一致：

1. 单 token、单 head/layer interaction 上，suppression 对 gold 与 conflict 的区分接近随机；所有 95% CI 均覆盖 0.5。
2. 把同类 token 的分数恢复后，gold repair 没有选择性改善答案；在 8K/32K 反而降低 gold 质量，64K 的改善也弱于 conflict/filler，说明收益来自非特异的 logit 扰动，而不是“救回真实证据”。

## 1. 证书能否区分 gold 与 conflict？

下表是 seed-macro AUROC；0.5 表示随机。`pre_suppression = pre_score - post_score`，`grid envelope` 允许在预先固定的相位网格中寻找最好分数。

| 长度 | 指标 | Gold vs conflict AUROC | 95% CI |
|---:|---|---:|---:|
| 8K | pre-suppression | 0.505 | [0.492, 0.518] |
| 8K | grid-envelope suppression | 0.504 | [0.487, 0.520] |
| 32K | pre-suppression | 0.493 | [0.482, 0.504] |
| 32K | grid-envelope suppression | 0.493 | [0.480, 0.505] |
| 64K | pre-suppression | 0.499 | [0.488, 0.509] |
| 64K | grid-envelope suppression | 0.499 | [0.486, 0.511] |

因此 suppression 能说明“RoPE 改变了这对 Q/K 的分数”，但不能说明 token 是真实、虚假、有用或有害。

## 2. 修复不同类别后，答案是否选择性改善？

数值是相对 no-op instrumented baseline 的变化。`ΔNLL < 0` 和 `Δmargin > 0` 才是改善。

| 长度 | 被修复类别 | ΔGold NLL [95% CI] | ΔGold-vs-conflict margin [95% CI] |
|---:|---|---:|---:|
| 8K | gold | +0.043 [-0.055, +0.132] | -0.563 [-1.000, -0.156] |
| 8K | conflict | +0.210 [+0.059, +0.356] | -0.656 [-1.141, -0.141] |
| 8K | lexical-format | -0.057 [-0.095, -0.017] | +0.375 [+0.016, +0.719] |
| 8K | filler | -0.036 [-0.089, +0.012] | -0.016 [-0.297, +0.250] |
| 32K | gold | +0.407 [-0.051, +0.912] | -0.453 [-1.031, +0.156] |
| 32K | conflict | +0.287 [-1.034, +1.549] | -0.609 [-2.579, +1.656] |
| 32K | lexical-format | -0.147 [-0.923, +0.649] | +0.031 [-0.922, +1.031] |
| 32K | filler | -0.210 [-0.502, +0.078] | -0.125 [-0.828, +0.500] |
| 64K | gold | -0.313 [-0.936, +0.266] | +0.344 [-0.250, +1.000] |
| 64K | conflict | -0.513 [-1.326, +0.023] | +0.141 [-1.031, +1.266] |
| 64K | lexical-format | -0.328 [-0.684, +0.005] | +0.406 [+0.016, +0.781] |
| 64K | filler | -0.257 [-0.615, -0.008] | +0.266 [+0.000, +0.641] |

最关键的反例是：

- 8K/32K 中，恢复 gold 的平均 NLL 与 margin 都向错误方向移动；
- 64K 中所有类别都可能改善，且 conflict 的 NLL 改善大于 gold；
- lexical-format/filler 也能获得相当或更强收益。

这排除了“只要把被 RoPE 压低的远程证据抬回来，就会改善答案”的简单因果解释。

## 3. 探索性 token/line 聚合

把同一证据 token 跨 36 层、32 heads 聚合后，某些事后选择的 reducer 在单个长度上可达到约 0.62--0.67 的 Gold-vs-conflict AUROC；但 seed 标准差约 0.22--0.32，最佳 reducer 随长度变化，line-level 胜率也不稳定。这是多 reducer 探索后的描述性信号，不是预注册、可泛化的 selector，不能推翻上面的正式 per-interaction 结果。LOSO 等权组合另行报告。

## 4. 方法决策

- **NO-GO：** 以 `pre-post suppression > threshold` 直接选择或修复 token。
- **NO-GO：** 把 suppression 称为 correctness / truth certificate。
- **可保留：** suppression 作为 RoPE 相位干扰的机制量，用于解释分数变化。
- **下一步只允许：** 加入独立、label-free 的内容/关系信号后再做选择，并与 conflict、lexical、filler 的匹配干预比较；或者把研究中心转向完整的 QK → mass → Value/residual → output-margin 因果链。

## 5. 对应产物

- `intervention_summary.csv`：按长度和 intervention 类别汇总及 seed bootstrap CI。
- `certificate_aurocs.csv`：正式 per-interaction AUROC。
- `case_rows.csv`：每个 seed/长度/干预的输出结果。
- `summary.json`：完整机器可读汇总。
- 相邻目录 `../safety_gqa_block_aggregation_8seed/`：探索性 token/line 聚合。

