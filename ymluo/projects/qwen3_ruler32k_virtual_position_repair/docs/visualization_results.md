# 虚拟位置修复：图表结论

## 1. Alpha 冒烟扫描

![Alpha sweep](../outputs/analysis/smoke_alpha_sweep.png)

- x 轴：从原始位置向 Query 附近虚拟位置移动的比例 $\alpha$。
- 左图：完整输出的 RULER 官方分数；右图：首答案 token NLL。
- 紫色虚线：根据两条冒烟样例冻结的 $\alpha=0.05$。

UUID 样例随 $\alpha$ 在正确与错误之间多次翻转，说明“位置更近”不是单调增益。$\alpha=1$ 真正把远程候选移到 Query 前约 382 token，但 UUID 从 1.00 降至 0，多值任务从 1.00 降至 0.75。

## 2. 冒烟样例的 mass 与 QK

![Smoke mechanism](../outputs/analysis/smoke_mass_qk.png)

- 左图：答案证据获得的 attention mass。
- 右图：gold 候选 QK 增量减去 non-gold 候选 QK 增量；低于 0 表示位置修复更偏向非证据。

所有非零 $\alpha$ 的 gold-minus-non-gold QK 差都为负。UUID 样例的证据 mass 从 $\alpha=0$ 的 0.836% 降至 $\alpha=1$ 的 0.167%。因此完整近移没有选择性增强证据。

## 3. 未参与调参的 RULER-32K 任务

![Held-out task scores](../outputs/analysis/heldout_task_scores.png)

图中排除了两条用于选择 $\alpha$ 的样例。5% 修复相对 `local_global_postscore` 只改变两条样本：CWE 提高 0.1，但 `niah_multikey_2` 有一条由正确变为错误，任务均值从 100% 降至 50%。

## 4. Held-out NIAH 的 QK 选择性

![Held-out QK delta](../outputs/analysis/heldout_niah_qk_delta.png)

- x 轴：non-gold 候选的平均 QK 增量。
- y 轴：gold 候选的平均 QK 增量。
- 红色虚线：gold 与 non-gold 增益相同。

14 条 NIAH 样例中，9 条位于红线下方，8 条的 gold QK 本身下降。整体均值为 gold $-0.039$、non-gold $+0.100$。颜色较暗的错误样例同时具有显著负 gold 增量。

## 允许的结论

当前逐 token、统一强度、推理时位置近移方案没有改善 RULER-32K；它对 RoPE 相位的扰动是非单调的，并更容易增强大量 non-gold 候选。该结论只否定当前实现，不否定频率受限、分块保序或训练期适配的位置修复。
