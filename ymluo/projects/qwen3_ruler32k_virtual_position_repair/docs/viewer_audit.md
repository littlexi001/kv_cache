# 图表视觉核验

核验日期：2026-08-03。

- `smoke_alpha_sweep.png`：两条样例、完整分数与 NLL 的单位分离；$\alpha=0.05$ 已标注；非等间隔 alpha 使用真实数值位置，没有伪装成等间隔。
- `smoke_mass_qk.png`：mass 使用百分比，QK 使用原始 logit 单位；零线与冻结 alpha 清楚；两种指标未混用同一坐标轴。
- `heldout_task_scores.png`：明确写明排除两条调参样例；每个柱为该任务剩余样本均值，样本数另存于 CSV。
- `heldout_niah_qk_delta.png`：gold/non-gold 同单位，等值线清楚；颜色只表示官方分数，不暗示因果强度。
- 四张图标题、图例、刻度均可读，无截断或重叠；图中数字与 `summary.json`/CSV 一致。

限制：样本量只有 24 条 held-out，置信区间较宽；图只支持“当前位置修复没有改善且机制方向不利”，不支持对所有可能的位置修复方案下结论。
