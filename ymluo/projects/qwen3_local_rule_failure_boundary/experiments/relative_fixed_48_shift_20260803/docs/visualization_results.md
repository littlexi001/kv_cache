# 可视化契约与结果

## 作图前契约

主图使用三个离散条件，固定顺序为 `original_47`、`gap_plus_48`、`co_shift_plus_48`。

- Panel A：`P(basket)` 与最强错误 token 概率，纵轴固定为 0--1。
- Panel B：全词表输出 margin，绘制零决策边界。
- Panel C：四个原子证据 token 的全模型平均 attention mass，纵轴为百分比。
- 每个点直接标注数值；图注必须明确 `co_shift_plus_48` 保持相对距离，而非声称隐藏状态不变。

## 预期诊断

若 `co_shift_plus_48` 的相对距离校验失败，图不生成。若校验通过，则以 Panel A/B 回答概率问题，以 Panel C 判断概率变化是否伴随证据读取变化。

## 实测结果

三项位置不变量全部通过。主图位于 `outputs/analysis/relative_fixed_48_shift.png`。

图中最重要的对照是：把相同的 48 个 filler 放到证据之后时，`P(basket)` 从 45.34% 降到 6.52%；把它们放到证据之前、令证据与 Query 同移时，`P(basket)` 为 42.11%。同移组保持正 margin，而距离增加组穿过零决策边界。

可视化检查：三个 panel 均使用同一条件顺序；概率轴为 0--1；margin 图标出了零边界；attention mass 标为 36x32 个 head 的平均值。未发现裁切、标签遮挡或坐标含义错误。
