# Query 二阶矩收缩系数敏感性：结果

## 实验设置

实验定义、严格配对键、指标和通过阈值见 `design.md` 与
`experiment_design.md`。正式结果只能来自
`20260810_qksieve_shrinkage_sensitivity_v1/summary.json`；在该文件通过完整性校验
之前，本页不填写数值，也不支持“0.75 对模型和主题稳定”的结论。

## 待填结果

需要报告五个系数在 1%、2%、4% 候选比例下的 top-2% recall、候选 attention
mass 与 score RMSE，并给出相对 `0.75` 的 cluster-bootstrap 95% 区间。主文只保留
一张紧凑敏感性表，逐模型、逐主题结果和失败条件放入附录。

## 当前结论边界

代码和协议已完成，GPU 实验尚未完成。当前不能声称固定收缩系数已经通过跨模型
敏感性验证。
