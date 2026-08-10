# Query 二阶矩收缩系数敏感性：问题与方法

## 研究问题

冻结的 QKSieve-Robust 使用请求内 prompt 最后 8 个 Query 估计二阶矩，并固定
收缩系数 `lambda=0.75`。需要回答的可证伪问题是：在模型和文本主题改变时，
`0.75` 是否仍接近检索质量最好的固定系数，而不是某一个开发样本上的偶然选择。

## 数值先验与数学模型

每个 layer、KV head 的原始 Query 二阶矩记为 `Cq`，各向同性尺度为
`mu=tr(Cq)/d`。实验只改变

`Cq(lambda) = (1-lambda) Cq + lambda mu I`。

`lambda` 较小时保留八个 prompt Query 的方向性，但有限样本噪声和小特征值会被
放大；`lambda` 较大时条件数更稳定，但真实的 Query 各向异性会被抹平。坐标变换、
240-bit 位宽分配、Key stride、精确候选消费和所有其他变量保持不变。

## 实现合同

- 输入：Llama-3.1-8B 与 Qwen3-4B，各自的体育、医学 32K trace。
- 校准：只用 prompt 最后 8 个 Query；所有 decode Query 都是 held-out。
- 系数：`0, 0.25, 0.5, 0.75, 0.9`。
- 方法：QK-balanced、15 个物理 rate unit、stride-32 Key 样本。
- 候选比例：1%、2%、4%；exact top-2% 作为检索目标。
- 输出：top-2% recall、候选 attention mass、top-2% mass recall、score Pearson、
  score RMSE，以及相对 `0.75` 的按“模型/主题/layer”聚类配对 bootstrap 区间。

该实验是冻结超参数的敏感性消融，不用于重新选择 `lambda`。若 `0.75` 失败，结论
应改成明确局限，而不是在测试 trace 上重新调参。
