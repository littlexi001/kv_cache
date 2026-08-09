# ValueSketch 去留消融：问题与方法

## 可证伪假设

在每个 head 最多保留 1,280 个 token 的 QKSieve 中，未选 token 的 rank-16 INT4 ValueSketch 补偿对普通长文本的平均质量收益低于 0.3%，因此不值得放入论文主方法。

## 先验与数学对象

对候选集合 `S`，无补偿版本只计算候选上的精确 attention：

`y_S = sum_{i in S} exp(s_i) v_i / sum_{i in S} exp(s_i)`。

补偿版本额外用 rank-16、block-256、INT4 的 ValueSketch 近似未选集合 `T` 的 softmax 分子与分母，再与候选精确结果合并。若 `S` 已覆盖绝大部分 attention mass，尾部项对输出和目标 token NLL 的影响应很小。

## 实现合同

- 相同 QK-balanced request-local 低比特 Key 索引。
- 相同 sampled-quantile selector 和 top-1,280 上限。
- 相同原始 FP16 K/V、prompt、目标 token 和模型权重。
- 唯一变量：是否启用 rank-16 INT4 ValueSketch 尾部合并。
- 输出：NLL、相对 Full 的质量保持率、top-1 一致率、稳态 step 延迟、固定准备成本和显存。

## 判定

- 支持删除：三条流的平均质量增益小于 0.3%，且没有单条流出现超过 1% 的明显修复。
- 支持保留：平均质量增益至少 0.3%，或它稳定修复无补偿版本的显著失败。
- 证据不足：不同流方向冲突或样本方差大，需要体育、医学、LongBench/RULER 再判定。
