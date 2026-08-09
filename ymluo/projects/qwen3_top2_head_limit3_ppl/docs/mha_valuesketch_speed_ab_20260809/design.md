# MHA 下 ValueSketch 有无补偿速度对照：设计

## 问题

在 QKSieve 已选出每个 attention head 的候选 token 后，ValueSketch 还要扫描未选 token，并估计它们对 softmax 分母和 Value 分子的合计贡献。本实验只回答一个问题：在 MHA 模型上，保留这项补偿会增加多少 attention 延迟和整模型稳态 decode 延迟？

可证伪假设：在候选集合完全相同的条件下，ValueSketch 会使 QKSieve attention 路径变慢，但在 64K 以上仍能相对 Full attention 保持明显加速。

## 先验与数学对象

令候选集合为 `S`，未选集合为 `T`，attention 分数为 `s_i`。无补偿版本只计算：

`y_fast = sum(i in S) exp(s_i) v_i / sum(i in S) exp(s_i)`。

有补偿版本额外用 rank-16、block-256、INT4 ValueSketch 近似：

`Z_T = sum(i in T) exp(s_i)`，

`N_T = sum(i in T) exp(s_i) v_i`，

最后返回：

`y_robust = (sum(i in S) exp(s_i) v_i + N_T) / (sum(i in S) exp(s_i) + Z_T)`。

因此补偿的新增计算应出现在两处：selector 扫描时累计 `Z_T` 和低秩系数；最终 selected attention 中合并尾部输出。

## 实现合同

- 模型：`NousResearch/Yarn-Llama-2-7b-128k`，标准 HuggingFace Llama 实现。
- 结构：32 个 Query heads、32 个 KV heads、head dimension 128，即真实 MHA。
- Key selector：相同 QK-balanced plain index、相同有限样本分位数、相同候选上限 1,280。
- 无补偿：只对候选原始 FP16 K/V 做精确 attention。
- 有补偿：候选不变，额外启用 rank-16、block-256、INT4 ValueSketch。
- 禁止项：router、任务规则、Full fallback、按长度切换方法。
- 公平性硬条件：两条 synthetic MHA 路径的 32 个 head 必须满足阈值、候选数量和候选 token 集合完全一致。

## 输出

1. 每层 attention 路径：Full、无补偿、有补偿的独立 CUDA-event 延迟。
2. 真实模型：32K、64K、128K 下生成 64 token，跳过前 16 token 后的平均稳态延迟。
3. 一次性 QK 索引和 ValueSketch 构建时间单独报告，不混入稳态延迟。
