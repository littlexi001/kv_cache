# Value 尾部补偿的无阈值收缩设计

## 问题

QKSieve 对每个 attention head 精确计算候选集合 `S`，并用 rank-16 INT4 ValueSketch 近似未选集合 `T`。当前实现把近似尾部完整加入 attention 输出，即固定使用 `alpha=1`。Llama-3.1-8B 的 LongBench 弱项实验显示，固定尾部补偿可能比只使用精确候选更差。

需要回答的可证伪命题是：

> ValueSketch 尾部估计存在请求、层和 head 相关的噪声；根据估计信号与噪声自动收缩尾部校正，可以在不使用长度阈值、任务规则、router 或 Full fallback 的情况下，同时避免短文本退化并保留长文本收益。

## 先验条件

1. 候选 token 使用原始 FP16 K/V 精确计算，因此候选部分没有 Value 量化误差。
2. 未选 token 的 Value 由低秩 INT4 表示，误差主要来自低秩残差和 INT4 量化残差。
3. 尾部由许多 token 加权求和。尾部权重越分散，有效样本数越大，残差越容易抵消；权重集中时，低秩误差更危险。
4. 是否需要尾部补偿取决于当前数值状态，不应直接由 `N < 64K` 这类长度规则决定。

## 数学模型

设 `S` 为选中 token，`T` 为未选 token，精确 softmax 的分子和分母为：

```text
A_S = sum_{i in S} exp(s_i) v_i
Z_S = sum_{i in S} exp(s_i)
A_T = sum_{i in T} exp(s_i) v_i
Z_T = sum_{i in T} exp(s_i)
y   = (A_S + A_T) / (Z_S + Z_T)
```

ValueSketch 给出尾部估计 `A_T_hat` 和 `Z_T_hat`。定义连续收缩输出：

```text
y_hat(alpha) = (A_S + alpha A_T_hat) / (Z_S + alpha Z_T_hat)
alpha in [0, 1]
```

`alpha=0` 是 selected-only，`alpha=1` 是当前固定尾部补偿。令：

```text
c_hat = y_hat(1) - y_hat(0)
```

把 `c_hat` 看作真实尾部校正加估计噪声。ValueSketch 在未选 token 上的每 token 残差二阶矩记为 `sigma_v^2`。使用 proxy 权重 `w_i_hat`，尾部输出噪声估计为：

```text
sigma_tail^2 = sigma_v^2 * sum_{i in T}(w_i_hat^2)
               / (Z_S + Z_T_hat)^2
```

对应的尾部有效 token 数为：

```text
n_eff = (sum_{i in T} w_i_hat)^2 / sum_{i in T}(w_i_hat^2)
```

两种不需要训练的收缩系数为：

```text
alpha_SURE  = clip(1 - sigma_tail^2 / ||c_hat||^2, 0, 1)
alpha_Ridge = ||c_hat||^2 / (||c_hat||^2 + sigma_tail^2)
```

当尾部校正很弱或残差很大时，`alpha` 自动趋近 0；当尾部信号强、有效 token 多且残差小，`alpha` 自动趋近 1。长度只通过权重分布间接影响结果，不进入决策公式。

## 实现契约

### 输入

- 当前冻结 QKSieve 的候选索引和精确 K/V。
- ValueSketch 的 `value_mean`、`value_basis` 和 INT4 codes。
- 融合检索 kernel 已有的 `Z_T_hat` 和尾部低秩系数。
- 每个 KV head 的 ValueSketch 残差二阶矩 `sigma_v^2`。

### 新增计算

1. 建索引时从已有 Value 采样计算一个 residual MSE 标量，不增加逐 token 索引。
2. 检索 kernel 在累加 `sum(w)` 时同时累加 `sum(w^2)`。
3. attention reduce kernel 计算 `||c_hat||^2`、`sigma_tail^2` 和一个标量 `alpha`。
4. 使用同一融合 kernel 输出 `y_hat(alpha)`。

### 不允许改变

- QK-balanced 坐标、qMSE 位宽、c64 采样阈值。
- 动态 token 预算和候选集合。
- 原始 FP16 K/V 与 exact sparse attention。
- 不引入训练、任务标签、长度阈值或 Full fallback。

### 调试输出

- 每层/head 的 `alpha`、`n_eff`、`sigma_tail^2`、`||c_hat||^2`。
- alpha 的均值、P10/P50/P90，以及与实际输出误差的关系。
- alpha=0、alpha=1、SURE、Ridge 和 oracle alpha 的误差对照。

## 通过条件

1. 离线真实 Q/K/V trace：SURE 或 Ridge 的平均与 P90 输出误差不劣于 `min(alpha=0, alpha=1)`。
2. 32K、64K、128K PPL：质量保持率相对两个固定 alpha 的最佳值下降不超过 0.2%。
3. LongBench 弱三任务：恢复 alpha=0 已观察到的质量，不重新出现固定 alpha=1 的下降。
4. 额外逐 token kernel 时间低于当前稳态 decode 的 2%；若融合后更快，则报告实际收益。

## 失败解释

- 若离线误差改善而 PPL 不改善，说明局部 attention L2 不是合适的下游风险指标，需要改用 `W_o` 度量。
- 若 `alpha` 总接近 0，ValueSketch 在当前预算下没有足够收益，应删除它而不是继续增加索引。
- 若 `alpha` 总接近 1，但 LongBench 仍退化，问题来自 proxy tail partition，而不是 Value 重建残差。
- 若不同 head 的最优 alpha 差异很大，则全局 alpha 假设被否定，应保留 per-head 标量。

## 当前结论边界

目前已经证明固定 `alpha=1` 会导致三个短 LongBench 弱项下降，也证明固定 `alpha=0` 在单个 128K 流上会从 99.78% 降到 97.53%。这否定了任一固定端点作为通用方案，但尚未证明 SURE/Ridge 优于两个端点。正式主方法冻结前必须完成真实 Q/K/V、跨主题、跨模型和融合 kernel 验证。

## 更新后的误差模型：block 条件残差

真实 Q/K/V trace 表明，仅用残差方差得到的 SURE 系数几乎总是 1，无法解释短任务退化。更稳定的现象是：ValueSketch 残差不是独立零均值噪声，而是同时依赖局部文本 block 和可用于检索的 Key 坐标。

令 `v_i_hat` 为 rank-16 INT4 重建值，`r_i = v_i - v_i_hat`。对 block `b` 保存：

```text
n_b      = block 中的 token 数
R_b      = sum_{i in b} r_i
Z_b      = sum_{i in b} z_i
z_i      = 重建 Key 的前 8 个坐标
mu_r,b   = R_b / n_b
mu_z,b   = Z_b / n_b
```

使用每 32 个 token 的请求内样本闭式求解一个每 KV head 的线性映射：

```text
B = Cov(r_i - mu_r,b, z_i - mu_z,b)
    [Cov(z_i - mu_z,b) + lambda I]^{-1}
```

其中 `lambda = 1e-3 * mean(diag(Cov(z)))`，没有跨数据训练。假设 block 内残差满足：

```text
r_i = mu_r,b + B (z_i - mu_z,b) + epsilon_i
```

对当前未选集合 `T_b`，融合检索扫描累加：

```text
W_b  = sum_{i in T_b} w_i_hat
G_b  = sum_{i in T_b} w_i_hat z_i
```

于是未选 Value 残差的条件估计为：

```text
A_res_hat = sum_b [W_b mu_r,b + B (G_b - W_b mu_z,b)]
```

最终把 `A_res_hat` 加到原 ValueSketch 尾部分子。候选 token 仍使用原始 FP16 V 精确计算，因此需要从 block 统计量中减去候选的计数、残差和坐标贡献，避免重复计算。

### 额外存储和计算

- 每 256 token、每 KV head 保存 128 维残差均值和 8 维 Key 坐标均值；FP16 约为 `136*2/256 = 1.0625 Byte/token/head`。
- 相对 FP16 K+V 的 `512 Byte/token/head`，新增约 0.21%。
- 每 KV head 的 `128x8` FP16 矩阵 `B` 与序列长度无关。
- token 扫描时新增 8 维加权和；block reduce 时新增一次 `128x8` 小矩阵乘。

### 新的可证伪命题

若 block 条件残差确实补回了系统性 Value 误差，则减少 exact 候选数后，它应仍达到当前 `top-1280 + 全局 ValueSketch` 的输出误差。若只能在同一 trace 上改善而跨长度、跨任务失效，则该模型被否定，不能进入主方法。
