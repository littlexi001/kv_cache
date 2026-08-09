# 条件矩残差修正：设计

## 可证伪假设

QKSieve 的低秩 Value 近似残差并非完全独立于低比特 Key 坐标。若这种相关性在同一请求的留出 token 上仍成立，则只扫描前 8 个 Key 坐标的一阶加权矩，可以降低未选中 Value 尾部造成的 attention 输出误差；若相关性不成立，解析收缩系数应自动接近 0，使方法退化为现有的残差均值修正。

## 输入与固定条件

- 输入：真实模型 prefill 捕获的每层 Q/K/V、QKSieve 已有的低比特 Key 坐标、现有 rank-16 INT4 Value 近似。
- 固定：QK basis、位宽分配、候选集合、精确 top-k 消费端、Value rank 和量化位宽。
- 唯一变量：是否加入经留出样本校准的 8 维条件残差项。
- 不使用：任务标签、文本长度阈值、router、Full-attention 回退。

## 数学模型

对 token `i`，令 Value 近似残差为 `r_i = v_i - vhat_i`，低比特 Key 的前 `d=8` 维坐标为 `x_i`。训练子集上拟合：

`r_i - rbar_b = A (x_i - xbar_b) + epsilon_i`

其中 `b` 是 token 所在块。训练样本取每 64 个 token 一个，校准样本与其交错、同样每 64 个 token 一个。校准集上计算：

`gamma = clip(sum_i <p_i, y_i> / sum_i ||p_i||^2, 0, 1)`

其中 `p_i=A(x_i-xbar_b)`，`y_i=r_i-rbar_b`。在线使用 `gamma A`，而不是固定强度 `A`。

未选中尾部的线性修正为：

`Delta = gamma A (sum_i w_i x_i - W xbar_tail)`

它只需要在已有低比特 Key 扫描时额外累加 8 个加权矩，不应建立 FP32 token 坐标表。

## 实现约定

1. `gamma=0` 时输出必须严格退化为残差均值版本。
2. `gamma` 只能由当前请求 prefill 的交错留出 token 计算。
3. 第一阶段只验证真实 Q/K/V 层输出误差，不声明模型 PPL 或速度。
4. 只有跨主题、跨长度、跨 Qwen/Llama 均不劣后，才进入闭环 PPL。
5. CUDA 版本必须复用 packed Key code，且通过逐元素数值一致性测试后才能测速。

## 已知失败原因

- `gamma=1`：可能把训练样本中的偶然相关性全部加入，造成过修正。
- 留出样本太少：`gamma` 方差大；应报告有效校准数和增益分布。
- 固定 top-k 的 oracle 已失效：条件矩不能修复候选预算本身不足的问题，必须单独报告 exact-QK oracle。
