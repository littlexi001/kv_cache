# MomentSieve：输出误差驱动的条件矩 KV 检索（研究设计稿）

日期：2026-08-02

> `MomentSieve` 是暂定工作名。本文档记录方法推导、实现状态和实验门槛，不能把合成实验写成真实模型结论。

## 1. 为什么不能在旧版本之间继续打补丁

旧版本形成了清楚但不理想的 Pareto：

| 路径 | 优点 | 主要缺陷 |
|---|---|---|
| 112-bit `[4,1]` | 索引小、扫描快 | 低比特边界 token 容易错序，128K 存在长尾失败 |
| QKSieve-Fast 240-bit | 选择器较快 | 只计算 selected token，遗漏 Value 总贡献 |
| QKSieve-Robust | 质量接近 Full | 对所有遗漏 token 累积 Value 矩，扫描和 exact consumer 都更慢 |
| General | 分长度选择上述动作 | 能工作，但更像策略拼接，不是统一数值原理 |

新方法必须同时回答两个问题：

1. 哪些 token 值得进入 exact K/V attention？
2. 没有进入 exact attention 的 token，如何用很小成本保留其总贡献？

## 2. 面向多轮与 Agent 会话的约束

目标场景是长历史被连续复用：

- dense prefill 或历史写入只发生一次；
- Key 索引、Value 统计量和线性映射可以跨多轮复用；
- 每轮只增量编码新增 token；
- 每个 decode step 不能训练 router，也不能扫描完整 FP16 K/V；
- 不使用 Full-attention fallback；数值异常只能切换不同的稀疏估计路径。

因此要区分一次性成本与每步成本：

```text
T_agent(T) = T_build + T * T_sparse_step
T_full(T)  = T * T_full_step
break_even = T_build / (T_full_step - T_sparse_step)
```

正式实验必须报告 `T_build`、`T_sparse_step` 和 break-even 轮数，不能只报告稳态 attention kernel。

## 3. 统一的残差分解

对一个 KV head，把低比特 QK-balanced Key 坐标记为 `x_i`，Value 记为 `v_i`。在 block `b(i)` 内使用共享线性条件均值：

```text
v_i = mu_v[b] + A * (x_i - mu_x[b]) + r_i
```

其中：

- `mu_x[b]` 和 `mu_v[b]` 是 block 均值；
- `A` 是该 KV head 在当前请求内共享的 ridge 最小二乘映射；
- `r_i` 是 Key 坐标无法解释的 Value 残差。

`A` 的闭式解为：

```text
A = Cov(V, X) * (Cov(X, X) + lambda * I)^(-1)
```

这一步不使用任务标签、答案或测试 query。统计量可在 prefill 中增量累积：

```text
sum(x), sum(v), sum(x*x^T), sum(v*x^T)
```

## 4. 候选不应只按 QK 分数排序

设 query head `g` 的完整 softmax 权重为 `p[g,i]`，其 output projection 切片为 `W[g]`。若集合 `S` 使用 exact K/V，而遗漏 token 使用条件均值，则 layer 输出误差满足：

```text
||Delta y||
<= sum_{i not in S} sum_g p[g,i] * ||W[g] r_i||
<= sum_{i not in S} sum_g p[g,i] * ||W[g]||_2 * ||r_i||
```

在固定候选数 `k` 下，使这个上界最小的集合是按下面的 token 风险取 top-k：

```text
risk_i = sum_g p[g,i] * ||W[g]||_2 * ||r_i||
```

实际系统在 log 域计算：

```text
log_risk_i = logsumexp_g(
    proxy_score[g,i]
    - estimated_log_partition[g]
    + log_spectral_norm(W[g])
    + quantized_log_norm(r_i)
)
```

关键点：

1. `proxy_score` 来自低比特 Key 索引；
2. 每个 query head 的 `log_partition` 用 256 个 exact QK probe 估计；
3. `log ||r_i||` 按 block 做 4-bit min-scale 量化；
4. `||W[g]||_2` 是模型静态常数，不增加逐 token 索引；
5. 四个 GQA query head 共用一个 candidate set。

直接对四个原始 QK 分数取最大值不严格，因为 softmax 对每个 head 的任意常数平移不变。减去各自 `log_partition` 后，候选规则也具有相同不变性。

### 4.1 不使用指数函数的快速近似

严格 `logsumexp` 每 token 需要多次指数。部署 kernel 可用：

```text
fast_log_risk_i = max_g(
    proxy_score[g,i]
    - estimated_log_partition[g]
    + log_spectral_norm(W[g])
    + quantized_log_norm(r_i)
)
```

对于 GQA-4：

```text
fast_log_risk_i <= log_risk_i <= fast_log_risk_i + log(4)
```

因此它是有界的 kernel-friendly 近似，不是经验 router。是否足以保持候选质量必须由真实 Q/K/V 消融决定。

## 5. 不逐 token 扫描的条件 Value 尾部

QKSieve-Robust 慢的主要原因是对每个遗漏 token 计算指数权重，并累积 8 个 Key 矩。新方法用 block 的二阶矩直接计算 softmax 尾部。

若 block 内低维 Key 坐标近似：

```text
X | block b ~ Normal(mu_b, Sigma_b)
score = a^T X + c
```

则指数倾斜有闭式解：

```text
Z_b = n_b * exp(c + a^T mu_b + 0.5 * a^T Sigma_b * a)

E[exp(score) * X]
= Z_b * (mu_b + Sigma_b * a)
```

结合 Value 条件均值，block 尾部 numerator 为：

```text
N_b = Z_b * (mu_v[b] + A * Sigma_b * a)
```

selected token 仍使用原始 BF16 K/V 和 exact QK。其 proxy 指数质量从 `Z_b` 与一阶矩中扣除，剩余部分才作为 tail completion。

这把 Robust 的逐 token 尾部成本：

```text
O(N * rank)
```

变为：

```text
O(N / block_size * score_dim + k * score_dim)
```

其中第二项只处理已经选出的约 1,280 个 token。

## 6. 重尾异常的可检测修复

二阶闭式估计在重尾分布上可能出现：

```text
estimated_all_block_mass < exact_selected_proxy_mass
```

这不是隐蔽误差，而是可直接检测的不可能事件。处理方式不是 Full fallback：

1. 普通 block 继续使用快速二阶闭式；
2. 仅对发生负剩余质量的 block，从其一、二阶和中扣除 selected token；
3. 对剩余 token 的矩重新计算闭式尾部；
4. exact attention 仍只访问共享候选 K/V。

该路径称为 `hybrid moment repair`。它只改变尾部估计，不回退 Full attention，也不需要任务规则。

## 7. 索引与显存账本

以 head dim 128、BF16 K+V 共 `4096 bit/token/KV-head` 为分母，预期默认索引为：

| 组成 | 约 bit/token/KV-head |
|---|---:|
| 第一 16-D QK-balanced band，INT4 + FP16 scale | 80 |
| 条件 Value 残差 log-norm，blockwise INT4 | 约 4.03 |
| rank-8 条件 Value block 均值与全局映射 | 约 1–4，取决于 block size |
| 对角高斯 score 矩 | 约 0.6–2.3，取决于 block size |
| 合计 | 约 86–91 |
| 相对完整 BF16 K+V | 约 2.1%–2.3% |

完整协方差只作为质量诊断；若对角版本接近它，正式部署不应承担完整协方差的带宽和计算。

## 8. GQA 共享 exact consumer

旧 consumer 即使候选相同，也会为四个 query head 重复读取候选 K/V。新 CUDA 路径按一个 KV head 启动：

1. 候选 K 只加载一次，同时计算四个 exact QK；
2. 候选 V 只加载一次，同时累积四组 softmax numerator；
3. partial output 仍按 query head 保存，之后与各自 tail 合并；
4. candidate indices 从 `[B, QH, C]` 降为 `[B, KVH, C]`。

该 kernel 已完成本地代码，尚需在 CUDA 服务器上编译、数值对齐和独立计时。

## 9. 当前合成压力实验

协议：8,192 token、4% exact candidate、16-D score、8-D 条件坐标、32-D Value；覆盖高斯、block 漂移、Student-t 重尾、1% 离群点和非线性 Value，每类 3 个 seed。

| 尾部方法 | 平均相对输出误差 | 最坏相对输出误差 |
|---|---:|---:|
| candidate only | 1.0164 | 1.7202 |
| block mean | 0.3080 | 0.5133 |
| 逐 token 条件尾部 | **0.2190** | **0.3786** |
| 对角二阶闭式 | 0.2246 | 0.3898 |
| 完整二阶闭式 | 0.2228 | 0.3837 |
| 对角闭式 + hybrid repair | 0.2233 | 0.3898 |
| 完整闭式 + hybrid repair | **0.2215** | **0.3837** |

边界：这些数字只证明公式实现和压力行为值得继续，不能证明真实模型 PPL、LongBench 或 RULER 质量。

## 10. 冻结前的实验门槛

按下面顺序推进，前一项失败就不跑大任务：

1. 真实 32K Q/K/V trace：80-bit 与 240-bit；per-head、raw shared max、归一化 max、严格 output bound。
2. 同一 trace：QK-only、4-bit residual leverage；逐 token、对角闭式、完整闭式和 hybrid repair。
3. CUDA 数值验证：现有 consumer 与 GQA-shared consumer 最大绝对误差。
4. 独立 CUDA 计时：query、scan、candidate merge、tail closure、exact consumer、完整 attention call，禁止 stage 求和冒充完整时间。
5. 六主题 held-out PPL：32K 小样本；只有候选方法明确进入 Pareto 才扩到 64K/128K。
6. 8K/16K/32K/64K/128K 直接速度；检查短文本 crossover。
7. 多轮 Agent 场景：固定历史、连续问答和增量 append，报告 build amortization 与 break-even。
8. 最后才跑完整 LongBench、RULER 和第二模型。

预定冻结条件：

- 128K 质量保持不低于 99.5%；
- 不使用 Full fallback；
- auxiliary index 不高于 2.5%；
- 128K attention 至少达到 BinaryPC 同协议 Pareto，或在相同速度下质量显著更高；
- 8K 不明显慢于 Full，16K 起有稳定正加速；
- 独立 held-out 与第二模型不出现主题性 cliff。

## 11. 当前状态

已完成：

- output-error priority、softmax-normalized GQA priority 和谱范数上界实现；
- 4-bit / per-group leverage 费率核算与单元测试；
- token exact、Gaussian diag/full、selected-conditioned 与 hybrid repair；
- 7 项本地数值单元测试；
- 合成重尾与长度压力实验；
- GQA-shared exact consumer CUDA 源码与验证入口；
- 两组 8-GPU 小实验启动脚本和自动汇总脚本。

未完成：

- 远端 CUDA 编译和真实 trace 实验。当前本机到 `10.176.37.31` 没有内网路由，SSH 持续超时；服务器恢复后优先执行，不先启动 LongBench。
