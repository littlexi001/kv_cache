# 大语言模型参数空间的奇异性：来源、机制与 MoE 解决方向

## 一句话总结

> 高频 pattern 通过梯度外积将 hidden state 的公共方向写入 $W_q, W_k$ → 形成共享高收益路由通道 → 残差裸路径绕过 RMSNorm 使之逐层累积 → 参数空间呈现少数大奇异值主导的结构。存在更均匀的参数空间和表征空间使学习效率更高（hard reweighting 验证——$E$: 2.29→1.40, $W_q$: 1000→134, $q$ vectors: 2470→239）。MoE 应从 $W_q, W_k$ 层切入，基于每层的 RMSNorm(H) 而非静态 E 做门控。

---

## 1. 事实：参数空间是奇异的

- $W_q, W_k$（注意力路由矩阵）表现出显著的谱集中：$\sigma_1 / \sigma_{512} \approx 8-15\times$（Qwen3-0.6B），toy 中可达 134×
- $W_v$（内容矩阵）几乎不集中（$\sim 1.2\times$）：内容分散在多个方向，不需要单一高增益通道
- Residual stream 的主奇异方向从 L2 开始锁定至 L26（$\cos \approx 1.000$），但 RMSNorm 后与原始残差的余弦仅 0.19——归一化完全改变了 Q/K 的输入空间
- Q/K 输出侧（$U_Q[0]$ vs $U_K[0]$）在深层趋向完全对齐（$\cos \to 1.0$）：注意力路由坍缩为一维

---

## 2. 机制：参数大奇异方向的三步因果链

### 2.1 Hidden states 产生公共方向

- **直接原因**：Zipf 频率 + 嵌套语言结构 → 高频 token/pattern 在 batch 中反复出现 → hidden state 矩阵 $H \in \mathbb{R}^{N \times d}$ 的 SVD 出现主导方向
- **物理本质**：该方向 = 最常被预测的 token 的 embedding 方向的加权平均。不是从 E 的静态 SVD 结构继承的（Qwen 中在 L0 就被 LayerNorm 清零至 $\cos = 0.03$），而是训练动力学创造的

### 2.2 梯度外积将该方向写入参数

- **数学**：$\partial L / \partial W_q = \sum (\partial L / \partial q_s) \otimes h_s$
- 每个样本贡献一个 rank-1 矩阵，其右奇异方向 = $h_s$（normalized）
- 所有 $h_s$ 共享主导方向 → 累加的梯度矩阵 $gW_q$ 以该方向为主 → 梯度下降数千步后 $W_q$ 的 $Vh[0]$ 收敛到 $H$ 的 $Vh[0]$
- **实验证据**：toy 残差模型中 $\cos(Vh_H[0], Vh_{W_q}[0]) = 0.99$；Qwen 中 $\cos(H_{\text{norm}}, Q) = 0.06$–$0.34$（RMSNorm 压制但不消灭）

### 2.3 反馈闭环放大 + 残差绕过 RMSNorm

- $W_q.Vh[0] \approx H.Vh[0]$ → $q$ 沿此方向被 $\sigma_1$ 放大 134×(toy) → $q$ 坍缩为一维 → 注意力失去区分度 → 更多梯度往此方向流 → $\sigma_1$ 更大
- **残差路径的作用**：Pre-Norm 中 $h_{\text{new}} = h + \text{Attn}(\text{LN}(h))$。残差的裸 identity path 使公共方向逐层无损传导（$\cos = 1.000$ for L2→L26），而 RMSNorm 只过滤进入 Q/K 的前向路径，不过滤残差加法路径
- **实验证据**：去掉裸 identity path 后（double norm: $h = \text{LN}(h) + F(\text{LN}(h))$），Raw H 不再冻结（$\cos$ 在 0.26–0.95 间波动），但深层 $W_q$ 变得极端集中（$\sigma_1/\sigma_4 = 46\times$ vs standard 的 $1.3\times$）

---

## 3. 存在性：更均匀的空间学习效率更高

### 3.1 硬证据：hard reweighting 对照

| 指标 | no_rew（plain） | hard_rew（$\alpha=1.0$） |
|------|---------------|----------------------|
| 收敛准确率 | 50%（卡死） | **100%** |
| $E$ 的 $\sigma_1/\sigma_4$ | 2.29 | **1.40** |
| $W_q$ 的 $\sigma_1/\sigma_4$ | 1000× | **134×**（7.5× 改善） |
| $q$ vectors 的 $\sigma_1/\sigma_4$ | 2470× | **239×**（10× 改善） |
| $H \leftrightarrow Q$ 对齐 $\cos$ | — | **0.99**（稳态） |
| 梯度在 $v_1 \sim v_4$ 分布 | 不均衡 | **近乎均等**（1.3× 偏好） |

**更均匀的空间不仅存在，而且恰好对应了 no_rew 永远到不了的收敛终点。**

### 3.2 维度鲁棒性

维度从 4D 扩展到 24D，hard_rew 的 $\cos(H, Q)$ 始终保持 > 0.5 且远超随机基线（4–94×）。更均匀的空间在不同维度下都可达到。

### 3.3 梯度偏好消解

hard_rew 收敛后，梯度在 $v_1 \sim v_4$ 四个方向几乎均等。均匀空间是梯度下降的**稳态吸引子**——一旦构造出来，梯度就不会再把它推回集中状态。

---

## 4. 对 MoE 设计的指导

### 4.1 门控应基于每层的 RMSNorm(H)，而非静态 E

| 层深度 | H 的状态 | 门控依据 |
|--------|---------|---------|
| embed | H ≈ E[token]，$\cos(H, E) = 0.40$ | 可用 token identity |
| L0→L1 | LayerNorm + Attention 清零 E 结构，$\cos(H, E) \to 0.03$ | **必须用当前 H** |
| L2→L26 | H 方向锁死（$\cos = 1.000$），但与 E 几乎无关（$\cos = 0.068$） | E 完全不可靠，门控可复用（方向不变） |
| L27 | 出口重塑，$\cos$ 陡变 | 重新用当前 H |

- **Qwen 中 $\cos(H_{\text{norm}}, Q)$ 只有 0.06–0.34，不是 0.99**：因为 RMSNorm 剥离了均值方向，Q 对齐的是一个去均值的 H。MoE 门控如果只看 E，在深层会完全失效
- H 从 L2→L26 方向锁死意味着**门控决策可以复用 24 层**——不需要每层重新路由

### 4.2 Wq, Wk 是最需要 MoE 的模块，Wv 不需要

| 模块 | $\sigma_1/\sigma_{512}$ | 是否需要拆分 |
|------|----------------------|------------|
| $W_q$ | 8–15×（集中） | **是——核心瓶颈** |
| $W_k$ | 8–15×（集中） | **是** |
| $W_v$ | 1.2–1.4×（平坦） | 否——内容自然分散 |
| $E$ | 12×（中等） | 可共享——不是瓶颈 |

### 4.3 综合方案

```
E  (共享)  →  H₀  →  [Gate₀(H_norm₀)]  →  Expert₀{Wq₀, Wk₀, Wv₀}
                  →  [Gate₁(H_norm₀)]  →  Expert₁{Wq₁, Wk₁, Wv₁}
                  →  ... 
                  →  H₁ = H₀ + Attn_out
                      
                  →  [Gate复用*]  →  Expert₀{Wq, Wk, Wv}   (L2→L26, H方向锁死)
                                     ...
                  →  H_final → lm_head
```

关键设计：
1. **门控在 RMSNorm(H) 上**，不在 E 上
2. **L2 之后门控可冻结复用**（H 方向锁死）
3. **$W_q, W_k$ 分 expert，$W_v$ 和 $E$ 可共享**
4. **结合 loss reweighting** 压低 $W_q$ 初始集中度

---

## TODO

- [ ] 在 toy 中构建 MoE prototype，验证 expert 拆分 $W_q, W_k$ 是否打破 Q/K 输出坍缩
- [ ] 验证 L2 后冻结门控是否可行
- [ ] 在真实 LLM 上确认 H 方向锁死是否普遍成立（当前仅 Qwen3-0.6B 数据）
- [ ] 设计 expert 的语义分配策略（按 frequency？按 domain？按 syntactic role？）
