# 大语言模型参数空间的奇异性：两阶段理论、实证验证与 Subspace Split 解法

## TL;DR

1. **最大奇异方向为何出现**：最初的大奇异方向主要来自 Zipf 分布中的高频 token / phrase / shared pattern。这些 pattern 先在梯度中形成 common mode，再通过 tied embedding、residual stream 和梯度外积写入表征空间与参数空间。Nested structure 更像传播和放大机制，而不是最初来源。Toy 模型中 $\cos(Vh_H, Vh_{W_q}) \approx 0.99$；Qwen 中部分层/模块的 activation-parameter input-side 对齐高于随机基线，但不是所有层所有矩阵都强对齐。

2. **奇异值为何持续增大**：两阶段理论解释了方向和 gain 的分离：方向对齐有饱和项 $(1-c^2)$（Phase 1，方向先稳定）；但 CE 对 finite margin 仍有正 residual pressure，奇异值方程没有同样的方向饱和项（Phase 2），因此方向稳定后 $\sigma$ 仍会继续增长，只是增长速度随 margin 增大而下降。Tied embedding 可能让输入侧和输出侧方向共同漂移，使方向不完全冻结；但持续奇异值增长的核心仍是 CE 对 finite margin 保留正 residual pressure。

3. **大奇异值为何伤害长尾 feature 学习**：在 tied embedding + shared dense parameter path 中，tail margin 可以写成 $m_T = b\sigma_t\tau - a\sigma_c\rho$。其中 $b\sigma_t\tau$ 是 tail 自己方向的有用 logit，$a\sigma_c\rho$ 是 common high-gain direction 在 tail 样本上的错误 logit 竞争项。当 common gain $\sigma_c$ 持续增大、tail 表征中含 common 投影 $\rho$ 时，tail 必须先抵消 common logit，才能学习自己的 tail direction（no_rew 实验：$\sigma_1/\sigma_4 = 115\times$，准确率仅 38%）。

4. **方法**：Subspace Split / CRS — 将 hidden state 拆成 common component 和 residual component，分别送入独立参数矩阵。它的目的首先是验证机制：如果 long-tail hurt term 是 $a\sigma_c\rho$，那么结构上降低 $\rho$、并让 residual path 不共享 common gain $\sigma_c$，就应该改善 long-tail 学习。在 trigram toy 的 5-seed matched-lr 实验中，Split α=0.3 在不做 loss reweighting 的情况下达到 100% accuracy（同 lr 下 no_rew=38%, hard_rew=75%），且 residual branch 谱明显更平（Wq_c=1.1×, Wq_r=5.3×, E=1.9×）；Codex 的独立 5-seed causal attention 实验也验证了 CRS 显著加速 tail 收敛。该方法目前主要作为机制验证（2-token trigram + 单层 attention + 固定 lr 下的结论），尚未在真实 LLM 上测试。

---

## 1. 参数空间奇异性的事实

### 1.1 现象的四个关键特征

| 特征 | Toy 证据 | Qwen3-0.6B 证据 |
|------|---------|---------------|
| $W_q, W_k$ 的光谱集中度远高于 $E, W_v$ | $W_q$=134×, $W_v$=1.2× | $W_q$=8-15×, $W_v$=1.3× |
| 奇异方向对齐输入 H 而非静态 E | $\cos(Vh_H, Vh_{W_q}) \approx 0.99$ | $\cos(H, Q)$=0.26 vs $\cos(E, Q)$=0.06（4.4× 优势）|
| 残差流的 common 方向在 L2-L26 间冻结 | — | $\cos(Vh_H^{\ell}, Vh_H^{\ell+1})$=1.000 |
| RMSNorm 改变 hidden state 的 scale 和方向分布，使 common 方向在 QKV 输入中的贡献减弱；但残差 identity path 绕过该变换 | $\cos(Raw, Norm) \approx 0.19$ | 同 |

### 1.2 梯度外积是耦合机制

对任意线性层 $y = W \cdot h$：

$$\frac{\partial L}{\partial W} = \sum_{\text{samples}} \underbrace{\frac{\partial L}{\partial y}}_{\text{1×d}} \otimes \underbrace{h}_{\text{1×d}}$$

这是一个外积。每个样本贡献一个 rank-1 矩阵，其**右侧（输入空间）的奇异方向 = $h$ 的方向**。所有 sample 的 $h$ 共享一个公共方向 $\bar{x}$（全体 hidden states 的质心，对齐于高频 token K 的方向）→ 累积梯度以该方向为主 → $W$ 的 $Vh[0]$ 收敛到 $\bar{x}$。

Toy 中：$\cos(Vh_H[0], Vh_{W_q}[0]) = 0.99$。

---

## 2. 两阶段理论 + Tied Embedding 扩展

### 2.1 原始两阶段分析（rank-1 模型，$u$ 固定）

对于模型 $m = \sigma \cdot u^\top \cdot (v^\top x)$，其中 $v$ 是输入侧奇异向量，$u$ 是输出方向（固定）：

**Phase 1 — 方向发现**：
$$\frac{dc}{dt} = r(m) \cdot \sigma \cdot (1-c^2)$$
其中 $c = v^\top x$ 是方向对齐度。$(1-c^2)$ 是饱和因子：$c \to 1$ 时方向旋转趋近于零。

**Phase 2 — 增益放大**：
$$\frac{d\sigma}{dt} = r(m) \cdot c$$
**没有饱和因子。** $c \approx 1$ 时 $d\sigma/dt \approx r(m)$——全油门。这是 $\sigma$ 持续增长的根本原因。

两相变形的闭合解：$1 - c(t)^2 = (1-c_0^2) \cdot \exp(-(\sigma(t)^2 - \sigma_0^2))$。对齐误差随 $\sigma$ 增长指数衰减——方向在 Phase 1 就已收敛，Phase 2 主要做增益放大。

### 2.2 Tied Embedding 扩展

在 tied embedding 下，$u$（输出侧方向）不再固定，而是 $E[K]$——和输入侧共享同一个空间，且 $u \approx v$（两者都是全体 token 的质心方向）。

**三体动力学**：$(\sigma, v, u)$ 共同演化。Phase 2 不是 $\sigma$ 独自增长——而是 $v$ 和 $u$ 在 $\bar{x}$（hidden states 的慢漂移质心）的牵引下协同漂移。

**关键的 $\epsilon$ 机制**：$\bar{x}$ 在训练中缓慢漂移 → $c = v^\top u$ 永远不能精确达到 1.0 → $(1-c^2)$ 因子不会精确归零 → 残留 $\epsilon > 0$ 给 $\sigma$ 提供持续的增长信号。

### 2.3 实验验证

**no_rew (RMSNorm+残差注意力, lr=0.03)**：
- Phase 1 转 Phase 2 在 step ~2401
- $c_{vx} \to 0.98$（接近 1.0，但 $1-c^2 = 0.04$ 永不归零）
- $\sigma_1/\sigma_4$ 从 557 涨到 1125 且仍在增长 — **完全验证 Phase 2 无饱和**
- 但准确率仅 12.5% — $\sigma_1$ 增长独占梯度资源，长尾方向饿死

**hard_rew ($\alpha$=1.0, lr=0.08)**：
- $c_{vx} \to 0.92$（不如 no_rew 对齐好）
- $1-c^2 = 0.15$（更大）
- 但 100% 收敛 — **$\sigma_1$ 更小但谱更平，longtail 获得了梯度资源**

---

## 3. 长尾伤害机制：SIR 衰减

一个 longtail bigram $(T_1, T_2)$ 的梯度在 $v_1$（common 方向）上的投影：

$$\|\text{proj}_{v_1}(\partial L/\partial E_T)\| \approx \sigma_1 \cdot \sum_{k} \text{softmax}_k \cdot |u_{k,1}| \propto f_K \cdot \sigma_1$$

$$\text{SIR}_T \approx \frac{f_T \cdot C_1}{f_K \cdot \sigma_1 \cdot C_2} = O\!\left(\frac{1}{\sigma_1}\right)$$

当 $\sigma_1$ 随训练增长时，长尾 token 的有效梯度中 $v_1$ 分量占比越来越大 → 其表示被拽向 common 方向 → 内部循环（如 $G2G0 \to G1$）因输入被「稀释」而无法准确预测。

**实验验证**：no_rew 下 $G2G0 \to G1$ 永不收敛（no_rew 38% 卡死）；hard_rew 压平 $\sigma_1/\sigma_4$ 后 $G2G0 \to G1$ 在 101 步收敛。

---

## 4. 解法：Subspace Split (CRS)

### 4.1 方法定义

$$h^{(c)} = P_{v_K} h = v_K (v_K \cdot h), \qquad h^{(r)} = h - h^{(c)} = (I - P_{v_K}) h$$

$$y = \alpha \cdot W_c h^{(c)} + W_r h^{(r)}$$

其中 $v_K = \bar{E} / \|\bar{E}\|$（词表级质心，全局参数，无信息泄露）。

### 4.2 梯度解耦

$$\frac{\partial L}{\partial W_r} = \sum \frac{\partial L}{\partial y_r} \otimes \underbrace{h^{(r)}}_{=(I-P_{v_K})h}$$

$h^{(r)} \perp v_K$ 由构造保证 → 每个 rank-1 外积的输入方向不含 $v_K$ 分量 → $W_r$ 的 top singular direction 永远不会被拉向 $v_K$。

### 4.3 实验结果（5-seed, matched lr=0.05, 3000 steps）

| Model | acc | G0G1→K | G1K→G2 | KG2→G0 | G2G0→G1 | E $\sigma_1/\sigma_4$ | Wq $\sigma_1/\sigma_4$ |
|---|---|---|---|---|---|---|---|
| Baseline no_rew | 38%±0% | 2301±0 | **N/A** | 51±0 | 901±0 | 3.0±0.0 | 115±0 |
| Baseline hard_rew | 75%±0% | N/A | 401±0 | 51±0 | 101±0 | 3.1±0.0 | 2722±0 |
| Split α=1.0 | 25%±0% | N/A | N/A | 51±0 | 901±0 | 2.7±0.0 | c:2.8 r:7.8 |
| **Split α=0.3** | **100%±0%** | **1451±0** | **1851±0** | **51±0** | **1451±0** | **1.9±0.0** | **c:1.1 r:5.3** |

- 所有方差为零 — 现象稳定
- **matched lr 下，Split α=0.3 是唯一 100% 收敛的方法**（hard_rew 需要更高 lr=0.08 才能 100%，同 lr 仅 75%）
- $Wq\_c$（common branch）近乎各向同性（1.1×）
- $Wq\_r$（residual branch）的谱集中度仅为 baseline 的 1/20（5.3× vs 115×）
- **Split 不修改 loss 函数**，无需知道 $f_{\text{target}}$

### 4.4 交叉验证

独立 CRS 实现（conflict-free cycle data + causal prefix direction estimator + full-sequence causal attention）验证了一致结论：CRS 将 tail 稳定收敛从 dense 的 ~1256 步压到 ~152-168 步（8.3× 加速）。差异在于 α 压制：full-sequence context 下 α=1.0 即 work；我们的 2-token trigram context 下需要 α=0.3 — context 越短，common 方向在 hidden state 中占比越大，α 压制越必要。

### 4.5 与 Loss Reweighting 的对比

| | Loss Reweighting | Subspace Split |
|---|---|---|
| 机制 | 修改 loss 权重降低 common token 梯度贡献 | 参数分解，common 和 residual 梯度流天然正交 |
| 干预阶段 | Phase 1（压低起源） | Phase 1（阻止对齐） |
| 是否需要先验 | 是（$f_{\text{target}}$） | 否（$v_K$ 从 E 自适应追踪） |
| LR 鲁棒性 | 需要更高 lr | 标准 lr 即 work |
| 参数增加 | 0 | 2× for Q/K/V（common + residual） |

---

## 5. 未解决问题

1. **α 压制的最优调度**：能否从 α=1.0 衰减到 0.3？这可能在保留早期脚手架作用的同时达到更好的终态谱平度。
2. **多层 Transformer**：单层下残差 identity path 的部分 bypass 是否在多层下更严重？
3. **$v_K$ 在线追踪**：词表质心 $\bar{E}$ 在训练中漂移，需要多频繁更新？
4. **Rank-k common subspace**：真实 LLM 中可能需要 top-k 奇异向量的分解而非 rank-1。
5. **真实 LM 上的 retro-fit**：如何在已有的 pre-trained model 上实现 CRS 而不需重新训练？

---

## 6. 关键代码与文档

| 文件 | 内容 |
|------|------|
| `c3s_and_scale_free_docs/two_phase_singular_mode_learning_proof_and_test.md` | 两阶段理论证明 |
| `tied_embedding_two_phase_extension.md` | Tied embedding 三体扩展 |
| `subspace_split_architecture.md` | CRS 方法定义与单 seed 结果 |
| `crs_common_residual_split_experiment.md` | Codex 独立验证（causal prefix estimator + 5-seed） |
| `scripts/subspace_split_architecture.py` | CRS 独立可运行脚本 |
