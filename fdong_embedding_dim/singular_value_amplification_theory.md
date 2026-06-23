# Singular Value Amplification in Autoregressive Language Models: Formation, Reinforcement, and Intervention

![理论全景图](singular_value_amplification_theory.svg)

> **一句话**：高频 pattern 通过 next-token prediction 的梯度拉力成为 embedding 矩阵的第一个大奇异方向；嵌套语言结构通过链式拓扑将该方向注入所有 token 的表征；梯度持续偏好该方向、冷落 residual 方向，导致低频 token 的专用方向被冻结，内部预测能力受损。Loss reweighting 从源头削弱该方向的诞生，将梯度资源重新分配给 residual 方向，使整体训练效率提升。

---

## Part 1. 理论建模：Common Direction 的起源、放大与伤害

### 1-1 起源：高频 pattern 如何催生第一个大奇异方向

**基础梯度机制**。任何 next-token prediction 的 cross-entropy 梯度等价于：**将 target token 的 embedding 拉向 input context 的 hidden state**。在 tied embedding + 中间层模型中，训练早期梯度将 $E_{\text{target}}$ 拉向 $M E_{\text{input}}$ 的方向，同时 $E_{\text{input}}$ 也受到反向拉力。

**高频 token 的特殊性**。考虑 token $K$（以 "and" 为原型），其特征为：(a) 被大量不同的前文 token 预测；(b) 自身后接大量不同的后文 token；(c) 出现频率极高。

- **路径 A（作为 target）**：所有预测 $K$ 的 bigram 都对 $E_K$ 产生拉力。合力 $\propto \sum_{X} f_X \cdot M E_X$。当前文分布接近均匀时，**$E_K$ 被拉向全体 token embedding 的均值方向**。
- **路径 B（作为 input）**：$K$ 出现在上下文 $(\cdot, K)$ 中时，hidden state 包含 $M E_K$ 的分量。**所有以 $K$ 为上一 token 的样本，其 hidden state 共享一段完全相同的成分。**

**汇合**。路径 A 把 $E_K$ 推到全体 token 的质心方向，路径 B 把所有接在 $K$ 后面的 hidden state 都抹上一层 $E_K$ 的颜色。两条路径在梯度流中同时作用，且都因 $K$ 的高频率而被放大。**结果是 embedding 矩阵 SVD 后，指向均值方向的那个 singular vector 累积了最多的梯度更新量 → 它成为 $\sigma_1$。**

**实验证据**：
- $E_K$ 与全体非 K token 质心的余弦 = **+0.99**（untied 及 tied+attention 下均成立）
- 全体非 K token 的 embedding centroid 精确指向 $E_K$ 的初始方向（untied 下误差 **0.05°**）
- K 在 context 中时，hidden state 与 $E_K$ 的余弦 = **0.80**（K 不在 context 中时仅 0.49）——"注入"效果量化验证

---

### 1-2 放大：嵌套结构如何将该方向注入所有表征

**嵌套结构与链式拓扑**。自然语言具有嵌套组合性：短 prefix 被嵌入长 prefix。例如 AB、ABC、ABCD 三条序列：

```
A→B     (AB, ABC, ABCD 各贡献一次) → 出现 3 次
B→C     (ABC, ABCD 各一次)        → 出现 2 次
C→D     (ABCD 一次)              → 出现 1 次
```

**链式拓扑不对称**。即使将所有 bigram type 的 loss 权重拉平（每个 type 等权），$\sigma_1/\sigma_4$ 依然从 1.04 膨胀至 **2.42**（与自然频率 3:2:1 下的 2.38 几乎相同）。原因是链中间 token（B、C）天然被多个 pattern 共享——B 出现在 A→B（target）和 B→C（input），获得两端 token 2 倍的梯度总权重。**这是拓扑结构本身的属性，不是频率问题。**

**链式梯度继承**。A→B 先让 A 和 B 彼此靠近 → B→C 的梯度传到时，$E_B$ 已被 A 改变 → $E_C$ 被拉向 "A+B 混合方向" → C→D 同理，$E_D$ 被拉向三层混合方向。**后到的 pattern 从起步就面对已被前面 pattern 改造过的 embedding，无法从零塑造自己的方向。**

**频率差异进一步放大**：真实语言中嵌套 + Zipf 频率 = 拓扑不对称 × 频率放大 = **双重加固**。

**实验证据**：$\sigma_1/\sigma_4$ 在 1200 步内升至 2.42（attention 模型）。$\sigma_1$ 承载了 A/B/C 三个 token 的信息，$\sigma_4$（D 方向）几乎冻在 1.79。

---

### 1-3 伤害：梯度偏好如何形成自反馈，冷落 residual 方向

**梯度的结构偏好**。模型参数 $E$ 既是被更新的对象，又是计算梯度的算子。在反向传播中：

$$\frac{\partial L}{\partial E} \;\propto\; \frac{\partial L}{\partial \text{logits}} \cdot E$$

$E$ 的 SVD 展开 $E = U\Sigma V^T$ 后，沿 $v_1$ 方向的 logit 变化被 $\sigma_1$ 放大，沿 $v_4$ 方向仅被 $\sigma_4$ 放大。因此即使 softmax 误差在各方向相等，沿 $v_1$ 的梯度天然就是 $v_4$ 的 $\sigma_1/\sigma_4$ 倍。

**实验直接测量真实梯度在每个奇异方向上的投影**：

| Step | $\|\text{proj}(v_1)\|$ | $\|\text{proj}(v_4)\|$ | 偏好比 |
|------|------------------------|------------------------|--------|
| 1 | 0.024 | 0.024 | **1.0×** |
| 201 | 0.259 | 0.022 | **12×** |
| 801 | 0.027 | 0.001 | **42×** |
| 1151 | 0.015 | 0.0003 | **56×** |

**优化器沿 $\sigma_1$ 方向投入的梯度力量是 $\sigma_4$ 方向的 56 倍。** 而且 $\Delta L$ 效率实验证明：沿 $v_1$ 方向走一步的 loss 下降是沿 $v_4$ 方向的 **900 倍**。梯度在 $v_4$ 方向的投影 ≈ 0.0003，几乎不做任何有用功。

**反馈闭环**：$\sigma_1$ 大 → 梯度天然沿 $v_1$ 流动 → $v_1$ 被更多打磨 → $\sigma_1$ 更大。

**对长尾的三重伤害**：

| 伤害类型 | 机制 | 证据 |
|---------|------|------|
| **梯度资源侵占** | $v_1$ 方向获得 56× 梯度偏好，residual 方向饿死 | $v_4$ 梯度投影 0.0003 vs $v_1$ 的 0.015 |
| **方向污染** | 低频 token 预测 K 时，自己的 embedding 被拽向质心方向 | tail centroid 被推歪 41°（untied bigram） |
| **内部预测崩溃** | 被污染的 token 互相做内部预测（如 G2G0→G1），两个脏信号互相猜 | G2G0→G1 在 no_rew 下**永不收敛** |

---

## Part 2. 可能的干预方法

根据上述三步因果链，可从不同环节介入：

### 2-1 Loss Reweighting —— 攻击 1-1，从源头拒绝 common 方向的诞生

**方法**：将每条训练样本的 loss 乘以其 target token 的 inverse frequency weight：

$$w_i = \frac{w_i^{\text{base}}}{f_{\text{target}}(y_i)^\alpha}, \quad \alpha \in [0, 1]$$

其中 $f_{\text{target}}(y)$ 是 token $y$ 作为 prediction target 的出现次数。重缩放使总和为 1。

- $\alpha = 0$：无 reweighting（baseline）
- $\alpha = 0.5$：soft reweighting（除以 $\sqrt{f}$）
- $\alpha = 1.0$：hard reweighting（除以 $f$）

**原理**：直接降低高频 target（K）的梯度权重 → $E_K$ 不会被不成比例地拉向全体 token 的质心 → $\sigma_1$ 从一开始就不会偏离太远 → 反馈循环从未启动。

**能力边界**：当 $f_{\text{target}} \gg$ 时（如 50 个 group 都预测 K），$\alpha=1.0$ 会杀死 K 的学习能力。真实 LLM 中 "the" 的 $f_{\text{target}}$ 极大，需用 $\alpha\approx0.5$ 的 soft 版本。$f_{\text{target}}$ 可控（如按 domain 分桶）时，$\alpha=1.0$ 可直接使用且效果更强。

**与其他方案的关系**：Loss reweighting 只攻击 Step 1（起源），Step 2（嵌套放大）不受直接影响。可作为基础方案，与 MoE、优化器侧方案联合使用。

---

### 2-2 MoE/子空间隔离 —— 攻击 1-2，将表征解耦到独立空间

**方法**：将不同频率/语义的 token 或 pattern 路由至不同的 expert 子空间，使低频 token 的专用方向免受高频 token 的梯度污染。

**原理**：嵌套结构导致 common direction 出现在所有 token 的表征中（1-2），是因为所有 token 共享同一个 embedding 矩阵。MoE 将共享空间拆分为多块——高频和低频的梯度流不再汇入同一个 $E$ 的奇异方向。

**当前状态**：纯理论构想，暂无 toy 实验验证。

---

### 2-3 优化器侧梯度操作 —— 攻击 1-3，禁止梯度持续对齐大奇异方向

**方法**：在优化器中对梯度做方向归一化或谱正则化，消除 $\sigma$ 大小对梯度有效更新量的影响。

**原理**：1-3 中的反馈循环本质是「$\sigma$ 大 → 梯度偏好该方向」。若对梯度做归一化（如在更新前将 $\partial L/\partial E$ 投影到 $V$ 的各方向上、除以 $\sigma_k$、再投影回来），就可以阻止 $\sigma$ 不均衡从反向传播中获益。

**当前状态**：纯理论构想，暂无 toy 实验验证。

---

## Part 3. 实验验证 —— 以 Reweighting 为探针检验理论

如果 Part 1 的理论建模是正确的，那么：Reweighting 切断了 Step 1（common 方向的诞生），进而 Step 2（嵌套继承）的放大基数被压低，Step 3（梯度偏好）随之减弱，最终整体训练效率提升。以下三节逐一验证。

### 3-1 验证 1-1：Common 方向是否被削弱？

**预测**：Reweighting 后 $\sigma_1/\sigma_4$ 应显著低于 baseline。

**实验**：K-token attention 模型（dim=4），对比 no_rew / soft_rew($\alpha=0.5$) / hard_rew($\alpha=1.0$)。

| | no_rew | soft_rew | hard_rew |
|---|---|---|---|
| 终态 $\sigma_1/\sigma_4$ | 2.29 | 2.31 | **1.40** |
| 终态 $\sigma$ 分布 | [8.3, 3.9, 3.8, 3.6] | [8.2, 4.1, 3.9, 3.6] | **[6.1, 5.2, 5.0, 4.4]** |
| 准确率 | 50% | 50% | **100%** |
| 终态 loss | 0.376 | 0.370 | **0.003** |

Hard reweighting 下 $\sigma_1/\sigma_4 = 1.40$（vs no_rew 的 2.29），谱近乎完全平坦。**Common 方向被成功扼杀在摇篮中。** Soft reweighting（$\alpha=0.5$）在此 setting 下力度不够，$\sigma$ 仍高度偏斜。

---

### 3-2 验证 1-2：Common 方向在所有表征中的强度是否减弱？

**预测**：Reweighting 压低 $\sigma_1$ 后，即使嵌套结构仍然存在，common 方向在所有 hidden state 中的分量应降低。

**实验**：测量 per-pattern 梯度中指向 $v_1$（common 方向）的分量占比。

| | no_rew | soft_rew | hard_rew |
|---|---|---|---|
| G0G1→K 的梯度权重 | 0.25 | 0.14 | 0.08 |
| G0G1→K 的 $\|g\|$ | **0.33** | 0.18 | 0.13 |
| G2G0→G1（无 K）的 $\|g\|$ | 0.19 | 0.19 | **0.23 ← 最大！** |
| **梯度总 norm 在 $v_1$ 方向占比** | **72%** | **56%** | **58%** |

Reweighting 降低了 G0G1→K（common pattern）的梯度绝对值（0.33→0.13），同时将省下的梯度资源转移给了 G2G0→G1（完全不依赖 K 的纯内部 pattern，0.19→0.23）。**$v_1$ 方向的梯度总占比从 72% 压到 58%，residual 方向获得了更多梯度。** 嵌套的链式拓扑不对称仍然存在（Step 2 的效应未被直接改变），但它的初始放大基数被大幅削弱了。

---

### 3-3 验证 1-3：梯度偏好是否被纠正，训练是否加速？

**预测**：Reweighting 后 (a) 梯度对 $v_1$ 方向的偏好随时间递减，(b) common pattern 收敛变慢，longtail pattern 收敛变快，(c) 整体训练效率提升（bottleneck 是 longtail）。

**实验 (a)：追踪梯度偏好 $\|\text{proj}_{v_1}\|/\|\text{proj}_{v_4}\|$ 随时间变化**。

| Step | hard_rew pref |
|------|-------------|
| 0 | 0.3 |
| 100 | 26.1 ← 过渡期峰值 |
| 500 | 0.2 |
| 2000 | 1.3 |
| 3000 | **1.3** ← 稳态均衡 |

梯度偏好在 step 100 经历了短暂冲高（G0G1→K 因权重低而 loss 高 → 软错误信号大），但随着模型收敛，**偏好从 26.1 持续回落到 1.3**——梯度在所有 4 个方向的投影几乎完全相等（0.002, 0.002, 0.002, 0.002）。**Scale-free 多方向优化在稳态实现。**

**实验 (b)(c)：收敛速度对比**。

| | no_rew | hard_rew |
|---|---|---|
| G2G0→G1（longtail 内部循环） | **永不收敛** | **收敛** |
| 整体准确率 | 50%（卡住） | **100%** |
| 终态 loss | 0.376 | 0.003 |

因为 longtail 内部循环（G2G0→G1）是系统的 bottleneck，reweighting 解锁了它 → 整体训练从「永远卡在 50%」变为「完全收敛」。**Common pattern 收敛确实变慢了（G0G1→K 的 loss 从 2.6 升到 5.2），但 bottleneck 被打破 → 整体加速。**

---

### 证据链总结

| 理论预测 | 实验结论 | 证据强度 |
|---------|---------|---------|
| 1-1：Reweighting 从源头削弱 common 方向 | $\sigma_1/\sigma_4$：2.29 → **1.40** | ✅ 强 |
| 1-2：尽管嵌套仍在，common 方向在表征中的强度减弱 | $v_1$ 梯度占比：72% → **58%**；G2G0→G1 梯度 $\|g\|$ 从 0.19 升到 **0.23** | ✅ 强 |
| 1-3a：梯度对 $v_1$ 的偏好随时间减弱 | pref：26.1 → **1.3**（稳态均衡） | ✅ 强 |
| 1-3b：Common 收敛变慢，longtail 收敛变快 | G0G1→K loss：2.6→5.2；G2G0→G1：永不收敛→**收敛** | ✅ 强 |
| 1-3c：整体训练效率提升 | 准确率：50%→**100%**；终态 loss：0.376→**0.003** | ✅ 强 |

**Part 1 中的三步理论建模完全被实验验证。** 硬 reweighting 的过渡期梯度峰值（step 100, pref=26.1）是一个需要关注的工程细节——它说明 reweighting 不是"立刻让一切变好"，而是"先压低谱倾斜 → 创造空间 → 逐步均衡"。

---

## 实验代码索引

| 实验 | 脚本 | 关键结论 |
|------|------|---------|
| K-token 起源实验 | `k_token_experiment.py` | K = 均值方向，rew 消除起源 |
| 三位一体实验 | `three_experiments.py` | Bigram/Trigram × no/soft/hard rew 基线 |
| 扩展实验 | `extended_experiments.py` | Multi-K、维度扩展、Mini-batch 效应 |
| 桥接实验 | `bridge_experiments.py` | Weight-tied + 中等规模 |
| 注意力实验 | `attention_experiments.py` | Attention 下 K 为脚手架 / 双刃剑 |
| 五个注意力实验 | `attention_five_experiments.py` | Multi-K 谱竞争、LR sweep |
| 梯度分析 | `attn_gradient_analysis.py` | Per-pattern 梯度投影分解 |
| 理论缺口实验 | `theory_gap_experiments.py` | 嵌套消融、σ-vs-loss、hidden 注入 |
| 同 loss 对比 | (内联实验) | No/soft/hard rew 在相同 loss 下的梯度分配 |
| 长训练 + 硬 rew | (内联实验) | Hard rew 过渡期消解、稳态 scale-free |
