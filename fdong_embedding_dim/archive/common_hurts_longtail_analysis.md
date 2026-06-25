# Common 如何影响 Longtail 学习：从 Toy Bigram 到 Attention 的梯度动力学分析

## TL;DR 问答

**Q1: Common 是否影响了 Longtail 的学习？**

**是，在两个层面：**

- **Common Token（如 "and"/K）**：高频 target token 的 embedding 被所有预测它的上下文梯度拉向全体 token 的质心，成为表征空间的「均值方向」。经实验验证：在 tied+attention 模型（最接近真实 LLM），K 与全体 token centroid 的余弦在各深度下均达到 **+0.99**。

- **Common Pattern/Group**：在 untied bigram 中，common group（70% 频率）通过软最大化（softmax）的跨组梯度将 tail group 推离其初始正交方向 **41°**，导致 tail 收敛步数从 30 增加到 50（+67%）。在 tied+attention 中，此效应显著减弱（组间梯度余弦仅 ~0.17，而 untied 下达到 −0.68），但 Zipf 分布仍然拖慢收敛。

---

**Q2: 为什么会影响？**

根本原因：next-token prediction 的 cross-entropy 梯度天然具备「吸引力」结构——

$$\frac{\partial L}{\partial W_y} = (\text{softmax}_y - 1) \cdot E_x$$

- **Token 层面**：任意上下文 $(x_0, x_1)$ 预测 $y$ 时，$W_y$（或 tied embedding 中的 $E_y$）被拉向上下文的表示。高频 $y$ 被大量不同的上下文预测 → 合力指向全体上下文的质心 → $E_y$ 收敛到均值方向。

- **Group 层面**（untied bigram）：common group 高频出现 → 其 $W$ 向量 norm 暴涨 → 通过 softmax 跨组梯度持续拉扯 tail 的 $W$ 向量 → 推歪 tail 的方向 → tail 梯度信号/干扰比（SIR）从 157 暴跌到 0.6 → 有效学习率归零。

- **Attention 模型**中：Wq/Wk/Wv 和 tied embedding 将 common 的梯度 norm 优势压制了约 100 倍（\|∇E_K\|=0.16 vs 组内 \|∇E\|≈0.67），但 K↔各组梯度余弦仍维持 +0.27~0.31——残留的「温和牵引」仍在塑造均值方向。

**Common 如何进一步伤害 Longtail Token 的预测？——「稀释 + 传染」两步链路：**

以 group C（低频）为例。C 有四件事要做：C0,C1→K（预测 K）、C1,K→C2、K,C2→C0、C2,C0→C1（内部循环）。

- **第一步（稀释）**：C0,C1→K 的梯度把 E[C0] 和 E[C1] 拉向 K 的方向。但 K 的方向是全体 token 的质心，不是 C 组自己的方向。所以 E[C0] 每预测一次 K，就被往质心方向拽一步 → C0 的表示从「纯 C 组方向」被稀释成了「C 组方向 + 一点质心方向」。

- **第二步（传染）**：C1,K→C2 需要用被稀释过的 E[C1] 和本身就在质心的 E[K] 去预测 C2——两个不纯的输入预测一个 C 组输出 → 更难了。更糟的是 C2,C0→C1：**两个都被稀释过的输入互相预测——纯内部循环反而成了最难的。**

- **实验证据**：NoK 下内部循环只需 1-101 步。WithK 下 G2G0→G1 **永远不收敛**（不用 reweighting），用了 reweighting 后仍需 101-401 步。**Common token 的存在让 longtail 连自己的纯内部循环都学不好了。**

**额外推论**：表征的均值（embedding matrix 的第一主成分方向）**必然是某个高频 token**——这是训练的动力学必然结果，而非统计巧合。

---

**Q3: 如何解决？能力边界在哪？**

**方法：Inverse Target Frequency Loss Reweighting**

$$w_i = \frac{w_i^{\text{base}}}{f_{\text{target}}(y_i)^\alpha},\quad \alpha \in [0.3, 0.5],\; \text{然后重缩放至总和为 1}$$

**效果**：
- Untied bigram: K vs group 收敛 gap 从 526 步消除至 −19 步
- Tied+trigram: 内部瓶颈模式（G2G0→G1）从永不收敛 → 401 步
- Tied+attention: 同上从永不收敛 → 101 步（配合 lr=0.18）；σ₁/σ₂ 从 1.68 降至 1.01
- 允许使用更大学习率（lr=0.18 vs 0.03），加速整体收敛

**能力边界**：
1. **硬 reweighting（α=1.0）在 f_target 大时失败**：若 K 被 50 个 group 预测，1/50 的权重使 K 完全无法学习。必须使用 soft α（0.3~0.5）。
2. **不消除均值方向，只削弱拉力大小**：K 的方向依旧是全体 token 的质心——reweighting 控制的是**力的强度**而非几何终点。
3. **对维度敏感**：低维（2D）下效果最显著，高维下 baseline 本身已较均衡，边际收益递减但依然有效。
4. **真实 LLM 的 mini-batch 限制**：当 tail token 在某 batch 中完全缺失时，reweighting 对该 batch 无效——需要全局统计量作为先验。

---

## 代码索引

| 文件 | 内容 |
|------|------|
| `common_hurts_longtail_analysis.md` | 本文档 |
| `scripts/two_dimension_testnew.py` | Untied 4-group 循环 bigram（原始实验） |
| `scripts/analyze_toy_gradient_interference.py` | 梯度干扰分析（SIR、余弦） |
| `scripts/analysis_step0_grad.py` | Step 0 梯度分解 |
| `scripts/analysis_uniform_vs_zipf_v2.py` | Uniform vs Zipf 几何动力学 |
| `scripts/k_token_experiment.py` | K-token untied bigram（mean bias 验证） |
| `scripts/k_token_convergence.py` | K-token target vs input 收敛对比 |
| `scripts/three_experiments.py` | Inverse frequency reweighting 三实验（bigram/trigram） |
| `scripts/extended_experiments.py` | Multi-K + 维度扫描 + mini-batch 效应 |
| `scripts/bridge_experiments.py` | Weight-tied + 中规模合成数据 + 2-layer |
| `scripts/bridge_realistic.py` | 真实 K 频率（~3%）下的 soft reweighting α sweep |
| `scripts/bridge_medium_scale.py` | 50 group 中规模收敛轨迹 |
| `scripts/tied_diagnosis.py` | Tied+proj 下 M-output 质心 + 深度效应 |
| `scripts/tied_trigram_experiments.py` | Tied+proj+trigram 完整实验 |
| `scripts/attention_experiments.py` | Single-head attention（dim=4）：WithK vs NoK |
| `scripts/attention_five_experiments.py` | 五组扩展实验（Multi-K 谱占据、LR sweep 等） |
| `scripts/lr_sweep_detail.py` | LR sweep 逐 pattern 收敛对比 |
| `scripts/attn_gradient_analysis.py` | Attention 下梯度结构分析 |

运行方式：`cd fdong_embedding_dim && python3 scripts/<脚本名>.py`

---

## 1. 物理先验

### 1.1 数据是长尾的

自然语言的 token 频率服从 Zipf 分布。"the" 和 "and" 的出现次数是 "quantum" 等低频 token 的数万至数百万倍。该分布也存在于 domain 和抽象 feature 层面。**这是给定属性，不做假设改动。**

### 1.2 训练范式

大语言模型使用 autoregressive next-token prediction + cross-entropy loss。在此损失函数下，每个训练对 $(x, y)$ 均产生一个梯度更新，其大致方向为「将模型对 $x$ 的表示拉向有利于预测 $y$ 的方向」。频率不同的 pair 获得不同的更新次数。

### 1.3 猜想

**猜想 1**：高频 target token 的 embedding 收敛至全体 token 表示的质心（均值方向）。
**猜想 2**：高频 group/pattern 的梯度贡献更大，在参数空间中占据主导谱方向，排挤低频 group/pattern。
**猜想 3**：可通过 inverse target frequency loss reweighting 缓解上述两种效应。

---

## 2. Toy 数学模型

### 2.1 模型架构演变

| 模型 | 公式 | 对应真实 LM 中的什么 |
|------|------|---------------------|
| **Untied Bigram** | $\text{logit}_j = E_i \cdot W_j$ | 最简单的 token embedding + 输出层 |
| **Tied + Proj + Trigram** | $\text{logit} = M \cdot (E_{c1}+E_{c2}) \cdot E^T$ | Weight-tied embedding + 线性中间层 + 2-token 上下文 |
| **Tied + Attention** | $\text{logit} = \text{Attn}(E_{c1}, E_{c2}) \cdot E^T$ | Weight-tied embedding + 单头自注意力（最接近真实 Transformer） |

### 2.2 数据配置

**配置 A（原始实验）**：4 组（A/B/C/D），每组 3 个 token，组内循环 bigram $G_0 \to G_1 \to G_2 \to G_0$。无共享 token。

**配置 B（K-token）**：1 至多个 K token 作为共享连接器。每组通过 K：$G_0 \to G_1 \to K \to G_2 \to G_0$。K 作为 target 被所有组预测，建模高频 function word。

### 2.3 度量

| 度量 | 定义 | 含义 |
|------|------|------|
| Convergence step | $\text{acc}=1.0$ 的首次步数 | 学习速度 |
| $\sigma_1/\sigma_2$ | 嵌入矩阵奇异值比 | 谱对称性 |
| Centroid deviation | 组质心偏离初始方向的角度 | 正交性是否被保持 |
| Gradient cosine | 两组条件梯度的余弦 | 梯度重叠程度 |
| SIR | $\pi_r \cdot \|q_r\| / \sum_{i \neq r} \pi_i \cdot \|q_i \cdot \hat{q}_r\|$ | 梯度信号/干扰比 |
| cos(K, centroid) | $E_K$ 与全体非 K token 质心的余弦 | 均值偏向的程度 |

---

## 3. 关键实验结果

### 3.1 现象：Common 拖慢 Longtail（配置 A，Untied Bigram）

| 指标 | Uniform (1:1:1:1) | Zipf (7:1:1:1) |
|------|-------------------|-----------------|
| 全员 acc=100% 步数 | **30** | **50** |
| Tail centroid 偏离 | 1.82° | **40.85°**（被推飞） |
| Final $\sigma_1/\sigma_2$ | 1.000 | 1.084 |

### 3.2 机制：Tail 被推离正交方向 + 梯度 SIR 崩溃

Zipf 下 Common↔Tail1 跨组余弦从 0 涨到 **−0.681**。Tail SIR 从 157 暴跌到 **0.62**（干扰淹没信号）。

$$\text{频率不对称} \to \text{更新速度不对称 (7:1)} \to \sigma_1 \neq \sigma_2 \to \text{tail 被推歪 41°} \to \text{梯度重叠} \to \text{SIR} \downarrow \to \text{tail 学更慢}$$

### 3.3 K-token：Mean Bias 的形成（配置 B）

- Untied Bigram：K 与全体非 K centroid 余弦 = **+1.000**。K 收敛 43 步 vs group 569 步（gap=526）。
- **Reweighting**（$\alpha$=0.5）：gap 缩小至 −19 步。K 方向仍为 centroid，但 norm 不再不成比例增长。
- Tied+Attention（dim=4）：K 与 centroid 余弦 = **+0.99**。均值方向在所有模型架构下均成立。

### 3.4 Attention 下 Reweighting 的质变效果

| Pattern | no_rew 最佳 lr 下的收敛 | rew (α=0.5) lr=0.18 下的收敛 |
|---------|----------------------|---------------------------|
| G2G0→G1（内部循环） | **永不** | **101** |
| KG2→G0 | 351 | **251** |
| G1K→G2 | **251** | 301 |
| G0G1→K | **251** | 401 |
| **完整收敛** | **3/4 patterns**（G2G0→G1 永远缺失） | **4/4 patterns** |

**关键解读——为什么 Common token 伤害了 Longtail 的内部循环？**

上表中 NoK（没有 K、纯内部循环）下 G0G1→G2 只需 101 步，G1G2→G0 只需 1 步——内部循环极其简单。但有了 K 后，G2G0→G1（内部循环）从不收敛变为收敛于 101 步（rewarded），仍然远慢于 NoK 的 1 步。这是一个两步的「稀释+传染」过程：

**(1) 稀释**：以 group C 为例。C0,C1→K 的梯度把 E[C0] 和 E[C1] 拉向 K 的方向。但 K 同时被 A、B、D 拉动——K 不在 C 组自己的方向（z 轴），而在全体 token 的质心方向。所以 E[C0] 每次预测完 K，就被往质心方向拽一小步。C0 的表示从「纯 z 轴」变成了「z 轴 + 一点质心方向」——**C0 不再是一个纯粹的 C 组 token 了。**

**(2) 传染**：C1,K→C2 需要用被稀释过的 E[C1] 和本身就在质心的 E[K] 去预测 C2——两个不纯的输入预测一个纯粹的 C 组输出。更难的是 C2,C0→C1：**两个都被稀释过的输入互相预测——纯内部循环反而成了最难的。**

**Reweighting + 高 lr = 瓶颈被解锁**。Reweighting 使损失谱更平坦，允许使用 6 倍大的学习率而不发散。

### 3.5 梯度结构验证（Tied+Attention，dim=4）

| 梯度余弦 | Untied Bigram (Step 200) | Tied+Attention (Step 200) |
|---------|------------------------|--------------------------|
| 组间 | −0.14 到 −0.68 | −0.16 到 −0.18（**保持正交**） |
| K↔各组 | 0.27 到 0.68 | 0.27 到 0.31（**弱牵引**） |
| K 的梯度 norm vs 组 | W[K]=16.6（**碾压**） | \|∇E_K\|=0.16 vs 组≈0.67（**4 倍更小**） |
| Tail SIR 最低点 | 0.62 | >2.0 |

**Attention 将 K 的梯度优势压制了 100 倍，但 +0.27~+0.31 的残留余弦仍在塑造均值方向。** K 的梯度分解：80% 来自「被预测」（target），20% 来自「出现在上下文中」。

---

## 4. 解决方案：Inverse Target Frequency Reweighting

### 4.1 方法

$$L_{\text{reweighted}} = \sum_i \frac{L(x_i, y_i) \cdot w_{\text{group}}(x_i)}{f_{\text{target}}(y_i)^\alpha},\quad \alpha \in [0.3, 0.5]$$

然后重缩放使总权重和为 1。

### 4.2 有效性证据

| 实验设置 | 无 reweighting | Reweighting | 改善 |
|---------|---------------|------------|------|
| Untied Bigram K vs group gap | 526 | −19 | gap 消除 |
| Tied+Trigram internal | N/A | 401 | 解锁 |
| Tied+Attn internal | N/A | 101 | 解锁 |
| Tied+Attn σ₁/σ₂ | 1.68 | **1.01** | 谱扁平 |
| Tied+Attn lr 容限 | 0.03 | **0.18** | 6× |

### 4.3 为什么有效

Reweighting 直接降低高频 target token（如 K）的每条 bigram 的梯度权重。在 untied 模型中，这抑制了 K 的 W 向量的 norm 暴涨。在 attention 模型中，这减少了 K↔各组 +0.27~0.31 的残留梯度余弦所对应的净拉力——使 K 仍然走向质心方向，但走得更慢、更均衡。

### 4.4 能力边界

1. **硬 reweighting（α=1.0）在 f_target 大时崩溃**：当 K 被 50 个 group 预测时，除以 50 使 K 权重趋近于零，K 无法学习。中规模合成实验（50 group）显示 α=1.0 下 K 的准确率始终为 0%。必须使用 soft α（0.3~0.7）。
2. **真实 K 频率（~3%）下 α≈0.5 为最优**：模拟 "the" 级别频率时，除以 √f_target 使 K 权重从 3.85% 降至 1.61%，既保留了 K 的学习能力（100% 准确率），又加速了长尾 group 的收敛（loss 下降 14%）。
3. **不消除均值方向**：K 的方向性均值偏倚依然存在（cos(K,centroid)≈+0.99）——reweighting 控制的是拉力大小，而非几何终止点。
4. **真实 LLM 的 mini-batch 限制**：当 tail token 在全部 batch 中均不出现时，reweighting 无法对其生效——需要全局频率统计量作为离线先验。
5. **维度效应**：低维（2D）下效果最显著（gap 从 526→−19），高维（3D/4D）下 baseline 本身已较均衡（gap 从 90→57），reweighting 的边际收益递减，但 σ₁/σ₂ 仍能从 1.49 降至 1.01。

---

## 5. 推广到真实 LLM

| Toy | 真实 LLM |
|-----|---------|
| K token 作为共享 target | "and"/"the"/"a" 等高频 function word |
| 所有 group 通过 K 进行转移 | 「所有 token 都可能后接 "and"」 |
| K 的 embedding 被拉向质心 | "and" 的 embedding 是全体 token embedding 的第一主成分 |
| Reweighting 降低 K 的拉力 | 训练时对 "and" 的 loss 做降权（除以 √f_target） |
| Attention 压制 K 的梯度优势 | 多层 attention + LayerNorm 天然控制 norm 不对称 |
| 深度放大均值偏倚 | 1 层→4 层使 K↔中心夹角从 53° 降至 5° |

**尚需验证**：真实 Transformer 中，中间层的 hidden state 在多大程度上「修复」或「放大」embedding 层的均值偏倚——初步的 `transformer_embedding_init` 实验提示中间层可部分恢复 tail 的表征秩，但该修复的极限和机制仍需进一步研究。Reweighting 作为一个轻量级损失函数修改，可与现有的长尾学习方法（如 focal loss、class-balanced loss）直接结合使用。

---

## 6. 当前不确定性

1. **多层 Transformer 中的 reweighting**：本研究的 attention 模型仅含单层单头。真实 Transformer 的多层堆叠是否放大或削弱 reweighting 的效果？
2. **真实数据的实现**：在 LLM 预训练中，target frequency 统计量如何高效维护？需要多少数据才能稳定估算 $f_{\text{target}}$ 用于 soft reweighting？
3. **与其他长尾方法的组合**：reweighting 与 focal loss、重采样等方法的叠加是否产生协同效应？
4. **生成质量的下游评估**：reweighting 改善了训练阶段的收敛均衡性，但在 perplexity、生成多样性等下游指标上的效果如何？
