# Common 如何影响 Longtail 学习：梯度动力学分析

## 核心问答（TL;DR）

问：数据天然服从长尾分布，这个先验我们需要改吗？  
答：不改。长尾分布是人类语言世界的固有属性。我们要理解它如何塑造模型。

---

问：在这种分布下，高频（common）数据是否影响低频（longtail）数据的学习？  
答：**是。** Common 会拖慢 longtail 的收敛。在 toy 实验中，Zipf 分布（70% common, 10%×3 tail）下 longtail 收敛步数从 30 增加到 50（+67%）。

---

问：为什么会影响？  
答：**因为 next-token prediction 的 cross-entropy 梯度把每个 token 的表示拉向它的 next token。** 高频 token 的梯度更新次数多 → 出力大 → 建立梯度优势 → 把低频 token 的方向推歪（tail centroid 偏离初始方向 41°）→ 破坏了训练初期各方向之间的正交性 → 低频 token 的梯度信号被干扰淹没（SIR 从 157 暴跌到 0.6）。链条：

$$\text{频率不对称} \to \text{更新速度不对称 (7:1)} \to \sigma_1 \neq \sigma_2 \to \text{tail 被推歪 41°} \to \text{梯度重叠} \to \text{SIR} \downarrow \to \text{tail 学更慢}$$

---

问：那高频 token 自己在做什么？它们会成为所有 token embedding 的"均值"吗？  
答：**是。** 高频 token 作为最多的训练样本 target，其 $W$ 向量被拉向全体前文 token 的平均方向；对称地，其 $E$ 向量被拉向全体后文 token 的平均方向。在当前设定下，它稳定在全体 token embedding 的质心。K-token 实验中，所有非 K token 的 embedding 质心精确指向 K 初始方向（误差 0.05°）。这就解释了真实 LLM 中"and"为什么会是 embedding mean bias。

---

问：那 K 作为 target（被很多人预测）和作为 input（预测很多人），哪个学得快？  
答：**K 作为 target 极快（43 步），K 作为 input 永不收敛。** 因为作为 target 时，多个 input 指向同一个 output → 梯度合力一致 → 收敛快。作为 input 时，一个 input 要指向多个 output → 梯度互相抵触 → 不可能。这解释了"and"为什么好预测（→ and 方向固定），以及从"and"往后预测为什么难（→ 千奇百怪）。

---

问：这能在 Transformer 里复现吗？  
答：底层机制相通。Transformer 的 gradient 同样把上下文表示拉向预测目标，高频 pair 拉得更狠。但 Transformer 有 LayerNorm 和多层非线性变换——中间层可以部分修复 embedding 层的 tail 压缩（你们的 `transformer_embedding_init` 实验已观察到 hidden states 中 tail effective rank 恢复）。需要进一步在中间层上验证这个机制。

---

问：有没有办法解决 common token 导致的 mean bias（embedding 被拉向高频 token）？  
答：**有。Inverse target frequency loss reweighting。** 把每条数据的 loss 除以它的 target token 被预测的次数 $f_{\text{target}}$，然后重缩放到总和为 1。这样 $W_K$ 不再收到 4 倍梯度拉力（因为 K 作为 target 被 4 个不同 group 预测），所有 target token 的梯度贡献均等化。

Bigram+rew 实验中：K_target 收敛从 43 步拖到 408 步（和 groups 的 389-428 持平），gap 从 526 步缩小到 -19 步——**mean bias 被消除了**。Trigram+rew 实验中：K_input 首次收敛（204 步），所有 groups 在 142-278 步收敛——**比原版快 3-5 倍**。

---

## 代码索引

| 文件 | 内容 |
|------|------|
| `common_hurts_longtail_analysis.md` | 本文档：完整分析（结论、机制、实验） |
| `scripts/two_dimension_testnew.py` | 原始 4-group 循环 bigram 实验（配置 A） |
| `scripts/run_common_hurts_tail_2d_control.sh` | 配置 A Uniform vs Zipf 对照启动脚本 |
| `scripts/analyze_toy_gradient_interference.py` | 梯度干扰分析（SIR、梯度余弦） |
| `analysis_step0_grad.py` | 第 0 步梯度分解 |
| `analysis_uniform_vs_zipf_v2.py` | Uniform vs Zipf 几何动力学逐步对比 |
| `k_token_experiment.py` | K-token bigram 实验（配置 B） |
| `k_token_convergence.py` | K-token target vs input 收敛速度分析 |
| `three_experiments.py` | **inverse frequency reweighting 三实验**（trigram baseline / bigram+rew / trigram+rew） |

运行方式（以三实验为例）：
```bash
cd fdong_embedding_dim
python3 three_experiments.py
```

---

## 1. 物理先验

### 1.1 数据是长尾的

自然语言的 token 频率服从 Zipf 分布。高频 token（"the", "and", "a"）出现次数是低频 token（"quantum", "photosynthesis"）的数万到数百万倍。这一分布存在于三个层面：token、domain、以及尚不能显式定义的抽象 feature。

**不做假设改动**。这是人类语言的给定属性。

### 1.2 训练范式

大语言模型使用 autoregressive next-token prediction，损失函数为 cross-entropy：

$$L = -\frac{1}{N} \sum_{(x, y)} \log P_\theta(y \mid x)$$

其中 $P_\theta(y \mid x) = \text{softmax}(\text{logits}_\theta(x))_y$。

**核心观察**：在这个损失函数下，每个训练对 $(x, y)$ 都会对模型参数产生一个梯度更新，其大致方向是"把模型对 $x$ 的表示拉向有利于预测 $y$ 的方向"。频率不同的 pair，更新次数不同。

### 1.3 猜想

**猜想 1**：高频 token pair 的梯度更新多，会在参数空间里"推得更猛"，导致其占据主导谱方向。  
**猜想 2**：低频 token pair 在被高频 pair 主导的参数空间里学习时，会受到干扰（梯度信号被淹没），导致收敛变慢。  
**猜想 3**：被所有人预测的高频 token（如 "and"），其 embedding 方向会收敛到全体 token embedding 的质心。

## 2. Toy 数学模型

### 2.1 模型：Bigram LM

| 组件 | 定义 |
|------|------|
| 词汇表大小 $V$ | 12 tokens（4 groups × 3 tokens）或 13 tokens（K + 4 groups × 3）|
| 参数 | $E \in \mathbb{R}^{V \times 2}$, $W \in \mathbb{R}^{V \times 2}$ |
| 前向 | $\text{logit}_j = E_i \cdot W_j$ |
| 损失 | $L = -\log(\text{softmax}(\text{logit})_{\text{target}})$ |
| 初始化 | $E_0 = W_0$，4 个 group 的 token 放在 2D 平面上 0°/90°/180°/270° 方向，每个 token ±12° 偏移 |

### 2.2 数据：两种配置

**配置 A（原始实验）**：
- 每个 group 内部循环 bigram：$A_0 \to A_1,\; A_1 \to A_2,\; A_2 \to A_0$
- Uniform：A/B/C/D 各 25%
- Zipf：A=70%, B=C=D=10%

**配置 B（K-token 实验）**：
- 加入共享 token K。每个 group：$G_0 \to G_1,\; G_1 \to K,\; K \to G_2,\; G_2 \to G_0$
- K 作为 target 出现在 4/16 bigrams 中，作为 input 出现在 4/16 bigrams 中
- Uniform：A/B/C/D 各 25%
- K 初始方向：(0.707, 0.707)（45°，在所有 group 的对称轴上）

### 2.3 梯度结构

对 bigram $i \to j$：

$$\frac{\partial L}{\partial W_j} = (\text{softmax}_j - 1) \cdot E_i \quad (\text{系数为负} \to W_j \text{ 被拉向 } E_i)$$
$$\frac{\partial L}{\partial W_{k \neq j}} = \text{softmax}_k \cdot E_i \quad (\text{跨组拉力：common 数据拉 tail 的 } W)$$
$$\frac{\partial L}{\partial E_i} = \sum_k (\text{softmax}_k - \delta_{k,j}) \cdot W_k$$

## 3. 实验实现

### 3.1 代码路径

| 脚本 | 用途 |
|------|------|
| `scripts/two_dimension_testnew.py` | 配置 A 主实验 |
| `scripts/analyze_toy_gradient_interference.py` | 梯度干扰分析 |
| `scripts/run_common_hurts_tail_2d_control.sh` | 配置 A 的 Uniform vs Zipf 对照 |
| `k_token_experiment.py` | 配置 B（K-token）实验 |
| `k_token_convergence.py` | K-token 收敛速度分析 |
| `analysis_step0_grad.py` | 第 0 步梯度分解 |
| `analysis_uniform_vs_zipf_v2.py` | Uniform vs Zipf 几何动力学对比 |

### 3.2 超参数

| 参数 | 值 | 理由 |
|------|-----|------|
| dim | 2 | 最小可观察正交性破坏的维度 |
| lr | 0.03 | 快速收敛，允许在 2000 步内观察完整动态 |
| theta_deg | 12° | group 内 token 需要一定区分度才能完成循环预测 |
| steps | 200-2000 | 足以观察从初始化到收敛的完整过程 |
| batch | full | 所有 12/16 条 bigram 的加权梯度求和，消除 minibatch noise |

### 3.3 度量

| 度量 | 定义 | 含义 |
|------|------|------|
| Convergence step | $\text{acc}_{\text{group}} = 1.0$ 的首次步数 | 学习速度 |
| $\sigma_1/\sigma_2$ | E 矩阵奇异值比 | 谱对称性 |
| Centroid deviation | group 质心偏离初始方向的角度 (°) | 正交性保持 |
| Cross-group cosine | 两个 group centroid 的余弦 | 表征重叠 |
| Tail SIR | $\pi_r \cdot \|q_r\| / \sum_{i \neq r} \pi_i \cdot \|q_i \cdot \hat{q}_r\|$ | gradient 信号/干扰比 |
| Common-tail gradient cosine | $q_{\text{common}}$ 和 $q_{\text{tail}}$ 的余弦 | 梯度正交性 |

## 4. 实验结果与证据

### 4.1 现象：Common 拖慢了 Longtail（猜想 1+2，配置 A）

| 指标 | Uniform (1:1:1:1) | Zipf (7:1:1:1) |
|------|-------------------|-----------------|
| 全体 acc=100% 步数 | **30** | **50** |
| Common 收敛 | 30 | 33 |
| Tail 收敛 | 30 | 44-50 |
| Final $\sigma_1/\sigma_2$ | 1.000 | 1.084 |

**结论**：Common 拖慢了 tail（50 vs 30），$\sigma_1/\sigma_2$ 偏离 1.0（谱不对称）。

### 4.2 机制证据 1：Tail 被推离正交方向（猜想 2）

| Group | Uniform (step 200) | Zipf (step 200) |
|-------|-------------------|-----------------|
| Common | 1.82° | 2.08° |
| Tail1 | 1.82° | **40.85°** |
| Tail2 | 1.82° | **41.10°** |
| Tail3 | 1.82° | 1.96° |

跨 group 余弦：

| 余弦对 | Uniform (step 200) | Zipf (step 200) |
|--------|--------------------|-----------------|
| Common·Tail1 | 0.000 | **-0.681** |
| Tail1·Tail2 | -1.000 | **-0.140** |

**证据**：Uniform 下所有 group 对称偏离（均 1.82°），正交性保持。Zipf 下 tail1/tail2 被推飞 41°，原先的正交性被打破。Tail3（在 x 轴上，和 common 同线）没被推歪——只有和 common 正交的 y 轴方向（tail1/tail2）被跨组力推开。

### 4.3 机制证据 2：梯度干扰增强（猜想 2）

配置 A Zipf 的 tail SIR 和 common-tail 梯度余弦变化：

| Step | Tail SIR | Common-tail grad cosine |
|------|----------|------------------------|
| 0 | **157** | -0.004 |
| 20 | 0.80 | -0.108 |
| 30 | **0.62** | -0.158 |
| 40 | 0.61 | -0.231 |
| 200 | 0.94 | -0.218 |

**证据**：SIR 从 157 暴跌到 0.62（干扰淹没信号）。SIR 最低点（step 20-40）恰好对应 common 刚收敛、tail 还在挣扎的阶段。梯度余弦从 0 涨到 -0.22。

### 4.4 机制证据 3：频率权重直接决定 centroid 速度（猜想 2）

配置 A 的 per-step centroid 位移（step 2）：

| | Uniform | Zipf |
|---|---|---|
| Common | 0.001337 | **0.003743** |
| Tail | 0.001337 | **0.000535** |

比率 Zipf common/Zipf tail = 7.0 = 0.70/0.10。频率差异直接被编码为运动速度差异。

### 4.5 K-token 实验：mean bias 的形成（猜想 3）

| 指标 | Uniform K-exp | Zipf K-exp |
|------|---------------|-------------|
| K 作为 target（G1→K）首次 acc=100% | **43** | **53** |
| K 作为 input（K→G2）首次 acc=100% | **永不** | **永不** |
| 首个 group 整体收敛 | 569 | 115（A） |

| K 几何 (Uniform, step 600) | 值 |
|---------------------------|-----|
| K initial direction | (0.7070, 0.7070) |
| Non-K centroid direction | (0.7076, 0.7066) |
| **Angle(K init, centroid)** | **0.05°** |

**证据**：K 作为 target（被预测）学得极快（43 vs 569），K 作为 input（预测别人）永远学不会。K 初始方向精确成为全体非 K token embedding 的质心——这是 embedding mean bias 的 toy 级再现。

## 5. 失败分析与改进

### 5.1 配置 B K-input 永不收敛（bigram 设定）

**原因**：在 bigram 设定下，输入只有当前 token K，需要预测 4 个不同的 target（A2, B2, C2, D2），而 K 本身不携带「我属于哪个 group」的信息 → 准确率天花板 25%。$E_K$ 被 4 个方向的 $W$ 同时拉扯，在 2D 中无法同时对齐四个正交方向。

### 5.2 改进 1：Trigram 上下文

将模型从 bigram（只看当前 token）升级为 trigram（看前两个 token）。输入为 $(x_{t-2}, x_{t-1})$，表示 $h = E[x_{t-2}] + E[x_{t-1}]$，预测 $x_t$。这样 (A1, K) → A2 和 (B1, K) → B2 成为不同的输入——K_input 的天花板被消除。

### 5.3 改进 2：Inverse Target Frequency Loss Reweighting

**动机**：K 作为 target 被 A1, B1, C1, D1 四个不同输入预测 → $f_{\text{target}}(K) = 4$，而其他 target 的 $f_{\text{target}} = 1$。在标准 loss 下，$W_K$ 收到 4 倍梯度拉力 → 过快收敛 → 过度拉扯其他 token → mean bias。

**方法**：

$$L_{\text{reweighted}} = \sum_i \frac{L(x_i, y_i) \cdot w_{\text{group}}(x_i)}{f_{\text{target}}(y_i)}$$

然后重缩放使总权重和为 1。效果：K 作为 target 的总梯度权重从 $4 \times 0.0625 = 0.25$ 降到 $4 \times 0.015625 = 0.0625$——和任何其他 target 完全一致。

### 5.4 三组实验对比

| | Bigram 原版 | Bigram+rew | Trigram | Trigram+rew |
|---|---|---|---|---|
| K_target 收敛 | **43** | 408 | **45** | 257 |
| K_input 收敛 | N/A | N/A | N/A | **204** |
| A 收敛 | 703 | 405 | 282 | **143** |
| B 收敛 | 695 | 428 | N/A | **187** |
| C 收敛 | 671 | 397 | 262 | **278** |
| D 收敛 | 569 | 389 | 267 | **142** |
| **最快 group** | **569** | **389** | **262** | **142** |
| **K vs groups gap** | **526** | **−19** | — | ~115 |

**关键结论**：

1. **Bigram+rew**：K_target 收敛从 43 → 408（9.5× 慢），与 groups（389-428）持平。**mean bias 被消除**——K 不再享有拉力优势。

2. **Trigram**：K_input 仍然不收敛。虽然上下文区分了 (G1, K)，但 K_target 仍在 step 45 抢先收敛 → $W_K$ 暴涨 → $E_K$ 被过早冻结 → K_input 无法学习。

3. **Trigram+rew**：唯一的 K_input 收敛方案（204 步）。Reweighting 拖慢了 K_target（45→257），给了 $E_K$ 时间在 K_target 和 K_input 之间找到平衡。所有 groups 在 142-278 步收敛——**比原版快 3-5 倍**。

**物理对应**：Inverse frequency reweighting 相当于在 llm 训练中对 "and""the" 等高频 token 的 loss 做降权——防止它们的 embedding 被过度拉扯、成为不健康的 mean bias。Trigram 上下文对应真实 Transformer 的 attention（可以看到更长前文），解决了"and→???" 的歧义。

## 6. 推广到真实 LLM

| Toy | Transformer |
|-----|-------------|
| Bigram $i \to j$ 的梯度 | Self-attention + FFN 梯度被链式法则反向传播 |
| $E, W$ 独立训练 | $E$ 和 $W_{\text{out}}$ 通常 weight-tied |
| Common 跨组力 $\to$ tail 被推歪 41° | 低频 token embedding 被拉向高频 token 方向 |
| K 成为 embedding 质心 | "and" 是 embedding 矩阵第一主成分 |
| Norm 从 1 涨到 13 | LayerNorm 限制 norm，但方向性偏移仍然发生 |

**需要进一步验证的**：Transformer 中间层（hidden states）是否以及在多大程度上修复 embedding 的 tail 压缩。已有的 `transformer_embedding_init` 实验提示 hidden states 中 tail effective rank 有所恢复——这意味着中间层的非线性变换提供了"逃逸通道"。但这个修复的机制和极限需要进一步研究。

## 7. 当前不确定性

1. **维度效应**：2D 是极端情况（Welch bound 要求严格正交不可能）。在更高维度下（d=3,4,5），tail 被推歪的角度是否会减小？
2. **packed_common 初始化**：如果初始化时所有 group 挤在同一方向，梯度动力学如何"掰开"它们？频率差异会导致某些 group 永远被"挤在"低维子空间吗？
3. **Transformer 中间层**：需要在 controlled Transformer 实验中跟踪 per-group hidden state 的 centroid deviation，确认 toy 的"推力"机制在多层非线性变换后是否仍然成立。
