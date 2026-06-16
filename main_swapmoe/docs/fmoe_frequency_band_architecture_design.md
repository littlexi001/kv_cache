# Frequency-Based MoE 模型结构设计说明

## 1. 核心设计原则

一句话概括：

> **越高频的 feature 越共享、激活越稠密、组合越少，因此使用少量较大的常驻 expert；越长尾的 feature 越稀疏、组合越多，因此使用大量较小的按需 expert，并用更高 recall 的预取策略降低预测错误的代价。**

本设计面向 batch 约为 1 的端侧逐 token decode。最终优化目标不是让所有 expert 都具有最高的 exact-ID 可预测性，而是最小化模型质量约束下的预期动态搬运成本：

$$
\mathbb{E}[C_{\mathrm{swap}}]
=
\sum_e P(e\text{ activated})\,S_e\,(1-\operatorname{HitRate}_e),
$$

其中 $S_e$ 是 expert 参数量。高频计算通过常驻消除搬运，中频计算通过较强的可预测性进行 prefetch，长尾计算则通过减小单个 expert 的尺寸降低预测错误和 cache miss 的代价。

## 2. 表征空间与 Feature 定义

以第 $l$ 层 attention output projection 为例：

$$
W_O^{(l)}=U^{(l)}\Sigma^{(l)}V^{(l)\top}.
$$

设进入 MoE 的 post-attention residual 表征为 $h_t^{(l)}$。在当前实现约定下，用与输入表征维度匹配的谱方向 $\phi_i^{(l)}$ 表示第 $i$ 个输入侧 feature，定义 token 在该方向上的激活强度为：

$$
z_{t,i}^{(l)}
=
\sigma_i^{(l)}\left\langle h_t^{(l)},\phi_i^{(l)}\right\rangle.
$$

对应的 feature energy 为：

$$
e_{t,i}^{(l)}=\left(z_{t,i}^{(l)}\right)^2.
$$

对数据集中的 token 统计 feature $i$ 被纳入能量覆盖集合的频率：

$$
f_i^{(l)}
=
\frac{1}{T}\sum_{t=1}^{T}
\mathbf{1}\left[i\in\mathcal{A}_{\tau}(h_t^{(l)})\right].
$$

按 $f_i^{(l)}$ 从高到低排列 feature，并划分为 common、middle-frequency 和 long-tail 三个频段。频段边界可以先使用 0%-10%、10%-20%、20%-100% 作为初始设置，最终应由实际激活统计和模型 ablation 决定。

## 3. 决定 Expert 结构的两个变量

每个频段不能只用“包含多少个 feature directions”描述。结构设计需要同时考虑两个变量。

### 3.1 单 Token 激活稀疏度

设频段 $B$ 包含 $d_B$ 个 feature，token $t$ 在该频段激活的 feature 数量为：

$$
a_{t,B}=\left|\mathcal{A}_{\tau}(h_t)\cap B\right|.
$$

定义频段内平均激活密度：

$$
\delta_B
=
\mathbb{E}_t\left[\frac{a_{t,B}}{d_B}\right].
$$

- $\delta_B$ 高：多数 token 会同时使用该频段中的大量 feature，计算更接近共享路径。
- $\delta_B$ 低：每个 token 只使用该频段中的少量 feature，适合细粒度条件计算。

### 3.2 Feature 组合多样性

令 token 在频段 $B$ 上的激活集合为：

$$
S_{t,B}=\mathcal{A}_{\tau}(h_t)\cap B.
$$

即使两个频段包含相同数量的 feature，它们也可能具有完全不同的组合多样性。我们可以用激活集合的经验熵、聚类数量或 pairwise Jaccard distance 衡量这种多样性：

$$
D_B=H(S_{t,B}).
$$

- 激活稠密时，不同 token 的激活集合高度重合，$D_B$ 较低，需要的专家类型较少。
- 激活稀疏且 feature 总数较多时，不同 token 可以形成大量不同组合，$D_B$ 较高，需要更多 expert 承载不同 specialization。

因此：

```text
Expert 数量主要由组合多样性 D_B 决定；
单个 Expert 大小主要由每条路由需要承载的计算容量决定；
频段维度 d_B 本身不能直接决定单个 Expert 的大小。
```

## 4. 三频段 F-MoE 结构

### 4.1 Common Band：窄频段、稠密激活、单个大 Expert

Common band 包含数量较少、但几乎被所有 token 激活的高频 feature。

设计方式：

```text
Feature 范围:      较窄，例如最高频的 0%-10%
Token 激活密度:    高
Feature 组合数:    少，token 间高度重合
Expert 数量:       1 个 common expert
单 Expert 容量:    较大
部署方式:          始终常驻 HBM
是否需要预测:      不需要
```

这里“频段较窄”和“expert 较大”并不矛盾。频段宽度描述 router 用来识别 common computation 的 feature 范围；expert hidden size 描述所有 token 都要经过的共享计算容量。Common expert 服务全部 token，因此应提供足够容量，但无需参与动态分发和 swapping。

### 4.2 Middle-Frequency Band：中等密度、少量较大 Expert

Middle-frequency band 中的 feature 不会被所有 token 激活，但在局部语境、同一 sequence 或同一语义阶段内具有持续性。

设计方式：

```text
Feature 范围:      中间频段，例如 10%-20%
Token 激活密度:    中等或偏高
Feature 组合数:    有限
Expert 数量:       少量，例如 2-4 个
单 Expert 容量:    中等或较大
Routing 粒度:      粗粒度 top-1 / top-k
部署方式:          常驻、缓存，或基于预测提前 prefetch
预期可预测性:      较高
```

由于同一 sequence 内的 context feature 通常具有连续性，middle-frequency routing 应当比 long-tail routing 更稳定。此前对无 residual attention output 的分析也显示，同一 sequence 内不同 token 的 attention context 表征具有较高相似度，这为中频 expert 的跨 token persistence 和 prefetch 提供了依据。

### 4.3 Long-Tail Band：宽频段、稀疏激活、大量小 Expert

Long-tail band 包含大量低频 feature，但单个 token 只激活其中很小一部分。其关键属性不是“频段很宽”，而是“每 token 的有效支持集很小，同时不同 token 的支持集组合非常丰富”。

设计方式：

```text
Feature 范围:      较宽，例如 20%-100%
Token 激活密度:    低
Feature 组合数:    多
Expert 数量:       多，例如 8、16 或更多
单 Expert 容量:    小
Routing 粒度:      细粒度 top-1 / top-k
部署方式:          load-on-demand，并使用预测 prefetch
预期可预测性:      exact-ID 可预测性较低
```

长尾 expert 的数量多，是为了覆盖丰富的稀疏 feature 组合；每个 expert 较小，是因为每个 token 实际只需要其中少量 feature 对应的增量计算。这样即使 exact-ID prediction 较难，单次预测错误导致的额外搬运成本仍然可控。

## 5. 前向计算定义

设 common expert 为 $C$，middle-frequency experts 为 $\{M_i\}_{i=1}^{N_m}$，long-tail experts 为 $\{L_j\}_{j=1}^{N_l}$。

三个频段的 routing feature 分别为：

$$
z_{t,\mathrm{mid}}=z_t[B_{\mathrm{mid}}],
\qquad
z_{t,\mathrm{tail}}=z_t[B_{\mathrm{tail}}].
$$

路由为：

$$
\mathcal{R}_m(h_t)
=
\operatorname{TopK}
\left(G_m(z_{t,\mathrm{mid}}),K_m\right),
$$

$$
\mathcal{R}_l(h_t)
=
\operatorname{TopK}
\left(G_l(z_{t,\mathrm{tail}}),K_l\right).
$$

模型输出定义为：

$$
y_t
=
C(h_t)
+
\sum_{i\in\mathcal{R}_m(h_t)}\alpha_i M_i(h_t)
+
\sum_{j\in\mathcal{R}_l(h_t)}\beta_j L_j(h_t).
$$

需要保持当前已经验证的原则：**feature-band projection 只用于 routing，进入所有 expert 的仍然是完整表征 $h_t$。**

建议初始容量关系为：

$$
k_c > k_m > k_l,
\qquad
1 < N_m \ll N_l,
$$

其中 $k_c,k_m,k_l$ 是三类 expert 的 hidden expansion，$N_m,N_l$ 是 middle 和 tail expert 数量。

## 6. 预测与 Prefetch 策略

### 6.1 Common Expert

Common expert 始终常驻，不需要预测，也不产生动态搬运。

### 6.2 Middle-Frequency Experts

Middle-frequency expert 数量少、单个 expert 较大，但预期具有较强的跨 token persistence。因此系统应优先提高它们的 precision，并提前 prefetch 目标 expert。

跨层预测时，不应直接比较前层和目标层的同名 group。应将前层表征投影到目标层的 routing feature basis：

$$
\tilde z_{p\rightarrow q,B}
=
h^{(p)}\Phi_{B}^{(q)}\Sigma_{B}^{(q)},
$$

再用 $\tilde z_{p\rightarrow q,B}$ 预测第 $q$ 层的 local expert activation。

### 6.3 Long-Tail Experts

Long-tail exact-ID prediction 可以较低，但系统可以通过以下方式提高 recall 上界：

1. 预测 top-$K$ 个候选 expert，而不是只预测 top-1。
2. 根据 predictor confidence 动态调整 $K$。
3. 保留最近使用的 tail experts，利用局部 cache 命中。
4. 将多个高度相关的小 expert 放入同一传输块，降低单次 miss 的启动开销。

因此，tail predictor 的主要指标不应只有 top-1 accuracy，还应包括 recall@$K$、额外预取字节数和最终 cache miss cost。

## 7. 错误路由是否可以容忍

预测错误不一定意味着模型质量立即崩溃，因为预测路径和模型真实 routing 可以解耦：

```text
Prefetch 预测错误:
  模型仍按真实 router 选择 expert，只是发生一次 cache miss 或延迟加载；
  影响主要是时延，不影响模型输出。

将错就错地执行预测 expert:
  省去 miss 后的等待，但可能损失模型质量；
  是否可接受需要单独实验，不能默认安全。
```

“将错就错”值得验证，因为模型始终保留 common path 和 middle-frequency path，而 tail expert 只承担较小的增量计算。若不同 tail experts 的功能存在冗余或局部相似性，偶尔使用预测 expert 可能只造成有限质量下降。

建议评估以下四种模式：

| 模式 | 执行方式 | 目的 |
|---|---|---|
| Oracle routing | 始终执行真实 expert | 模型质量上界 |
| Prefetch only | 预测只用于搬运，最终执行真实 expert | 标准无损方案 |
| Predicted routing | 直接执行预测 expert | 测量错误路由的质量敏感性 |
| Confidence fallback | 高置信度时执行预测 expert，低置信度时等待真实 expert | 质量与时延折中 |

需要报告 prediction accuracy、validation loss、下游任务、额外传输量和端到端时延。特别应绘制“允许预取的 expert 数量 $K$ / 额外传输量”与“recall / latency / quality”的 Pareto 曲线。

## 8. 建议的首轮模型配置

为了先验证结构关系，建议在总参数量和每 token FLOPs 尽量可比的条件下，从以下配置起步：

```text
Common:
  1 个较大 expert，始终激活并常驻。

Middle:
  2-4 个中等 expert，每 token 选择 1 个；
  主要测试跨 token persistence 和目标层 basis 对齐预测。

Long-tail:
  8-16 个小 expert，每 token 选择 1 个；
  主要测试 recall@K、动态搬运量和错误路由质量损失。
```

至少需要三组容量 ablation：

| variant | common capacity | middle experts | tail experts | 目的 |
|---|---:|---:|---:|---|
| A | 大 | 少量、中等 | 多量、小 | 主方案 |
| B | 中 | 少量、中等 | 多量、中等 | 检验 tail expert 是否仍过大 |
| C | 大 | 多量、小 | 多量、小 | 检验 middle 过度细分是否损害可预测性 |

所有配置应匹配或报告总参数量、每 token 激活参数量和 FLOPs，避免把结构收益与容量增加混在一起。

## 9. 合作者需要完成的实验

### 9.1 验证三频段统计假设

逐层统计：

- 每个频段的平均激活密度 $\delta_B$；
- 每个 token 在各频段平均激活多少 feature；
- 激活集合的 Jaccard similarity、聚类数量或经验熵；
- 同 sequence 相邻 token 在各频段的激活集合相似度；
- 各频段的跨 token persistence。

目标是确认以下顺序是否成立：

$$
\delta_{\mathrm{common}}
>
\delta_{\mathrm{mid}}
>
\delta_{\mathrm{tail}},
$$

$$
D_{\mathrm{common}}
<
D_{\mathrm{mid}}
<
D_{\mathrm{tail}}.
$$

### 9.2 训练结构对比

- Baseline flat MoE；
- 当前 SV-route + full-input F-MoE；
- 新三频段异构容量 F-MoE；
- 参数量与激活 FLOPs 对齐的 ablation。

报告训练 loss、validation loss 和下游任务，先确认模型能力不下降。

### 9.3 预测与 Swapping 对比

- Middle 和 tail 分开报告 top-1 accuracy、recall@$K$；
- 测试目标层 basis 对齐 predictor；
- 报告 resident parameter ratio；
- 报告每 token 动态换入参数量；
- 报告预测错误后的实际 cache miss 和等待成本；
- 测试 predicted routing / confidence fallback 的模型质量。

## 10. 预期结论

该结构不要求所有频段同时做到高可预测性。它把不同频段的统计性质转化为不同的系统策略：

```text
Common feature:
  高频、稠密、共享 -> 单个大 expert 常驻，不需要预测。

Middle-frequency feature:
  中等频率、组合有限、具有上下文持续性 -> 少量较大 expert，重点做高精度 prefetch。

Long-tail feature:
  低频、稀疏、组合丰富 -> 大量小 expert，允许较低 exact-ID accuracy，依靠 recall@K 和低搬运成本控制代价。
```

最终评价标准不是单一 routing accuracy，而是在模型质量不下降的前提下，同时降低 resident HBM、动态传输量和端到端 decode latency。
