# Attention Mask 如何通过最终表征影响 Loss

## 0. 本轮研究目的

我们已经观察到两类看似矛盾的现象：

1. 屏蔽某些 attention token 后，perplexity 上升，但模型仍然能够回答正确；
2. 只保留部分 score-top token 后，perplexity 反而低于 full attention，模型仍然能够回答正确。

因此，本轮不再只问“保留多少 token”，而是要解释：

> **前面所有 Transformer 层采用不同 attention mask 后，最终输出位置的 hidden representation 如何变化；这种变化经过 LM head 后，为什么会提高或降低正确 token 的概率、loss 与最终答案正确率？**

研究链条为：

```text
attention token mask
        ↓
最终位置表征 X 发生变化
        ↓ LM head
vocabulary logits 发生变化
        ↓ softmax + cross entropy
正确 token 概率 / margin / loss 发生变化
        ↓ autoregressive decoding
答案保持正确或发生翻转
```

本轮只研究整次 Transformer forward 的输入和最终输出，不追踪扰动在中间层如何逐层传播。

---

## 1. 输出端的数学建模

对样本 (i) 的某个预测位置，记 Transformer decoder 在 final norm 之后、LM head 之前的输出为：

\[
x_i \in \mathbb{R}^{d}
\]

LM head 记为：

\[
W \in \mathbb{R}^{|V|\times d}
\]

于是 vocabulary logits 为：

\[
z_i = W x_i
\]

softmax 概率为：

\[
p_i(j)=\frac{\exp(z_{i,j})}{\sum_k\exp(z_{i,k})}
\]

若正确 token 为 (y_i)，单 token cross-entropy loss 为：

\[
L_i=-\log p_i(y_i)
\]

数据集平均 loss 与 perplexity 为：

\[
\bar L=\frac1N\sum_iL_i,
\qquad
\operatorname{PPL}=\exp(\bar L)
\]

因此，perplexity 不是“答案是否正确”的直接度量。它衡量模型对所有正确 token 分配概率的几何平均水平。

### 1.1 Loss 与答案正确率为什么可能不一致

设正确 token 为 (y)，当前最高竞争 token 为 (c)。定义 logit margin：

\[
m_{y,c}=z_y-z_c=(W_y-W_c)^\top x
\]

改变 attention mask 后：

\[
x^{M}=x^{F}+\Delta x
\]

margin 的变化为：

\[
\Delta m_{y,c}=(W_y-W_c)^\top\Delta x
\]

于是，同一个判别方向上的不同位移可以产生三种结果：

| 表征位移结果 | Loss / PPL | Argmax / 答案 |
|---|---|---|
| 正确 token 概率下降，但 margin 仍大于 0 | 上升 | 仍然正确 |
| margin 降到 0 以下 | 通常明显上升 | 答案翻转 |
| 正确 token 概率或 margin 上升 | 下降 | 仍然正确且置信度更高 |

因此需要同时观察：

- 正确 token logit 与 probability；
- 最强竞争 token 及其 logit；
- correct-vs-competitor margin；
- cross-entropy loss / PPL；
- teacher-forced token accuracy；
- autoregressive answer correctness。

不能用其中任何单一指标替代其余指标。

---

## 2. 最终表征空间的 SVD 建模

从大量样本、多个预测位置收集 full-attention 最终表征：

\[
X^{F}=
\begin{bmatrix}
(x_1^{F})^\top\\
(x_2^{F})^\top\\
\vdots\\
(x_N^{F})^\top
\end{bmatrix}
\in\mathbb{R}^{N\times d}
\]

本研究首先对 (X^F) 做**未中心化 SVD**：

\[
X^{F}=U\Sigma V^\top
\]

其中：

- (v_k) 是 hidden space 中的第 (k) 个右奇异向量；
- (sigma_k) 表示 full-attention 表征沿该方向的绝对能量；
- top singular direction 允许包含 common direction，不将其预先消除。

任意最终表征可以投影到这个固定坐标系：

\[
a_{ik}^{F}=v_k^\top x_i^{F}
\]

并写成：

\[
x_i^{F}=\sum_{k=1}^{d}a_{ik}^{F}v_k
\]

### 2.1 为什么主分析不先中心化

mask 后的表征并不保证仍来自与 full attention 相同的分布。不同 mask 可能改变：

- common direction 的强度；
- 整体表征均值；
- 各奇异方向的投影；
- 表征能量谱；
- full-attention 子空间之外的 residual。

若分别对每种条件中心化，就会消除均值漂移：

\[
\Delta\mu=\mu_M-\mu_F
\]

而该漂移经过 LM head 后可能直接改变 logits：

\[
\Delta z_{\mathrm{mean}}=W\Delta\mu
\]

因此，本研究以 full-attention 未中心化 SVD 作为统一坐标系。中心化 PCA 只作为补充，用于进一步区分：

```text
整体分布平移
+
分布内部方差与子空间结构变化
```

---

## 3. SVD 方向本身是否影响预测

奇异值大只说明该方向在表征中能量大，不自动说明它对 LM loss 重要。因此需要直接做方向消融。

### 3.1 单方向消融

删除 (x_i) 在方向 (v_k) 上的投影：

\[
x_i^{(-k)}=x_i-(v_k^\top x_i)v_k
\]

然后直接经过原 LM head：

\[
z_i^{(-k)}=Wx_i^{(-k)}
\]

对应 logits 变化为：

\[
\Delta z_i^{(k)}=-(v_k^\top x_i)Wv_k
\]

因此，一个方向的预测作用由两部分共同决定：

```text
样本在该方向上的投影大小
×
LM head 对该方向的映射敏感度
```

### 3.2 连续谱段消融

可进一步比较：

- 只保留 top-(r) 奇异方向；
- 删除 top-(r) 奇异方向；
- 删除 tail-(r) 奇异方向；
- 分 band 删除连续的奇异方向区间。

例如，只保留 top-(r) 方向：

\[
x_i^{\mathrm{top-}r}=\sum_{k=1}^{r}(v_k^\top x_i)v_k
\]

对每种消融重新计算 logits、loss、PPL、margin 和 accuracy，得到两张谱：

1. **Representation energy spectrum**：(sigma_k)；
2. **Prediction sensitivity spectrum**：删除方向或谱段后的 (Delta L)、(Delta m) 与 accuracy change。

这一分析要验证：

> 高能方向是否一定重要？低能 tail directions 是否可能承载少量但决定答案的判别信息？

---

## 4. Attention Mask 如何改变最终表征

对同一输入分别运行 full attention 与某种 mask 条件：

\[
x_i^{F},\qquad x_i^{M}
\]

定义表征扰动：

\[
\Delta x_i=x_i^{M}-x_i^{F}
\]

将两种表征投影到同一个 full-attention SVD 基底：

\[
a_{ik}^{M}=v_k^\top x_i^{M}
\]

\[
\Delta a_{ik}=a_{ik}^{M}-a_{ik}^{F}=v_k^\top\Delta x_i
\]

从而可以观察每种 mask：

- 增强或削弱了哪些 singular directions；
- 是否主要改变 common/top directions；
- 是否主要改变低能但高 loss-sensitivity 的方向；
- 是否产生 full-attention 主子空间无法解释的新 residual；
- 这些变化如何映射为 correct-token logit、margin 与 loss 变化。

对 full top-(r) 子空间，mask 表征的子空间外 residual 为：

\[
r_{i,r}^{M}=x_i^{M}-V_rV_r^\top x_i^{M}
\]

该量用于判断 mask 只是改变原有方向上的坐标，还是将最终表征推向了新的空间方向。

---

## 5. 需要比较的 Mask 条件

第一阶段应至少包含：

| 条件 | 已知行为 | 希望解释的问题 |
|---|---|---|
| Full attention | 基线 | 建立原始 (X)、logits 与 SVD 坐标系 |
| Oracle score top 2% | 能正确回答，answer PPL 低于 full | 哪些表征方向被增强或去噪，使 loss 下降？ |
| Top 2% without front | 仍能正确回答，PPL 略升 | front 被移除后，哪些方向改变但没有跨越决策边界？ |
| Top 2% without answer | 回答错误，PPL 暴涨 | 少量 answer token 如何扰动任务敏感方向并导致答案翻转？ |
| Top 2% without end | 回答错误 | query / decode-control 信息对应哪些最终判别方向？ |
| Top 2% without other | 回答错误 | answer 与 query 之间的关系信息如何体现在最终表征中？ |

后续可加入不同 score ratio，比较从 `1%` 失败到 `2%` 成功时，最终表征在哪些方向上跨过了关键阈值。

---

## 6. 两条互补的分析主线

### 6.1 Logit / Loss 主线

直接研究最终任务输出：

```text
X 的变化
→ correct logit 的变化
→ competitor logit 的变化
→ margin 的变化
→ loss / PPL 的变化
→ 是否发生答案翻转
```

这条主线回答：

> 为什么 PPL 可以变化而 accuracy 不变，以及什么时候这种变化会跨过决策边界。

### 6.2 Representation / SVD 主线

研究 (X) 在高维表征空间中的变化：

```text
full-attention X 构建固定 SVD 基底
→ 测量各方向的 prediction sensitivity
→ 将不同 mask 的 X 投影到同一基底
→ 观察各方向坐标和 residual 的变化
→ 对齐 logits、margin 与 loss
```

这条主线回答：

> Attention mask 改变了最终表征的哪些空间方向，而这些方向为什么会对模型输出有益、无害或致命。

---

## 7. 当前核心研究假设

### 假设一：表征变化大小不等于任务影响大小

某些 mask 可能造成较大的 (|\Delta x|)，但变化主要落在 LM head 对当前答案不敏感的方向，因此答案仍然正确。

### 假设二：低能方向也可能具有高任务敏感度

少量 answer token 可能只改变若干统计能量较低的方向，但这些方向与 correct-vs-competitor 判别轴高度对齐，因此会导致 loss 暴涨或答案翻转。

### 假设三：适度 score-top pruning 可能是一种表征去噪

Top 2%-4% 可能减少了 full attention 中某些使 correct-token margin 下降的方向分量，从而使 PPL 低于 full attention。

### 假设四：Common direction 可能变化，但不一定决定答案

front token 具有很高 attention mass，却可在当前任务中被移除而不影响答案。它可能显著改变 common/top directions，但这些变化未必投影到当前答案的关键判别轴。

---

## 8. 本轮最终要形成的解释

理想情况下，我们希望把实验现象解释成如下形式：

> 某种 attention mask 使最终表征 (x) 在 full-attention SVD 基底中沿若干方向移动；这些方向经过 LM head 后，提高或降低了正确 token 相对竞争 token 的 logit margin。若 margin 只降低但仍为正，则 PPL 上升而答案保持正确；若 margin 跨过零点，则答案翻转；若 mask 去除了使 margin 下降的干扰方向，则 PPL 反而低于 full attention。

这条解释建立后，我们才能进一步回答：

- 哪些 KV token 对最终预测真正重要；
- 哪些高-mass token 主要是结构性成分；
- score-top pruning 为什么可能优于 full attention；
- future selector 应优化 attention mass、表征重建，还是最终 loss-sensitive directions。

当前阶段先完成上述问题形式化，下一步再据此设计数据采样、hidden-state 收集、SVD 消融和 mask 对照实验。
