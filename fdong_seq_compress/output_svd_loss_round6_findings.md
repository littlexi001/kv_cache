# Final-X SVD 与 Attention Pruning Loss: Round6 Findings

Date: 2026-06-12

## 0. 三个核心问题与回答

|核心问题|一句话回答|
|---|---|
|Final (X) 的不同 SVD feature directions 对最终 loss 有何不同作用？|Top singular subspace 构成预测骨架，而其余方向的能量与 loss 敏感性并不一致，删除不同方向可能使 loss 上升，也可能使 loss 下降。|
|为什么只保留 attention score-top 2% token 反而可能降低 loss？|Score-top pruning 主要压低了大量错误 vocabulary logits，使 softmax 中正确 token 的相对概率上升，表现为一种输出端去噪。|
|Attention score-tail token 与 final (X) 的 SVD directions 是什么关系？|Attention mask 并非定向删除某组有害奇异方向，而是引起全谱表征重组；当降低 loss 的方向贡献占优时，最终 loss 才会净下降。|

本轮从模型输出端沿以下链条理解 attention pruning：

```text
attention token mask
→ final hidden state X
→ LM head logits z = W X
→ softmax / cross-entropy loss
→ token accuracy 与自由生成结果
```

具体实验先建立 final (X) 的未中心化 SVD 空间，再分别观察：

- 删除各奇异方向后，LM loss 对该方向有多敏感；
- attention mask 如何改变 final (X)、logits、margin 与 loss。

需要严格区分两类 tail：

```text
SVD tail directions：final X 空间中低奇异值的方向。
Score-tail tokens：attention 中 QK score 较低、被 oracle pruning 删除的历史 token。
```

当前实验表明二者都可能包含低价值或轻微有害成分，但 score-tail token 并不只通过 SVD-tail directions 影响输出。

---

## 1. Final-X SVD 是否可靠

离线采集结果：

|项目|结果|
|---|---:|
|模型|Qwen3-0.6B|
|hidden size|1024|
|采样 final (X)|4096|
|文本来源|14 份多领域长文本|
|chunk 数|64|
|SVD basis 正交最大误差|(2.3\times10^{-6})|
|全 basis 重建相对误差|(1.6\times10^{-6})|
|非零奇异值|1024 / 1024|

未中心化能量谱高度集中：

|累计能量|所需方向数|
|---:|---:|
|50%|2|
|80%|30|
|90%|113|
|95%|245|
|99%|592|

direction 0 单独解释约 `39.4%` 的绝对能量，说明 final (X) 中存在很强的 common/top direction。

---

## 2. 奇异方向的预测敏感度

普通文本评估基线：

```text
loss = 1.383
PPL = 3.988
token accuracy = 70.31%
mean correct-vs-competitor margin = 4.195
```

当前单方向实验每隔 8 个方向测试一次，共测试 `129 / 1024` 个方向。横轴是被删除的奇异方向 index，纵轴是：

\[
\Delta L_k=L(x-(v_k^\top x)v_k)-L(x)
\]

其中：

- (Delta L_k>0)：该方向被删除后 loss 上升，说明该方向总体有益；
- (Delta L_k<0)：该方向被删除后 loss 下降，说明该方向在当前数据上存在净干扰。

这里的“净干扰”是一个数据分布上的平均量：它表示在当前采样 token 上删除该方向后，平均 loss 更低；并不表示该方向对每个样本都无用，也不意味着模型可以无条件永久删除该方向。它的直接意义是证明 SVD energy 不能替代 prediction utility，若要形成压缩方案，还需要进一步验证方向作用的跨样本稳定性和可预测性。

![Final-X SVD sensitivity](outputs/output_svd_sensitivity_20260612_210055/svd_sensitivity_summary.png)

上图同时给出奇异值能量谱、单方向删除后的 loss 变化，以及连续 top/tail 子空间消融结果，用于区分 representation energy 与 prediction sensitivity。

### 2.1 Direction 0 同时高能且高敏感

删除 direction 0：

```text
Delta loss = +0.726
PPL ratio = 2.07x
accuracy = -3.91 percentage points
margin = -2.12
```

因此 common/top direction 不能被视为纯冗余，它承担了大量输出预测功能。

### 2.2 其余方向的能量不能代表任务价值

除 direction 0 外，奇异值与删除方向后 (Delta loss) 的 Spearman 相关约为 `0.14`。例如：

|方向|删除后的 (Delta loss)|解释|
|---:|---:|---|
|8|+0.0227|有益方向|
|16|+0.0240|有益方向|
|32|-0.0061|删除后略有改善|
|64|-0.0060|删除后略有改善|
|72|+0.0073|轻微有益|

因此：

> 奇异值描述表征能量，而不是 LM head 的 prediction sensitivity；二者必须分别测量。

### 2.3 Top 与 tail band 的作用不同

|消融|PPL ratio|Accuracy|
|---|---:|---:|
|删除 top 1% directions|18.6x|53.5%|
|删除 top 5% directions|292x|28.5%|
|删除 top 10% directions|1806x|18.4%|
|删除 tail 10% directions|0.992x|69.9%|
|删除 tail 20% directions|0.987x|69.9%|

当前数据支持：

> Top singular subspace 构成 final (X) 的主要预测骨架；tail 20% directions 对平均预测几乎没有正贡献，删除后 loss 还略有下降。

---

## 3. Attention Mask 如何改变 Loss

任务样本为 NIAH：

```text
Question: What is the best thing to do in San Francisco?
Expected answer: eat a sandwich and sit in Dolores Park
```

![Attention-mask output summary](outputs/mask_output_svd_shift_20260612_210514/mask_output_summary.png)

|条件|PPL|Accuracy|Correct logit|Top competitor logit|Margin|X cosine to full|Relative L2|
|---|---:|---:|---:|---:|---:|---:|---:|
|Full|4.21|88.9%|24.55|18.11|6.44|1.000|0.000|
|Top 2%|1.41|88.9%|22.99|17.32|5.67|0.960|0.258|
|Top 2% - front|1.86|88.9%|22.89|17.48|5.41|0.953|0.271|
|Top 2% - end|3.76|88.9%|20.07|17.11|2.96|0.858|0.460|
|Top 2% - answer|118.39|11.1%|15.62|18.62|-2.99|0.810|0.603|
|Top 2% - other|4.11|77.8%|19.56|17.29|2.27|0.880|0.477|

### 3.1 Top 2% 的 loss 改善不是因为 correct logit 上升

Top 2% 相对 full：

```text
correct logit: -1.56
strongest competitor logit: -0.79
correct-vs-top1 margin: -0.76
loss: -1.09
```

也就是说：

> 正确 token 的 raw logit 与 top-1 margin 都变低，但 cross-entropy loss 显著下降。

Cross-entropy 为：

\[
L=-z_y+\log\sum_j\exp(z_j)
=\log\left(1+\sum_{j\ne y}\exp(z_j-z_y)\right)
\]

因此当前结果意味着：虽然最强竞争 token 相对 correct token 更接近，但大量其余错误 token 的相对 logits 必须整体下降得更多，使整个错误词表的 log-sum-exp 从 full 的有效竞争量约 `3.21` 降到 top 2% 的 `0.41`。

当前输出端机制可以概括为：

> Oracle top-score pruning 通过压低广泛的错误词表激活来降低 softmax 分母，而不是简单提高正确 token 的 raw logit 或 top-1 margin；从输出行为上看，这是一种去噪。

### 3.2 跨长度与 needle 位置的稳定性

固定保留 score-top 2%，在 `1000 / 2000 / 4000` 三种长度和七种 needle depth 上测试，共得到 21 个样本：

|结果|数值|
|---|---:|
|loss 下降|18 / 21|
|保持 teacher-forced token accuracy|18 / 21|
|median (Delta loss)|-0.963|
|median PPL ratio|0.382|
|系统性失败条件|depth = 0|

![Score-top stability across lengths and needle depths](outputs/score_top_stability/score_top_stability.png)

除 needle 位于 context 最前端的 `depth=0` 外，其余 18 个样本全部表现为 loss 下降且 token accuracy 不变。三个 depth-0 样本则全部发生 loss 上升和 accuracy 下降，说明 top 2% 的去噪收益具有明显的位置边界，而不是无条件成立。

### 3.3 不同类别的作用

- **front**：移除后仍正确，且 final (X) 与普通 top 2% 接近；其高 attention mass 不直接等于答案信息价值。
- **end**：使 (X) 发生较大位移并显著降低 margin；teacher-forced token accuracy 暂未翻转，但自由生成已经失败。
- **answer**：产生最大任务破坏，correct logit 大幅下降、margin 变负，直接跨越判别边界。
- **other**：造成部分 token 翻转，说明其中含有连接 query 与 answer 的关系信息。

![Mask-induced SVD shift](outputs/mask_output_svd_shift_20260612_210514/mask_svd_shift_heatmap_top128.png)

![Mask-induced absolute coefficient shift](outputs/mask_output_svd_shift_20260612_210514/mask_svd_projection_shift.png)

Mask 对 final (X) 的影响不是只集中在单一方向：answer/end/other 消融都会在较宽的 SVD 谱段产生位移。其中 answer 消融的整体扰动最强，top 2% 与 top 2% - front 的谱位移模式最接近。

### 3.4 分阶段删除 score-tail token

为了避免只比较 full 与极端 top 2%，下一步固定同一个样本，依次删除 attention score-tail 的 `20% / 50% / 80% / 98%`，即分别保留 score-top 的 `80% / 50% / 20% / 2%`。每个阶段同时测量：

- final (X) 在各未中心化 SVD 方向上的带符号系数变化；
- 每个方向对 cross-entropy (Delta loss) 的 integrated-gradient attribution；
- 所有方向 attribution 之和与真实 (Delta loss) 的 completeness error。

阶段性结果为：

|删除 score-tail|保留 score-top|(Delta loss)|PPL ratio|结论|
|---:|---:|---:|---:|---|
|20%|80%|+0.015|1.015|略微变差|
|50%|50%|+0.019|1.019|略微变差|
|80%|20%|-0.528|0.590|明显改善|
|98%|2%|-1.093|0.335|大幅改善|

这说明 score-tail pruning 存在明显阈值：删除少量低分 token 时，正负方向贡献近似抵消且净效果略差；删除达到 80%-98% 后，降低 loss 的方向贡献开始占优。

以下图片给出各阶段 final (X) 的逐方向系数位移和带符号 loss attribution：

#### 删除 score-tail 20%

![Tail 20 percent SVD loss attribution](outputs/mask_output_svd_tail_sweep/tail20/mask_svd_signed_loss_attribution.png)

#### 删除 score-tail 50%

![Tail 50 percent SVD loss attribution](outputs/mask_output_svd_tail_sweep/tail50/mask_svd_signed_loss_attribution.png)

#### 删除 score-tail 80%

![Tail 80 percent SVD loss attribution](outputs/mask_output_svd_tail_sweep/tail80/mask_svd_signed_loss_attribution.png)

#### 删除 score-tail 98%

![Tail 98 percent SVD loss attribution](outputs/mask_output_svd_tail_sweep/tail98/mask_svd_signed_loss_attribution.png)

Integrated Gradients 的 completeness absolute error 均小于约 `4.4e-4`，因此逐方向贡献之和能够准确复现真实 (Delta loss)。正负贡献的总量为：

|条件|降低 loss 的方向贡献|提高 loss 的方向贡献|净贡献|
|---|---:|---:|---:|
|删除 score-tail 80%|-1.749|+1.220|-0.528|
|删除 score-tail 98%|-3.049|+1.956|-1.093|

负贡献广泛分布在 top、middle 和 tail SVD directions 中，并非只集中在 SVD tail。例如删除 score-tail 98% 时，前 10% 奇异方向的净贡献约为 `-0.428`，其余改善由中部和尾部方向共同完成。

因此第三个问题的当前答案是：

> Attention pruning 不是把 final (X) 中一组固定的“有害 directions”直接抹掉，而是经过多层 attention 与 residual stream 引起全谱重组；该重组同时包含有益和有害位移，最终 loss 取决于二者的净和。

---

## 4. 关于“1024 维空间容量不足”的猜想

当前 intuition 可以更严格地写为：

> Hidden dimension 有限可能迫使大量语义 feature 在同一组方向上 superpose。Score-tail token 若向这些共享方向注入与当前任务无关或冲突的分量，就可能同时激活大量错误 vocabulary logits；oracle pruning 删除这些贡献后，softmax 分母下降，loss 反而改善。

但以下表述不成立：

```text
1024 维空间不能容纳超过 1024 个向量。
```

1024 维空间可以包含无限多个向量，只是最多有 1024 个线性独立方向。大量 feature 共享方向也不必然产生干扰；模型本来就依赖 distributed representation 与 superposition。

所以真正需要验证的是：

1. score-tail token 是否更容易把 final (X) 推向会激活错误词表的共享方向；
2. 这些方向是否同时承载多个互相干扰的 feature；
3. hidden dimension 越小，这种错误词表激活与 pruning benefit 是否越强。

这个猜想目前合理，但尚未被现有实验直接支持。

---

## 5. 竞争性解释

除了有限维度 superposition，还存在至少四种解释。

### H1. Softmax sharpening / renormalization

删除低分 attention token 后，剩余 attention 权重重新归一化。Loss 改善可能主要来自分布变尖，而不是被删除 token 的 V 本身有害。

### H2. Score-tail V 的方向性干扰

大量低权重 V 单独贡献很小，但总和可能把 residual stream 推向错误词表方向。删除后这种累计偏移消失。

### H3. Oracle selection 的后验优势

当前 top-ratio 使用真实 QK score，是 query-dependent oracle。它可能利用模型当前状态进行后验去噪，不能说明任意可实现 selector 都能获得相同改善。

### H4. 数据与任务特异性

普通文本上 top 4% 优于 full，但长程检索的最优比例不同。改善可能依赖文本片段、答案位置和模型规模。

---

## 6. 下一步应做的验证实验

### 6.1 精确分解 Cross-Entropy 的变化

对每个预测 token 保存：

```text
correct logit
top-k competitor logits
wrong-vocabulary logsumexp
correct probability
rank and margin
```

验证 top 2%-4% 是否系统性降低：

\[
\log\sum_{j\ne y}\exp(z_j-z_y)
\]

而不仅是当前使用的均值汇总。

### 6.2 区分“删除有害 V”与“softmax sharpening”

比较：

|干预|目的|
|---|---|
|Top-ratio mask + renormalize|当前方法|
|删除 tail contribution，但保留 full 权重、不 renormalize|测 tail V 的净方向贡献|
|Full attention + temperature sharpening|只改变 attention 尖锐度|
|随机删除同数量 token|排除“变稀疏本身”|
|保留 tail token，但将其 V 替换为零/均值|区分 score 与 content 作用|

### 6.3 建立 SVD direction 到 Loss 的带符号 attribution

该分析已实现。对每个样本计算：

\[
\Delta a_{ik}=v_k^\top(x_i^M-x_i^F)
\]

以及方向 (k) 对 logits 的精确变化：

\[
\Delta z_{i,k}=Wv_k\Delta a_{ik}
\]

当前使用从 full (X) 到 masked (X) 的直线路径 integrated gradients，计算每个方向对 loss 的带符号贡献。这样可以回答：

> Top pruning 改变的哪些 SVD directions 在降低错误词表激活？

### 6.4 直接连接 score-tail token 与 final-X directions

分别只删除不同 attention score bands：

```text
tail 0%-10%
tail 10%-20%
...
top 0%-2%
```

测量每个 band 引起的 final (Delta a_k)，再与 direction sensitivity 对齐。该实验用于验证 score-tail 与 SVD-tail 是否存在真实因果连接。

### 6.5 验证 hidden-dimension / superposition 假设

最直接的证伪方式是跨模型宽度比较：

- 相同任务与 context length；
- 不同 hidden size 的模型；
- 比较 full-vs-top-ratio 的 loss benefit；
- 比较错误词表 logsumexp、feature overlap 和 SVD sensitivity。

若“小 hidden dimension 导致干扰”成立，应观察到：

> 模型越窄，score-tail 引起的错误词表激活越强，oracle pruning 带来的 loss 改善越大。

也可以在同一模型 final (X) 上人为施加不同 rank bottleneck，观察 pruning benefit 是否随有效维度降低而增强。

---

## 7. 当前 Claim Boundary

当前已经支持：

1. final (X) 的未中心化 SVD 能量高度集中；
2. top subspace 对预测关键，tail 20% directions 平均上近似可删；
3. singular energy 与 prediction sensitivity 不等价；
4. top 2% attention 的 loss 改善不是由 correct logit 或 top-1 margin 提高造成，而是来自错误词表整体竞争量下降；
5. score-tail pruning 会重组整个 final-X SVD 谱，而不是只改变 SVD-tail directions；
6. 在 21 个 NIAH 长度/位置样本中，18 个非 depth-0 样本稳定获得 loss 改善，三个 depth-0 样本稳定失败；
7. answer/front/end/other mask 会产生显著不同的 final-X、logit 和 loss 后果。

当前尚未支持：

1. score-tail token 的有害作用一定来自 SVD-tail directions；
2. 有限 hidden dimension / feature superposition 是 loss 改善的根本原因；
3. pruning benefit 能跨问题、答案、文本内容、模型和任务稳定复现；
4. inference-time selector 能以低成本复现 oracle top-score 的效果。
