# QKSieve 理论补充：Query 协方差与随机旋转

本文补充两条与新增因果消融直接对应的理论结论。它们不声称下游任务质量必然提高，而是明确说明 QK-balanced 坐标和 Query-weighted bit allocation 分别解决什么问题，以及实验应验证什么。

## 1. 不使用 Query 协方差的代价

固定坐标系、量化器和物理 bit 预算。记某个 bit allocation \(b\) 产生的 Key 量化误差二阶矩为

$$
C_e(b)\succeq 0.
$$

Key-MSE allocator 最小化

$$
J_K(b)=\operatorname{tr}(C_e(b)),
$$

而 QKSieve allocator 最小化 QK score MSE

$$
J_{QK}(b)=\operatorname{tr}(C_q C_e(b)).
$$

假设 Query 二阶矩满足

$$
\mu I\preceq C_q\preceq L I,\qquad \mu>0.
$$

令 \(b_K\) 和 \(b_{QK}\) 分别是两个目标在相同可行集中的最优解，则

$$
J_{QK}(b_K)
\le
\frac{L}{\mu}J_{QK}(b_{QK})
=
\kappa(C_q)J_{QK}(b_{QK}).
$$

证明只需连续使用三步：

$$
\begin{aligned}
J_{QK}(b_K)
&\le L\operatorname{tr}(C_e(b_K))\\
&\le L\operatorname{tr}(C_e(b_{QK}))\\
&\le \frac{L}{\mu}J_{QK}(b_{QK}).
\end{aligned}
$$

含义：

- 若 \(C_q=\alpha I\)，则 \(\kappa(C_q)=1\)，Key-MSE 和 QK-MSE 等价；
- Query 越各向异性，Key-only allocation 的最坏保证越弱；
- 若 \(C_q\) 奇异，则不存在有限的乘法保证，Key-MSE 可能把 bit 花在 Query 完全不敏感的方向。

对应实验必须区分两个层次：

| 方法 | 坐标 | bit allocation | 回答的问题 |
|---|---|---|---|
| Key-PCA + Key-MSE | 不使用 Query | 不使用 Query | 完全移除 Query 后会怎样 |
| QK-balanced + Key-MSE | 使用 Query | 不使用 Query | 只移除 Query-weighted allocation 会怎样 |
| QK-balanced + qMSE | 使用 Query | 使用 Query | 完整 QKSieve |

实现方法名分别为：

- `qksieve_fullprompt_keypca_autokey_fulltopk`
- `qksieve_fullprompt_qkbalanced_autokey_fulltopk`
- `qksieve_fullprompt_auto_plain_fulltopk`

## 2. 随机正交旋转为何不是 QK balancing

设 \(R\) 是 Haar 分布的随机正交矩阵，\(E_g\) 是一个固定 \(d_g\) 维 band 的坐标投影。定义

$$
P_g=R E_g R^\top.
$$

对任意半正定二阶矩 \(C\)，有

$$
\mathbb E_R\operatorname{tr}(C P_g)
=
\frac{d_g}{d}\operatorname{tr}(C).
$$

证明：Haar 不变性说明 \(\mathbb E_R[P_g]\) 与任意正交矩阵可交换，因此只能是 \(cI\)。又因为 \(\operatorname{tr}(P_g)=d_g\)，所以 \(c=d_g/d\)。

这说明：

- 同一个正交旋转同时作用于 Query 和 Key 时，完整 FP score 仍精确保持；
- 但随机旋转的 band 标签在期望上可交换，没有“前几个 band 更重要”的数据相关顺序；
- QK balancing 的价值不是“做了旋转”，而是把 Query 和 Key 共同敏感的 score 方向显式排序。

对应 256-bit 同码率实验为：

- random rotation + uniform 1-bit；
- Key-PCA + uniform 1-bit；
- QK-balanced + uniform 1-bit。

实现方法名分别为：

- `qksieve_fullprompt_random_uniform1_fulltopk`
- `qksieve_fullprompt_keypca_uniform1_fulltopk`
- `qksieve_fullprompt_qkbalanced_uniform1_fulltopk`

## 3. 可证伪实验

先运行：

```bash
scripts/launch_qksieve_uniform1_ablation_longbench_m20_5gpu_20260728.sh
```

该脚本只准备了八方法同样本、同 prompt、同 active-token budget、同 exact sparse-attention kernel 的筛选实验；当前遵守 GPU 暂停要求，尚未启动。

理论支持以下可证伪预测：

1. QK-balanced uniform 1-bit 应优于 random uniform 1-bit；否则不能把收益归因于 QK-aligned 坐标。
2. QK-balanced qMSE allocation 应优于 QK-balanced Key-MSE allocation；否则 Query-weighted allocator 的必要性不足。
3. Query moment 越各向异性，Key-MSE 相对 qMSE 的 held-out score-MSE gap 应越大。
4. band sensitivity 的 AM/GM 比越大，uniform 相对 automatic mixed-bit 的 held-out qMSE gap 应越大。

若这些趋势不成立，应收缩论文主张，而不能用下游平均分掩盖机制证据。
