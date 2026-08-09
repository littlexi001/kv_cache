# 固定总 bit 下的长度与请求条件化量化

## 1. 可以严格支持的结论

希望解释的经验现象是：

> 不同长度、不同请求、不同层、不同 head、不同 band 可能需要不同的量化
> 策略；只要同一总 bit 预算内仍存在满足排序精度要求的策略，就不需要增加
> 总 bit 数。

这里的“量化策略”不是只指某个 head 使用多少 bit，而是包含：

1. 每层、每个 KV head 的 QK-balanced 双正交坐标；
2. 128 维中八个 16-D band 的 `0/1/2/4/8 bit` 分配；
3. 各 band 的量化规则和逐 token scale；
4. 固定总预算下的候选选择目标。

当前物理索引预算保持：

```text
R(b) = 16 * sum_g [b_g + I(b_g > 0)] <= 240 bit/token/KV-head.
```

下面不会声称“240 bit 对任意请求和任意长度都足够”。能够证明的是：

```text
若固定 240-bit 可行集合中存在满足当前请求排序证书的策略，
则提高总 bit 数不是必要条件；切换到该条件最优策略已经足够。
```

这是一条有条件、可审计、可被实验证伪的结论。

## 2. 形式化问题

把条件记为：

```text
c = (request x, history length N, layer l, KV head h, decode position t).
```

为简化符号，暂时省略 `c,l,h,t`。QK-balanced 变换满足：

```text
q'_g = A_g^T q
k'_{i,g} = D_g^T k_i
sum_g q'_g^T k'_{i,g} = q^T k_i.
```

因此未量化的完整变换不产生 score 误差。第 `i` 个历史 token 的精确 score 为：

```text
s_i = sum_g q'_g^T k'_{i,g}.
```

band `g` 使用 `b_g` bit 后：

```text
khat'_{i,g}(b_g) = Q_{b_g}(k'_{i,g})
e_{i,g}(b_g)     = k'_{i,g} - khat'_{i,g}(b_g)
shat_i(b)        = sum_g q'_g^T khat'_{i,g}(b_g)
xi_i(b)          = shat_i(b) - s_i.
```

完整 score 误差是：

```text
xi_i(b) = -sum_g q'_g^T e_{i,g}(b_g),
```

再加上运行时 INT8 Query 的独立误差项。由于 softmax 和排序不受所有 score
共同平移影响，定义中心化误差：

```text
xibar(b) = (1/N) * sum_i xi_i(b)
eta_c^2(b) = (1/N) * sum_i [xi_i(b) - xibar(b)]^2.
```

`eta_c(b)` 是当前条件 `c` 下真正与排序有关的 score RMSE。

## 3. 为什么不同 head 和 band 应有不同失真权重

对任意有限 Query 样本和 Key 量化误差样本，有精确恒等式：

```text
E[(q'_g^T e_g)^2] = trace(C'_{q,g} C_{e,g}(b_g)).
```

其中：

```text
C'_{q,g} = E[q'_g q'_g^T]
C_{e,g}(b) = E[e_g(b) e_g(b)^T].
```

因此 band `g` 的 score distortion 是：

```text
D_{c,g}(b) = trace(C'_{q,c,g} C_{e,c,g}(b)).
```

这个量天然依赖：

- 请求内容，因为当前 prompt 的 Query/Key 二阶矩不同；
- 上下文长度，因为加入的新 token 会改变 Key 分布、边界候选和后续 Query；
- 层和 head，因为不同层/head 的 Query 协方差与 Key 误差方向不同；
- band，因为 QK-balanced joint spectrum 和量化误差不同。

忽略可审计的 cross-band 项时：

```text
eta_c^2(b) approximately sum_g D_{c,g}(b_g).
```

这已经说明不存在“所有层、所有 head、所有请求使用相同 band 权重”的理论
依据。

## 4. 确定性的 top-k crossing 证书

设精确 score 从大到小为：

```text
s_(1) >= s_(2) >= ... >= s_(N).
```

最终 proxy 选择 top-`B`，希望至少保住精确 top-`r` 核心，其中 `r <= B`。
定义核心到候选池外部的 margin：

```text
m_{r,B}(c) = s_(r) - s_(B+1) > 0.
```

### 定理 1：固定条件下的核心漏失上界

令：

```text
eta_c^2(b) = (1/N) * sum_i [xi_i(b) - xibar(b)]^2.
```

则 proxy top-`B` 漏掉的精确 top-`r` token 数不超过：

```text
miss_{r,B}(b;c)
<= min(r, floor(4N * eta_c^2(b) / m_{r,B}^2(c))).
```

证明：

1. 把满足下面条件的 token 称为 bad token：

```text
|xi_i - xibar| >= m_{r,B}/2.
```

2. 因为全部中心化平方误差之和为 `N * eta^2`，Markov 计数给出：

```text
#bad * (m_{r,B}^2/4) <= N * eta^2.
```

所以：

```text
#bad <= 4N * eta^2 / m_{r,B}^2.
```

3. 对任意非 bad 的精确 top-`r` token `i` 和候选池外 token `j`：

```text
shat_i - shat_j
= (s_i - s_j) + [(xi_i-xibar) - (xi_j-xibar)]
> m_{r,B} - m_{r,B}
= 0.
```

因此非 bad 核心 token 不会被非 bad 外部 token 反超。每漏掉一个核心 token，
至少需要一个 bad token 占据或破坏对应位置，于是得到上界。

### 推论 1：无漏失证书

若：

```text
4N * eta_c^2(b) / m_{r,B}^2(c) < 1,
```

则精确 top-`r` 必然全部包含在 proxy top-`B` 中。

这条公式直接展示了三个因素：

```text
风险 proportional to N
风险 proportional to score error eta^2
风险 proportional to 1 / margin^2
```

即使 `eta` 不变，长度翻倍也会使这个最坏情况证书恶化两倍；如果边界 margin
同时缩小，恶化会更快。

## 5. 概率性的 crossing 界

确定性证书很保守。为了得到更接近平均情况的表达，条件于精确 scores，假设每个
token 的中心化 score 误差是零均值 sub-Gaussian，并且：

```text
xi_j - xi_i is sub-Gaussian with variance proxy v_{ij}(b).
```

对精确 top-`r` token `i` 和精确 top-`B` 外 token `j`，记：

```text
Delta_{ij} = s_i - s_j >= m_{r,B}.
```

发生反超必须满足：

```text
xi_j - xi_i >= Delta_{ij}.
```

sub-Gaussian tail bound 给出：

```text
P(j crosses i)
<= exp[-Delta_{ij}^2 / (2 v_{ij}(b))].
```

### 定理 2：核心漏失概率

若每个单 token 误差的 variance proxy 不超过 `sigma^2`，且两个 token 的误差
条件独立，则 `v_{ij} <= 2 sigma^2`，union bound 给出：

```text
P(exact top-r is not contained in proxy top-B)
<= r(N-B) * exp[-m_{r,B}^2 / (4 sigma^2)].
```

为了让失败概率不超过 `delta`，一个充分条件是：

```text
sigma^2
<= m_{r,B}^2
   / [4 log(r(N-B)/delta)].
```

这里长度通过两条路径收紧允许误差：

1. `r(N-B)` 增大，参与竞争的 crossing pair 更多；
2. `m_{r,B}` 通常随候选数增大而缩小。

这不是对实际量化误差独立性的声明。论文中应同时保留不需要概率假设的定理 1，
把定理 2 作为平均情况解释。

## 6. 为什么长度增加通常会缩小边界 margin

先考虑一个可严格计算的例子。若 scores 是独立的 `Uniform(0,1)`，则任意相邻
order-statistic gap 的期望都是：

```text
E[s_(B) - s_(B+1)] = 1 / (N+1).
```

因此长度翻倍时，期望边界 margin 近似减半。

对一般连续分布 `F`，令：

```text
u_N = F^{-1}(1-B/N),
```

在 `u_N` 附近 density `f(u_N)` 平滑且非零时，probability integral transform
给出局部 spacing 近似：

```text
m_{B,N} approximately 1 / [N f(u_N)].
```

如果 active ratio `B/N` 固定，`u_N` 近似固定，margin 通常为 `O(1/N)`。
如果 `B` 固定，`u_N` 会进入分布尾部，缩放由 tail density 决定，但允许的
量化误差仍会随长度改变。

真实 attention scores 有位置、语义和 token 间相关性，不能直接假设 i.i.d.
Uniform。这个 order-statistic 结果只证明“候选数量增加可以系统性压缩
top-k margin”，实际 margin 必须逐层、逐 head 测量。

## 7. 面向 crossing 的固定预算分配目标

普通 Key-MSE 或 score-MSE 平均对待所有 token，但 top-k 只关心边界 crossing。
可以定义条件化的 crossing distortion：

```text
D_cross,c,g(b)
= sum_{i in top-r, j outside top-B}
    E[(xi_{j,g}(b) - xi_{i,g}(b))^2]
    / Delta_{ij}^2.
```

由于 crossing event 蕴含：

```text
|xi_j - xi_i| >= Delta_{ij},
```

Markov inequality 和 union bound 给出：

```text
P(core crossing)
<= sum_g D_cross,c,g(b_g) + cross-band residual.
```

这比平均 Key reconstruction MSE 更贴近最终目标：

- margin 小的边界 pair 权重大；
- 对当前 Query 不敏感的 band 权重低；
- 只在某些 layer/head 出现危险 crossing 时，bit 会集中到对应位置；
- 目标显式依赖 `request, N, layer, head, band`。

实际离散优化仍然很小：

```text
b_c^*
= argmin_b sum_g D_cross,c,g(b_g)
  subject to R(b) <= 240,

b_g in {0,1,2,4,8}.
```

八个 band、五种 bit level 和 15 个归一化预算单位，可以继续使用当前的完整枚举
或动态规划，不需要训练 router。

## 8. 固定总 bit 的 water-filling 解释

用连续高率近似解释分配方向：

```text
D_cross,c,g(b_g) = alpha_{c,g} * 2^(-2b_g),
b_g >= 0,
sum_g b_g <= R.
```

拉格朗日最优解是：

```text
b_{c,g}^*
= [0.5 * log2(alpha_{c,g}/tau_c)]_+,

sum_g b_{c,g}^* = R.
```

如果所有 band 都 active：

```text
b_{c,g}^*
= R/G
  + 0.5 * log2(alpha_{c,g}/geomean_u(alpha_{c,u})).
```

因此总预算 `R` 可以完全不变，但只要 `alpha_{c,g}` 随请求、长度、层或 head
变化，最优 bit 分配就会变化。

需要强调：

```text
长度造成的统一 margin 缩放只会收紧总体误差目标；
只有 band sensitivity 或边界 pair 组成发生非均匀变化时，
最优的 band 相对分配才会变化。
```

所以数学上正确的说法不是“长度翻倍必然把某个 band 从 2 bit 改成 4 bit”，
而是“条件化风险曲线改变时，固定分配一般不再最优”。

## 9. 固定策略不可能对所有请求统一最优

### 定理 3：两 band 反例

考虑两个一维 band，总预算只允许一个 band 无损存储，另一个 band 置零。

请求 A：

```text
q_A = (1,0).
```

精确 top-1 信息全部位于 band 1。把 bit 给 band 1 可以无损排序；把 bit 给
band 2 会让精确 top token 与外部 token 在 proxy 中变成 tie，并可在固定
tie-break 下选择错误 token。

请求 B：

```text
q_B = (0,1).
```

情况完全相反，必须把 bit 给 band 2。

因此：

```text
不存在一个固定的一-band分配同时对 A 和 B 最优；
request-adaptive 分配可以在完全相同总 bit 下同时成功。
```

把 A、B 解释为不同层或不同 head，就得到 layer/head 条件化的必要性。

### 长度反例

保持 Query：

```text
q = (1,1).
```

短上下文 `N1` 中，精确 top token 的区分信息只位于 band 1，因此 band 1
高精度最优。扩展到 `N2>N1` 后，加入一个 score 略高、但区分信息只位于
band 2 的新候选。此时 band 1-only 策略会继续选择旧 token，band 2-only
策略才能选择新 top token。

于是同一个 Query、同一个总 bit，仅仅增加历史候选就可以改变条件最优策略。

这两个反例证明的是“不存在统一最优固定模板”，不是说每一个真实请求都必须
使用独一无二的模板。

## 10. “不增加总 bit”的严格条件

令固定 rate `R` 下的可行策略集合为：

```text
Theta_R
= {score-preserving QK transform and mixed-bit b : R(b) <= R}.
```

定义当前条件下的最优证书值：

```text
C_R(c)
= min_{theta in Theta_R}
    4N * eta_c^2(theta) / m_{r,B}^2(c).
```

### 定理 4：固定 rate 的条件充分性

若：

```text
C_R(c) < 1,
```

则存在一个总预算不超过 `R` 的策略，使精确 top-`r` 核心全部进入 proxy
top-`B`。因此在该条件 `c` 下，增加总 bit 不是必要条件。

对一组条件 `C`，若：

```text
for every c in C, C_R(c) < 1,
```

则同一个总预算 `R` 对整组条件都足够，但每个条件可以选择不同的
`theta_c^*`。这不要求存在单个冻结策略满足：

```text
max_c C(theta_fixed,c) < 1.
```

证明直接来自定理 1 和 `C_R(c)` 的定义。

这正是“固定容量、条件化策略”的数学表达。

### 定理 5：同预算条件化策略支配冻结策略

上面的定理回答“某一个条件下，固定 rate 是否可能足够”。还可以进一步比较
一组条件上的平均风险。令 `L(c,theta) >= 0` 是任意可积风险，例如：

```text
L(c,theta) =
  centered score MSE eta_c^2(theta)，或
  crossing upper bound，或
  omitted attention mass，或
  一阶 attention output distortion.
```

在同一个固定预算可行集 `Theta_R` 中，冻结策略和条件化策略的最优风险分别为：

```text
L_frozen(R)
= min_{theta in Theta_R} E_c[L(c,theta)],

L_adapt(R)
= E_c[min_{theta in Theta_R} L(c,theta)].
```

则总有：

```text
L_adapt(R) <= L_frozen(R).
```

证明只需要对任意固定 `theta` 使用逐点不等式：

```text
min_{u in Theta_R} L(c,u) <= L(c,theta),
```

再对 `c` 取期望，最后对右侧的 `theta` 取最小。

当 `Theta_R` 是有限集合时，等号成立当且仅当至少存在一个冻结策略
`theta_star`，它对几乎所有条件都是逐点最优：

```text
theta_star in argmin_theta L(c,theta), almost surely.
```

否则不等式严格：

```text
L_adapt(R) < L_frozen(R).
```

这是“不增加总 bit 仍能提高质量”的最直接数学结论。它不依赖高率量化近似，
也不依赖 attention score 独立或高斯。不同请求、长度、层或 head 只要以正概率
偏好不同策略，冻结模板就会产生严格的平均风险代价。

相应的条件化 distortion-rate envelope 为：

```text
D_adapt(R)  = E_c[min_{theta in Theta_R} L(c,theta)],
D_frozen(R) = min_{theta in Theta_R} E_c[L(c,theta)].
```

因此对任意目标风险 `epsilon`：

```text
R_adapt(epsilon) <= R_frozen(epsilon).
```

这里的含义是条件化可以节省达到同一风险所需的 rate；并不是说任意小的固定
rate 都一定足够。若某个条件下：

```text
min_{theta in Theta_R} L(c,theta) > epsilon,
```

那么只重新分配仍无法达标，必须提高 rate、扩大候选或改变表示族。

### 实际估计器的代价

真正运行时不知道 `L(c,theta)`，只能从 prefill 中估计
`Lhat(c,theta)` 并选择：

```text
theta_hat(c) = argmin_theta Lhat(c,theta).
```

若可行策略集合有限，则有逐条件确定性 regret 界：

```text
L(c,theta_hat)
- min_theta L(c,theta)
<= 2 * max_theta |Lhat(c,theta)-L(c,theta)|.
```

证明是在真实风险和估计风险之间加减两次，并使用
`Lhat(c,theta_hat) <= Lhat(c,theta_star)`。这条式子明确区分：

```text
条件化本身的理论收益
vs.
有限采样、Query drift 和近似风险估计造成的实现损失。
```

因此论文不能只报告 oracle 条件化结果，还必须报告 request-local 估计器的
held-out regret、构建开销和跨 decode 位置稳定性。

### 跨 layer/head/band 的固定总预算

上面的 240 bit 是“每个 KV head 各自固定预算”。更一般地，把索引单元记为：

```text
a = (layer l, KV head h, band g).
```

考虑整个模型或一层内共享的总预算：

```text
min_{b_a >= 0} sum_a alpha_{c,a} * 2^(-2b_a)
subject to sum_a w_a b_a <= R_total.
```

`w_a` 是该单元每增加 1 bit 的物理存储成本。KKT 条件给出：

```text
b_{c,a}^*
= [0.5 * log2(alpha_{c,a}/(tau_c w_a))]_+.
```

因此在总 bit 完全固定时，bit 可以同时在 layer、head 和 band 之间移动：

```text
高 alpha_{c,l,h,g} / 单位成本的单元获得更多 bit，
低敏感度单元获得更少 bit 或直接置零。
```

若系统要求每个 head 恰好 240 bit，只需给每个 `(l,h)` 使用独立的
`tau_{c,l,h}`；此时只能在该 head 的 band 间移动。若采用 ragged packed
layout，则可以使用一个全局 `tau_c`，进一步在 head 和 layer 间共享预算。
两者的理论目标相同，物理布局和 kernel 复杂度不同。

### 坐标条件化同样不增加 bit

正交消融表明，不能把“策略”缩窄成 bit allocation。令当前条件的 QK joint
matrix 为：

```text
M_c = C_{q,c}^{1/2} C_{k,c}^{1/2}
    = U_c Sigma_c V_c^T.
```

在 Query 和 Key 独立采样、二阶矩固定的分析模型下，用 rank-`r` 双线性映射
`B` 近似精确 score 的风险为：

```text
L_c(B)
= || C_{q,c}^{1/2}(I-B)C_{k,c}^{1/2} ||_F^2.
```

Eckart--Young 定理给出 request-local 最优值：

```text
min_{rank(B)<=r} L_c(B)
= sum_{j>r} sigma_{c,j}^2.
```

对任意从其他条件 `c0` 冻结的同 rank 映射 `B_{c0}`：

```text
L_c(B_{c0}) - L_c(B_c^*)
= ||M_c-C_{q,c}^{1/2}B_{c0}C_{k,c}^{1/2}||_F^2
  - sum_{j>r} sigma_{c,j}^2
>= 0.
```

若第 `r` 与 `r+1` 个奇异值有严格 gap，等号要求冻结映射仍然张成当前条件的
最优奇异子空间。只要 joint singular directions 随请求或长度旋转，冻结坐标
通常就严格次优，而 request-local 坐标使用完全相同的 rank 和 bit 数。

一个二维反例更直观。总预算只能无损保留一个坐标，且 `sigma_1 > sigma_2`：

```text
条件 c0: M_0 = diag(sigma_1, sigma_2)，冻结策略保留坐标 1；
条件 c1: M_1 = diag(sigma_2, sigma_1)，当前最优策略保留坐标 2。
```

在 `c1` 上：

```text
冻结坐标误差 = sigma_1^2，
条件化坐标误差 = sigma_2^2.
```

当 `sigma_1/sigma_2` 很大时，两者差距可以任意大，但存储的坐标数和总 bit
完全相同。QKSieve 的 mixed-bit 情况不是纯 rank 截断，不过其尾部 0-bit、
中间 1-bit、头部 4-bit 结构遵循同一机制：过期坐标可能把当前高 score-energy
方向旋转进低 bit 或 0-bit band；重新计算 QK-balanced 坐标会把它们重新排到
高精度 band。

### 为什么 layer/head 的敏感度会不同

单个 attention head 的输出为：

```text
o = sum_i p_i v_i,       p = softmax(s).
```

对 score 做一阶扰动 `delta s`，softmax Jacobian 给出精确的一阶项：

```text
delta o
= sum_i p_i (v_i-o) delta s_i
+ O(||delta s||^2).
```

因此 band `g` 的局部输出失真可定义为：

```text
D_out,c,l,h,g(b)
= E[
    ||sum_i p_i(v_i-o) delta s_{i,g}(b)||_2^2
  ].
```

若只为解释而额外假设不同 token 的误差条件不相关，则：

```text
D_out,c,l,h,g(b)
= sum_i p_i^2 ||v_i-o||_2^2
  Var[delta s_{i,g}(b)].
```

这说明同样大小的 QK 误差对输出并不同价：

- attention 概率大的 token 权重大；
- `v_i` 与当前输出 `o` 差异大的 token 权重大；
- 当前 Query 在某个 band 上能量大时，该 band 的 score 误差更大；
- 不同 head 的 `p_i`、`v_i` 和 Query 方向不同，权重自然不同。

再令 `J_{>l}` 表示第 `l` 层 attention 输出到最终 logits 的局部 Jacobian，
`W^O_l` 表示该层输出投影，则：

```text
delta z
approximately
sum_l J_{>l} W^O_l delta o_l.
```

忽略跨层误差相关项时，第 `l` 层的输出感知失真为：

```text
D_logit,c,l,h,g(b)
= E[
  delta o_{l,h,g}^T
  (W^O_l)^T J_{>l}^T J_{>l} W^O_l
  delta o_{l,h,g}
].
```

所以 `alpha_{c,l,h,g}` 同时包含 QK crossing 风险、Value leverage 和下游层
放大率。它依赖请求、长度、层、head 和 band 并非经验上的偶然，而是由
attention Jacobian 和网络复合结构直接导出的。实际方法不必计算昂贵的完整
`J_{>l}`；crossing-aware score surrogate 是一个 training-free、可实现的近似，
而 output-aware 式子用于解释 layer/head 条件化的理论来源。

## 11. 从 top-k 证书到 attention 输出

设最终选择集合为 `S`，Full attention probability 为 `p_i`，遗漏质量为：

```text
epsilon_S = sum_{i not in S} p_i.
```

QKSieve 在 `S` 上重新使用原始 FP16 K/V 计算精确 attention，因此：

```text
||p - p_tilde||_1 = 2 epsilon_S,

||o - o_tilde||_2
<= epsilon_S * diameter({v_i}).
```

如果 top-`r` 核心全部保留，则：

```text
epsilon_S <= 1 - sum_{i=1}^r p_(i).
```

结合定理 2，可得到概率性输出界：

```text
P(
  ||o-o_tilde||_2
  > [1-sum_{i=1}^r p_(i)] * diameter(V)
)
<= delta.
```

因此理论链条是：

```text
条件化 QK/band 统计
-> 固定 rate 下的 crossing-aware bit allocation
-> top-r 核心保留
-> omitted attention mass
-> attention output error
-> 后续 decoder logit stability.
```

## 12. 当前实验如何对应理论

严格四窗口、256 个 paired token 的正交消融结果：

| 256K, top-1,280/head | 坐标 | 240-bit allocation | PPL retention (95% CI) | Top-1 | KL | Steady | Online |
|---|---|---|---:|---:|---:|---:|---:|
| Exact FP16 QK | 原始 FP16 | 不适用 | 100.240% [99.48, 100.88] | 90.625% | 0.00516 | 1.088x | 1.088x |
| Frozen baseline | 冻结 | 冻结 | 80.771% [77.84, 83.92] | 67.188% | 0.24246 | 7.785x | 6.668x |
| Frozen-basis + realloc | 冻结 | request-local Key-MSE | 89.911% [84.13, 94.28] | 77.344% | 0.13332 | 7.766x | 5.259x |
| Request-local + fixed | request-local | `(4,1,1,1,1,1,0,0)` | **100.443% [99.80, 101.02]** | 91.016% | **0.00620** | 7.671x | **4.955x** |
| Request-local + realloc | request-local | request-local Key-MSE | 100.430% [99.78, 100.99] | 91.016% | 0.00622 | 7.733x | 4.055x |

另一个独立控制中，冻结坐标但把所有维度提高到 INT8，也达到
100.267% retention、0.00508 KL、1.882x steady 和 1.846x online。

同一冻结 240-bit 策略在 128K 的保持率为 98.993%，到 256K 变成 80.771%；
Exact-QK top-1,280 在两个长度都约为 100%。这符合：

```text
oracle budget 仍足够，
但固定 proxy 的 eta_N / margin_N 跨过了 ranking boundary。
```

更重要的是，使用 request-local 坐标但固定一个极其简单的 240-bit
allocation：

```text
(4,1,1,1,1,1,0,0)
```

就恢复到 100.443%。这构成“提高总 bit 不是必要条件”的直接经验例证。

正交消融进一步完成归因：

```text
只改 allocation:
80.771% -> 89.911%，提升 9.140 个百分点，但仍显著低于 Full。

只改坐标:
80.771% -> 100.443%，完整恢复。

local 坐标下再做自动 allocation:
100.443% -> 100.430%，没有可分辨质量收益，
且 64-token online speed 从 4.955x 降到 4.055x。
```

因此针对当前 256K 断崖，严格结论是：

```text
主因：冻结 QK-balanced 坐标跨请求/长度失配；
次因：冻结 allocation 失配；
不是：top-1,280 oracle budget 不够；
也不是：240-bit 物理 rate 本身不够。
```

这也修正了最初过强的猜测：不能说“256K 必须动态改变每个 band 的 bit”。
在这四个窗口上，request-local 坐标加固定 allocation 已经足够，而且更快。

这不等于动态 allocation 没有研究价值。已有独立 32K sports/medicine
机制诊断覆盖 `36层 x 8 KV heads = 288` 个 layer/head pair：

```text
平均物理 rate: 1.857 bit/Key dimension
平均 active bands: 3.573 / 8
两个请求的精确 allocation agreement: 45.83%
即 54.17% 的 layer/head allocation 发生变化
```

所以论文应分别陈述：

1. 理论上，固定 rate 下的条件化坐标和 allocation 都可严格支配冻结模板；
2. 实验上，256K 质量恢复主要来自 request-local 坐标；
3. request/head 级 allocation 确实会变化，但它何时带来净质量收益需要
   crossing-risk 与 setup cost 共同判断，不能由这组 256K 结果过度声称。

## 13. 可以形成的新方法

比“按长度查表增加 bit”更有研究价值、也更符合消融结果的方法是：

### Conditional-coordinate fixed-rate QKSieve

1. prefill 过程中按 layer/head 增量估计当前 Q/K moments；
2. 构造 request-local、score-preserving QK-balanced 坐标；
3. 默认使用已经验证的固定 240-bit allocation
   `(4,1,1,1,1,1,0,0)`，避免不必要的 calibration setup；
4. 从 prompt-tail Query 与分层 Key 样本估计冻结坐标和 local 坐标的
   score/crossing risk；
5. 对确有明显 allocation regret 的 layer/head，才在同一 240-bit 预算内
   运行八 band 常数规模 DP；否则保留固定 allocation；
6. 用 `C_R(c)`、经验 margin 或其上界报告每个 layer/head 的风险证书；
7. 将 moment 累积、SVD 和 index construction 融进或重叠于 prefill；
8. decode 只扫描低比特索引，最终只在 top-`B` 原始 FP16 K/V 上计算。

第 5 步不是 learned router。可以用完全数值化的选择规则。若
`Lhat_fixed` 与 `Lhat_auto` 是同一组 held-out prompt-tail 样本上的 crossing
风险估计，则只有当：

```text
Lhat_fixed - Lhat_auto
> 2 * uniform_estimation_error + setup_cost_penalty
```

才启用动态 allocation。否则使用固定 allocation。这样避免把采样噪声当成
需要重新分配 bit 的信号，也解释了为何当前 256K 应选择 local-coordinate +
fixed-bit 版本。

这个方案具有以下特点：

- training-free；
- 不使用任务标签和 learned router；
- 不回退 Full attention；
- 总 index rate 固定；
- 主适应对象是 request/layer/head 条件化 QK 坐标；
- 仅在数值证据充分时才改变 band allocation；
- 风险目标直接面向 top-k crossing，而不是平均 Key reconstruction；
- 理论目标、系统实现和实验指标可以一一对应。

## 14. 论文中建议使用的表述

可以写：

> The rate required by a retrieval index and the allocation of that rate are
> distinct. Context length tightens ranking margins and increases the number
> of potential crossing pairs, while request-, layer-, and head-conditional
> QK statistics rotate the joint score subspace and change bandwise
> distortion. We derive a fixed-rate crossing certificate and show that
> condition-specific QK coordinates can satisfy it without increasing index
> bits, even when no single frozen template is uniformly valid. A strict
> 256K split attributes the recovery primarily to the coordinates rather
> than to additional rate or dynamic bit allocation.

不能写：

> 240 bit 在所有长度和请求上理论上必然足够。

不能写：

> 256K 失效来自每个 head 的 bit 数分配，因此所有请求都需要动态 allocation。

当前证据严格支持的是：

```text
冻结低比特策略失效；
冻结高位宽和 request-local 固定 rate 都能修复；
固定 rate 的条件化策略具有数学充分条件；
256K 修复主要来自 request-local QK 坐标；
动态 allocation 有条件化理论价值，但在当前 256K 上没有超过固定 allocation。
```
