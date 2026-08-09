# 超长上下文质量退化：诊断与方法设计

## 可证伪假设

`64K` 不是模型中的自然分界。固定保留 1,280 个 token 时，质量随长度下降的主要原因是遗漏集合承载的 softmax 质量持续增加；真正决定输出误差的是遗漏质量与遗漏 Value 残差的加权方向。若该判断正确，补偿 Value 尾部应比继续提高 Key 位宽更有效。

截至 2026-08-03 的因果实验进一步把问题分成两类：

1. 旧 256K 灾难性下降来自冻结 Q/K 坐标与当前请求失配；请求局部 QK-balanced 坐标已经消除该故障。
2. 请求局部坐标下仍存在的最差主题退化，主要来自 Value-tail 近似；religion 128K 中，exact top-1,280、QK proxy 和 rank-16 Value tail 的质量保持率依次为 99.20%、98.87% 和 97.28%。

因此当前的新假设是：普通 Value PCA 优化了原始 Value 重构误差，但模型真正使用的是经过输出投影 `W_o` 的误差；同时，Value-tail 的最优强度在同一请求内具有时间稳定性，可以由少量 prefill query 闭式估计，而不需要长度规则或训练 router。

## 物理先验

1. QKSieve 的低比特 QK 排序在 32K--128K 仍较稳定，因此排序误差不是当前首要矛盾。
2. 固定 token 数无法在任意长度保持固定 attention mass。
3. Value 的低秩残差不是独立白噪声；独立噪声假设给出的 SURE 风险严重偏小，不能用作当前证书。
4. 方法应由当前 query 的可测误差决定预算，而不能写成 `N > 64K` 的任务补丁。
5. 对 GQA 的一个 KV head，Value 重构误差应使用共享该 KV head 的全部 query head 对应 `W_o` 块来度量。

## 数学模型

令选中集合为 `S`，遗漏集合为 `T`，`w_i = exp(q k_i^T)`。完整 attention 输出为：

```text
o = (sum_{i in S} w_i v_i + sum_{i in T} w_i v_i)
    / (Z_S + Z_T)
```

只计算 `S` 时，精确误差恒等式为：

```text
o_sparse - o = (1 - p_S) * (mu_S - mu_T)
```

因此长度只通过 `p_S` 和 `mu_T` 间接影响误差。

将 Value 写成低秩重构与残差：

```text
v_i = vhat_i(r) + e_i(r)
```

### 输出投影加权的 Value 子空间

对第 `g` 个 KV head，令共享它的 query head 集合为 `H(g)`，`W_{o,h}` 是 `W_o` 中作用于第 `h` 个 head 输出的列块。定义：

```text
G_g = sum_{h in H(g)} W_{o,h}^T W_{o,h}
```

普通 PCA 最小化 `sum_i ||v_i-vhat_i||_2^2`。新子空间最小化模型真正看到的二次误差：

```text
sum_i (v_i-vhat_i)^T G_g (v_i-vhat_i)
```

令 `G_g = L_g L_g^T`，先对 `(V-mu)L_g` 做 rank-r PCA，再用 `L_g^{-T}` 映射回 Value 空间。它不增加每个 token 的编码长度，也不增加 decode 扫描维度；只改变一次性构造的 encoder/decoder 基。

### 请求内闭式 tail 校准

令只使用 top-k 的输出为 `s_t`，使用低秩 Value-tail 的输出为 `y_t`，校正方向为 `d_t=y_t-s_t`。对 prefill 最后 `m` 个 query，完整 prefill 已给出精确输出 `o_t`。每层只存一个系数时：

```text
alpha_l = clip_[0,1](
    sum_t <W_o(o_t-s_t), W_o d_t>
    / sum_t ||W_o d_t||_2^2
)
```

GQA 细化版为每个 KV head 存一个系数。将各组校正经 `W_o` 投影后记为 `D_t`，解一个带微小岭项的 8x8 线性系统，再把结果截断到 `[0,1]`。两者都没有训练参数、任务标签或长度阈值；decode 只增加一个标量乘法。

### 输出误差感知的精确 token 选择

固定 `alpha` 实验显示，当前最差 128K 请求的主要误差来自未选 token 的 Value 残差，而不是 QK 排序。令：

```text
e_i = v_i - vhat_i
rho_i = sqrt(e_i^T G_g e_i)
```

其中 `rho_i` 是 token `i` 的 Value sketch 残差经过共享 GQA 输出投影后的大小。若 selected token 使用精确 Value、其余 token 使用 `vhat_i`，由三角不等式：

```text
||Delta o|| <= sum_{i not in S} p_i rho_i
```

在固定 `|S|=k` 时，使该上界最小的集合正好是最大的 `p_i rho_i`。softmax 分母对排序是公共常数，因此等价于：

```text
priority_i = proxy_score_i + log(rho_i)
S = top-k(priority)
```

这与当前只按 `proxy_score_i` 选 token 有本质区别：高 attention 但已被 Value sketch 准确表示的 token 不必浪费精确槽位；中等 attention 但 Value 残差很大的 token 应被取回。`log(rho_i)` 在建索引时计算一次，可量化为每个 KV token/head 一个 INT4 或 INT8 风险码。INT8 的额外逻辑存储仅为完整 FP16 K/V 的约 `1/512=0.195%`，decode 扫描只多一次解码和加法。

该准则的失败条件也明确：如果主要误差来自 QK score 而不是 Value，或不同 query head 对同一 KV residual 的 `W_o` 敏感方向差异过大，共享 `G_g` 风险码可能不足。实验必须同时报告普通 top-k、FP32 风险、INT8 风险、INT4 风险，以及使用 exact score 的诊断上界。

### 无长度阈值的输出风险预算

固定 `k` 仍然无法覆盖从 10K 到 10M 的上下文。令低比特 Key 给出的近似 logit 为 `ztilde_i`，将上面的 Value 风险向上量化为 `rhobar_i >= rho_i`。首先定义可部署的风险质量：

```text
rtilde_i = exp(ztilde_i) * rhobar_i
```

按 `rtilde_i` 降序排列，选择使累计风险质量达到目标比例 `eta` 的最小集合：

```text
S_eta = smallest S such that
        sum_{i in S} rtilde_i / sum_j rtilde_j >= eta
```

该规则不读取任务名或上下文长度。若注意力仍集中且 Value sketch 能准确表示多数 token，即使 `N` 很大，所需 `k` 仍小；若长度增加产生了许多不可由 sketch 表示的重要 token，`k` 会由数值分布自动增加。只按 attention mass 选择是消融项，它把 `rhobar_i` 固定为 1。

进一步，对每个 256-token Key 量化块保存各投影坐标的最大重构误差 `epsilon_b,j`。当前 query 的精确 QK 误差满足：

```text
delta_b(q) = sum_j |qprime_j| * epsilon_b,j / sqrt(d)
|z_i - ztilde_i| <= delta_b(q),  i in block b
```

因此未选 Value 残差的严格上界为：

```text
B_V(S) = [sum_{i not in S} exp(ztilde_i + delta_b(i)) * rhobar_i]
         / Z_S
Z_S    = sum_{i in S} exp(z_i)
||sum_{i not in S} p_i W_o e_i|| <= B_V(S)
```

其中 `Z_S` 在取回候选的精确 K 后可直接计算。算法从较小候选集开始；若 `B_V(S)` 超过误差容限，则从同一次扫描保留的后续候选中扩展，直到证书通过。它是由当前 query、当前缓存和量化误差共同决定的 progressive budget，不存在 `N>64K` 的人工分支。

`B_V` 只证明 Value 重构项。proxy softmax 权重引入的误差必须用独立的 QK 区间项 `B_K` 报告；在该项尚未实现和验证前，方法只能称为“风险覆盖预算”，不能声称完整 attention 输出得到严格认证。

实现失败条件：风险覆盖目标与真实输出误差不相关；达到稳定质量需要接近 Full 的预算；QK 区间上界过于保守；或者逐步扩展造成的二次访存抵消稀疏 attention 收益。

### 跨 head 的全局风险分配与一致权重

> 2026-08-03 闭环验证结论：下面的“无约束全局 top-B”只在使用
> Full query 的单层离线重放中成立，已被整模型实验否定，不能作为方法。
> 在 Llama 4K 的第一个真实 decode step，它使部分 head 只剩 24--51
> 个 token；第 0 层 hidden-state 相对偏差约 0.36%，第 1 层即放大到
> 约 5.94%，第 2 层约 47.84%。固定每 head 1,280 的对应偏差在全部
> 32 层均低于 0.75%。这说明跨 head 硬阈值对 query 扰动不连续。

初步实验否定了“每个 head 独立覆盖相同比例风险”的规则。原因是该规则先除以每个 head 自己的总风险，丢掉了 head 之间的绝对尺度：一个总输出风险很低的 head 和一个高风险 head 会得到相同覆盖率，但后者应得到更多精确 token。

对 query head `h` 和 token `i`，定义 head-specific 输出残差：

```text
rho_h,i = ||W_o,h (v_i - vhat_i)||_2
c_h,i   = ptilde_h,i * rho_h,i
```

所有 head-token 对共享一个总预算 `B` 时，选择全局最大的 `B` 个 `c_h,i`，正好最小化下述 proxy Value 误差上界：

```text
||Delta y_value||_2
<= sum_h sum_{i not in S_h} ptilde_h,i * rho_h,i
```

这会自然产生不同的 `k_h`，而不使用层号、head 号、任务或长度规则。若使用误差容限，则选择使全局剩余风险质量低于 `tau` 的最小集合。head-specific INT4 风险码对 GQA-4 的额外逻辑存储为每 KV token 16 bit，约占完整 FP16 K/V 的 `16/4096=0.391%`。

为了让误差分解和预算单调性更清楚，候选输出使用同一套 proxy softmax 权重：

```text
ohat = sum_{(h,i) in S} ptilde_h,i * v_i
     + sum_{(h,i) not in S} ptilde_h,i * vhat_i
```

即精确槽位只取回原始 Value，不再把 selected token 的权重突然替换成 exact QK。于是总误差可以分成两个独立项：

```text
||o_exact - ohat||
<= E_score(QK proxy, full Value)
 + E_value(S, proxy weights)
```

`E_score` 与 Value 预算无关，由 QK 索引和可选的数值温度校准决定；`E_value` 则由全局 top-`c_h,i` 最小化。系统上该形式还可能省去 top-k token 的精确 K 读取与 QK 重排，只读取选中 Value。它必须先通过 real-QKV 和模型 PPL 验证；若 coherent proxy 的 score-only 误差过大，则继续保留 exact selected QK 的混合路径。

### 闭环稳定的逐 head 输出误差证书

最终目标不是让一次 teacher-forced attention 的误差和最小，而是保证
每层扰动后产生的新 query 仍落在稳定区域。为此不再固定全局总槽位，
而对每个 query head 独立限制最坏输出误差。

令低比特 logit 为 `ztilde_h,i`，由精确探针或块级量化界得到当前
query 的 score 误差尺度 `delta_h,i`。Value sketch 残差经过该 query
head 输出投影后的大小为：

```text
rho_h,i = ||W_o,h (v_i-vhat_i)||_2
```

令 `ohat_h` 为 proxy 权重与 Value sketch 给出的 head 输出，并定义：

```text
d_h,i = ||W_o,h (vhat_i-ohat_h)||_2
c_h,i = ptilde_h,i * [rho_h,i + delta_h,i * d_h,i]
```

第一项近似 Value 残差风险；第二项是一阶 score 扰动对输出的风险。
候选按 `c_h,i` 降序排列。对每个 head，选择满足下式的最小前缀：

```text
sum_{i not in S_h} c_h,i
-------------------------------- <= tau
||W_o,h ohat_h||_2 + epsilon_scale
```

其中 `tau` 是所有长度、任务、层和 head 共用的输出误差容限，
`epsilon_scale` 只防止分母接近零。规则不输入上下文长度，也不规定
固定 token 数；候选池变密、score 误差增大或 Value 残差增大时，预算
都会自动增加。

稳定性来自两个约束：

1. 每个 head 必须独立通过误差条件，不能被其他 head 抢空预算。
2. 若两个 query 的 `c_h` 向量相差至多 `epsilon`，则尾部风险只会按
   该扰动连续变化；不会出现全局硬水位跨过后一次转移数千槽位的情况。

这仍是待验证的候选，不是已成立结论。首先用真实 Q/K/V trace 测试
`tau` 扫描下的实际局部误差、p99、最大误差和预算；只有 4K、32K、
128K 与第二模型同时通过，才接入模型闭环。若可用 `tau` 随模型或主题
明显变化，或需要接近 Full 才通过，则该证书操作化失败。

对每个 256-token 块 `b`，用 QKSieve 已有的低维 Key 坐标 `x_i` 建立解析条件均值：

```text
e_i(r) = mu_e,b + A_r (x_i - mu_x,b) + epsilon_i
```

`A_r` 由 prefill 缓存上的岭回归闭式求解，不训练 router，也不读取答案 token。运行时扫描低比特 Key 时同时累积每块的 `sum(w_i)`、`sum(w_i^2)` 和 `sum(w_i x_i)`，从而估计遗漏 Value numerator。

```text
Nhat_T = sum_{i in T} wtilde_i *
         [vhat_i(r) + mu_e,T,b + A_r(x_i - mu_x,T,b)]
```

其中块内 tail 均值由预存块统计量减去已选 token 的精确统计得到。最终输出为：

```text
ohat = (N_S,exact + Nhat_T) / (Z_S,exact + Zhat_T)
```

对回归后残差 `epsilon_i`，定义可在线计算的相对风险：

```text
R_abs(r,d)^2 = sum_b S2_b * sigma_e,b(r,d)^2 / Zhat^2
R_rel(r,d)   = R_abs(r,d) / ||ohat||_2
S2_b         = sum_{i in T_b} wtilde_i^2
```

候选方法按 `r in {16,32,64,96,128}`、`d in {0,8,16,32}` 从便宜到昂贵选择首个满足 `R_rel <= tau` 的配置。`tau` 必须在独立验证集冻结；决策不直接使用长度、任务名或答案标签。

## 实现契约

输入：当前 query、低比特 Key 索引、GPU 上的原始 K/V、rank-16 INT4 Value sketch、模型静态 `W_o`、每 token 的低比特 Value 风险码，以及可选的 Key 块误差上界。

固定参数：Value rank 16、Value INT4、Key rate 15 bit/coordinate、风险覆盖率 `eta` 或证书容限 `tau`。`top_k=1280` 仅作为固定预算对照，不是最终规则。

步骤：

1. 用 `G_g` 构造 W_o-metric Value basis，编码 rank-16 INT4 Value 坐标，并为每个 token 计算 `rho_i`。
2. 将 `log(rho_i)` 按块向上量化为 INT4/INT8 风险码；Key 索引可选保存块级量化误差上界。
3. decode 的融合扫描同时计算 proxy score、风险优先级、总风险质量和分块 top candidates。
4. 按固定 `k` 或风险覆盖率取得候选，对其原始 K/V 做 exact attention，同时用 Value sketch 聚合未选 tail。
5. 证书版计算 `B_V`；若不通过则扩展候选，否则输出当前 attention 结果。

输出：attention 向量、实际每 head 预算、风险覆盖率、`B_V/B_K`、selected mass、QK 边界统计、索引构建与各 decode 阶段耗时。

通过条件：风险排序在独立主题上优于普通 QK 排序；风险覆盖预算在相同平均 token 数下优于 attention-mass 预算；独立主题和第二模型上平均局部误差不高于 3%、p99 不高于 10%；整模型 PPL 质量保持不低于 99.5%，且不依赖长度规则。

失败条件：prefill 与 decode 的最优系数明显漂移、W_o-metric 只改善离线 L2 而不改善 PPL、额外校准无法在多轮复用中摊销，或独立模型 PPL 明显下降。

## 预期存储

rank-16 INT4 Value sketch 已占完整 KV 的约 1.96%。当 `d=8`、块大小 256、块均值使用 FP16 时，条件统计额外约占完整 KV 的 0.21%，全局矩阵 `A` 在长序列下可忽略。总索引预计约 8.0%，但最终以真实分配和元数据统计为准。

## 候选冻结算法：双证书信息约束 QKSieve

### 可证伪假设

固定 `top-1280` 随长度退化包含两类不同风险：大量小的 QK 分数误差会改变整条 softmax 分布，少数异常大的分数误差会漏掉单个关键 token。若该拆分正确，则只用平均 RMSE 或只用均匀探针都不能同时覆盖两类风险；`cross-fit softmax KL` 与逐 token 分数上界的组合应当能在不读取长度和任务名的情况下选择安全位宽与安全 token 数。

### 输入与固定参数

- 输入：当前层的 query、请求内 QK-balanced 坐标、原始 GPU K/V、量化 Key 候选、rank-16 INT4 Value sketch、INT4 Value 风险码。
- 候选 Key rate：`R={15,19,23,27}` bit/coordinate。候选集合在所有长度、模型和任务上保持不变。
- 探针：每层 256 个确定性分层位置；偶数位置与奇数位置互为拟合组和验证组。
- 当前发现阶段的 KL 容限：`tau_KL=0.20`。该值必须在 discovery 集冻结，再在 held-out 主题上验证。
- token 误差容限：使用同一个归一化输出 RSS 容限；不设置 `N>64K` 分支，也不设置 Full 回退。

### 阶段 1：稳定坐标与候选位宽

1. 用 prefill 末尾 8 个 query 和抽样 Key 构造请求局部 QK-balanced 坐标。
2. 对 query 二阶矩使用 OAS 向各向同性矩阵收缩，避免 8 个 query 在 128 维空间中的秩亏过拟合。
3. 对每个 `r in R`，按收缩后的 QK-MSE 闭式失真分配各 16 维 band 的位宽。
4. 当前 decode query 只选择总 rate，不重新学习 band 顺序；单个 query 重新分配 band 已被实验否定。

### 阶段 2：cross-fit softmax KL rate 证书

对某个候选 rate `r`，在 256 个探针上得到精确分数 `s` 与代理分数 `stilde_r`。在偶数探针拟合正斜率仿射校准，在奇数探针计算 KL；再交换两组并取平均：

```text
D_r,h = 0.5 * [KL(softmax(s_B) || softmax(a_A stilde_B))
               + KL(softmax(s_A) || softmax(a_B stilde_A))]
D_r   = p90_h D_r,h
```

截距在 softmax 中抵消，仅保留正斜率。选择满足 `D_r <= tau_KL` 的最小 rate；若没有候选通过，选择 `R` 中最大 rate。该决策每层独立，首次 decode 后复用；只有探针 KL 或 query 数值漂移超过容限时才重检。

### 阶段 3：罕见漏检上界与 token 预算

Key 量化残差给出逐 token 分数上界 `u_i`。对候选集合 `S`：

```text
L_S = logsumexp_{i in S}(stilde_i - u_i)
U_T = logsumexp_{i not in S}(stilde_i + u_i)
mass_tail_upper = exp(U_T - logaddexp(L_S, U_T))
```

若 `u_i` 是有效上界，则 `mass_tail_upper` 是未选 softmax 质量的确定性上界。token 先按 QK/Value 联合 RSS 风险排序，再扩大到归一化输出风险和 `mass_tail_upper` 都通过。它保护“代理分数很低但真实分数很高”的单针反例；均匀 KL 探针无法单独保证这一点。

### 输出与调试产物

- 每层选择的 rate、各 band 位宽、cross-fit KL 的每 head 分布。
- 每 head 实际 token 数、尾部 RSS、`mass_tail_upper`、是否因 KL 或罕见事件上界升级。
- 索引构建时间、探针时间、扫描时间、候选选择时间、精确稀疏 attention 时间和 Value-tail 时间。

### 通过、失败与声明边界

- 通过：冻结 `tau_KL` 后，在未参与设计的主题和 seed 上，位宽决策不低估危险条件；相同或更低总流量下优于固定 rate-23；闭环 PPL 保持不低于 99.5%。
- 失败：256 探针 KL 在 held-out 上系统性低估；逐 token 上界要求接近 Full 才通过；相邻 decode step 频繁切换；或额外扫描抵消 attention 加速。
- 声明边界：在逐 token 上界尚未实现和验证前，只能称为“信息约束自适应 rate 候选”，不能称为严格证书。任何有限稀疏预算都不可能在完全对抗的注意力分布上保证与 Full 相同；若真实注意力接近均匀，数学规则应诚实地增加 token 数，而不是隐藏失败。
# 概率越界救援：当前候选方法

## 可证伪假设

低比特 QK 代理在 32K--128K 没有发生突变，但固定 `top-1280` 的遗漏质量随长度增加。与此同时，代理边界附近的误排序不是均匀发生的：某个 token 越接近代理 top-k 边界，并且它的 Key 量化残差越大，它越可能从边界外越入真实 top-k。若该假设正确，只需精确读取少量“可能越界”的 Key，就能使低码率索引达到高码率索引的排序质量。

## 数学模型

令真实 logit、代理 logit 和误差分别为 `s_i`、`stilde_i`、`e_i=s_i-stilde_i`，代理 top-k 边界为 `t`。由 Cauchy--Schwarz 不等式，Key 重构残差给出自然尺度：

```text
|e_i| <= scaling * ||qprime||_2 * ||kprime_i-khatprime_i||_2.
```

将右侧记为 `a_i`。用当前请求的 256 个确定性分层 exact-QK 探针拟合仿射校准，并估计标准化误差 `z_i=e_i/a_i` 的单侧尾部。当前实现比较两种估计：

```text
Gaussian:  p_i = 1 - Phi((t-stilde_i)/(sigma * a_i))
Empirical: p_i = [1 + #{z_probe >= (t-stilde_i)/a_i}] / (m+1)
```

`p_i` 是边界外 token 越过边界的估计概率。令 `mu=sum_i p_i`，在独立或弱相关越界事件近似下，失败概率 `delta` 对应的救援数为：

```text
r = ceil(mu + sqrt(2*mu*log(1/delta)) + 2*log(1/delta)/3).
```

选择 `p_i` 最大的 `r` 个边界外 token，读取它们的原始 Key，与原代理 top-k 合并后做一次 exact QK 重排，最终仍只保留原来的 k 个 token。长度没有作为输入；当历史变长、边界变密或误差变大时，`sum_i p_i` 会自然增加。

## 实现契约

- 输入：当前 query、低比特 Key 索引、每 token 的 Key 残差范数、GPU 常驻原始 K/V。
- 固定参数：探针数 256、`delta=0.01`、基础索引 rate-15、最终 token 数由既有 6% 且上限 1280 的规则给出。
- 中间量：仿射校准、标准化误差分布、每 token crossing probability、期望越界数和 Bernstein 救援数。
- 输出：exact 重排后的 k 个 token、救援数、预测越界数、真实诊断召回率和阶段耗时。
- 通过：在 4K/32K/96K 的独立主题上，以低于 rate-23 的总读取量达到不差于 rate-23 的 top-k mass 和局部输出误差；真实模型 PPL 保持率不低于 99.5%。
- 失败：重尾或块相关误差使 256 探针系统性低估；隐藏 needle 的残差范数与普通 token 无法区分；额外 exact-K 读取与重排抵消 attention 收益。
- 声明边界：Bernstein 公式依赖越界事件独立或弱相关，不是任意对抗输入上的确定性证书。完全不可区分的隐藏 needle 不可能由任何次线性扫描保证发现。

## 候选保留：不丢弃已经精确读取的风险 token

### 可证伪假设

当前实现已经为 `r` 个风险候选读取原始 Key 并计算 exact QK，却在重排后再次压回固定的 `k` 个 token。若 4K 的质量缺口和长序列退化主要来自固定预算遗漏的 softmax/Value 质量，那么保留已读取的风险候选应当在几乎不增加检索流量的情况下改善输出；若质量不变，则问题不在最终 token 预算，而在代理分数或 Value 近似。

### 数学规则

令 `B` 是代理 top-k，`R` 是按经验越界概率选出的 Bernstein 救援集合。旧规则为：

```text
S_old = exact_topk(B union R, k).
```

候选保留规则为：

```text
S_keep = unique(B union R).
```

`R` 的大小仍由 `sum_i p_i` 决定，不读取任务名或上下文长度。因为 `B` 和 `R` 的原始 Key 已经为 exact 重排读入 GPU，`S_keep` 相比旧规则只增加最终稀疏 attention 的 K/V 消费，不增加风险扫描和 exact-Key 候选读取。该规则不是额外的经验预算函数，而是停止丢弃已经由越界风险检验判定为有价值的证据。

### 实现契约

- 输入：代理 top-k 索引、经验越界概率、Bernstein 救援数、原始 K/V。
- 参数：与越界救援相同，即 256 个探针、`delta=0.01` 和 rate-15 Key 索引；不增加长度阈值或任务参数。
- 过程：构造去重的 `B union R`，对其中 Key 计算 exact QK，并对整个并集执行一次稀疏 attention；未选 tail 仍使用 rank-16 INT4 Value sketch。
- 输出：实际并集 token 数、attention mass、局部输出相对 L2、真实模型 PPL，以及风险扫描、exact 候选和稀疏 attention 的独立耗时。
- 通过：4K/32K/96K 的局部输出误差均优于“重排回 k”，真实模型质量保持率达到 99.5%，新增稀疏 attention 时间小于 20%。
- 失败：风险候选 exact 分数普遍很低、并集质量无改善、并集快速膨胀，或稀疏 attention 增量抵消系统收益。
- 声明边界：候选保留改善的是固定预算遗漏，不会修复代理空间中完全不可辨识的隐蔽 needle。
