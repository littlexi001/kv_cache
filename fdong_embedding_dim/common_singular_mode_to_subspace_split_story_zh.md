# 从 Common 奇异模态到表征/参数双拆分：机制、伤害与结构性解法

## 0. 一句话核心结论

本文的核心逻辑分四步：

1. **大奇异方向的来源**：最初的大奇异方向来自 Zipf 数据分布中的高频 token / phrase / pattern。这些 common pattern 在 tied embedding 和梯度外积更新下，先形成 common gradient mode，再被写入表征空间和参数空间。

2. **为什么奇异值继续增大**：在 common 方向已经稳定后，cross-entropy loss 仍会继续追求更大的 margin。方向更新有饱和项，但奇异值增长没有相同的方向饱和项，所以会出现“奇异向量先稳定，奇异值继续增大”的两阶段过程。

3. **为什么大奇异值方向伤害长尾学习**：common high-gain direction 会在 long-tail 样本上形成错误 logit 竞争项。最小 margin 模型里，tail margin 是

$$
m_T=b\sigma_t\tau-a\sigma_c\rho
$$

其中 $a\sigma_c\rho$ 就是 common direction 对 tail margin 的伤害项。$\sigma_c$ 越大，tail 样本中 common 投影 $\rho$ 越强，long-tail 学习就越慢。

4. **为什么要做表征/参数双拆分实验**：为了验证上面的机制是否正确，我们构造一个结构性干预：对原本的线性变换 $y=Wh$，先把输入表征 $h$ 分成 common component 和 residual / long-tail component，再让两部分分别经过不同参数矩阵。也就是让 common 信息走 common 参数通道，让去 common 后的 residual 信息走 residual 参数通道。这个实验的目的不是先声称提出最终 solution，而是检验：如果 long-tail 的伤害确实来自 $a\sigma_c\rho$，那么同时降低 tail 表征里的 common 投影 $\rho$，并阻止 tail path 共享 common 参数增益 $\sigma_c$，是否会让 long-tail 学得更快。

更简洁地说：

$$
\text{Zipf common pattern}
\rightarrow
\text{common direction}
\rightarrow
\text{CE-driven singular gain growth}
\rightarrow
\text{tail margin 被 common logit 压制}
$$

这个结构性干预可以写成：

$$
y_k
=
\alpha W_cP_{c,k}h_k
+
W_rP_{\perp,k}h_k
$$

其中 common component 和 residual / long-tail component 在表征和参数上同时分离。

## 1. 最大奇异方向如何产生

### 1.1 最初的大方向不是 nested structure 自己凭空产生的

我们现在的理解是：最初的 common direction 主要来自数据分布中的共享统计源，尤其是高频 token、高频 phrase、功能性短语、短程 common pattern。

在最小 tied embedding 语言模型里，输出 logit 是：

$$
z_j=e_j^\top h
$$

其中 $e_j$ 同时是 token $j$ 的输入 embedding 和输出 classifier weight。

对一个 target token $y$，cross-entropy 对输出 embedding 的梯度会把 $e_y$ 拉向所有预测它的 hidden states 的均值方向。若某个 token $K$ 高频成为 target，则它在早期梯度中占比很大：

$$
\Delta e_K
\propto
\sum_{s:y_s=K} h_s
$$

如果 $K$ 前面的上下文很多，那么这个和式不是某个单一语义方向，而更接近“很多上下文的公共均值方向”。于是，高频 target $K$ 会先形成一个 common direction。

这解释了为什么 uniform-disjoint 数据不应产生强 common direction：如果每组 token 和 pattern 都均匀且互不共享，那么没有一个 target 或 prefix 在梯度中反复出现，初始 common 梯度模式就不会显著出现。

### 1.2 tied embedding 让输出侧 common direction 回流到输入侧

tied embedding 的关键在于同一个 $E$ 同时用于输入和输出：

$$
x=E[\mathrm{token}]
$$

$$
z=hE^\top
$$

所以高频 target $K$ 被输出梯度拉向 common direction 后，$e_K$ 又会作为输入 embedding 出现在大量上下文里。于是：

$$
\text{K as target}
\rightarrow
e_K \text{ 被拉向 common mean}
\rightarrow
\text{K as input}
\rightarrow
\text{更多 hidden states 含有 common component}
$$

这形成了输入侧和输出侧的反馈循环。

### 1.3 参数矩阵为什么也会吸收这个方向

对任何线性层：

$$
y=Wh
$$

单个样本的梯度是：

$$
\frac{\partial L}{\partial W}
=
\delta h^\top
$$

其中：

- $h$ 是输入这个矩阵的 activation；
- $\delta=\frac{\partial L}{\partial y}$ 是输出侧误差信号。

这是一个 rank-1 外积。它的右侧方向由输入 $h$ 决定。因此，如果很多样本的 $h$ 都含有 common direction，那么参数矩阵 $W$ 的输入侧 / 右奇异方向就会被反复拉向这个 common direction。

所以：

$$
\text{activation common direction}
\rightarrow
\text{parameter right singular direction}
$$

这也是我们在 toy 模型和 Qwen 分析里看到的现象：并不是每个矩阵都同等强烈地对齐 common direction，而是那些最能利用该方向降低 loss 的模块更容易吸收它。

### 1.4 nested structure 的角色：放大和传播，而非最初来源

自然语言中的 nested structure、共享前缀、共享短程句法结构，会让 common direction 更容易出现在很多 token 的 residual stream 中。

但我们现在不把 nested structure 当作“第一个 common direction 的唯一来源”。更准确的链条是：

$$
\text{Zipf / high-frequency shared pattern}
\rightarrow
\text{first common direction}
\rightarrow
\text{nested structure reuses and propagates it}
$$

也就是说，nested structure 主要更像 amplifier 和 carrier；在某些不完全对称的链式拓扑中，它也可能独立产生轻度 common direction，但在我们当前实验尺度下，其贡献弱于 Zipf 高频 / shared pattern。

## 2. 为什么奇异值会不停增大

这一部分主要和老板的 two-phase singular-mode theory 对齐。

### 2.1 两阶段学习：方向先稳定，gain 后增长

考虑一个 rank-1 参数模态：

$$
W=\sigma uv^\top
$$

对某个输入方向 $x$，若右奇异方向 $v$ 与 $x$ 的 alignment 是：

$$
c=v^\top x
$$

则这个方向上的 margin 可以写成：

$$
m=\sigma c
$$

cross-entropy loss 对 margin 的残余压力可以写成：

$$
r(m)=-\phi'(m)>0
$$

对于 logistic / softmax CE，只要 margin 不是无穷大，就有：

$$
r(m)>0
$$

方向动力学里有饱和项：

$$
\frac{dc}{dt}
\propto
r(m)\sigma(1-c^2)
$$

当方向已经对齐：

$$
c\approx 1
$$

则：

$$
1-c^2\approx 0
$$

所以方向更新停止或显著变慢。

但奇异值动力学没有这个方向饱和项：

$$
\frac{d\sigma}{dt}
\propto
r(m)c
$$

当：

$$
c\approx 1
$$

仍然有：

$$
\frac{d\sigma}{dt}
\approx
r(m)>0
$$

所以会出现：

> 奇异向量先稳定，奇异值继续增长。

### 2.2 tied embedding 下的补充

在 tied embedding 里，输入方向和输出方向不是完全独立的，因为输出 token embedding 也会作为未来输入 embedding 使用。

一个更贴近 tied setting 的局部模型是：

$$
m=\sigma ab
$$

其中：

- $a=u^\top e_y$ 是输出侧奇异方向与 target embedding 的 alignment；
- $b=v^\top x$ 是输入侧奇异方向与输入 activation 的 alignment。

方向阶段：

$$
a\rightarrow 1,\quad b\rightarrow 1
$$

gain 阶段：

$$
\frac{d\sigma}{dt}
\approx
r(\sigma)>0
$$

因此 tied embedding 不推翻老板的 two-phase theory，而是补充了一个输入/输出共同空间中的反馈路径。

### 2.3 我们的实验支持什么

我们做的 two-phase diagnostic 显示，在 plain CE training 中：

- 方向 drift 后期显著下降；
- 但 top singular energy $\sigma_1^2$ 后期仍然继续增长；
- Wq/Wk/Wv/Wo/Bqk 等 attention 参数在训练足够久后都能看到这个趋势；
- Bqk 在 no-O-projection 对照里尤其明显。

因此实验支持老板理论的定性预测：

$$
\text{direction discovery}
\rightarrow
\text{gain amplification}
$$

但边界是：我们验证的是离散训练轨迹上的定性两阶段现象，不是对连续 ODE 的逐项拟合证明。

## 3. common 奇异值增长如何伤害 long-tail 学习

### 3.1 最小 margin 模型

设 common direction 是 $u$，tail direction 是 $v$，且：

$$
u^\top v=0
$$

common token $K$ 的 embedding 主要沿 $u$：

$$
e_K=au
$$

tail token $T$ 的 embedding 主要沿 $v$：

$$
e_T=bv
$$

参数矩阵有两个方向：

$$
W=\sigma_c uu^\top+\sigma_t vv^\top
$$

其中：

- $\sigma_c$ 是 common direction 的 gain；
- $\sigma_t$ 是 tail direction 的 gain。

tail 样本的输入 hidden state 通常不是纯 tail direction，而含有 common component：

$$
x_T=\rho u+\tau v
$$

其中：

- $\rho$ 是 tail 表征里的 common 投影；
- $\tau$ 是 tail 表征里的 tail 投影。

经过 $W$ 后：

$$
h_T=Wx_T=\sigma_c\rho u+\sigma_t\tau v
$$

tail token 自己的 logit：

$$
z_T=e_T^\top h_T=b\sigma_t\tau
$$

common token 的错误竞争 logit：

$$
z_K=e_K^\top h_T=a\sigma_c\rho
$$

于是 tail 相对 common 的 margin 是：

$$
m_T=z_T-z_K
$$

即：

$$
m_T=b\sigma_t\tau-a\sigma_c\rho
$$

这是整个故事里最关键的公式。

### 3.2 common gain 增大会直接压低 tail margin

对时间求导：

$$
\frac{dm_T}{dt}
=
b\tau\frac{d\sigma_t}{dt}
-
a\rho\frac{d\sigma_c}{dt}
$$

第一项是 tail 学习带来的正贡献。第二项是 common gain 增长带来的负贡献。

如果：

$$
a\rho\frac{d\sigma_c}{dt}
>
b\tau\frac{d\sigma_t}{dt}
$$

则：

$$
\frac{dm_T}{dt}<0
$$

也就是说，tail 自己的方向即使在学，只要 common direction 的 gain 长得更快，tail 的相对 margin 仍然可能变差。

更常见的情况是：

$$
\frac{dm_T}{dt}>0
$$

但被第二项显著拖慢。于是表现为 long-tail 收敛慢。

### 3.3 为什么 Zipf 会放大这个问题

common token / pattern 的有效梯度规模近似是：

$$
p_Kr_K
$$

tail token / pattern 的有效梯度规模近似是：

$$
p_Tr_T
$$

Zipf 分布下：

$$
p_K\gg p_T
$$

即使 common token 已经学得不错，$r_K$ 变小，只要不是 0，乘上高频 $p_K$ 后仍可能推动 $\sigma_c$ 继续增长。

所以 tail margin 的动力学可以粗略写成：

$$
\frac{dm_T}{dt}
\approx
p_Tr_T b^2\tau^2
-
p_Kr_K a^2\rho\rho_K
$$

第二项就是 common pattern 对 tail 学习的持续压制。

### 3.4 reweighting 为什么有用，但不是结构性解法

loss reweighting 做的是：

$$
p_K \rightarrow \tilde{p}_K
$$

其中：

$$
\tilde{p}_K<p_K
$$

于是 common gain 增长变慢：

$$
\frac{d\sigma_c}{dt}\downarrow
$$

tail margin 中的负项也变小：

$$
a\rho\frac{d\sigma_c}{dt}\downarrow
$$

这解释了我们看到的现象：

- common 收敛变慢；
- tail 收敛变快；
- 多个奇异值一起增长；
- $\sigma_1/\sigma_4$ 变小；
- 谱更平。

但 reweighting 仍然没有改变这个结构：

$$
m_T=b\sigma_t\tau-a\sigma_c\rho
$$

它只是让 $\sigma_c$ 长慢一点，没有从结构上消除 $\rho$，也没有让 tail path 避开 common high-gain 参数通道。

所以 reweighting 是 strong baseline，但不是最 fundamental 的结构性解法。

## 4. 表征和参数双拆分为什么从原理上是对的

### 4.1 方法定义

原始线性层是：

$$
y_k=Wh_k
$$

我们把它替换为：

$$
y_k
=
\alpha W_cP_{c,k}h_k
+
W_rP_{\perp,k}h_k
$$

其中：

$$
P_{c,k}=u_{c,k}u_{c,k}^\top
$$

$$
P_{\perp,k}=I-P_{c,k}
$$

$$
h_{c,k}=P_{c,k}h_k
$$

$$
h_{r,k}=P_{\perp,k}h_k
$$

所以：

$$
y_k=\alpha W_ch_{c,k}+W_rh_{r,k}
$$

这里：

- $u_{c,k}$ 是当前位置的 common direction 估计；
- $h_{c,k}$ 是 common component；
- $h_{r,k}$ 是 residual / long-tail component；
- $W_c$ 是 common branch 参数；
- $W_r$ 是 residual / long-tail branch 参数；
- $\alpha$ 是 common branch 的 gain control。

在 causal LM 里，$u_{c,k}$ 不能用未来 token。我们的 prefix-mean CRS 用：

$$
u_{c,k}
=
\mathrm{stopgrad}
\left(
\frac{\sum_{i<k}h_i}
{\left\|\sum_{i<k}h_i\right\|}
\right)
$$

Workbuddy 的 subspace split 版本用的是全局词表 embedding 均值方向：

$$
v_K=\frac{\bar{E}}{\|\bar{E}\|},\quad
\bar{E}=\frac{1}{V}\sum_{i=1}^{V}E_i
$$

两者都是 common direction estimator，但含义不同：

- prefix mean 是 sequence-position causal estimator；
- vocabulary centroid 是 global embedding-space estimator。

### 4.2 为什么它直接对应伤害项

我们前面得到的 long-tail hurt term 是：

$$
a\sigma_c\rho
$$

其中：

- $\rho$ 是 tail 表征中的 common 投影；
- $\sigma_c$ 是 tail 路径共享到的 common 参数增益。

表征拆分处理 $\rho$：

$$
h_r=P_\perp h
$$

使 residual / tail branch 的 common 投影变成：

$$
\rho'\ll \rho
$$

参数拆分处理 $\sigma_c$：

$$
W_c \neq W_r
$$

使 tail path 不再经过 common branch 的 high-gain 参数：

$$
\sigma_c^{(r)}\ll \sigma_c
$$

因此 tail margin 从：

$$
m_T=b\sigma_t\tau-a\sigma_c\rho
$$

变成：

$$
m_T^{\mathrm{split}}
=
b\sigma_t^{(r)}\tau
-
a\sigma_c^{(r)}\rho'
$$

如果：

$$
\rho'\approx 0
$$

且：

$$
\sigma_c^{(r)}\approx 0
$$

那么 common 对 tail 的结构性伤害项就被直接压掉。

这就是为什么这个方法比简单调 loss 更 fundamental：

> reweighting 改变哪些样本更强地要求 margin；subspace split 改变哪些参数子空间被允许回应这些 margin request。

### 4.3 梯度外积视角

原始线性层：

$$
\frac{\partial L}{\partial W}
=
\delta h^\top
$$

如果 $h$ 含有 common direction，梯度就会把 common direction 写进 $W$ 的输入侧 / 右奇异方向。

拆分后 residual branch：

$$
\frac{\partial L}{\partial W_r}
=
\delta_r h_r^\top
$$

其中：

$$
h_r=P_\perp h
$$

并且：

$$
h_r\perp u_c
$$

所以每个 rank-1 外积的输入侧方向都不含 common component。更精确地说，$W_r$ 的右奇异方向不会被输入侧梯度直接拉向 common direction。

注意这里的严谨边界是：

> residual branch 的 input-side / right-singular gradient 被去 common；但输出侧误差信号 $\delta_r$ 仍然可能含有 common direction。

因此不能泛泛说“整个梯度没有 common component”，只能说“输入侧外积方向没有 common component”。

## 5. 实验结果如何支持这个故事

### 5.1 最大方向来源实验

已有 shared-K / Zipf / uniform-disjoint 实验支持：

- uniform-disjoint 数据不产生强 common gradient mode；
- shared high-frequency target 会先在梯度里产生 common mode；
- 参数/表征谱集中出现在梯度集中之后。

这支持：

$$
\text{frequency/shared target}
\rightarrow
\text{common direction nucleation}
$$

### 5.2 参数吸收 activation common direction

toy attention 和 Qwen 分析支持：

- 参数矩阵的右奇异方向常与输入 activation PC1 / common direction 对齐；
- 对齐强度在不同模块之间有选择性；
- Q/K/V/O/MLP 并不是每个矩阵都同样吸收 common direction，而是取决于该模块是否能利用这个方向降低 loss。

这支持：

$$
\text{input activation common direction}
\rightarrow
\text{parameter input-side singular direction}
$$

### 5.3 two-phase 实验

two-phase singular dynamics 实验支持：

- 训练后期方向 drift 下降；
- $\sigma_1^2$ 仍有正增长；
- Bqk、Wq、Wk、Wv、Wo 在足够训练后都能观察到两阶段签名。

这支持老板理论：

$$
\text{direction stabilizes}
\rightarrow
\text{CE keeps amplifying singular gain}
$$

### 5.4 CRS conflict-free causal 实验

在 conflict-free synthetic causal task 上，单层 residual attention：

| 方法 | tail 稳定到 100% accuracy 的平均 step | final tail loss | final Bqk $\sigma_1/\sigma_4$ |
|---|---:|---:|---:|
| dense | 1256 | 0.00897 | 492 |
| dense + reweight | 1100，且 4/5 seeds 达到 | 0.0442 | 1059 |
| CRS, $\alpha=1$ | 152 | 0.0000156 | 359 |
| CRS, $\alpha=0.5$ | 168 | 0.0000082 | 241 |

结论：

> CRS 显著加速 tail 收敛；hard split 是主要收益来源，$\alpha$ 压制更多体现为谱更平。

### 5.5 有 label conflict 的实验

真实语言数据不可能 100% deterministic。因此我们也跑了有 token conflict 的版本。这里不能看 100% accuracy，而要看是否更快接近理论可达上限和 Bayes-optimal distribution。

对于 conflict prefix：

$$
K\rightarrow A/B/C/D
$$

若四类等概率，则 top-1 ceiling 是：

$$
0.25
$$

Bayes CE 是：

$$
\log 4=1.386
$$

实验结果：

| 方法 | 达到整体 tail accuracy ceiling 的 step | final conflict CE | final conflict KL to Bayes |
|---|---:|---:|---:|
| dense | 520 | 2.783 | 1.397 |
| dense + reweight | 未达到 | 2.605 | 1.219 |
| CRS, $\alpha=1$ | 200 | 1.840 | 0.453 |
| CRS, $\alpha=0.5$ | 300 | 1.818 | 0.432 |

按 conflict KL 阈值看：

| KL threshold | dense | dense + reweight | CRS $\alpha=1$ | CRS $\alpha=0.5$ |
|---|---:|---:|---:|---:|
| KL ≤ 1.0 | 未达到 | 未达到 | 140 | 160 |
| KL ≤ 0.75 | 未达到 | 未达到 | 160 | 200 |
| KL ≤ 0.5 | 未达到 | 未达到 | 180 | 300 |
| KL ≤ 0.45 | 未达到 | 未达到 | 320 | 340 |

结论：

> 即使在有 label conflict、更接近真实语言的 setting 中，CRS 也更快接近理论可达分布；它不是靠突破不可能的 100% accuracy，而是更快降低 conflict CE/KL。

### 5.6 Workbuddy subspace split 交叉验证

Workbuddy 的独立 trigram 脚本也支持同一方向：

| 方法 | accuracy | 参数谱 |
|---|---:|---|
| no reweighting | 37.5% | $W_q \sigma_1/\sigma_4=115$ |
| hard reweighting | 100% | $W_q \sigma_1/\sigma_4=14104$ |
| Subspace Split $\alpha=0.3$ | 100% | $W_q^{(c)} \sigma_1/\sigma_4=1.1,\ W_q^{(r)} \sigma_1/\sigma_4=5.3$ |

这说明即使不用 loss reweighting，只靠结构拆分，也可以让模型学会 tail pattern，并让 residual 参数谱显著更平。

需要注意的边界：

- Workbuddy 用的是 vocabulary centroid $v_K$，不是 prefix mean $u_{c,k}$；
- 它是单 seed trigram toy；
- 文档中“RMSNorm implicitly subtracts mean”的说法应修改，因为 RMSNorm 不减均值；
- 梯度解耦应表述为 input-side / right-singular direction 解耦，而不是整个梯度完全无 common component。

## 6. 这个方法为什么是 fundamental solution

老板关注 fundamental understanding，所以这部分必须强调：我们的方法不是经验 trick，而是由 margin hurt term 推出来的结构性干预。

理论伤害项是：

$$
a\sigma_c\rho
$$

它由两个结构因素组成：

| 符号 | 含义 | 为什么伤害 long-tail | 我们如何处理 |
|---|---|---|---|
| $\rho$ | tail 表征里的 common 投影 | tail 样本会产生错误 common logit | 表征拆分：$h_r=P_\perp h$ |
| $\sigma_c$ | common high-gain 参数通道 | common gain 在 tail path 上继续放大 | 参数拆分：$W_c$ 与 $W_r$ 分离 |
| CE margin pressure | 方向稳定后仍推高 gain | common direction 不停追 margin | $\alpha W_c$ 控制 common branch |

因此 CRS / Subspace Split 不是简单让模型容量变大，而是改变学习几何：

$$
\text{dense model: one shared space answers all margin requests}
$$

变成：

$$
\text{split model: common margin and residual margin use different subspaces}
$$

这就是它 fundamental 的地方：

> 它不是只改变样本权重，而是改变参数更新允许写入哪些方向。

## 7. 当前 claim boundary

目前可以说：

1. 高频 shared pattern 是 common direction 的主要来源之一；
2. 参数矩阵会通过梯度外积吸收输入 activation common direction；
3. CE loss 会在方向稳定后继续放大对应奇异值；
4. common high-gain direction 会通过 $a\sigma_c\rho$ 项伤害 long-tail margin；
5. 表征/参数双拆分能在 toy causal attention 中显著改善 long-tail 学习；
6. 有 label conflict 时，CRS 仍能更快接近 Bayes-optimal conflict distribution。

还不能说：

1. CRS 已经在真实 LLM 上验证；
2. rank-1 common direction 足够描述真实模型；
3. 每个矩阵都必须 split；
4. prefix mean、EMA mean、vocabulary centroid 哪个 estimator 最好；
5. 多层 Transformer 中 residual bypass 是否会削弱拆分效果。

## 8. 下一步实验

建议下一步按以下顺序做：

1. projection ablation：只 split Q、只 split K、只 split V、只 split O、split QK、split QKV、split QKVO；
2. common direction estimator ablation：prefix mean、EMA prefix mean、vocabulary centroid、activation PC1；
3. rank-$k$ common subspace：从 rank-1 扩展到多个 common directions；
4. multi-layer residual bypass：检查 common direction 是否绕过 split 在更深层重新出现；
5. small real LM validation：在小真实文本模型上检查 loss、tail token CE、参数谱和 common alignment。

## 9. 给老板汇报时的短版

我们现在的 story 是：

> Zipf 高频 pattern 先产生 common singular direction；老板的 two-phase theory 解释了方向稳定后 cross-entropy 为什么还会继续放大它的 singular value；这个 common gain 在 tied embedding 和 dense shared parameter path 中会作为错误 logit 竞争项进入 tail margin，形式上是 $a\sigma_c\rho$。因此，真正 fundamental 的解法不是只调 loss 权重，而是把 common 与 residual 的表征和参数通道拆开，使 tail branch 既不读取 common component，也不共享 common high-gain parameter channel。实验上，CRS 在 conflict-free 和 conflict 数据中都显著加速 tail 学习，并让 residual 参数谱更平。
