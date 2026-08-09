# QKSieve

## 1. 要解决的问题

对一个 query head，标准自回归 attention 在长度为 $N$ 的历史 KV 上计算

$$
\ell_i=\frac{q^\top k_i}{\sqrt d},\qquad
o=\frac{\sum_{i=1}^{N}e^{\ell_i}v_i}
        {\sum_{i=1}^{N}e^{\ell_i}}.
$$

每生成一个 token，都要读取全部 K/V，计算量和显存带宽均随 $N$ 线性增长。

> 不扫描完整 FP16 K/V，如何为每个 query head 找到少量高权重 token，近似 Full Attention计算？

## 2. 先验经验

1、对于模型的每层每个head只需要retrival attention最高的top2%的token就能够保持推理质量。

## 3. 当前效果

模型为 Llama-3.1-8B-Instruct，质量在 held-out `mixed_b` 的 32 个teacher-forced token 上测量；质量保持率定义为$\exp(\mathrm{NLL}_{\rm full}-\mathrm{NLL}_{\rm sparse})$。超过 100% 只代表该小 probe 的 NLL 略低，不能解释为普遍超过 Full。

| 历史长度 | 实际 token/head | Active 比例 | 质量保持 | Top-1 | Full decode | QKSieve decode |       加速 |
| -------: | --------------: | ----------: | -------: | ----: | ----------: | -------------: | ---------: |
|      32K |        1,276.17 |      3.895% | 100.812% |  100% |    87.46 ms |       45.52 ms | **1.921x** |
|      64K |        1,276.58 |      1.948% | 101.358% |  100% |   151.22 ms |       62.77 ms | **2.409x** |
|     128K |        1,278.86 |      0.976% | 100.526% |  100% |   277.64 ms |       89.89 ms | **3.089x** |

补充资源结果：

- 低比特 Key 索引约占完整 FP16 K+V 的 **5.8%**；ValueSketch 约占**1.61%**；总辅助索引约为 **7.4%**。
- 一次性索引成本在 32K/64K/128K 分别约为 `3.17/3.35/3.58 s`；按实测
  decode 斜率，分别生成约 `76/38/19` 个 token 后回本。

完整 LongBench 的已有 QKSieve full-proxy 参考路径在 16 个英文任务、3,750样本上取得 `0.458852`，Full 为 `0.459398`，即 **99.881%**。

## 4. 方法

当前冻结原型为
`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`，流程如下。

1. **请求内 QK-balanced 坐标。** Prefill 后，按层和 KV head 估计 Query/Key
   二阶矩，构造保持完整 dot product 的双正交坐标。
2. **混合位宽 band 索引。** 将 128 维分成八个 16-D band；每个 band 从
   `0/1/2/4/8 bit` 中选择，使用 qMSE 和 OAS 收缩，在每 token/head 240-bit
   总预算下自动分配位宽。
3. **确定性 sampled-quantile 检索。** 目标候选数为
   `min(max(6%N,256),1280)`；采样数随目标尾部概率缩放，使阈值样本中预计保留
   64 个上尾锚点。随后只做一次低比特全历史扫描，以阈值直接产生逐 query-head
   候选，不执行通用 top-k 或 exact rerank。
4. **候选精确计算。** 对候选读取 GPU 常驻的原始 FP16 K/V，计算精确 QK、
   softmax 分母和 Value 分子。
5. **ValueSketch 尾部补偿。** 对未选 token 使用 $W_O$-aware rank-16、block-256、INT4 ValueSketch，累计尾部 softmax 分母和 16 维低秩分子，并与候选的精确结果在同一指数标尺下合并。
6. **融合与确定性执行。** 代理扫描、候选 bitmask、尾部统计一次完成；候选使用固定前缀扫描压缩，尾部使用固定树归约。新 token 只增量追加 Key 编码和
   ValueSketch 编码。

## 5. 核心数学推导

### 5.1 QK-balanced 坐标

对一个 KV head，分别用当前请求的 Query 与 Key 样本估计非中心二阶矩

$$
\widehat C_q=\frac1{m_q}\sum_j q_jq_j^\top,
\qquad
\widehat C_k=\frac1{m_k}\sum_i k_ik_i^\top.
$$

对 Query 矩做各向同性收缩，对 Query/Key 矩做特征值 floor，得到正定矩阵$\widetilde C_q,C_k$。当前实现中，坐标构造使用 `0.5` Query 收缩；后面的
band qMSE 使用数据自适应 OAS 收缩。

令

$$
\widetilde C_q^{1/2}C_k^{1/2}=U\Sigma V^\top,
$$

并定义

$$
A=\widetilde C_q^{-1/2}U\Sigma^{1/2},
\qquad
D=C_k^{-1/2}V\Sigma^{1/2}.
$$

因为

$$
AD^\top=I,
$$

所以对任意 $q,k$，

$$
q'=A^\top q,\qquad k'=D^\top k,
\qquad {q'}^\top k'=q^\top k.
$$

完整 128 维变换本身不产生 score 误差。同时，

$$
A^\top\widetilde C_qA=D^\top C_kD=\Sigma,
$$

在构造时采用的二阶模型
$\mathbb E[qq^\top]=\widetilde C_q$、$\mathbb E[kk^\top]=C_k$ 下，其逐步
推导为

$$
\begin{aligned}
\mathbb E[q'q'^\top]
&=A^\top\mathbb E[qq^\top]A\\
&=A^\top\widetilde C_qA\\
&=\Sigma^{1/2}U^\top
\widetilde C_q^{-1/2}\widetilde C_q
\widetilde C_q^{-1/2}U\Sigma^{1/2}\\
&=\Sigma,
\end{aligned}
$$

以及

$$
\mathbb E[k'k'^\top]=D^\top C_kD=\Sigma.
$$

由于 $\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_d)$，对任意坐标
$r,s$，

$$
\mathbb E[q'_rq'_s]=\sigma_r\mathbf{1}[r=s],
\qquad
\mathbb E[k'_rk'_s]=\sigma_r\mathbf{1}[r=s].
$$

特别地，

$$
\boxed{
\mathbb E[(q'_r)^2]
=\mathbb E[(k'_r)^2]
=\sigma_r
}.
$$

这里是**非中心二阶矩**。只有当坐标均值为零时，它才等于通常意义下的方差。
若使用原始经验 Query 矩而不是构造变换时的收缩矩，则

$$
\frac1{m_q}\sum_jq'_j{q'_j}^\top
=\Sigma+\Delta_q,
\qquad
\Delta_q=A^\top(\widehat C_q-\widetilde C_q)A.
$$

因此原始样本上准确的坐标结论是

$$
\frac1{m_q}\sum_j(q'_{j,r})^2
=\sigma_r+(\Delta_q)_{rr},
$$

而不是无条件严格等于 $\sigma_r$；Key 侧同理。$\Delta_q,\Delta_k$ 应在实验中
直接测量。

进一步，在 Query 与 Key 按各自二阶分布独立抽样的模型下，

$$
q^\top k={q'}^\top k'=\sum_{r=1}^{d}q'_rk'_r.
$$

第 $r$ 个坐标的 score 二阶能量为

$$
\mathbb E[(q'_rk'_r)^2]
=\mathbb E[(q'_r)^2]\mathbb E[(k'_r)^2]
=\sigma_r^2.
$$

不同坐标的交叉项为零，因此

$$
\mathbb E[({q'}^\top k')^2]
=\sum_{r=1}^{d}\sigma_r^2.
$$

所以第 $g$ 个 16-D band 的理论 score 能量为

$$
E_g=\sum_{r=16(g-1)+1}^{16g}\sigma_r^2.
$$

前部 band 因奇异值更大而承载更多联合 QK score 能量。这说明坐标按 Query
与 Key 的联合 score 能量排序，而不是只按 Key 重构能量排序。真实 attention
中的 Query 与 Key 可能相关，此时 $\sigma_r^2$ 分解是二阶独立模型下的结论，
需要用真实轨迹的逐 band score 能量验证；当前 `91.22%/96.57%` 正是这一验证。

在 Query/Key 独立二阶模型下，最优 rank-$r$ 双线性近似的误差为

$$
\min_{\operatorname{rank}(B)\le r}
\mathbb E\left[(q^\top k-q^\top Bk)^2\right]
=\sum_{j>r}\sigma_j^2.
$$

这给出了按联合奇异值顺序切分 band 的依据。

### 5.2 Band 量化与 qMSE 位宽分配

将 $q',k'$ 划分为八个 16 维 band：

$$
q'=(q'_1,\ldots,q'_8),\qquad
k'=(k'_1,\ldots,k'_8).
$$

第 $g$ 个 band 使用 $b_g\in\{0,1,2,4,8\}$ bit。$b_g=0$ 时该 band 置零；
$b_g\ge2$ 时使用逐 token 对称整数码：

$$
a_{i,g}=\frac{\|k'_{i,g}\|_\infty}{2^{b_g-1}-1},
\qquad
\widehat k'_{i,g}=a_{i,g}\,
\operatorname{clip}\!\left(
\operatorname{round}\frac{k'_{i,g}}{a_{i,g}}
\right).
$$

1-bit band 使用 sign code 和平均绝对值 scale。每个 active band 还保存一个
FP16 scale，相当于每坐标多 1 bit，因此物理成本为

$$
R(b)=16\sum_{g=1}^{8}\left(b_g+\mathbf{1}[b_g>0]\right)\le240\ \text{bit}.
$$

令 $e_{i,g}(b)=k'_{i,g}-\widehat k'_{i,g}(b)$。第 $g$ 个 band 的分数失真为

$$
D_g(b)
=\frac1{m_qm_k}\sum_{j,i}
\left({q'_{j,g}}^\top e_{i,g}(b)\right)^2
=\operatorname{tr}\!\left(C_{q,g}C_{e,g}(b)\right).
$$

由于每个 head 只有少量 prefill Query，使用 OAS 将经验 Query 二阶矩向各向
同性矩阵收缩：

$$
\overline C_q=(1-\alpha)\widehat C_q
+\alpha\frac{\operatorname{tr}(\widehat C_q)}dI,
$$

$$
\alpha=\operatorname{clip}_{[0,1]}
\frac{(1-2/d)\operatorname{tr}(\widehat C_q^2)
+\operatorname{tr}(\widehat C_q)^2}
{(m_q+1-2/d)
\left[\operatorname{tr}(\widehat C_q^2)
-\operatorname{tr}(\widehat C_q)^2/d\right]}.
$$

最终 allocation 为

$$
b^*=\arg\min_b\sum_{g=1}^{8}D_g(b_g)
\quad\text{s.t.}\quad R(b)\le240.
$$

八个 band、五种位宽和 15 个归一化成本单位可用动态规划精确求解：

$$
F(g,r)=\min_{b:r_g(b)\le r}
\left[F(g-1,r-r_g(b))+D_g(b)\right].
$$

因此，在冻结坐标、量化器和校准 qMSE 下，该分配不差于任何同预算固定
`[4,1]`、`[4,2,2,1]` 等配置。完整 score MSE 还包含跨 band 项

$$
\Gamma=\sum_{g\ne h}\operatorname{tr}(C_{q,gh}C_{e,hg}),
$$

其绝对值满足

$$
|\Gamma|\le
\sum_{g\ne h}\|C_{q,gh}\|_F\|C_{e,hg}\|_F.
$$

所以 band 可分是可测近似，不是无条件假设。在线 Query 还按 band 量化为
INT8；总代理误差可写成

$$
q'^\top k'-\widehat q'^\top\widehat k'
=q'^\top(k'-\widehat k')
+(q'-\widehat q')^\top\widehat k'.
$$

qMSE allocation 直接优化第一项，第二项由 INT8 Query 精度和内核等价性实验
单独控制。

### 5.3 长度连续的 sampled-quantile 候选

候选目标为

$$
k(N)=\min\left(
N,\max\left(256,\min(\lceil0.06N\rceil,1280)\right)
\right),
\qquad p=\frac{k(N)}N.
$$

固定采样数会使长序列中的目标上尾样本越来越少。令期望上尾锚点数
$c=64$，采用

$$
m(N)=\operatorname{clip}_{[256,8192]}
\left(256\left\lceil\frac{c/p}{256}\right\rceil\right).
$$

每个 query head 使用分层等距、确定性相位偏移的位置采样代理 logit
$\widehat\ell_i$，取样本的目标上尾 order statistic 为阈值 $\tau$，然后

$$
S=\{i:\widehat\ell_i\ge\tau\},\qquad T=\{1,\ldots,N\}\setminus S.
$$

该规则使阈值估计在不同长度下都约有 64 个上尾锚点。它不保证每个 head
恰好选中 $k$ 个 token，但避免了完整分数物化和通用 top-k 排序。

### 5.4 $W_O$-aware rank-16 ValueSketch

候选使用原始 Value；ValueSketch 只近似 $T$。对一个 KV head 关联的 GQA
query heads，令对应输出投影块为 $W_{O,a}$，定义输出敏感度矩阵

$$
G=\sum_{a\in\mathcal G(h)}W_{O,a}^\top W_{O,a}.
$$

取 $G+\varepsilon I=LL^\top$。对 Value 样本均值 $\mu$，在变换空间

$$
y_i=L^\top(v_i-\mu)
$$

上做 PCA，取前 $r=16$ 个特征向量 $U_r$。定义

$$
z_i=U_r^\top L^\top(v_i-\mu),
\qquad
B=L^{-\top}U_r,
$$

则

$$
v_i\approx\widehat v_i=\mu+Bz_i.
$$

这等价于优先最小化经过 $W_O$ 后的 Value 重构误差，而不是原始 Value 的
普通欧氏误差。

将每 256 个 token 作为一块，对每个低秩维度独立做 INT4 仿射量化：

$$
s_{b,r}=\frac{z^{\max}_{b,r}-z^{\min}_{b,r}}{15},
\quad
c_{i,r}=\operatorname{clip}_{[0,15]}
\operatorname{round}\frac{z_{i,r}-z^{\min}_{b,r}}{s_{b,r}},
$$

$$
\widehat z_{i,r}=z^{\min}_{b,r}+s_{b,r}c_{i,r}.
$$

16 个 INT4 系数占 8 Byte/token/KV-head；block min/scale 均摊约 0.25 Byte，
合计约为完整 FP16 K+V 的 **1.61%**。

### 5.5 尾部 softmax 补偿

代理扫描已经遍历所有 token，因此在判断 $i\in S$ 的同时，对 $i\in T$
累计阈值中心化权重

$$
\widetilde w_i=e^{\widehat\ell_i-\tau}.
$$

只需保存 17 个充分统计量：

$$
\widetilde D_T=\sum_{i\in T}\widetilde w_i,
\qquad
\widetilde c_T=\sum_{i\in T}\widetilde w_i\widehat z_i.
$$

尾部 Value 分子为

$$
\widetilde N_T
=\sum_{i\in T}\widetilde w_i\widehat v_i
=\widetilde D_T\mu+B\widetilde c_T.
$$

候选使用原始 FP16 K/V。令

$$
m=\max_{i\in S}\ell_i,
$$

$$
D_S=\sum_{i\in S}e^{\ell_i-m},
\qquad
N_S=\sum_{i\in S}e^{\ell_i-m}v_i.
$$

将尾部从阈值标尺转换到候选最大值标尺：

$$
\rho=e^{\tau-m}.
$$

当前版本不使用学习增益，补偿系数固定为 1，最终输出为

$$
\widehat o=
\frac{N_S+\rho\widetilde N_T}
     {D_S+\rho\widetilde D_T}.
$$

若代理 logit 精确，即 $\widehat\ell_i=\ell_i$，且 ValueSketch 精确，即
$\widehat v_i=v_i$，则

$$
\rho\widetilde D_T=\sum_{i\in T}e^{\ell_i-m},
\qquad
\rho\widetilde N_T=\sum_{i\in T}e^{\ell_i-m}v_i,
$$

从而 $\widehat o=o$。因此补偿机制在理想条件下严格恢复 Full Attention；实际
误差只来自低比特 score 权重与 rank-16 INT4 Value 近似，而不是来自稀疏集合
内重新归一化。

进一步令同一标尺下的真实尾部统计量为

$$
D_T=\sum_{i\in T}e^{\ell_i-m},
\qquad
N_T=\sum_{i\in T}e^{\ell_i-m}v_i,
$$

并定义近似误差

$$
\Delta_D=\rho\widetilde D_T-D_T,
\qquad
\Delta_N=\rho\widetilde N_T-N_T.
$$

则存在精确误差恒等式

$$
\widehat o-o
=\frac{\Delta_N-o\Delta_D}
       {D_S+D_T+\Delta_D},
$$

并有

$$
\|\widehat o-o\|
\le
\frac{\|\Delta_N\|+\|o\|\,|\Delta_D|}
     {D_S+\rho\widetilde D_T}.
$$

该式把剩余风险拆成两个可独立测量的量：尾部 partition 误差
$|\Delta_D|$ 和尾部 Value 分子误差 $\|\Delta_N\|$。

## 6. 复现入口

- 方法配置与 ValueSketch：`src/run_head_top2_targeted_ppl_20260714.py`
- 低比特扫描与确定性压缩：`src/mixedblock_spectral_cuda_20260729.py`
- 候选/尾部 softmax 合并：`src/qksieve_valuesketch_cuda_20260801.py`
- 当前结果：`docs/deterministic_sampled_valuesketch_20260804/visualization_results.md`
- 当前 variant：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`
