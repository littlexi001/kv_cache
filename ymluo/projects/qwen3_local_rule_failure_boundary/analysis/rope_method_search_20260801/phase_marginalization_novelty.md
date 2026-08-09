# RoPE 多锚点相位边缘化：近期文献与新颖性审计

**检索截止：2026-08-01**  
**审计对象：** 对同一个远程语义 Key/Value，在多个候选相对位置或 RoPE 相位下计算 QK；随后用 log-mean-exp、ensemble、group averaging 等方式合并，并只让该语义 token 的同一个 Value 写入一次。  
**说明：** 这是论文新颖性风险审计，不是法律意义上的专利查新。结论只覆盖下列原论文、官方实现和本轮关键词检索。

## 结论先行

### 最终判断：作为独立方法贡献，当前版本 **NO-GO**

本轮没有找到与下面“裸 log-mean-exp 单分母”公式逐字完全相同的 RoPE 长上下文方法：

$$
s_{j,m}
=
\frac{q_t^\top R(-\delta_m)k_j}{\sqrt d}
+b_m,
$$

$$
\widetilde{s}_j
=
\log\sum_{m=1}^{M}\pi_m\exp(s_{j,m}),
\qquad
\alpha_j
=
\frac{\exp(\widetilde{s}_j)}
{\sum_{\ell}\exp(\widetilde{s}_{\ell})},
$$

$$
o_t=\sum_j\alpha_jv_j.
$$

但这个算子的**核心结构已经被高度覆盖**。最直接的先例是 NeurIPS 2024 的 **MoICE**：它在同一个 head 内，对同一组语义 Q/K/V 使用多个 RoPE angle，聚合各 angle 的 attention distribution，再用同一个 V 计算输出。差别主要只剩：

- MoICE 是“每个相位先 softmax，再按 router 权重混合”；
- 当前公式是“先对同一 token 的多相位 logits 做 LME，再做一个 token softmax”；
- MoICE 改变 RoPE base；当前设想更偏向给远程 QK 指定多个相对距离 anchor。

因此，以下宽泛表述都不安全：

- “首个 multi-phase RoPE attention”；
- “首个让同一 KV 使用多个 RoPE 相位的方法”；
- “首个共享 V 的多相位 attention ensemble”；
- “首个 token/head-wise 多 RoPE 聚合”；
- “首个 phase marginalization / group-averaged positional attention”。

如果最终论文只把聚合顺序从 mixture-of-softmax 改成 softmax-of-LME，审稿人很可能将其视为 **MoICE 的归一化变体**，而不是足以支撑 ICLR 主方法的新架构。

### 可继续研究的窄版本：**CONDITIONAL GO**

可以把 LME 当作组件而不是核心 novelty，并把方法收缩为：

> 在冻结模型中，仅对被反事实诊断为“内容匹配仍在、但被原生 RoPE 相位压低”的远程 query–key pair，构造少量、pair/head/frequency-selective 的相对距离假设；保持局部顺序和未触发交互完全不变，在候选集内做可校准的单次稀疏消费，并用因果实验验证其恢复的是被相位抑制的真实证据而非扩大 attention budget。

这个组合仍需与 MoICE、Attention Buckets、RoPE++、SelfExtend、InfLLM、FASA、SALS 做公式和计算量对齐后，才能称为可守的论文贡献。

---

## 1. 目标算子的精确定义

设远程 query 为 $q_t$，第 $j$ 个语义 token 的 pre-RoPE key/value 为 $(k_j,v_j)$，其候选相对位置集合为

$$
\mathcal A_j=\{\delta_{j,1},\ldots,\delta_{j,M_j}\}.
$$

第 $m$ 个相位假设的 score 为

$$
s_{j,m}
=
\frac{q_t^\top R(-\delta_{j,m})k_j}{\sqrt d}+b_{j,m}.
$$

若 $\pi_{j,m}\ge 0$ 且 $\sum_m\pi_{j,m}=1$，则 token-level log-mean-exp score 为

$$
\widetilde{s}_j
=
\log\sum_m\pi_{j,m}\exp(s_{j,m}).
$$

再在语义 token 之间做 softmax：

$$
P(j\mid q_t)
=
\frac{\exp(\widetilde{s}_j)}
{\sum_{\ell}\exp(\widetilde{s}_{\ell})}.
$$

### 1.1 与“虚拟副本 + 共享 V”完全等价

把每个 $(j,m)$ 看成一个虚拟 key，但让同一 $j$ 的所有副本共享 $v_j$：

$$
P(j,m\mid q_t)
=
\frac{\pi_{j,m}\exp(s_{j,m})}
{\sum_{\ell,r}\pi_{\ell,r}\exp(s_{\ell,r})}.
$$

对相位变量求和：

$$
P(j\mid q_t)=\sum_mP(j,m\mid q_t).
$$

最终输出为

$$
o_t
=
\sum_{j,m}P(j,m\mid q_t)v_j
=
\sum_jP(j\mid q_t)v_j.
$$

所以不需要复制 Value，也不会让同一语义 token 写入多次；多相位只改变它获得的总概率。

### 1.2 不能混淆的四种聚合

1. **score mean**

$$
\bar s_j=\frac1M\sum_ms_{j,m}
$$

它会把正负相位直接抵消，不等于 LME。

2. **hard max**

$$
\widetilde s_j=\max_ms_{j,m}
$$

这是零温极限，只保留最佳相位；它不是对相位不确定性的平滑积分。

3. **LME 后统一 softmax**

$$
P_{\mathrm{LME}}(j)
=
\frac{\sum_m\pi_m e^{s_{j,m}}}
{\sum_{\ell,m}\pi_m e^{s_{\ell,m}}}.
$$

4. **每个相位先 softmax，再混合**

$$
P_{\mathrm{mix}}(j)
=
\sum_m\pi_m
\frac{e^{s_{j,m}}}{Z_m},
\qquad
Z_m=\sum_\ell e^{s_{\ell,m}}.
$$

第四种正是 MoICE 的核心聚合形式。两者一般不相等。

### 1.3 一个必须正视的理论问题：裸 LME 不一定是真正的“相位边缘化”

若 $m$ 是外生 nuisance variable，先验为 $\pi_m$，而每个相位给出条件分布

$$
P(j\mid m,q)=\operatorname{softmax}_j(s_{j,m}),
$$

那么概率论意义上的边缘化应为

$$
P(j\mid q)
=
\sum_m\pi_mP(j\mid m,q)
=
\sum_m\pi_m\frac{e^{s_{j,m}}}{Z_m}.
$$

它也可写成

$$
P(j\mid q)
=
\operatorname{softmax}_j
\left(
\log\sum_m
\pi_m e^{s_{j,m}-\log Z_m}
\right).
$$

因此，若声称“对未知相位做严格 marginalization”，LME 中应包含 $-\log Z_m$。裸 LME

$$
\operatorname{softmax}_j
\left(\log\sum_m\pi_me^{s_{j,m}}\right)
$$

等价于对各相位 softmax 使用一个被分区函数重加权的权重

$$
\widehat\pi_m
=
\frac{\pi_mZ_m}{\sum_r\pi_rZ_r}.
$$

这意味着 score 整体偏大的相位会自动获得更高权重。它还不满足每个相位独立的加法平移不变性：把某个分支全部 logits 加上常数 $c_m$，该分支的普通 softmax 不变，但裸 LME 的结果会改变。

这不是说裸 LME 一定无效，而是论文必须明确二选一：

- 若目标是规范的 nuisance marginalization，应使用或近似 $-\log Z_m$，此时与 MoICE 的公式距离非常近；
- 若目标是 joint energy pooling，应解释为什么让 $Z_m$ 参与 phase gating 是有益归纳偏置，并加入校准与消融。

---

## 2. 公式级重合总表

| 工作 | 是否同时使用多个位置/相位 | 是否复用同一语义 K | 是否复用同一 V | 聚合位置 | 是否为同一 token 的单分母 LME | 重合判断 |
|---|---:|---:|---:|---|---:|---|
| **目标算子** | 是 | 是 | 是 | 多相位 logits 先 LME，再做一次 token softmax | 是 | — |
| **MoICE** | 是；每 head/token 多个 RoPE base | **是** | **是** | 每个 angle 独立 softmax，再按 router 权重求和 | 否，但可写成带 $-\log Z_m$ 的 LME | **最高风险；核心结构已覆盖** |
| Attention Buckets | 是；多个 RoPE base | 否；是多次完整前向 | 否；各分支 hidden/V 都可能不同 | 最终词表概率加权 | 否 | 高层 ensemble 先例 |
| RoPE++ | 是；real/imag 两个互补相位分量 | 部分；共享 K 位置编码/缓存 | 分支输出后融合 | 独立 attention head 输出拼接/投影 | 否 | 双相位、共享缓存近邻 |
| SelfExtend | 同一 pair 可计算 neighbor/group 两种 score | 是 | 是 | `where` 硬选一个 score 后单次 softmax | 否 | 同 pair 多候选但不聚合 |
| DCA | intra/successive/inter 多种相位 | 否；key 分区互斥 | 否；每个 key 只出现一次 | 用各组 softmax-LSE 精确重构全局 softmax | 否 | LSE merge 先例，但对 disjoint keys |
| Ms-PoE | 是；不同 head 不同 position scale | 否；各 head 投影不同 | 否 | 标准 multi-head concat/projection | 否 | 多尺度 RoPE 先例 |
| InfLLM | 远程 token 使用统一固定距离 | 是 | 是 | 每个远程 token 只有一个 clipped phase | 否 | 单远程 anchor 先例 |
| CALIOPE | 是一套 chunk-aware remap | 是 | 是 | 每个 token 只分配一个新位置 | 否 | 推理时位置修复先例 |
| FoPE | 每个 coordinate 混合多个频率 | 不是离散位置副本 | 是 | 坐标内频率叠加后普通 attention | 否 | 频谱混合，不是相位假设边缘化 |
| FASA | 只用主导频带近似选候选 | 否 | 候选消费时原 V | proposal 后做原始全维 RoPE attention | 否 | frequency proposal–exact consumption 先例 |
| SALS | pre-RoPE 低秩 selector | 否 | 候选消费时原 V | RoPE-free Top-K 后恢复标准 RoPE | 否 | pre-RoPE proposal–RoPE consumption 先例 |
| Selective RoPE | 输入依赖单条累积相位轨迹 | 是 | 是 | 每个 pair 只有一个学习相位 | 否 | 内容条件 phase 先例 |
| TAPA | 内容决定单个 pair phase | 是 | 是 | 一个 token-aware phase score | 否 | pair-dependent phase 先例 |
| IHA | 每 token 有多个 virtual positions/phases | 不是同一 K；为跨 head 学习混合 | 不是同一 V | 扩展 attention 后学习 collapse | 否 | “virtual token + 多 phase”表述先例 |
| Frame/group averaging | 对一组变换运行并平均 | 取决于 backbone | 取决于 backbone | 通常平均输出/逆变换后的输出 | 否 | 一般数学思想已存在 |
| ViT Phase Marginalization | 多个 patch-grid phase | 多次前向产生 | 多次前向产生 | 对齐后平均 dense outputs | 否 | “Phase Marginalization”名称和思想已存在 |

---

## 3. 最接近工作的逐项比较

### 3.1 MoICE：决定当前方案不能宽泛 claim 的直接先例

[MoICE（NeurIPS 2024）](https://papers.neurips.cc/paper_files/paper/2024/file/91315fbb83ce353ae5538cba395f70d1-Paper-Conference.pdf) 把每个 RoPE angle/base 视为一个 in-context expert。对第 $h$ 个 head、第 $n$ 个 query，router 选出若干 angle，并计算

$$
A^{h,(m)}_{n,j}
=
\operatorname{softmax}_j
\left(
\frac{(q_n^h)^\top R_{\Theta_m,j-n}k_j^h}{\sqrt d}
\right).
$$

最终 attention distribution 是

$$
A^h_{n,j}
=
\sum_{m\in\operatorname{TopK}(q_n^h)}
p_n^h[m]A^{h,(m)}_{n,j}.
$$

随后仍用原 attention output

$$
o_n^h=\sum_jA^h_{n,j}v_j^h.
$$

所以 MoICE 已同时满足：同一 query；同一语义 pre-RoPE K；多个 RoPE 角度；同一 V；head/token-specific 聚合。它与目标方案的差异是归一化位置和 anchor 参数化，而不是“是否做同 KV 多相位聚合”。

更关键的是，MoICE 可精确写成带校正项的 token-level LME：

$$
A^h_{n,j}
=
\operatorname{softmax}_j
\left[
\log\sum_m
\exp\left(
s_{j,m}+\log p_n^h[m]-\log Z_m
\right)
\right].
$$

因此，“LME”这个符号本身不能建立结构性差异。

**重合结论：极高。** 如果没有新的触发机制、相对距离 anchor 构造、理论目标和机制证据，建议停止把当前算子作为主方法投稿。

### 3.2 Attention Buckets：多 RoPE ensemble 的更早先例

[Attention Buckets（ACL 2024）](https://aclanthology.org/2024.acl-long.601.pdf) 用不同 RoPE base 做 $N$ 次完整推理，每个分支输出词表分布 $p_m$，再按置信度加权：

$$
\widehat p=\sum_m\alpha_mp_m.
$$

它不是单层 attention 内的同 K/V 聚合，因为不同前向会改变所有层的 hidden states、Q/K/V；但它已经覆盖“用多个 RoPE 相位视图避开 attention waveform trough，再 ensemble”的问题动机。

**重合结论：概念高、算子中等。** 不能声称首次 multi-RoPE ensemble；可以强调单次稀疏 attention、共享原 K/V 和更低开销。

### 3.3 RoPE++：共享 K/V cache 的互补相位分量

[RoPE++（ICLR 2026）](https://arxiv.org/pdf/2512.07525) 同时保留复数内积的实部和虚部。其两条 score 分支可写为

$$
A^{\mathrm{Re}}_{t,j}
=
q_t^\top R_{j-t}k_j,
$$

$$
A^{\mathrm{Im}}_{t,j}
=
(R_{-\pi/2}q_t)^\top R_{j-t}k_j.
$$

它只额外旋转 query，复用 key 的位置编码与 KV cache；但两条分支作为不同 heads/分量分别计算 attention，最后经 concat/output projection 融合，并没有把同一 semantic token 的两个 logits 先 LME。

**重合结论：中高。** “共享 cache 的多相位视图”已存在；目标方案能区分的是同一 head 内、token-level、单分母的相位假设合并。

### 3.4 SelfExtend、DCA、InfLLM、CALIOPE：一个 pair 最终只保留一个位置

#### SelfExtend

[SelfExtend（ICML 2024）](https://proceedings.mlr.press/v235/jin24b.html) 同时构造 neighbor 与 grouped 两种候选 score，但使用硬掩码：

$$
a_{ij}
=
\begin{cases}
a^{\mathrm{neighbor}}_{ij},& |i-j|\le w_n,\\
a^{\mathrm{group}}_{ij},& \text{otherwise}.
\end{cases}
$$

实现上对应 `where(mask, ngb_attn, g_attn)`，被舍弃的候选不会进入 softmax。

#### DCA

[DCA（ICML 2024）](https://arxiv.org/pdf/2402.17463) 对 intra、successive、inter chunks 使用不同 query position rules，但每个 key 属于互斥分区。实现用每组返回的 softmax log-normalizer 合并输出：

$$
O=\sum_g\operatorname{softmax}(L)_gO_g,
\qquad
L_g=\log\sum_{j\in G_g}e^{s_j}.
$$

这是 LSE-weighted merge 的直接先例，但它只是把分块计算精确还原成一个覆盖**不重复 keys**的全局 softmax。

#### InfLLM

[InfLLM（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf) 构造

$$
C=\operatorname{Concat}(I,f(X,E),L)
$$

并在 initial、retrieved memory、local tokens 上统一 attention。被检索出的远程 token 最终被赋予相同的受限相对距离 $l_L$，即每个远程 key 只有一个固定 anchor，而不是多个同时存在的相位假设。

#### CALIOPE

[CALIOPE（Findings EACL 2026）](https://aclanthology.org/2026.findings-eacl.120.pdf) 在冻结模型推理时做确定性单调位置重映射：

$$
\Phi(t)=t+c(m(t)).
$$

每个 token 仍只有一个 remapped position。

**共同结论：** 这些工作覆盖了 local/remote 分工、远程压缩、chunk-aware remapping 和推理时位置修复；它们没有覆盖“同一个远程 key 的多个相位同时进入一个概率分布”。

### 3.5 Ms-PoE：多尺度在 heads 之间，不在同一 head 内边缘化

[Ms-PoE / Found in the Middle（NeurIPS 2024）](https://papers.neurips.cc/paper_files/paper/2024/file/6ffdbbe354893979367f93e2121e37dd-Paper-Conference.pdf) 对第 $h$ 个 head 使用缩放位置 $t/r_h$：

$$
R(t)\longrightarrow R(t/r_h),
$$

$$
r_i
=
R_{\min}
+
\frac{(i-1)(R_{\max}-R_{\min})}{n_h-1}.
$$

不同尺度对应不同 head 的 Q/K/V projections，并分别 softmax，最后走标准 multi-head 输出投影。它不是在同一 head 内复制同一个语义 K/V。

**重合结论：中等。** 多尺度 RoPE 已覆盖；同 head 多 anchor 和 token-wise latent aggregation仍可区分，但 MoICE 已进一步覆盖了后者的大部分。

### 3.6 FoPE、Selective RoPE、TAPA：学习或叠加相位，而不是枚举相对位置

#### FoPE

[FoPE](https://arxiv.org/pdf/2412.17739) 在同一 coordinate 内引入多频率分量：

$$
h_m(n)
=
H_m(n)
\left(
e^{i\omega_mn}+\sum_\omega a_\omega e^{i\omega n}
\right).
$$

它是频谱基函数混合，而不是同一 semantic key 在多个 relative-position hypotheses 下形成多个 attention logits。

#### Selective RoPE

[Selective RoPE（ICLR 2026）](https://openreview.net/pdf/437d371e1c06e7b684e3fed4fcbd8636e564cca6.pdf) 从输入产生并累计旋转角，形成一条 input-dependent phase trajectory；对一个 pair 仍只消费一个学习到的相位。

#### TAPA

[TAPA](https://arxiv.org/pdf/2509.12635) 用内容决定单一 pair phase：

$$
\operatorname{Attn}_{\mathrm{TAPA}}(q,k)
=
q^\top Mk
\cos\left(
\frac{2\pi|m-n|}{\alpha}\phi(q,k)
\right).
$$

**共同结论：** generic content-aware phase、learned phase 和 frequency mixture 都已有先例；目标方法不能把“相位取决于内容/频率”本身作为新颖性。

### 3.7 FASA、SALS：selector 与最终消费分离已经被覆盖

#### FASA

[FASA（ICLR 2026）](https://arxiv.org/pdf/2602.03152) 用少数主导 RoPE frequency chunks 近似完整 score：

$$
A_{t_1,t_2}
\approx
\sum_{i\in I_{\mathrm{dom}}}
q_{t_1}^{[i]}R_{\Delta t,\theta_i}
(k_{t_2}^{[i]})^\top,
$$

用它提出候选，再对候选做原始全维 RoPE attention。

#### SALS

[SALS（NeurIPS 2025）](https://papers.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf) 将 pre-RoPE key 投影到低秩空间：

$$
s_j=\widetilde q_{:r^\star}^\top\widetilde k_{j,:r^\star},
$$

用 RoPE-free score 选 Top-K，再恢复选中 keys、施加标准 RoPE 并计算 sparse attention。

**共同结论：** 若目标方案还包含 pre-RoPE retrieval，必须承认 proposal–consumption separation、frequency-aware proposal 与 RoPE-free proposal 都已有正式先例。可能的新意只能在“如何诊断必须修复的 pair、如何构造和校准多个远程 phase hypotheses”。

### 3.8 IHA、group averaging 与跨领域 Phase Marginalization

[Interleaved Head Attention（2026 预印本）](https://arxiv.org/pdf/2602.21371) 为每个原 token 构造多个 pseudo-Q/K/V 和不同 RoPE phases，再做扩展 attention，最后学习 collapse。它已经使用了“virtual token / multiple phase”语言，但 pseudo-K/V 是跨 head 学习混合，并非所有 phase 副本共享一个原始 V。

[Group Equivariant Stand-Alone Self-Attention（ICLR 2021）](https://openreview.net/forum?id=JkfYjnOEo6M) 和 [Frame Averaging](https://arxiv.org/abs/2110.03336) 说明：对一组变换求平均来获得不变性/等变性，是已有的一般数学框架。它们通常平均 backbone outputs 或逆变换后的 outputs，并非 RoPE attention 内的 token–phase 联合 Gibbs 分布。

此外，[Phase Marginalization for Patch-Grid Instability in Vision Transformers（2026）](https://arxiv.org/abs/2606.08132) 已使用 **Phase Marginalization** 这一名称：它运行多个 patch-grid phases、逆对齐 dense outputs 后平均。领域与算子不同，但不宜再把“Phase Marginalization”作为独占命名或宽泛 first claim。

---

## 4. 安全与不安全的论文表述

### 4.1 不安全

- “We are the first to aggregate multiple RoPE phases in attention.”
- “We introduce the first per-head/per-token mixture of RoPE angles.”
- “We are the first to reuse the same KV under several positional views.”
- “We introduce the first phase marginalization method.”
- “Prior methods choose only one phase, whereas ours considers several.”

MoICE、Attention Buckets、RoPE++、IHA 任何一项都足以反驳其中至少一部分。

### 4.2 只有完成对比后才可能使用的窄表述

较安全的候选表述是：

> To our knowledge, this is the first frozen-LLM mechanism that uses a causal pre/post-RoPE suppression test to instantiate pair- and head-specific relative-distance hypotheses only for remote interactions, and collapses those hypotheses into one semantic-token contribution under a calibrated single-attention energy model while leaving local and non-triggered interactions exactly unchanged.

中文：

> 据我们检索，这是首个面向冻结 LLM、由 pre/post-RoPE 反事实抑制证据触发、仅在远程 query–key pair 上构造 head-specific 相对距离假设，并在保持局部及未触发交互完全不变的前提下，将这些假设校准合并为一次语义 token 写入的方法。

即使如此，也必须在论文中明确说明：MoICE 已做多 RoPE angle 的同 KV attention mixture；我们的差异不是“多相位”本身，而是**因果触发、相对距离 hypothesis、局部严格不变、冻结推理、校准单分母和稀疏实现**这一整组约束。

---

## 5. Go / No-Go 实验门槛

### 5.1 必须加入的公式等价性检查

1. 显式展开 virtual copies $(j,m)$ 做一次 softmax；
2. token-level LME 后做 softmax；
3. 验证两者在 FP32 下逐元素一致；
4. 验证共享 V 后的 attention output 一致；
5. 若每个 token 的 anchor 数不同，必须使用归一化 $\pi_{j,m}$，防止仅因复制更多次而增加质量。

### 5.2 最小方法消融

在相同候选 token budget 和相同 QK 计算预算下比较：

1. 标准原生 RoPE；
2. exact post-RoPE Top-$k$；
3. pre-RoPE selector + 原位置消费（SALS-like）；
4. 一个固定远程 anchor（InfLLM-like）；
5. SelfExtend/grouped distance；
6. 多 anchor 的 score mean；
7. 多 anchor 的 hard max；
8. 裸 LME；
9. 带 $-\log Z_m$ 的规范 marginalization；
10. MoICE 式 mixture-of-softmax，uniform weights；
11. MoICE 式 mixture-of-softmax，learned/router weights；
12. RoPE++ 式 real/imag 两分量；
13. 只扩大候选或 attention budget、但不改 phase；
14. 随机 anchors 和等计算量随机 phase perturbation。

### 5.3 必须报告的结果

- Gold evidence recall；
- 两条证据链均命中率；
- gold attention mass；
- gold-vs-distractor QK margin；
- 正确答案 PPL 与首 token/完整答案准确率；
- attention entropy、校准误差；
- local task/perplexity 回归；
- 每 token/head 新增 QK 次数；
- prefill/decode latency、峰值显存、KV cache 增量；
- 不同 anchor 数量、位置、先验 $\pi_m$ 的敏感性。

### 5.4 最终决策条件

**GO：** 在独立 seeds、不同长度、不同 needle/distractor、至少两个模型上，校准多 anchor 显著优于 MoICE-style mixture、hard max、单 anchor 和等预算 selector；提升来自被诊断的相位抑制 pair，且 local 行为与无抑制样本几乎不变。

**NO-GO：** 出现任一情况就不宜把它作为 ICLR 主方法：

- 优势在与 MoICE mixture-of-softmax 对齐后消失；
- hard max 或 score mean 已达到相同效果；
- 收益主要来自 anchor duplication 或更大的 QK/attention budget；
- 不加 $-\log Z_m$ 时有效、校准后无效，且无法解释 partition-function bias；
- 只在单一人工 needle 样例有效；
- local PPL、短程顺序或冲突证据鲁棒性明显退化。

---

## 6. 对当前论文路线的建议

1. **不要把 LME 当作论文核心。** 它是标准 latent-variable / energy pooling 运算，且 MoICE 已给出更规范的 per-phase probability mixture。
2. **把 MoICE 提升为第一优先 baseline 和 related-work 近邻。** 它比 Ms-PoE、RoPE++ 更直接。
3. **保留 LME 作为实现候选与消融。** 如果裸 LME 比规范 mixture 更好，研究重点应转向“为什么 partition-function weighting 恰好能恢复远程证据”。
4. **核心贡献回到机制诊断。** 最可守的是从真实失败边界出发，证明哪些 QK 被 RoPE 反事实抑制，并只修复这些 pair；这比“又一种多相位平均”更有论文价值。
5. **将方法限定为远程补偿，而非替换 RoPE。** 局部 token 使用原始相对位置；远程候选才允许多个假设；未触发交互严格复现原模型。
6. **优先做小规模判别实验。** 先在现有 Qwen3-8B 失败边界上比较 bare-LME、normalized marginalization、MoICE mixture、max 和 fixed anchor；若不能稳定胜出，立即停止扩展工程。

### 一句话定位

当前“同 KV 多 RoPE 相位 + 聚合”的方法层面已经不新；真正仍可能有价值的是：**由因果抑制证据触发、只作用于远程 pair、保持局部几何不变的校准相位修复机制**。

---

## 7. 主要原始来源

- [MoICE, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/91315fbb83ce353ae5538cba395f70d1-Paper-Conference.pdf)
- [Attention Buckets, ACL 2024](https://aclanthology.org/2024.acl-long.601.pdf)
- [InfLLM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf)
- [SelfExtend, ICML 2024](https://proceedings.mlr.press/v235/jin24b.html)
- [DCA, ICML 2024](https://arxiv.org/pdf/2402.17463)
- [Ms-PoE / Found in the Middle, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/6ffdbbe354893979367f93e2121e37dd-Paper-Conference.pdf)
- [FoPE](https://arxiv.org/pdf/2412.17739)
- [SALS, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf)
- [RoPE++, ICLR 2026](https://arxiv.org/pdf/2512.07525)
- [FASA, ICLR 2026](https://arxiv.org/pdf/2602.03152)
- [Selective RoPE, ICLR 2026](https://openreview.net/pdf/437d371e1c06e7b684e3fed4fcbd8636e564cca6.pdf)
- [TAPA](https://arxiv.org/pdf/2509.12635)
- [CALIOPE, Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.120.pdf)
- [Interleaved Head Attention](https://arxiv.org/pdf/2602.21371)
- [Group Equivariant Stand-Alone Self-Attention, ICLR 2021](https://openreview.net/forum?id=JkfYjnOEo6M)
- [Frame Averaging](https://arxiv.org/abs/2110.03336)
- [Phase Marginalization for Patch-Grid Instability in Vision Transformers](https://arxiv.org/abs/2606.08132)

## 检索边界

本轮使用了方法名检索与公式/关键词组合检索，包括 `multiple RoPE angles`、`same KV multiple phases`、`virtual relative positions`、`phase marginalization`、`log-mean-exp attention positional`、`group averaging invariant attention` 等。除 MoICE 等上述近邻外，没有在截止日期前找到“同一语义 KV 的多个相对距离 anchor，以裸 LME 在 token softmax 前合并，并共享一次 V 写入”的逐项完全相同披露。这个“未找到 exact match”不能证明全球范围不存在，也不能抵消 MoICE 带来的高显然性风险；公式定型后仍需再做一次逐项查新。
