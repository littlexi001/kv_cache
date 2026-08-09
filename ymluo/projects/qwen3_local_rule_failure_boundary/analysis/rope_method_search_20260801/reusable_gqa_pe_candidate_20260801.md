# 可复用、GQA 一致的长程位置/检索候选审计

**日期：** 2026-08-01

**模型约束：** 冻结 Qwen3-8B；32 个 Query heads、8 个 KV heads、每 4 个 Query heads 共享一组 K/V；最终 support 约为序列的 2%。

**本阶段范围：** 只给出数学设计和最小可证伪实验；不实现 GPU runner，不启动服务器任务。

> **后续实验更新：** KVQ-R 的两个 8K smoke 虽然均覆盖两条 gold evidence，但都未通过 finite-difference audit；更决定性的是，简单 `pre_score_pair` 在两个 seeds 的 edge AUROC 都为 1.0，reverse edge 也与或高于 KVQ-R。该结果已触发本文第 4 条止损条件，所以 PFRI 从 Conditional GO 更新为 **NO-GO**，不再进入 sparse-consumer 实现。详见 [`kvq_relay_edge_smoke_2seed_merged/README.md`](kvq_relay_edge_smoke_2seed_merged/README.md)。

## 结论先行

1. **新位置编码（PE）：当前应判为 NO-GO。** 在冻结模型、K 必须按 GQA 共享、修复必须跨多个后续 Query 复用的条件下，所有自然的 K 相位修改都会落入已有类别：远程 NoPE/partial-RoPE、静态位置或 chunk remapping、head/frequency scaling、多相位副本，或逐 Query logit bias。当前证据不足以支持一个可守的新 PE。
2. **可复用检索方法：PFRI 已由后续 smoke 判为 NO-GO。** 候选名为 **PFRI（Prefix-Frozen Relay Index，前缀冻结接力索引）**。它原计划在文档 prefill 后仅建立一次由冻结模型内部算子定义的有向 block 图；但其核心有向 relay score 未提供超出普通端点相关性的独立信息。
3. PFRI 的潜在贡献不是“pre-RoPE 检索”，也不是“首次发现两跳 sequential query”，而只能是：

   > 在冻结 GQA LLM 的真实 KV cache 上，用一个与未来 Query 无关、可跨 decode token 复用的 \(K\rightarrow V\rightarrow Q\rightarrow K\) 转移索引，为严格 2% sparse attention 补全多跳证据。

4. 如果 PFRI 的有向边不能显著胜过 K--K 相似度、shuffled-V 和普通 exact-pre 检索，则整个方法方向立即 **NO-GO**，论文退回“RoPE 失效机制 + pre-RoPE retrieval baseline”。

---

## 1. 为什么当前不能再把 Query-specific phase repair 称为 PE

### 1.1 单 Query 下，它与标量 logit bias 完全等价

对一个 Query 和一个候选 Key，原生分数为

$$
s_i
=
\frac{q^\top R(t-p_i)k_i}{\sqrt d}.
$$

如果只针对当前 Query 把相位改成 \(R(t-p_i+\delta_i)\)，则

$$
s_i'
=
\frac{q^\top R(t-p_i+\delta_i)k_i}{\sqrt d}
=
s_i+b_i,
$$

其中 \(b_i=s_i'-s_i\)。因此

$$
\operatorname{softmax}(s')V
=
\operatorname{softmax}(s+b)V.
$$

这不是近似，而是逐元素恒等。若 \(\delta_i\) 由当前 Query 决定、没有生成一个可供后续 Query 复用的物理 K，那么它只是 attention-logit edit。

### 1.2 Qwen3-8B 的 GQA 排除了逐 Query-head K 修复

令 \(g(h)\) 表示 Query head \(h\) 对应的 KV head。Qwen3-8B 中

$$
H_Q=32,
\qquad
H_{KV}=8,
\qquad
|\mathcal H_g|=4.
$$

物理 cache 只有

$$
K_{l,g,i},
\qquad g=1,\ldots,8,
$$

而不是 32 份独立的 \(K_{l,h,i}\)。因此同一组中的四个 Query heads 必须看到同一个 Key：

$$
K_{l,h,i}=K_{l,h',i}=K_{l,g,i},
\qquad h,h'\in\mathcal H_g.
$$

若四个 Query heads 分别求出不同的 \(\delta_{l,h,i}\)，就不存在一份物理 K 能同时实现它们。

### 1.3 冻结模型下的静态 K 几何基本已被已有类别穷尽

假设一个方法在未来 Query 到来之前，给每个 cached Key 固定一个可复用变换

$$
K_{l,g,i}^{\star}=T_{l,g,i}\widetilde K_{l,g,i},
$$

其中 \(\widetilde K\) 是 pre-RoPE Key，并要求同一 KV head 的 K 唯一。对于保持 RoPE 二维频率平面结构的 \(T\)：

- 所有频率对使用同一虚拟位移：等价于静态 position/chunk remapping；
- 远程频率相位归零或只旋转部分维度：落入 RoPE-to-NoPE、partial-RoPE 或频带截断；
- 不同 head 使用不同频率或尺度：接近 AdaRoPE 一类 head-specific scaling；
- 为一个 Key 保存多个 \(T\) 并聚合：接近 MoICE/多相位专家；
- \(T\) 依赖当前 Query：重新退化为逐 Query score bias，并失去跨 Query 复用；
- 使用不保持原 RoPE 平面的通用内容变换：相当于替换已训练的 what/where 几何，冻结 Qwen3-8B 没有理由能直接解释它，且与 PoPE 类新编码需要训练的设定相冲突。

所以，在当前硬约束下继续搜索“新的 phase repair PE”没有足够的新颖性或可实现性。下文候选明确是一种 **support retrieval**，不是 PE。

### 1.4 一个满足复用但仍应拒绝的 PE 草案

可以在第一个问题到来时选中远程 blocks，把每个 block 整体 re-rotate 到固定的虚拟槽位，并在后续 decode tokens 中复用同一份 K。它确实可以做到 GQA 共享、block 内相对顺序不变和跨 token 复用；但其核心就是“检索后 chunk relocation + KV re-rotation”。它与 SelfExtend、InfLLM 式远程固定几何及各种 chunk/KV position remapping 高度重合，且冻结模型未必能适应新的跨 block 距离。因此该路线即使数值有效，也不具有当前论文所需的新颖性，判为 **NO-GO**。

---

## 2. 当前证据对候选设计的约束

现有实验中，exact pre-RoPE proposal + native post-RoPE consumer 是最稳定的正结果：

| 长度 | Full attention PPL | Exact post-RoPE Top-2% PPL | Exact pre-RoPE Top-2% PPL |
|---:|---:|---:|---:|
| 16K | 7.509 | 3.139 | **2.566** |
| 32K | 14.607 | 10.472 | **3.958** |
| 64K | 8.362 | 6.494 | **4.449** |

这些既有数值使用原实验的 per-Query-head selector，只证明 pre-RoPE proposal 有信号；它们不是 PFRI 的 matched GQA-group baseline。PFRI 实验必须重新计算 group-tied exact-pre/post controls。

它支持“proposal 与 consumption 分离”，但不能作为新方法，因为 SALS 已使用 RoPE-free latent proposal，FASA 也已覆盖 query-aware sparse proposal。

同时，当前负结果给出三条硬约束：

- suppression gap 不能区分 gold 与 conflict；64K safety smoke 的相关 AUROC 约为 0.50；
- 提高 evidence QK、recall 或 attention mass 并不必然改善 PPL；
- final-query phase repair 既有 logit-bias 等价性，又可能违反 GQA 的单 K 可实现性。

因此新候选必须解决的是 **多跳 support closure**，而不是再次设计一个通用 token importance 或 phase gate。

---

## 3. 候选：PFRI（Prefix-Frozen Relay Index）

### 3.1 核心思想

普通 pre-RoPE 检索只回答：

> 当前 Query 与哪个历史 Key 最相似？

两跳任务还需要回答：

> 如果第一条证据的 Value 被写入残差，它会把下一层 Query 推向哪一条历史 Key？

PFRI 在文档 prefill 后、问题 Query 到来前，使用文档自身的 K/V 和冻结层算子建立一个静态有向图：

$$
B\longrightarrow C.
$$

边表示 block \(B\) 的 Value write 在平均的冻结层局部几何中，与 block \(C\) 的 pre-RoPE Key 兼容。图只建立一次；之后的多个 Query 仅改变图的入口 seed，不改变节点、边、K、V 或 position。

### 3.2 Prefix epoch 与信息防火墙

令文档前缀为 \(X_{1:T}\)，问题和后续生成位于 \(t>T\)。索引定义为

$$
\mathcal G_X
=
\mathcal I_{\theta}(X_{1:T}),
$$

其中 \(\theta\) 是冻结 Qwen3-8B 参数。必须满足

$$
\frac{\partial \mathcal G_X}{\partial q_t}=0,
\qquad t>T.
$$

索引允许读取：

- 文档前缀的 pre-RoPE K、原始 V 和中间 residual；
- 冻结的 \(W_O,W_Q\)、RMSNorm 与 MLP；
- 固定 block 边界、固定 relay layers；
- 无标签的数值归一化统计。

索引禁止读取：

- gold/conflict span 或标签；
- 正确答案 token、unembedding direction、answer margin 或 loss gradient；
- 未来 Query 的 Q；
- 根据 seed、长度或答案结果选择 layer/head/阈值。

### 3.3 Label-free block 表示

将远程文档切为固定 \(C=32\) token 的 blocks。对 layer \(l\)、KV head \(g\)、block \(B\)，定义 query-independent 均值原型：

$$
\kappa_{B}^{l,g}
=
\operatorname{norm}
\left(
\frac{1}{|B|}\sum_{i\in B}\widetilde k_i^{l,g}
\right),
$$

$$
\nu_{B}^{l,g}
=
\frac{1}{|B|}\sum_{i\in B}v_i^{l,g}.
$$

这里 \(\widetilde k\) 是完成模型自身 K-normalization、但尚未施加 RoPE 的 Key。均值版本是最小实验；只有它通过后，才允许尝试两个确定性原型，以排除 block 平均造成的抵消。

### 3.4 GQA 一致的 Value write

令 \(\mathcal H_g\) 是共享 KV head \(g\) 的四个 Query heads，\(W_{O,l}^{[h]}\) 是 output projection 中属于 Query head \(h\) 的列块。定义一份 group-level 残差写入：

$$
u_{B}^{l,g}
=
\frac{1}{|\mathcal H_g|}
\sum_{h\in\mathcal H_g}
W_{O,l}^{[h]}\nu_{B}^{l,g}.
$$

这不是对真实 attention output 的声称，而是一个固定、归一化、GQA 可实现的 source descriptor。它不会为同一共享 V 构造四个互相冲突的 K 修复。

### 3.5 Prefix-frozen 的下一层 Query 转移算子

对 layer \(l\) 的 post-attention residual \(x\)，定义到下一层 pre-RoPE Query 的冻结映射：

$$
\Psi_l^{h'}(x)
=
\operatorname{QNorm}_{l+1}^{h'}
\left[
W_{Q,l+1}^{h'}
\operatorname{RMSNorm}_{l+1}
\left(
x+\operatorname{MLP}_l(\operatorname{RMSNorm}_{l,\mathrm{post}}(x))
\right)
\right].
$$

从每个文档 block 的末端取一个 residual landmark，构成固定集合 \(\mathcal R_l\)。不显式存储巨大 Jacobian，而只定义其平均 JVP：

$$
\overline J_l^{h'}[u]
=
\frac{1}{|\mathcal R_l|}
\sum_{r\in\mathcal R_l}
D\Psi_l^{h'}(x_r)[u].
$$

于是，source block \(B\) 的 Value 对 destination Query head \(h'\) 的静态 relay 向量为

$$
\rho_{B}^{l,g_s\rightarrow h'}
=
\overline J_l^{h'}[u_B^{l,g_s}].
$$

关键点是：\(\overline J\)、\(u_B\) 和 \(\rho_B\) 都只依赖文档前缀及冻结参数。它们不围绕 final-query residual 做线性化，也不按当前 Query 对 block 内 V 加权。

### 3.6 有向 relay edge

对 destination KV group \(g_d\)，定义

$$
e_l^{g_s\rightarrow g_d}(B\rightarrow C)
=
\operatorname{LME}_{h'\in\mathcal H_{g_d}}
\left[
\frac{
(\rho_{B}^{l,g_s\rightarrow h'})^\top
\kappa_C^{l+1,g_d}
}{\sqrt d}
\right],
$$

其中 LME 是按 head 数量校正的 log-mean-exp。随后只用各 source block 的非对角 destination 分布做 median/MAD 标准化：

$$
\widehat e(B\rightarrow C)
=
\frac{e(B\rightarrow C)-\operatorname{median}_{C'}e(B\rightarrow C')}
{\operatorname{MAD}_{C'}e(B\rightarrow C')+\epsilon}.
$$

只在固定 relay layers

$$
\mathcal L
=
\left\{
\left\lfloor\frac L4\right\rfloor,
\left\lfloor\frac L2\right\rfloor,
\left\lfloor\frac{3L}4\right\rfloor,
L-2
\right\}
$$

建立边，并为每个 \((B,l,g_s,g_d)\) 保存 top-\(r\) destination blocks。实际实现可把 \(\rho_B\) 作为 ANN query、\(\kappa_C\) 作为 ANN database，无需构造完整 \(O(M^2)\) 矩阵。

LME 跨四个 destination Query heads 时，可以分别做四次 ANN lookup，合并候选后再精确重排，无需把四个不同 Query-head 方向伪装成四份 K。以 64K、block size 32、4 个 relay layers、8×8 个 source/destination groups、每项 top-4 edge 估算，约有 210 万条稀疏边；用 int32 destination 与 FP16 score 存储约 12 MiB，因此索引结构本身是可实现的。一次性 JVP 建图可能较贵，必须由多 Query 复用摊销。

### 3.7 每个未来 Query 的 seed 与一次图扩展

未来 token \(t\) 到来后，允许使用它自己的 pre-RoPE Query 找入口 seed，但不重建图。对 KV group \(g\)：

$$
d_t^{l,g}(B)
=
\operatorname{LME}_{h\in\mathcal H_g}
\left[
\frac{(\widetilde q_t^{l,h})^\top\kappa_B^{l,g}}{\sqrt d}
\right].
$$

取 direct score 的 top-\(m\) blocks 为 \(\mathcal S_t^{\mathrm{seed}}\)。对 destination block \(C\)：

$$
r_t^{l,g_d}(C)
=
\max_{B\in\mathcal S_t^{\mathrm{seed}},g_s}
\left[
\widehat d_t^{l,g_s}(B)
+
\widehat e_l^{g_s\rightarrow g_d}(B\rightarrow C)
\right].
$$

最终 block admission score 使用无训练的 rank fusion：

$$
A_t^{l,g}(C)
=
\max
\left\{
\operatorname{RankNorm}(d_t^{l,g}(C)),
\operatorname{RankNorm}(r_t^{l,g}(C))
\right\}.
$$

只扩展一跳。若两跳以上路径才有效，优先认为方法过度依赖图传播，不在首轮增加复杂度。

首版只在 \(l\in\mathcal L\) 的四个固定 relay layers 使用 direct + graph support；其余层严格回退到 GQA-grouped exact pre-RoPE support。这样不会把尚未验证的 relay 信号扩散到全部 36 层，也能用四层消融直接判断收益是否来自有向边。

### 3.8 严格约 2% support 与近程顺序

对 \(t\ge8\mathrm K\)，每层、每个 KV group 的总预算为

$$
K_t=\left\lceil0.02t\right\rceil.
$$

固定保留：

$$
K_{\mathrm{local}}=64,
\qquad
K_{\mathrm{sink}}=4,
\qquad
K_{\mathrm{remote}}=K_t-68.
$$

其中 local set 是包含当前位置的最近 64 个 token：

$$
\mathcal S_{\mathrm{local}}(t)
=
\{\max(0,t-63),\ldots,t\}.
$$

按 \(A_t^{l,g}\) 选择完整的 32-token remote blocks；最后一个 block 超出预算时，只用 direct pre-RoPE token score 填满剩余名额。四个共享该 KV head 的 Query heads 使用完全相同的 support：

$$
S_{l,h,t}=S_{l,h',t}=S_{l,g,t},
\qquad h,h'\in\mathcal H_g.
$$

local 64 tokens、sink 和所有 selected remote tokens 都保留原 token index。不存在重新排序、压缩距离或虚拟位置。

### 3.9 原生 RoPE consumer

对最终 support 中任意位置 \(i\)：

$$
s_{t,i}^{l,h}
=
\frac{
(R_t\widetilde q_t^{l,h})^\top
(R_i\widetilde k_i^{l,g(h)})
}{\sqrt d},
$$

$$
a_{t,i}^{l,h}
=
\operatorname{softmax}_{i\in S_{l,g(h),t}}
(s_{t,i}^{l,h}),
\qquad
o_t^{l,h}=\sum_{i\in S_{l,g(h),t}}a_{t,i}^{l,h}v_i^{l,g(h)}.
$$

检索分数 \(d,r,A\) 到此全部丢弃。模型只消费原始 K/V 和原生 post-RoPE score。

---

## 4. 三个可审计的不变量

### 4.1 跨 Query 复用

在同一 prefix epoch 内，连续 32 个 decode tokens 或多个独立问题必须满足：

$$
\operatorname{hash}(\mathcal G_X)_{t}
=
\operatorname{hash}(\mathcal G_X)_{t+1}.
$$

每个 token 可以改变 seed 和 support，但不能改变 graph edge、block prototype 或历史 K/V。

### 4.2 GQA 一致

任何中间对象均不得出现按 Query head 单独存储的 repaired Key。合法索引只能是：

$$
K[l,kv\_head,token],
\qquad
S[l,kv\_head,query].
$$

同组四个 Query heads 的 gathered token indices 必须逐元素相等。

### 4.3 原生几何与 local exactness

对同一 selected support，PFRI consumer 的 QK logits 必须与未修改 Qwen3-8B 在这些位置上的 logits 在数值精度内相等：

$$
\max_{i\in S}
|s_{t,i}^{\mathrm{PFRI}}-s_{t,i}^{\mathrm{native}}|
\le 10^{-4}
$$

（FP32 重构审计；BF16 前向另报容差）。local 64-token 区间不允许任何 position、phase、K、V 或 score 修改。

---

## 5. 与最近工作的实质差异

| 方法 | 已有工作的核心 | PFRI 的差异与不可声称之处 |
|---|---|---|
| RoPE-to-NoPE hybrid / P-RoPE | 近程保留位置，远程用 NoPE 或交替 local/global layers | PFRI 的远程 consumer 仍是完整原生 RoPE；不声称 local/global PE。若改成远程 NoPE，创新立即消失。 |
| SALS | 在低秩 latent 中做 RoPE-free QK token proposal，再重构少量 K 做 sparse attention | PFRI 的 direct seed 与 SALS 同类，不能算贡献；唯一候选贡献是 prefix-frozen \(V\rightarrow Q\rightarrow K\) 有向边与路径补全。去掉 edge 后若效果不变，方法 NO-GO。 |
| FASA | 用少量 dominant RoPE frequency chunks 近似 query-aware token importance | PFRI 不选频带、不近似当前 post-RoPE QK；relay graph 在未来 Query 前建立。 |
| MoICE | 为当前 token/head 路由并聚合多个 RoPE angle experts，且训练 router | PFRI 只有一个原始 RoPE angle，无 KV 副本、无 attention mixture、无 router。 |
| PoPE | 用 polar-coordinate PE 解耦 what/where，并在相应编码下训练模型 | PFRI 不替换 PE，也不主张解决 what/where 表示学习；冻结 Qwen3-8B 的全部位置几何保持不变。 |
| AdaRoPE | 学习 head-specific frequencies 和 attention scaling | PFRI 不学习或缩放任何频率；selection 以 KV group 为物理单位，四个 Query heads 共享 support。 |
| chunk remapping / SelfExtend / KV re-rotation | 将远程 chunk 映射到新的、分组的或连续的 position ids | PFRI 不改变任何 position id，不 de-rotate/re-rotate K；block 只作为 retrieval unit。 |
| MoICE-like multi-phase marginalization | 同一 KV 在多个相位下计算并聚合 | PFRI 每个 KV 只有原始相位；图分数从不进入最终 attention logits。 |
| 当前 KVQ-R final-query edge probe | 用 final-query pre-RoPE 相关性聚合 V，并围绕 final-query residual 做有限差分 | PFRI 去掉这两处 Query 依赖：block V 原型是 prefix-only，JVP 在 prefix landmarks 上平均并永久冻结，因此一张图可被多个问题和 decode tokens 复用。 |

最直接的理论先例是 [How Do LLMs Perform Two-Hop Reasoning in Context?](https://arxiv.org/abs/2502.13913)：它已发现 sequential-query mechanism。因此不能声称首次发现 \(K\rightarrow V\rightarrow Q\rightarrow K\) 接力。PFRI 是否有方法新颖性，取决于“冻结真实 KV cache 上的可复用 relay index”能否在严格稀疏预算下产生超出普通语义图的独立收益。

相关主文献：

- [Periodic RoPE / P-RoPE](https://arxiv.org/abs/2605.27980)
- [SALS, NeurIPS 2025](https://arxiv.org/abs/2510.24273)
- [FASA, ICLR 2026](https://arxiv.org/abs/2602.03152)
- [MoICE](https://arxiv.org/abs/2406.19598)
- [PoPE, ICML 2026](https://arxiv.org/abs/2509.10534)
- [AdaRoPE, ICML 2026](https://arxiv.org/abs/2607.19363)
- [SelfExtend](https://arxiv.org/abs/2401.01325)
- [InfLLM](https://arxiv.org/abs/2402.04617)
- [Multi-Token Attention](https://arxiv.org/abs/2504.00927)

本次检索没有发现与 PFRI 完全相同的论文，但这不是“全球不存在”的证明。一般 graph retrieval、multi-hop RAG 和 sequential-query 分析都构成强先例；若边诊断通过，正式定稿前仍需按最终公式再做一次逐项查新。

---

## 6. 最小可证伪实验

### 6.1 实验对象

只做一个小而强的矩阵，不先扩成大 benchmark：

- 模型：冻结 Qwen3-8B；
- 长度：8K、32K；
- seeds：0--7；
- 条件：clean 两跳、gold + conflict 两跳；
- 每个 document prefix 放置 4 条互不相同的两跳链，从同一份 prefix cache 分叉提出 8 个问题；
- graph 在问题出现前只构建一次，8 个问题及每个问题的前 32 个 decode tokens 共用同一 graph；
- gold/conflict labels 只在 graph、scores、support 全部冻结后用于评估。

### 6.2 Arms

所有 sparse arms 严格使用相同 2% support、相同 local/sink 配额和原生 consumer：

1. exact_post_gqa_top2：组内四个 Query heads 聚合后的原生 post-RoPE exact Top-2%；
2. exact_pre_gqa_top2：组内聚合后的 full-dimensional pre-RoPE exact Top-2%；
3. direct_block_pre：只用 PFRI direct seed，不走图；
4. pfr_identity：令 \(\overline J=I\) 的便宜 relay control；
5. pfr_prefix_jvp：完整 PFRI；
6. kk_graph：相同 block、degree 和 budget 的 pre-RoPE K--K 图；
7. shuffled_v_graph：在 source blocks 间置换 \(\nu_B\)；
8. random_v_graph：使用 norm-matched 随机 V；
9. reverse_graph：使用 \(e(C\rightarrow B)\)；
10. random_degree_graph：匹配每个节点的 degree 与 block-distance 分布。

原有 per-Query-head exact-pre Top-2% 与 full attention 只作为质量上界/参照。所有直接方法比较必须使用相同的 GQA-grouped support 约束；否则 PFRI 会因为更严格的 gather 粒度而受到不公平惩罚。

### 6.3 先验证边，再运行 sparse consumer

第一阶段只回答 true first-hop \(\rightarrow\) true second-hop 是否比 matched distractor edge 更高：

- true-edge vs matched-negative AUROC；
- source block 和 destination block 的独立 proposal coverage；
- PFRI 相对 K--K、shuffled-V、random-V、reverse 的 AUROC 差；
- 每个 seed 单独 AUROC 与 seed bootstrap 95% CI；
- edge score 的 block-distance、Value norm、Key norm 偏相关。

只有边诊断通过，才运行第二阶段的 2% sparse attention。

### 6.4 End-to-end 指标

- Gold evidence token recall；
- 两条 gold 证据链均命中率；
- conflict admission rate；
- gold/conflict attention mass；
- 正确答案 NLL/PPL；
- 首答案 token accuracy 与 greedy exact match；
- paired answer margin；
- 32-token generation 中首次恢复正确答案的位置；
- local-order、copy、recent-reference 和 short-context PPL；
- graph build time、每 Query retrieval time、graph memory、实际 selected token 数；
- 8 个问题和 32 个 decode steps 中的 graph hash 与 GQA support-equality violations。

---

## 7. 硬性 GO / NO-GO 标准

以下任一项成立，立即停止 PFRI：

1. true-hop edge AUROC 的 seed 均值低于 0.65，或 95% CI 不能高于 0.5；
2. kk_graph 与 PFRI 的 AUROC 或 end-to-end 效果差距小于 0.03，说明所谓 relay 只是普通语义相似度；
3. shuffled-V 保留了 PFRI 超过随机部分的 80%：

   $$
   \frac{\operatorname{AUC}_{\mathrm{shuffle}}-0.5}
   {\operatorname{AUC}_{\mathrm{PFRI}}-0.5}
   \ge0.8;
   $$

4. 只有围绕当前 final Query residual 的 JVP 有效，而 prefix-averaged JVP 无效；这说明方法不能跨 Query 复用；
5. 在 32K 上，相对 exact_pre_gqa_top2 的 paired \(\Delta\)NLL 95% CI 不能低于 0，且两链均命中率提升不足 10 个百分点；
6. conflict admission 上升超过 2 个百分点，或 recall 上升但 PPL/accuracy 恶化；
7. one-hop、local-order、copy 或 recent-reference 任一集合准确率下降超过 1 个百分点；
8. 任一 graph hash、prefix immutability、GQA support equality、native-logit equality 或 2% budget audit 失败；
9. amortized 8-query 场景下，每 Query selector latency 超过 exact_pre_gqa_top2 的 2 倍且没有可行 ANN 加速；
10. 结果只在一个 block size、一个 seed 或 gold 恰好位于 block endpoint 时成立。

只有同时满足以下条件，才进入 64K 和公开 benchmark：

- PFRI edge 显著胜过全部结构与随机 controls；
- 32K 的 NLL、两链命中和准确率同时优于 exact-pre；
- conflict admission 不增加；
- 复用、GQA、预算、原生 consumer 四项审计零违规；
- local/order-sensitive 任务无可测伤害。

---

## 8. 风险与最终定位

### 8.1 最大技术风险

- prefix-averaged Jacobian 可能抹掉真实 Query state 的非线性依赖；
- block mean Value 可能发生语义抵消；
- relay compatibility 不能判断事实真伪，可能同样扩展 conflict chain；
- 选入正确证据后，原生 post-RoPE consumer 仍可能把其分数压低；
- graph build 的 JVP 成本可能只能在多 Query 复用场景中摊销；
- 一般 graph retrieval、multi-hop RAG 与 sequential-query 分析已有大量先例，方法 novelty 只能靠严格的 frozen-KV operator 与实证独立收益守住。

### 8.2 论文 claim 边界

即使实验全部通过，也只能声称：

> 我们提出一种 prefix-frozen、GQA-consistent 的 relay index，在不修改冻结 LLM 的位置编码或 KV cache 的情况下，用模型自身的 Value-to-next-Query 转移关系补全严格稀疏 attention support，并可由多个后续 Query/decode tokens 复用。

不能声称：

- 新的 RoPE 或新的通用 positional encoding；
- 首次发现 sequential-query/two-hop relay；
- 首次 pre-RoPE sparse retrieval；
- 首次 local/global attention；
- 首次 query-aware KV selection。

### 8.3 最终判定

$$
\boxed{
\text{Reusable new PE: NO-GO}
}
$$

$$
\boxed{
\text{Prefix-Frozen Relay Index: Conditional GO as retrieval}
}
$$

PFRI 是当前约束下唯一值得做一次小规模、强对照实验的可复用候选。它若不能在 edge AUROC、两链 closure 和 paired NLL 三条线上同时胜过 exact-pre、K--K 与 shuffled-V，就不应继续投入 GPU 预算或包装成 ICLR 方法。
