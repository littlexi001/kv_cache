# 可部署方法严格筛选：从 RoPE 诊断转向约束检索与证据接力

**日期：** 2026-08-01  
**目标：** 在不使用 gold evidence、正确答案、answer gradient 或校准标签的前提下，筛选仍可能具有论文价值的最小方法。  
**结论：** 只保留两个首轮候选；目前没有第三个方向达到同等优先级。二者都不修改 RoPE，不做远程 NoPE/local-global 混合，不做多相位修复，也不把 Value/output attribution 当作 token importance。

| 方向 | 当前决策 | 论文成立所需的第一个证据 |
|---|---|---|
| Query-Constraint Coverage（QCC） | **Conditional GO** | 在 matched gold/conflict 上显著优于 exact-pre、mean/max 与 lexical coverage |
| K→V→Q Relay Retrieval（KVQ-R） | **Conditional GO，后于 QCC** | 真两跳 edge 可分，打乱 Value 后收益消失，并在严格 2% budget 下改善 NLL |
| final-query MPR / phase repair | **NO-GO as PE** | 除非产生可复用的 K/position rule，并优于同 support、同 bias 的标量 logit control |

---

## 1. 现有结果给出的硬约束

### 1.1 可以保留的正结果

`exact pre-RoPE proposal + 原位置 post-RoPE consumption` 是当前最稳定的有效组件：

| 长度 | Full PPL | Exact post-RoPE Top-2% PPL | Exact pre-RoPE Top-2% PPL |
|---:|---:|---:|---:|
| 16K | 7.509 | 3.139 | **2.566** |
| 32K | 14.607 | 10.472 | **3.958** |
| 64K | 8.362 | 6.494 | **4.449** |

这说明 pre-RoPE 内容通道确实能找回一部分被原生 RoPE 排名压低的远程信息。但 SALS 已覆盖 RoPE-free/pre-RoPE proposal 后恢复标准 RoPE，FASA 已覆盖 frequency-aware proposal 后进行原始全维消费。因此，**pre-RoPE selector 只能作为共同底座，不能再作为论文方法本身。**

### 1.2 必须停止继续包装的方法

1. **Strict MPR 尚不成立。** 8K、seed 0 中，PPL 从 4.678 降到 3.422，但所有 arm 的首 token 都错误；`random + partition-preserve` 也达到 3.642；51.64% 的远程候选被触发，且定向 arm 约耗时 238 秒。它既不够稀疏，也没有稳定因果优势。
2. **suppression certificate 不是正确性证书。** 当前有效的 64K 单 seed safety smoke 中，`pre_suppression` 的 gold-vs-conflict AUROC 为 0.502，`grid_envelope_suppression` 为 0.500；各类 trigger rate 均约 83%--89%。更重要的是，修 gold evidence 使 Gold NLL **增加 0.348**，而修 conflict 和 filler 反而分别使 Gold NLL **降低 0.715 和 0.495**。这只是单 seed 描述性结果，但已足以否定“suppression 大就直接增强”的默认设计。
3. **oracle value derivative 不能直接变成部署 gate。** 8K 单 seed value smoke 中，target arms 的一阶预测与实际 margin 变化 Spearman 为 -0.258，informative sign accuracy 为 50%；平均预测为 -0.003，实际为 +0.313。它没有通过既定的 0.7 correlation / 80% sign 门槛，而且 gold/answer gradient 在推理时本来也不可用。
4. **提高 QK、recall 或 attention mass 均不充分。** Block transport 和 NPE 已出现 recall/mass 上升但 PPL 显著恶化；mass preservation 也没有消除退化。

### 1.3 Final-query MPR 的严格等价性风险

对固定 final Query、固定 head 和选中的 interaction $j$，任何 phase repair 最终都只产生一个新 scalar：

$$
s_j^{\mathrm{repair}}
=
\frac{q^\top R_{\Delta+\delta}k_j}{\sqrt d}
=
s_j^{\mathrm{native}}+b_j.
$$

只要 $V_j$ 不变，后续计算就是：

$$
a^{\mathrm{repair}}
=
\operatorname{softmax}
\left(s^{\mathrm{native}}+b_j e_j\right),
\qquad
o^{\mathrm{repair}}=\sum_i a_i^{\mathrm{repair}}V_i.
$$

因此，对这一个 Query，频率平面求解、虚拟位置移动和直接给同一个 attention logit 加 $b_j$ **行为上严格等价**。solver 的内部几何不能自动把它变成新的 positional encoding；当前 final-query PPL 改善最多证明“这个 query-specific logit intervention 改变了输出”。

这给所有 phase 路线增加一个硬性 GO 条件：必须定义可复用的 repaired K、position map 或训练时规则，使同一修复能约束至少两个后续 Query/decode token，并显著优于逐 Query、逐 interaction、同 support/同 $b_j$ 的 scalar-bias control。否则一律归类为 **logit calibration / causal probe，而不是新 PE**。本文下面两个候选刻意不声称修改 PE；它们只研究 support selection。

因此，后续方法必须满足：

- 不使用 suppression gap 作为“正确证据”标签；
- 不使用答案方向、gold margin 或其梯度；
- 不修改 position、phase、frequency 或原始 K/V；
- 只改变 sparse support 的形成方式，选中后仍按原位置 native post-RoPE score 消费；
- 与 exact-pre Top-2% 严格匹配 token、block 和 QK 预算；
- 必须分别解决 **冲突证据辨别** 或 **多跳证据闭合**，而不是再提高一个通用 importance score。

---

## 2. 候选一：Query-Constraint Coverage（QCC）

### 2.1 核心问题

现有 exact-pre 只用最终 Query 向量给每个 token 独立打分。它容易把“Xiaoming + age”的 gold 和 conflict 一起视为相关，却丢掉 Query 中的限定条件，例如 `according to the school register`。QCC 不预测哪个事实是真的，而是要求一个候选 span **同时覆盖 Query 中多个可观察约束**。

这不是新的位置编码，也不是把 pre/post 分数再混合一次。它把 sparse KV retrieval 从单向量 Top-K 改成一个无监督的、合取式 span 选择问题。

### 2.2 可部署输入信号

只使用推理时已有信息：

- 用户 Query 的 token 边界；
- Query suffix 中最多 $m=4$ 个高区分度 token 的 pre-RoPE Query 向量；
- prefix 中每个候选 sentence/block 的 pre-RoPE Keys；
- token 频次和标点边界；
- 选中 support 上原生的 post-RoPE QK 与 V。

Query pivot 不使用任务标签。令 prefix block 数为 $M$，Query token $u$ 在 prefix 中的 document frequency 为 $\mathrm{df}(u)$，定义归一化权重：

$$
\widetilde w_u
=
\log\frac{M+1}{\mathrm{df}(u)+1},
\qquad
w_u=\frac{\widetilde w_u}{\sum_{v\in U_m}\widetilde w_v}.
$$

保留权重最高的 $m$ 个非标点 Query token。实现时只需在 prefill 时额外保留很短的 Query suffix hidden states；不需要保存全序列 Query。

### 2.3 精确打分公式

对 Query pivot $u$、候选 block $B$、层 $l$、Query head $h$，定义长度校正后的 pre-RoPE late-interaction score。对 GQA 模型，$g(h)$ 表示 Query head $h$ 对应的 KV head：

$$
x_{uB}^{lh}
=
\tau_b
\log\sum_{j\in B}
\exp\left(
\frac{(q_{u}^{lh})^\top k_j^{l,g(h)}}
{\tau_b\sqrt d}
\right)
-\tau_b\log |B|.
$$

先在每个 $(u,l,h)$ 内对所有 blocks 做 robust normalization：

$$
z_{uB}^{lh}
=
\frac{x_{uB}^{lh}-\operatorname{median}_{B'}x_{uB'}^{lh}}
{\operatorname{MAD}_{B'}x_{uB'}^{lh}+\epsilon}.
$$

head/layer 聚合只取 trimmed mean，避免一个异常 head 决定结果：

$$
z_{uB}=\operatorname{TrimMean}_{l,h}(z_{uB}^{lh}).
$$

最终 block 分数使用 soft-min，而不是普通平均：

$$
\boxed{
C(B)
=
-\tau_c
\log
\sum_{u\in U_m}
w_u\exp\left(-\frac{z_{uB}}{\tau_c}\right)
}
$$

只要一个高权重 Query 约束没有被 block 覆盖，$C(B)$ 就会明显下降。按 $C(B)/|B|$ 做固定预算 knapsack，随后在入选 blocks 内恢复原始 token 位置，并完全使用 native post-RoPE attention：

$$
s_j^{\mathrm{consume}}
=
\frac{(R_tq_t^{l,h})^\top(R_jk_j^{l,g(h)})}{\sqrt d},
\qquad j\in S_{\mathrm{QCC}}.
$$

不入选 block 不参与 sparse softmax；所有入选 K/V 均不修改。

### 2.4 与最近工作的实质差异

| 工作 | 其核心 | QCC 的实质差异 |
|---|---|---|
| SALS | 一个 RoPE-free 低秩 final-query score 提议 Top-K | QCC 的贡献若成立，在于**多个 Query 约束的合取覆盖和 block-set objective**，不是 pre-RoPE proposal。 |
| FASA | 少量主导 RoPE 频块近似单个 QK score | QCC 不选择频带、不做离线 head 校准，并且评分对象是 Query-token × block 的覆盖矩阵。 |
| MoICE | 同一 KV 的多个 RoPE angle attention mixture | QCC 只有原生一个 position/phase，既不复制 KV，也不混合 attention distributions。 |
| VATP | attention × Value norm 的 token importance | QCC 不读取 Value；它评价 block 是否同时满足可见 Query 约束。 |
| CriticalKV | projected Value 导致的最坏输出扰动界 | QCC 不估计删除扰动或输出误差，也不以 projected Value 排序。 |
| LaProx | attention × projected Value 的跨 head/layer矩阵近似 | QCC 是集合覆盖，不拟合原 attention output。 |
| LOCOS | 将 OV write 投到候选答案 unembedding 方向 | QCC 不知道候选答案，不访问 unembedding，不使用答案方向。 |

**剩余 novelty 风险：** multi-vector late interaction、set cover 和 source-aware retrieval 在通用 IR/RAG 中已有大量先例。只有当“冻结 LLM 内部 Query constraints + sparse KV block support + 原生 RoPE consumption”在冲突和多跳任务上产生不可被普通 score mean/max 复现的效果时，它才值得形成论文方法。正式投入前必须补一次 ColBERT、multi-vector RAG、set-cover retrieval 的专项审计。

此外，[Multi-Token Attention (MTA, COLM 2025)](https://arxiv.org/abs/2504.00927) 已直接指出 single-query/single-key attention 无法可靠定位同时包含多个约束的片段，并用跨 query、key 和 head 的卷积实现多 token 条件化。它是 QCC 最直接的结构先例。因此，QCC 当前只能作为 **冻结模型、训练免费、RoPE-free sparse proposal** 的可部署 probe；若 soft-min/set-coverage 相对 MTA-style mean/max 邻居没有独立收益，就不能作为核心创新。

### 2.5 首个可证伪实验

不先做大规模 benchmark，只跑一个矩阵：

- 模型：Qwen3-8B；8 seeds；8K、16K、32K；
- 数据：当前 school-register gold/conflict safety set，加现有两跳 clean/conflict set；
- budget：与 exact-pre Top-2% 严格相同；block 长度和最终 token 数同时匹配；
- baseline：exact-pre final-query Top-2%、Query-pivot score mean、Query-pivot score max、BM25/lexical block coverage；
- primary metrics：gold-vs-conflict block AUROC、conflict admission rate、两跳 Both、Gold NLL/PPL、首 token Acc；
- safety：短程/local-order set 的 PPL 和准确率不得下降。

**Stop rule：满足任一项立即停止 QCC。**

1. gold-vs-conflict block AUROC 的 seed 均值低于 0.65，或 95% CI 不能高于 0.5；
2. 在 16K 和 32K 中，paired ΔNLL 相对 exact-pre 的 95% CI 均不能低于 0，或任一长度 Acc 下降；
3. 普通 Query-pivot mean/max 与 soft-min 的效果相同，说明“合取覆盖”没有独立作用；
4. BM25/lexical coverage 达到同等效果，说明贡献只是 source word matching；
5. 相同预算下在线 selector 时间超过 exact-pre 的 4 倍，且没有可验证的 block-index 加速路径。

---

## 3. 候选二：K→V→Q Relay Retrieval（KVQ-R）

### 3.1 核心问题

两跳证据不是两个独立的高分 token。第一条证据的 Value 应把桥接实体写进 Query residual，第二条证据才会在后续层变得可检索。固定的 per-token Top-K 无法直接表达这种依赖。

KVQ-R 不问“这个 Value 对最终答案 logit 有多重要”，而问：

> 如果候选 $i$ 的 Value 被当前 head 写入，它是否会把下一层 Query 朝候选 $j$ 的 Key 方向推进？

因此它把 KV cache 看成一个由模型自身算子定义的、有方向的证据接力图。最终仍只选择 support，不修改 attention、position、K 或 V。

### 3.2 可部署输入信号

只使用冻结模型在推理时可得的量：

- 当前 final-query residual $h_l$；
- 候选的 pre-RoPE $k_{li}$ 和 $v_{li}$；
- 当前 head 的 $W_O^{l,h}$；
- 下一层 RMSNorm、MLP/residual block 和 $W_Q^{l+1,h'}$；
- exact-pre 产生的至多 64 个候选 block，以及每个 block 内按 exact-pre 分数得到的无标签聚合权重。

不使用 gold span、answer token、unembedding direction 或 loss gradient。只在四个按网络深度固定的 relay 层 $\mathcal L=\{\lfloor L/4\rfloor,\lfloor L/2\rfloor,\lfloor3L/4\rfloor,L-2\}$ 上计算，不按数据或标签挑层。局部转移可以用冻结 block 的两次前向差分实现；若后续需要加速，可换成对同一函数的 batched forward-mode JVP。JVP 的目标是下一层 Query，不是答案或 loss。**不允许给 Top-4% 中的每个 token 单独跑一次 block 前向**：在线版本必须先压成至多 64 个 block 代表，再计算 relay。

### 3.3 精确打分公式

先用普通 exact-pre score 建立每层候选 block 池 $\mathcal C_l$。对 block $B$ 定义每个 head 的原始分数与跨 head 聚合分数：

$$
r_l^h(B)
=
\tau_r\log\sum_{i\in B}
\exp\frac{(q_l^h)^\top k_{li}^{g(h)}}{\tau_r\sqrt d}
-\tau_r\log|B|,
\qquad
r_l(B)
=
\operatorname{RobustNorm}
\left[\operatorname{Agg}_h r_l^h(B)\right].
$$

用同一组 pre-RoPE 分数在 block 内产生归一化、答案无关的 token 权重 $\pi_{li}^h$ 和 head 权重 $\gamma_{lB}^h$，形成一个 block 代表 write：

$$
\pi_{li}^h
=
\frac{\exp\left((q_l^h)^\top k_{li}^{g(h)}/(\tau_v\sqrt d)\right)}
{\sum_{r\in B}\exp\left((q_l^h)^\top k_{lr}^{g(h)}/(\tau_v\sqrt d)\right)},
\qquad
\gamma_{lB}^h
=
\frac{\exp(r_l^h(B)/\tau_h)}
{\sum_g\exp(r_l^g(B)/\tau_h)},
$$

$$
\delta h_{lB}
=
\sum_h\gamma_{lB}^h
W_O^{l,h}\sum_{i\in B}\pi_{li}^h v_{li}^{g(h)}.
$$

令 $F_l(h_l,\xi)$ 表示：在原生 attention residual write 之上额外加入扰动 $\xi$，再经过该层剩余的 residual/MLP 和下一层 normalization，得到下一层 Query 的冻结映射。因此，$F_l(h_l,0)$ 是原生前向基线，并不是“移除 attention”的零基线。使用全局固定的小幅度 $\eta$，计算答案无关的 block-to-Query relay：

$$
\Delta q_{l\rightarrow l+1}^{h'}(B)
=
\frac{
F_l^{Q,h'}(h_l,\eta\delta h_{lB})
-F_l^{Q,h'}(h_l,0)
}{\eta}.
$$

当 $\eta\rightarrow0$ 时，它等价于冻结 block 对 $\delta h_{lB}$ 的方向导数，但不需要任何输出目标。源 block $B$ 对下一跳 block $C$ 的有向 relay score 为：

$$
e_l(B\rightarrow C)
=
\operatorname{RobustNorm}
\left[
\operatorname{Agg}_{h'}
\left(
\tau_e\log\sum_{j\in C}
\exp\frac{
(\Delta q_{l\rightarrow l+1}^{h'}(B))^\top k_{l+1,j}^{g(h')}
}{\tau_e\sqrt d}
-\tau_e\log|C|
\right)
\right].
$$

两跳路径分数为：

$$
\boxed{
P_l(B,C)
=
r_l(B)
+\beta e_l(B\rightarrow C)
+r_{l+1}(C)
}
$$

在固定 token budget 下选择最高分路径端点的 union；再在入选 block 内按原 exact-pre token score 截取到恰好 2%：

$$
S_{\mathrm{KVQ}}
=
\operatorname{TopTokens}_{2\%}
\left(
\bigcup_{(B,C)\in\operatorname{TopPaths}(P)}
(B\cup C)
\right).
$$

最后仅在 $S_{\mathrm{KVQ}}$ 上使用各 token 原位置的 native post-RoPE score 和原始 V。路径只决定 admission，不改变消费几何。

### 3.4 与最近工作的实质差异

| 工作 | 其核心 | KVQ-R 的实质差异 |
|---|---|---|
| SALS | 低秩 pre-RoPE 单 token proposal | KVQ-R 需要一个**有向 candidate-pair edge**；第一条证据的 V 用来预测下一层 Query 会检索哪条证据。 |
| FASA | 主导频率近似当前 QK | KVQ-R 不选择 RoPE frequency，也不近似当前 QK；它估计跨层 K→V→Q→K relay。 |
| MoICE | 多个 RoPE angle 的同 KV attention mixture | KVQ-R 没有 phase expert 或 attention mixture；不同节点是不同证据，而不是同一 KV 的相位副本。 |
| VATP | attention × ‖V‖ 的标量重要性 | KVQ-R 不使用 Value norm；它保留 $W_{OV}$ 的方向，并测试该方向与**另一个 Key**的兼容性。 |
| CriticalKV | token 删除造成的最坏输出扰动界 | KVQ-R 不界定当前输出误差，也不以“保留原输出”为目标；它构造下一跳检索边。 |
| LaProx | 用 attention/projected Value 近似 layer output，含跨 head/layer依赖 | KVQ-R 不做矩阵重构；其对象是有语义方向的 evidence path，而不是 output approximation error。 |
| LOCOS | $\alpha W_{OV}$ 投影到答案 unembedding | KVQ-R 投影到下一层 Query space，再与另一个 Key 匹配；完全不需要答案候选或 unembedding。 |

**剩余 novelty 风险：** induction heads、associative memory、attention circuits、GraphRAG 和 multi-hop retriever 可能包含相邻的“Value 触发下一次检索”解释。可守点只能是：在冻结 decoder 的真实 KV cache 中，直接用 $W_{OV}\rightarrow W_Q$ 的无答案 relay operator 做 sparse support closure，并通过 shuffled-Value 和 matched-path intervention 证明它不是普通相关性或 Value norm。该方向正式扩展前也需要专项查新。

[How Do LLMs Perform Two-Hop Reasoning in Context?](https://arxiv.org/abs/2502.13913) 已经展示了从前置/桥接概念到答案的逐层 sequential-query mechanism。因此，KVQ-R 不能声称首次发现“前一跳写入会改变后一跳 Query”；其可守贡献只能是把该机制变成 **冻结真实 LLM KV cache 上、无答案监督的可计算 relay edge，并用于严格定额 sparse support closure**。

### 3.5 首个可证伪实验

先只做现有受控两跳数据，不做 64K：

- 模型：Qwen3-8B；8 seeds；8K、16K、32K；
- 候选池：exact-pre Top-4% token proposal，经固定边界合并并压成至多 64 个 block；最终消费 support 严格压回 2%；
- baseline：exact-pre Top-2%、扩大 proposal 后按单 token score 截回 2%、随机 pair、key-key similarity path；
- 关键 controls：同层同 head 内打乱 V、匹配 ‖V‖ 的随机 V、反转 $i\rightarrow j$、去掉 $W_O$ 或 $W_Q$、一跳数据；
- evaluation-only labels：真实两跳 edge AUROC、第一/第二证据 recall、Both；labels 不参与打分；
- end metrics：Gold NLL/PPL、首 token Acc、selector/QKV 额外耗时。

**Stop rule：满足任一项立即停止 KVQ-R。**

1. 真实 hop edge 对 matched distractor 的 seed-level AUROC 均值低于 0.65，或 95% CI 不能高于 0.5；
2. 打乱 V 后仍保留 80% 以上的 edge AUROC 或 PPL 收益，说明所谓 relay 实际只是 K/Q 相似度；
3. 在 16K 和 32K 中，paired ΔNLL 相对 exact-pre 的 95% CI 均不能低于 0，或 Acc 下降；
4. 一跳任务获得与两跳任务同样的相对收益，说明方法不是 chain closure，而是 generic Value perturbation；
5. 在至多 64 个 block 代表上，relay 计算超过 full final-query attention 时间的 20%，或实现仍需按 token/候选逐个前向。

---

## 4. 为什么不列第三个方法

下列三个直觉都没有通过本轮严格筛选：

1. **pre/post 或多 phase 一致性 gate：** 很容易退化为 FASA、SALS、MoICE 或 generic phase marginalization；当前 safety 数据也显示 suppression discrepancy 不能区分 gold/conflict。
2. **attention × Value / projected Value / output influence：** VATP、CriticalKV、LaProx、LOCOS 已占据主要结构空间；当前 oracle value smoke 自身也没有闭合。
3. **entropy、margin 或 selector disagreement 动态扩预算：** 可以作为系统安全 fallback，但本质上是通用 uncertainty routing。除非先证明一个新的、可校准的 long-context failure certificate，否则不足以单独支撑论文方法。

严格结论是：**先验证 QCC 是否解决“正确限定条件”，再验证 KVQ-R 是否解决“证据链闭合”。二者都失败时，应停止继续搜索轻量 RoPE repair，并把论文定位收缩为机制诊断 + exact-pre retrieval 的实证研究。**

---

## 5. 建议执行顺序

1. **先做 QCC。** 它实现最小、直接利用当前 safety 数据，而且一次实验即可判断 Query 的 source/entity/relation 约束能否区分 gold 与 plausible conflict。
2. **QCC 通过 safety 后，再做 KVQ-R 小规模 edge probe。** 先只保存 true/matched edge 分布，不做完整 replay；AUROC 达标后才接 sparse attention。
3. 任何方法都先过上述 stop rule，再扩到 64K、第二模型和自然 benchmark。不要先写大型 runner，也不要在单 seed PPL 改善上继续堆参数。
4. 若将来重启 phase 方法，第一项实验必须是“可复用 repair 对多个后续 Query”对“逐 Query matched scalar bias”；两者无差异时立即停止 positional-method claim。

## 6. 依据文件

- [`rope_method_search_report.md`](rope_method_search_report.md)
- [`literature_novelty_audit.md`](literature_novelty_audit.md)
- [`exact_pre_mpr_vs_exact_pre/summary.md`](exact_pre_mpr_vs_exact_pre/summary.md)
- [`strict_mpr_frozen_smoke_gpu6/README.md`](strict_mpr_frozen_smoke_gpu6/README.md)
- [`safety_gqa_64k_smoke_gpu7/certificate_aurocs.csv`](safety_gqa_64k_smoke_gpu7/certificate_aurocs.csv)
- [`safety_gqa_64k_smoke_gpu7/intervention_summary.csv`](safety_gqa_64k_smoke_gpu7/intervention_summary.csv)
- [`value_mediated_smoke_gpu6/first_order_prediction_summary.csv`](value_mediated_smoke_gpu6/first_order_prediction_summary.csv)
- [`value_mediated_smoke_gpu6/value_sample_summary.csv`](value_mediated_smoke_gpu6/value_sample_summary.csv)
