# RoPE 长程检索：方法搜索最终判定

**模型：** Qwen3-8B  
**日期：** 2026-08-01  
**实验设备：** 远程物理 GPU 6–7；未使用 GPU 0–5 或本机 GPU  
**判定口径：** 只有通过预注册审计、独立 seed、匹配预算和关键 control 的结果才可支持方法主张。

## 1. 最终决定

### 1.1 不建议把当前 SAGE-RoPE 写成“新的位置编码”

当前最强的有效组件仍是：

> 用 pre-RoPE 内容分数提出远程候选，再用候选原位置的 native post-RoPE 分数和原始 V 完成稀疏 attention。

它在受控两跳数据上明显优于 full attention 和 exact post-RoPE Top-2%，但 [SALS](https://arxiv.org/abs/2510.24273) 已直接覆盖“RoPE-free proposal + native RoPE sparse consumption”，[LongHeads](https://arxiv.org/abs/2402.10685) 与 [InfLLM](https://arxiv.org/abs/2402.04617) 也覆盖局部窗口、远程语义检索和稀疏消费的系统结构。因此，这一结果适合作为强 intervention/baseline，不足以单独支撑新 PE 的核心 novelty。

### 1.2 当前新 PE / phase repair 路线为 NO-GO

这不是因为某一个参数没调好，而是同时遇到三类硬约束：

1. **功能等价性：** 固定 support 和 V 时，针对当前 Query 的任意逐 token phase repair，都精确等价于加入相同的 attention-logit bias。
2. **物理可实现性：** Qwen3-8B 有 32 个 Q heads、8 个 KV heads；四个 Q heads 共享一个 cached K。逐 Q-head 选择不同 K 旋转一般无法由一份可复用 KV cache 实现。
3. **实验安全性：** phase suppression 不能区分 gold 与 plausible conflict；提高 QK、recall 或 attention mass 也没有稳定转化成更好的 PPL/accuracy。

### 1.3 最值得投稿的方向是机制论文，条件性 GO

可守的论文中心是：

> 在真实 8B RoPE 模型中，从受控失败边界出发，完整连接“二维频带相位变化 → 证据 QK → softmax mass → Value/residual 写入 → 后续 Query 漂移 → 输出 margin”，并区分位置相位、分母竞争与内容状态分叉。

现有第一层解析重构、逐层 BF16 更新重放、failure-boundary 扫描和 activation patching 已构成较强基础；8K/16K/24K 各 8 seeds 的局部高精度因果复核也已经完成。还需要第二模型、自然/公开任务、原生窗口内的完整中介消融，才能成为完整 ICLR 论文。

## 2. 所有候选路线的判定矩阵

| 路线 | 最关键结果 | 方法判定 |
|---|---|---|
| 远程 hard NoPE / 距离封顶 / 静态缩放 | 跨长度不稳定；多个变体显著恶化 PPL | **NO-GO**；且已有大量先例 |
| Exact pre-RoPE Top-2% proposal + native consumer | 16K/32K 留出数据显著优于 exact post-RoPE Top-2% | **有效 baseline**；novelty 不足 |
| 广泛/块级 phase transport | recall/mass 可升高，但 PPL 与准确率恶化 | **NO-GO** |
| Native Phase Envelope rollback | 8K/32K/64K 均无稳定收益，64K 明显退化 | **NO-GO** |
| 严格 token-sparse MPR | 单 seed top-1 有 PPL 改善；mass-preserve 取消收益；target 与 random score-effect 不匹配 | **NO-GO as PE**；仅是 score-lift probe |
| Suppression certificate gate | gold-vs-conflict AUROC 约 0.49–0.51；LOSO 跨长度不稳定 | **NO-GO as safety gate** |
| Query-span multi-vector proposal | token-max PPL 2.805；block PPL 154.317，均不优于 exact-pre 1.301 | **NO-GO** |
| Value-mediated singleton closure | BF16+同路径 no-op 在 8K/16K/24K 各 8 seeds 复现；证据坐标 Pearson 0.960/0.973/0.936 | **局部机制 GO**；非部署 selector，完整中介链未闭合 |
| KVQ-R directed relay edge | 两例 FD audit 均失败；pre-score pair 均为 1.0，reverse edge 与/高于 KVQ-R | **NO-GO**；没有独立方向性 |
| Prefix-frozen relay index（PFRI） | 核心 KVQ-R edge 未通过 | **停止实现** |

## 3. 四个决定性实验

### 3.1 Suppression 不是“正确性证书”

定义某 token 的 pre/post-RoPE suppression：

$$
c_j=s_j^{\mathrm{pre}}-s_j^{\mathrm{post}}.
$$

8 seeds、8K/32K/64K、真实 GQA 映射的 gold-vs-conflict AUROC 为：

| 长度 | Pre-suppression AUROC | 95% CI | Grid-envelope AUROC | 95% CI |
|---:|---:|---:|---:|---:|
| 8K | 0.505 | [0.492, 0.518] | 0.504 | [0.487, 0.520] |
| 32K | 0.493 | [0.482, 0.504] | 0.493 | [0.480, 0.505] |
| 64K | 0.499 | [0.488, 0.509] | 0.499 | [0.486, 0.511] |

固定等权、无标签标准化的 leave-one-seed-out line 组合也不稳定：`all_sampled` 的 gold 胜率在 8K/32K/64K 分别为 62.5%/50.0%/37.5%。因此，RoPE 可以同样压低真实证据、合理冲突和无关 token；只凭 suppression 大小不能决定应该增强谁。

### 3.2 多 Query 语义聚合会扩大“相关但错误”的匹配

去掉固定 sink 后，在 8K、4 seeds、相同 2% budget 下：

| 方法 | Gold PPL | Acc | Gold-conflict margin | Gold recall | Conflict recall |
|---|---:|---:|---:|---:|---:|
| Native | 1.262 | 75% | 4.344 | — | — |
| Exact final-query pre-RoPE | 1.301 | 75% | 4.281 | 39.37% | 39.48% |
| Query-span token-max | 2.805 | 75% | 3.031 | 36.14% | 40.68% |
| Query-span block | 154.317 | 50% | -1.250 | 42.29% | 60.61% |

block 版在 164 个总槽位中先保留 128 local + 1 current，只剩 35 个远程槽位，小于 64-token block。它因而退化成“押中 gold block 或 conflict block”的离散选择；两个 seeds 完全丢失 gold 并产生万级 PPL。token-max 没有块截断，但仍不能恢复 Query 中的来源/限定条件。

### 3.3 Value 接力边没有超出普通 relevance

KVQ-R 试图用源 block 的 V，经 $W_O$ 和下一层 $W_Q$，预测下一跳 Key。两个 8K seeds 均召回两条 gold evidence，prefix KV 的完整 SHA-256 也保持不变，但全部 case 都未通过 finite-difference audit。即使仅描述性查看无效 case：

| Seed | KVQ-R | Reverse edge | Pre-score pair |
|---:|---:|---:|---:|
| 0 | 1.000 | 0.969 | **1.000** |
| 1 | 0.750 | **0.813** | **1.000** |

高分可由“两个端点各自与 final Query 都相关”完整解释，不需要有向 $K\rightarrow V\rightarrow Q\rightarrow K$ 接力。因此继续实现 PFRI 没有证据基础。

### 3.4 高精度 singleton 支持局部 Value→margin 因果闭合

旧 NF4 singleton smoke 中 target 和 random 都出现约 `+0.23` margin 漂移；这来自 instrumented 与 replay 执行路径不一致，不能引用。新的非量化 BF16 v2 为每个 case 加入完全同路径 `epsilon=0` no-op；8/8 seeds 的 no-op 相对 instrumented margin 漂移都精确为 0，prefix cache 与单坐标回放审计也全部通过。

在 8 个独立 seeds、32 个 target singletons 上：

| 计划 | n | Pearson | seed-cluster 95% CI | Spearman | seed-cluster 95% CI | Sign accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Target | 32 | **0.833** | **[0.732, 0.932]** | **0.795** | **[0.667, 0.888]** | **78.1%** |
| Matched random | 32 | 0.416 | [-0.003, 0.654] | 0.003 | [-0.163, 0.328] | 50.0% |

只看 gold + conflict 的 16 个证据坐标，Pearson 为 **0.960**（seed-cluster 95% CI `[0.938, 0.993]`）、Spearman 为 **0.947**（`[0.849, 1.000]`）、符号正确率为 **15/16**，回归斜率为 `1.075`（`[0.931, 1.209]`）、截距为 `0.003`（`[-0.014, 0.032]`）。分类后，gold evidence 的 Pearson/Spearman 为 0.987/1.000，平均实际 $\Delta$margin 为 `+0.0750`；conflict evidence 为 0.968/0.952，平均实际 $\Delta$margin 为 `-0.0410`，但两类都有个别 seed 方向反转。

更关键的是，按 seed 聚合后 gold 与 conflict 的 attention probability 差和 suppression gap 差的 95% CI 都跨 0；而平均 $\partial m/\partial s$ 的 gold−conflict 差为 `1.248e-3`，95% CI `[0.826e-3, 1.702e-3]`。即 attention/phase suppression 不区分真假，但 Value/残差路径对答案 margin 的**有向效用**可以区分。相反，suppression gap 单独预测 actual $\Delta$margin 的 Pearson/Spearman 只有 0.091/0.143，baseline attention probability 也只有 0.212/0.078。该结果验证了一个局部环节：单个 attention score 的小幅变化，经 Value/residual 通道对输出 margin 的作用可由同路径一阶导数预测；“被 RoPE 压低”或“原 attention 高”本身都不是有益修复的充分条件。

16K 的 8-seed 独立复核进一步增强了结论：16 个 gold/conflict target 的 Pearson 为 **0.973**（95% CI `[0.946, 0.988]`）、Spearman 为 **0.976**（`[0.910, 1.000]`）、符号正确率为 15/16；matched-random evidence 的 Pearson/Spearman 只有 0.051/-0.047。24K 再次复现：Pearson **0.936**（`[0.849, 0.972]`）、Spearman **0.900**（`[0.757, 0.991]`）、符号 13/16；matched-random evidence 为 -0.226/-0.024。Gold−conflict 的 $\partial m/\partial s$ 差在 16K/24K 分别为 `1.284e-3` 与 `1.617e-3`，置信区间均不跨 0，而 attention probability 与 suppression gap 差在三个长度均跨 0。32K BF16 在两张 24GB 3090 上均于 native eager GQA `repeat_kv` 处 OOM，未产生任何可引用结果。

它仍是 answer-gradient oracle diagnostic：target ranking 使用正确答案 margin 梯度，正式统计只有 Qwen3-8B 的 8K/16K/24K，并且 instrumented 与 native BF16 路径的最大 baseline margin 差随长度达到 `0.464`，虽然各长度 8/8 top-1 决策不变。因此它增强的是机制论文，而不是 phase repair 或检索方法；也尚未闭合 phase、mass、Value、later-Query 各环节的中介比例。target/random 也不是严格的性能基线：证据 target 都是高敏感度坐标，而 random 不是；两者没有显著 PPL/accuracy 改善差异。

## 4. 为什么逐 Query phase repair 不能直接叫新 PE

设 native score、attention 和输出为：

$$
s_j=\frac{q^\top R(\Delta_j)k_j}{\sqrt d},
\qquad
a=\operatorname{softmax}(s),
\qquad
o=\sum_j a_jv_j.
$$

若当前 Query 为第 $j$ 个交互选择相位改变量 $\delta_j$：

$$
s'_j=\frac{q^\top R(\Delta_j+\delta_j)k_j}{\sqrt d}.
$$

定义：

$$
b_j=s'_j-s_j.
$$

则严格有：

$$
o'
=
\sum_j\operatorname{softmax}(s')_jv_j
=
\sum_j\operatorname{softmax}(s+b)_jv_j.
$$

所以，只要 repair：

- 针对当前 Query 单独计算；
- 不改变 V；
- 不被未来多个 Query 复用；

它就与逐 token logit bias 完全不可区分。要声称新的位置编码，必须让修复后的几何在 Query 到来前就固定，能被多个后续 Query 重用，并在 GQA 中按 KV group 一致。当前 MPR 不满足这些条件。

### 4.1 唯一形式可行的 semantic-phase escape hatch 仍不适合作为当前主方法

若 prefix token 只由自身 hidden state 产生、按 KV group 共享的 Key phase $\beta_{g}(h_p)$，Query 只由自身产生逐 Q-head phase $\alpha_h(h_q)$，则远程相位

$$
\phi_h(q,p)=\beta_{g(h)}(h_p)-\alpha_h(h_q)
$$

满足单份 KV cache、GQA 与多 Query 复用。更一般地，在“单份 cached Key + Query-side rotation + 每个二维平面只允许旋转”的约束下，可实现的 pairwise phase 必须具有这种可分离形式；任意非可分离 repair 都必须重算 Key、缓存多份 basis，或退化为 pairwise logit edit。

但它仍不适合成为当前论文的主方法：其公式与 content-dependent semantic position 的 [RePo](https://arxiv.org/abs/2512.14391) 高度接近，并叠加了 [AdaRoPE](https://arxiv.org/abs/2607.19363) 式逐频率/GQA 约束与 local/remote gate；若同时缓存 native 与 semantic Key，KV 开销显著增加；而当前实验又表明 phase 只能改 attention 权重，无法仅凭 suppression 区分有益 gold Value 与有害 conflict Value。最合理的处理是把“single-cache phase realizability”作为理论边界，把该模块留作 0.5B–1B proof-of-concept 或后续方法论文，而不是现在用两张 24GB GPU 强行训练成 8B/64K 主结果。

## 5. 文献边界

以下宽泛贡献均已有直接先例，不能再作为论文主 claim：

- 短程 RoPE、长程 NoPE：[RoPE-to-NoPE](https://arxiv.org/abs/2501.18795)、[Periodic RoPE](https://arxiv.org/abs/2605.27980)；
- pre-RoPE/RoPE-free proposal 后 native sparse attention：[SALS](https://arxiv.org/abs/2510.24273)；
- query/head-specific 多相位路由：[MoICE](https://arxiv.org/abs/2406.19598)；
- 逐 head/频率且 GQA-aware 的 RoPE 参数：[AdaRoPE](https://arxiv.org/abs/2607.19363)；
- 语义决定位置：[RePo](https://arxiv.org/abs/2512.14391)；
- 可跨 Query/布局复用的 semantic KV：[SemPIC](https://arxiv.org/abs/2607.28069)；
- 局部窗口 + 逐 head 远程语义块 + 稀疏 attention：[LongHeads](https://arxiv.org/abs/2402.10685)、[InfLLM](https://arxiv.org/abs/2402.04617)；
- 随机位置/相位训练：[PoSE](https://openreview.net/forum?id=3Z1gxuAQrA)、[Randomized YaRN](https://arxiv.org/abs/2606.23687)、[Shuffle the Context](https://arxiv.org/abs/2604.14339)。

因此，再做一个远程削弱、双通道、随机 phase augmentation 或静态 head gate，创新风险都很高。

## 6. 推荐的 ICLR 2027 论文重构

### 6.1 标题与中心问题

建议放弃“我们提出一个成熟的新 RoPE”的叙事，改为：

> **When Position Overrules Evidence: Tracing RoPE-Induced Retrieval Failures Through Transformer Depth**

核心问题：相同的远程证据为何会在只改变距离或 filler 后，从可用变成不可用？

### 6.2 可守的贡献

1. **受控失败边界：** 对每个 token 长度点定位成功/失败翻转，而不是只比较稀疏的 8K/32K 曲线。
2. **第一层精确分解：** 用 64 个二维频率对解析重构固定 pre-RoPE Q/K 的 post-RoPE 分数。
3. **跨层因果链：** 区分相位项、softmax 分母、Value 写入和后续 pre-RoPE Query 漂移，并用 no-op-matched intervention 与 activation patching 复核。
4. **部署/可辨识性定理：** 证明 query-specific phase repair 与 score bias 的等价性，以及 GQA + multi-query cache reuse 对新 PE 的约束。
5. **分析导出的 intervention baseline：** 将 exact pre-RoPE proposal + native consumer 作为“问题可修复”的证据，但明确不把其结构写成首创。

### 6.3 当前不能声称

- 首次发现 RoPE 会损害长程检索；
- 首次提出局部位置、远程语义；
- 首次使用 pre-RoPE 检索；
- SAGE-RoPE 是成熟、可部署或已证明加速的方法；
- attention mass、evidence recall 或 QK 提升必然改善答案；
- suppression 大的 token 就是真实证据；
- 当前单一模型、合成数据结果可以泛化到所有 LLM。

## 7. 投稿前最小补实验

按优先级排序：

1. **扩大高精度因果闭环：** 非量化 BF16、同路径 epsilon=0 no-op 的 8-seed/8K、16K、24K 均已稳定复现；32K 因 24GB 显存不足未完成。下一步先实现并验证不物化重复 KV heads 的等价 grouped-GQA control，再做 32K、第二模型及 phase/logit/mass/Value/residual 的完整中介消融。
2. **原生窗口内复现：** 在 8K/16K/32K 的多个成功→失败边界上做 activation patching 与 layerwise mediation，避免只依赖 143K 扩展位置。
3. **跨模型：** 至少 Qwen 与 Llama/Mistral 两个家族；至少一个原生长上下文 checkpoint。
4. **反平衡数据：** 独立随机化证据位置、冲突值、record 顺序、block 边界、filler 类型和 Query 模板。
5. **公开任务：** RULER + NoLiMa/自然多跳 + 一个局部顺序/代码依赖任务。
6. **最近邻 baseline：** SALS、FASA、LongHeads/InfLLM-style、exact pre/post Top-2%；严格匹配 token budget、local/sink quota 和扫描成本。
7. **系统边界：** 若仍报告稀疏方法，必须实现近似索引并报告真实 latency、gather bytes、KV cache 与吞吐量；否则只称 attention-rule intervention。

若第 1–4 项不能稳定复现，应把论文降为内部分析，不建议投 ICLR。若能复现，机制论文无需强行再造一个新 PE，也可能比当前增量方法更有说服力。

## 8. 最终一句话

**当前“新 RoPE 方法”不成熟，继续调 inference-time phase repair 的预期收益很低；8K/16K/24K 已复现的局部因果链使机制论文达到条件性 GO，真正有论文价值的是完整失败机制，以及对 query-specific repair、GQA 和可复用 KV 几何的严格可实现性边界。**

## 9. 主要证据文件

- `safety_gqa_formal_8seed/README.md`
- `suppression_certificate_block_aggregation_README.md`
- `strict_mpr_token_sparse_smoke_gpu7/README.md`
- `value_mediated_singleton_smoke_gpu6/README.md`
- `value_mediated_singleton_v2_bf16_noop_8seed_gpu67/README.md`
- `value_mediated_singleton_v2_bf16_noop_16k_smoke_gpu67/README.md`
- `value_mediated_singleton_v2_bf16_noop_16k_8seed_gpu67/README.md`
- `value_mediated_singleton_v2_bf16_noop_24k_8seed_gpu67/README.md`
- `value_mediated_singleton_v2_bf16_noop_32k_smoke_gpu67/README.md`
- `queryspan_prerope_sink0_smoke_gpu67/README.md`
- `kvq_relay_edge_smoke_2seed_merged/README.md`
- `method_equivalence_and_identifiability.md`
- `pe_novelty_frontier_20260801.md`
- `reusable_gqa_pe_candidate_20260801.md`
