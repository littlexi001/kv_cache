# Specialization Overview

我们的核心目标是实现 **MoE 模型中的 expert specialization**：不同 expert 应当对输入 token / context 中不同类型的 feature 形成稳定、可解释、可复用的分工。如果模型能够形成稳定的 feature-level expert specialization，它可能带来多个重要下游性质：

1. 预测与端侧部署；
2. KV cache reverse indexing；
3. 持续学习与遗忘控制。

其中，KV cache reverse indexing 是当前最直接的系统动机：如果 expert bucket 能够对应某类 feature，那么 decode 时当前 token 的 routing bucket 就可以作为索引，优先检索历史上属于相同或相关 expert bucket 的 KV，从而减少无关 KV 的访问。更长期看，如果 expert bucket 对应稳定 feature，也可能支持更可控的端侧部署、expert swap、持续学习和遗忘控制。

## 1. 如何定义 Specialization

要研究 specialization，首先需要定义什么叫“相同 feature”。

1. **Synthetic 数据：**
   在 synthetic 数据中，feature 是由数据生成规则显式给出的。例如，我们可以构造 hierarchical sequence：若干 token 组成一个 local slot，若干 local slot 再组成 higher-level slot。此时，同一个 local slot 或同一个 higher-level slot 就是 ground-truth feature group。

   在这个设定下，specialization 可以被形式化为：对于属于同一个 ground-truth feature group 的 token，MoE gate 应当把它们分发到相同或高度重叠的 expert；对于属于不同 feature group 的 token，MoE gate 应当产生可区分的 expert assignment。

   这一设定的优点是 ground truth 清晰，可以直接计算 local slot / higher-level slot 与 expert assignment 的一致性，例如 feature-to-expert purity、same-feature same-expert rate、MI / NMI 等指标。
2. **Real corpus：**
   一个可行定义是：如果两个 token 的 next-token logits 分布相似，说明模型认为它们应当被映射到相近的新状态，因此它们可以被视为表达相近 feature。

   更形式化地说，对每个 token position，可以取模型在该位置的 next-token logits 或概率分布作为语义状态表示。如果两个位置的预测分布接近，它们应当具有相近的 downstream behavior；理想的 expert specialization 应当让这些位置更倾向于进入相同或相近 expert bucket。

这两个设定从本质上是相通的：在 synthetic 数据中，属于同一 token 其 next-token logits 分布也是相似的，因此同一 slot 的 token 被分发到一个 expert，则这个 expert 中的 next-token logits 也是有限的。

## 2. 为什么现在的 MoE 没做到 Specialization

当前的关键问题是：标准 MoE 模型虽然有多个 expert，但这些 expert 是否真的形成了 feature-level specialization 并不明确。已有 synthetic 实验显示，baseline MoE 并没有自然按照 ground-truth local slot / higher-level slot 分发；相反，它更容易学到 token id、target token 或其他局部 shortcut。

### 2.1. 现在的 MoE gating 结果有什么特征？HRJ，synthetic & real。

部分 synthetic 结论见 [Round4 问题1：Baseline MoE 到底按什么分发](../fdong/inverse_kv_round4_plan.md#问题-1baseline-moe-到底按什么分发)。
这一问题的目标是诊断现有 MoE routing 的真实规律，而不是只证明它没有达到我们的预期。需要分析：

1. 同一个 expert 接收的 token / context 之间有什么共同点：
   1. Token ID 相同；
   2. 表征向量 cos-sim 更高；
   3. next-token logits 的 cos-sim 更高；
2. 不同 expert 接收的 token / context 之间有什么差异；
   1. Token ID 不同
   2. 表征向量 cos-sim 较低；
   3. next-token logits 的 cos-sim 较低；
3. Gating 是否主要由 token id、target token、token frequency、position、local context、attention bucket 或 representation cluster 解释；
   1. 真实数据：Gating 被表征空间的 SVD 头部子空间（top 5% 能解释 90% 的分发结果）决定；
   2. Synthetic 数据：头部子空间反而不重要，Gating 被表征空间 5%~20% 解释 90% 的分发结果。
4. 上述规律在不同 layer 中稳定存在；

### 2.2. 按我们的理想 specialization 定义，gating 结果有何特征？。CAR：real。

1. 同一个 expert 中的 token / context，其表征 cossim 高：～0.97；
2. 不同 expert 中的 token / context，表征 cossim 低，～0.20；
3. 也有反例，表征 cossim 高，但一起学的效果很差：～10%：
4. 线性分发无法支持 ground truth feature 分发

这个定义很重要，因为 specialization 不是简单追求所有 expert 负载均匀，也不是简单追求 routing sharp。一个 routing 可以非常 sharp 但没有语义；也可以负载非常均匀但破坏 feature locality。真正需要的是既有可用负载形态，又有可解释 feature structure 的 expert bucket。

### 2.3. 模型结构与训练范式如何影响 specialization。ZX：real。

1. **Load-balance loss：** 它能显著提升 expert usage 的均匀性，但不一定直接带来 feature specialization。依据是：加入 load-balance 以后，effective expert count 明显上升，说明流量分布被显著拉平；但与此同时，routing 与 feature 对齐相关的指标变化很小，expert purity 也没有同步提升。也就是说，load-balance loss 主要改变的是“token 是否更平均地分到各个 expert”，而不是“expert 是否更清楚地按 feature 分工”。
2. **残差链接：** 残差在这里更像是在帮助 gate，而不是干扰 gate。依据是：当 gate 使用标准的 residual-plus-normalized 表征时，feature 更容易被线性读出，最终 routing 与 feature 的对齐也更好；而当 gate 只看 pure attention output 时，这两个结果都会下降。也就是说，在当前 ordinary MoE 设定里，残差路径并没有明显削弱 gate 对 feature 的识别，相反，它更可能给 gate 提供了一个更容易利用的输入表示。
3. **Attention：** 如果问题是“attention 有没有学会 ground-truth feature relation”，那当前结果更像是：学到了一点，但没有证据表明存在某个 head 会让 token 几乎只在同一 feature 内部 attend。依据是：在本地 `qwen3-0.6B` 的正式 attention 分析里，用整层 attention output 做 feature probe，最好的 layer 也只有 `0.0688`，最好的单头 probe 只有 `0.0656`，整体绝对值仍然偏低；同时，直接看 attention pattern 时，表现最强的 head 对同 feature token 的偏好仍然很弱，而且这种偏好只在大约 `21%` 的位置上出现，远达不到“某个 head 基本只在同一 feature 内 attend”的程度。也就是说，当前 attention 最多只能说明它弱地捕捉到了一部分 feature relation，还不能说明它已经形成了清晰而强的 feature-internal attention structure。

## 3. 什么结构能做到 Specialization（DF & LYM：synthetic，LET：real）

基于上述对现有 MoE 为何没实现 specialization 的理解，我们针对性提出方案实现 specialization。
尤其对于 KV cache reverse indexing，routing signal 必须尽可能在 attention 前产生，否则它无法在同一层 attention 计算前减少 KV 访问。

### 候选技术方案：

1. **Oracle / ground-truth routing：**
   使用已知 ground-truth feature 直接分发 token，用来验证 feature-based expert bucket 是否本身有价值。这不是最终可部署方案，但可以提供上限和对照。已有 synthetic 结论见 [Round3 问题1：Ground-truth Routing 是否有好处](../fdong/inverse_kv_round3_plan.md#问题-1ground-truth-routing-是否有好处)。
2. **去掉或调整残差：**
   测试 gate 输入中残差信息是否阻碍 specialization。如果 gate 主要读到 token identity 或局部 shortcut，那么改变残差路径可能帮助它更依赖 attention / feature representation。
3. **SD on forward representation：**
   在前向表征上加入显式的 specialization-driving signal，使相似 feature 的 representation 更容易被路由到相同 expert，不同 feature 的 representation 更容易被分开。
4. **Head-level MoE：**
   每个 attention head 使用独立的 MoE routing。这个方向的直觉是，不同 head 可能捕捉不同 feature relation，因此 head-level expert 更可能形成细粒度 specialization。
5. **Hierarchical MoE：**
   使用层次化 expert 结构，让不同 expert bucket 对应不同粒度的 feature，例如 token-level、local-slot-level、higher-slot-level 或更抽象的语义 feature。
6. **Pre-attention routing：**
   为了服务最终的 KV cache reverse indexing，MoE input 不能依赖 attention output。可以考虑使用 layer input、q、k、v 或其他 attention 前可计算的 representation 作为 routing input。这样 decode 时才能在 attention 计算前根据 routing bucket 过滤历史 KV。

### 评价指标

1. **NTP Acc：** NTP accuracy / loss 不显著变差，最好在困难样本或长程依赖样本上有收益；
2. **Feature selectivity：** expert assignment 与 synthetic ground truth 或真实语料 proxy feature 显著对齐；
3. **Deployability：** routing signal 能够在需要的位置提前产生，并能服务 KV cache reverse indexing 或其他下游系统目标。

