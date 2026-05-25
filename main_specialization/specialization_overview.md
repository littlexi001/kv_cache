# Specialization Overview

我们的核心目标是实现 **MoE 模型中的 expert specialization**：不同 expert 应当对输入 token / context 中不同类型的 feature 形成稳定、可解释、可复用的分工。如果模型能够形成稳定的 feature-level expert specialization，它可能带来多个重要下游性质：

1. 预测与端侧部署；
2. KV cache reverse indexing；
3. 持续学习与遗忘控制。

其中，KV cache reverse indexing 是当前最直接的系统动机：如果 expert bucket 能够对应某类 feature，那么 decode 时当前 token 的 routing bucket 就可以作为索引，优先检索历史上属于相同或相关 expert bucket 的 KV，从而减少无关 KV 的访问。更长期看，如果 expert bucket 对应稳定 feature，也可能支持更可控的端侧部署、expert swap、持续学习和遗忘控制。

## 1. 如何定义 Specialization

要研究 specialization，首先需要定义什么叫“相同 feature”。

### Round1: Naive Synthetic Data

第一轮 synthetic 数据采用最干净的层次化构造：我们预先定义一个 token pool，再由若干 token 构成 local slot，由若干 local slot 构成 higher-level slot。不同 local slot 之间不共享 token，不同 higher-level slot 之间也不共享 local slot。

在这个设定下，每个 token 本身可以被视为一个独立 feature；同一 local slot 内的 token 共享一个 local-level feature；同一 higher-level slot 内的 local slots 共享一个更高层次的 compositional feature。因此，feature 的 ground truth 完全由数据生成规则给出。

对应地，specialization 可以被形式化为：对于属于同一个 ground-truth feature group 的 token / position，MoE gate 应当把它们分发到相同或高度重叠的 expert bucket；对于属于不同 feature group 的 token / position，MoE gate 应当产生可区分的 expert assignment。

这一设定的优点是 ground truth 清晰，可以直接计算 local slot / higher-level slot 与 expert assignment 的一致性，例如 feature-to-expert purity、same-feature same-expert rate、MI / NMI 等指标。它的缺点也很明确：它过于干净，弱化了真实语言中一词多义、同义改写和上下文依赖带来的歧义。

### Round2: Synthetic Data with Reused Tokens

第二轮 synthetic 数据放宽第一轮的“完全不重复”假设，允许 token 在不同 local slot 或不同 higher-level slot 中重复出现。这样做的动机是让 synthetic 数据更接近真实语言：真实语料中同一个 token 可以在不同上下文中表达不同含义，不同 token 也可以在相近上下文中表达相似功能。

例如，如果数据中同时存在：

```text
ABC
ABD
```

那么 `C` 和 `D` 可能代表同一语义模式下的两个可替换 token，也可能对应两个不同的 next-token behavior。又例如，如果数据中同时存在：

```text
ABC
EBF
```

那么 token `B` 是否应当被分到同一个 expert，取决于我们认为 feature 是由 token identity 决定，还是由上下文中的 local slot / sequence pattern 决定。

在这一轮设定中，我们更关心的是：**当 token identity 与 slot-level feature 不再一一对应时，MoE gate 到底会按什么分发。** 一种朴素但重要的假设是，理想 specialization 不应只按 token id 分发，而应更接近数据生成时定义的 local slot 或 higher-level slot；也就是说，expert bucket 应当捕捉 context-dependent feature，而不是只捕捉 surface token identity。

这个设定也自然连接到 real corpus。对真实语料而言，我们通常无法提前知道某个 token position 的 ground-truth feature label，因此可以用 downstream behavior 来定义 feature：如果两个 token position 的 next-token logits 分布相似，说明模型认为它们应当被映射到相近的新状态，因此它们可以被视为具有相似 feature。

更形式化地说，对每个 token position，可以取模型在该位置的 next-token logits 或概率分布作为语义状态表示。如果两个位置的预测分布接近，它们应当具有相近的 downstream behavior；理想的 expert specialization 应当让这些位置更倾向于进入相同或相近 expert bucket。

因此，Round1 / Round2 synthetic 与 real corpus 的 feature 定义并不是两套互不相干的定义。它们的共同核心是：**feature 不只是 token id，而是决定模型后续行为的可复用结构。** Synthetic 数据给出可控的 ground truth，real corpus 则需要通过 next-token logits、representation similarity 或 attention retrieval pattern 等 proxy 来近似这一结构。

## 2. 为什么现在的 MoE 没做到 Specialization

当前的关键问题是：标准 MoE 模型虽然有多个 expert，但这些 expert 是否真的形成了 feature-level specialization 并不明确。已有 synthetic 实验显示，baseline MoE 并没有自然按照 ground-truth local slot / higher-level slot 分发；相反，它更容易学到 token id、target token 或其他局部 shortcut。

### 2.1. Baseline MoE 到底按什么分发？HRJ，synthetic & real。

部分 synthetic 结论见 [Round4 问题1：Baseline MoE 到底按什么分发](../fdong/inverse_kv_round4_plan.md#问题-1baseline-moe-到底按什么分发)。
这一问题的目标是诊断现有 MoE routing 的真实规律，而不是只证明它没有达到我们的预期。

#### Round1: clean synthetic

已有结论是：baseline MoE 没有自然按照 ground-truth local slot / higher-level slot 分发，而是更接近 token id、target token 或局部 shortcut。

需要保留的核心证据包括：

1. 同一个 token id 大约有 70% 会被分到同一个 expert；
2. 同一个 expert 内部的表征向量 cosine similarity 更高，约为 0.5；
3. 同一个 expert 内部的 next-token logits cosine similarity 只有约 0.1，说明 token-id / representation shortcut 强于真正的 downstream-behavior feature；
4. synthetic 数据中，gating 不主要由表征空间 SVD 头部子空间解释，而更依赖 5% 到 20% 的中间子空间。这可能说明 clean synthetic 里的 common feature 还不够接近真实语料中的 high-frequency feature。

#### Round2: reused-token synthetic

Round2 需要专门回答：当 token id 与 slot-level feature 解耦后，baseline gate 仍然按 token id 分发，还是开始按 context-dependent feature 分发。

建议后续补充以下结论：

1. **same-token different-slot same-expert rate：** 同一个 token 出现在不同 local / higher-level slot 时，是否仍然被分到同一个 expert；
2. **different-token same-slot same-expert rate：** 不同 token 属于同一个 local / higher-level slot 时，是否会被分到同一个 expert；
3. **token-id NMI / local-slot NMI / high-slot NMI / target-token NMI：** 直接比较 routing 更像哪一种 feature label；
4. **conditional purity：** 固定 token id 后，expert 是否还能区分不同 slot；固定 slot 后，expert 是否能忽略 token id 差异；
5. **next-token logits similarity vs same-expert rate：** 判断同 expert token 是否真的具有相似 downstream behavior。

#### Real corpus

真实数据中没有显式 ground-truth slot label，因此需要用 proxy feature 解释 gating。当前结论是：真实数据里的 gating 更容易被表征空间的 SVD 头部子空间解释，top 5% 子空间可以解释约 90% 的分发结果。

后续需要继续判断：真实 MoE routing 更接近 token identity、representation cluster、next-token logits cluster、frequency / Zipf rank、position shortcut，还是 attention retrieval bucket。


### 2.3. 按我们的理想 specialization 定义，gating 结果应当有何特征？CAR：real。

#### Real data

理想 specialization 定义：分到同一 expert 的 token，其 next-token logits 分布应当相似。

1. 同一个 expert 中的 token / context，其表征 cossim 高：～0.97；
2. 不同 expert 中的 token / context，表征 cossim 低，～0.20；
3. 也有反例，表征 cossim 高，但一起学的效果很差：反例比例约～10%：
4. 线性分发无法支持 ground truth feature 分发

#### Reused token synthetic


### 2.4. 模型结构与训练范式如何影响 specialization。ZX：real。

1. **Load-balance loss：** 它能显著提升 expert usage 的均匀性，但不一定直接带来 feature specialization。依据是：加入 load-balance 以后，effective expert count 明显上升，说明流量分布被显著拉平；但与此同时，routing 与 feature 对齐相关的指标变化很小，expert purity 也没有同步提升。也就是说，load-balance loss 主要改变的是“token 是否更平均地分到各个 expert”，而不是“expert 是否更清楚地按 feature 分工”。
2. **残差链接：** 残差在这里更像是在帮助 gate，而不是干扰 gate。依据是：当 gate 使用标准的 residual-plus-normalized 表征时，feature 更容易被线性读出，最终 routing 与 feature 的对齐也更好；而当 gate 只看 pure attention output 时，这两个结果都会下降。也就是说，在当前 ordinary MoE 设定里，残差路径并没有明显削弱 gate 对 feature 的识别，相反，它更可能给 gate 提供了一个更容易利用的输入表示。
3. **Attention：** 没有证据表明存在某个 head 会让 token 几乎只在同一 feature 内部 attend。依据是：在本地 `qwen3-0.6B` 的正式 attention 分析里，用整层 attention output 做 feature probe，最好的 layer 也只有 `0.0688`，最好的单头 probe 只有 `0.0656`，整体绝对值仍然偏低；同时，直接看 attention pattern 时，表现最强的 head 对同 feature token 的偏好仍然很弱，而且这种偏好只在大约 `21%` 的位置上出现，远达不到“某个 head 基本只在同一 feature 内 attend”的程度。也就是说，当前 attention 最多只能说明它弱地捕捉到了一部分 feature relation，还不能说明它已经形成了清晰而强的 feature-internal attention structure。

## 3. 什么结构能做到 Specialization（DF & LYM：synthetic，LET：real）

基于上述对现有 MoE 为何没实现 specialization 的理解，我们针对性提出方案实现 specialization。
尤其对于 KV cache reverse indexing，routing signal 必须尽可能在 attention 前产生，否则它无法在同一层 attention 计算前减少 KV 访问。

### 结论：

在 synthetic 数据上

1. gate input representation:
   1. query/key vector 最好：NTP 能达到最优的同时，分发结果更逼近我们定义的 specialization：purity \~97%。
   2. layer input 其次：specialization purity \~80%。
2. gate granularity：
   1. head-level gating 优于 full token gating；
   2. SVD-based hierarchical gating 实现困难：SVD 结果不稳定：受采样数据的影响、受参数变化的影响。
3. expert input representation：full token vector 显著好于去掉 residual 的 attention output。NTP acc 约为 94% vs. 90%。
4. regularization：
   1. 合成数据具体实现：让 attention score 相关性高的 token，它们的 router logits cossim 尽可能大。
   2. 真实数据上的一种实现：约束输入同一 expert 的 token，其最终输出的 next-token logits 相似性高

### 候选技术方案：

后续可行结构可以按五个互相独立但需要组合设计的维度来组织：router input、router input shape、expert input、expert input shape 和 regularization。

#### 1. Router Input: 用什么表征决定分发

Router input 决定 gate 依据什么信息判断当前 token / context 应进入哪个 expert bucket。

1. **Residual / hidden state：** 使用标准 residual stream 或 layer-normalized hidden state 作为 gate 输入。这是 ordinary MoE 的默认方案，预测能力通常稳定，但容易捕捉 token id、target token 或局部 shortcut。
2. **Attention output without residual：** 使用去掉 residual 的 attention output 做 routing。直觉是减少 residual stream 中 token identity 的 domination，让 gate 更依赖 attention 聚合出的上下文信息。但已有结果显示，pure attention output 往往会伤 NTP，且不一定带来稳定 specialization。
3. **Layer input：** 使用 attention 前的 layer input 做 routing。它更接近 pre-attention routing，部署上更友好，NTP 表现也较稳定，但 specialization 通常不如 `k/head`。
4. **Q / K / V：** 使用 attention projection 后的 query、key 或 value 表征做 routing。其中 `k` 当前最值得关注：它本身就是 attention retrieval 中用于匹配历史 token 的表示，因此更可能与 feature bucket / retrieval bucket 对齐。
5. **Pre-attention routing input：** 为了服务 KV cache reverse indexing，routing signal 最好能在 attention 计算前得到。因此，layer input、q、k、v 是比 attention output 更可部署的候选。
6. **Spectral / SVD representation：** 将 hidden states 投影到 SVD / PCA basis 后，在谱空间中做 routing。它试图让不同频率或不同抽象层次的 feature 落到不同子空间，但当前实现中 SVD basis 对 batch 采样和训练动态敏感，效果不稳定。

#### 2. Router Input Shape: 如何切分用于分发的表征

Router input shape 决定 gate 是对完整 token 表征做一次分发，还是对表征的不同子空间分别分发。

1. **Full-token routing：** 对完整 hidden vector 做一次 routing。实现简单、表达稳定，但容易把多种 feature 混在一个 gate 决策里。
2. **Head-level routing：** 将 router input 按 attention head 切分，每个 head 单独 routing。当前 synthetic 结果显示，head-level routing 通常比 full-token routing 更利于 specialization，尤其是 `k/head`。
3. **True head/head MoE：** router 和 expert input 都按 head 切分。它能增强 feature bucket 与 attention retrieval bucket 的对齐，但会明显伤 NTP，说明只让 expert 看 head 子空间会损失表达能力。
4. **Spectral-band routing：** 将表征按 SVD / PCA 方向切成不同 spectral bands，并对不同 band 分别 routing。它理论上接近 hierarchical / feature-subspace ownership，但当前结果显示稳定性不足，暂时不是主线。

#### 3. Expert Input: expert 应该处理什么表征

Expert input 决定被选中的 expert 实际处理哪一份 token state。它不一定要和 router input 相同。

1. **Full residual token vector：** 让 router 使用更 feature-selective 的输入，但 expert 仍处理完整的 `attention output + residual`。这是当前最稳的方案：它保留完整预测信息，同时允许 gate 在更合适的表征空间中做分发。
2. **Attention output without residual：** 让 expert 只处理 attention 聚合出的信息。它有时能提升 attention bucket 与 expert bucket 的重合度，但 NTP 通常不如 full residual expert input 稳定。
3. **Layer input / q / k / v：** 让 expert 处理更早期或更局部的投影表征。已有 synthetic 结果显示，这类 expert input 往往明显伤 NTP，因此不应作为当前主线。
4. **Same as router input：** router 和 expert 使用同一表征。这个设计更“纯”，但容易把 routing 诊断问题和 expert 表达能力问题混在一起，实验解释更困难。

#### 4. Expert Input Shape: expert 处理完整表征还是子空间

Expert input shape 决定 expert 的 computation ownership 是作用在完整 token state 上，还是只作用在某个子空间上。

1. **Full expert input / output：** expert 输入和输出都是完整 hidden size。当前最推荐的默认方案是 `router=head`，但 `expert=full`：每个 head 可以独立决定 routing，但被选中的 expert 仍处理完整 token state。
2. **Head expert input / output：** 每个 head 的 expert 只处理该 head 子空间。它会带来更强的 specialization inductive bias，但已有结果显示 NTP 明显下降，因此只适合作为 ablation 或机制分析。
3. **Spectral-band expert input / output：** expert 只处理某个 spectral band，并写回对应子空间。这是更强的 feature-subspace ownership，但当前尚未证明可稳定训练。
4. **Hybrid design：** router 使用子空间，expert 使用 full token vector。这是目前最有价值的中间方案：既让 gate 看到更干净的 feature signal，又不牺牲 expert 的完整表达能力。

#### 5. Regularization: 如何让 routing 更硬地对齐 feature

自然训练得到的 specialization 仍然不够强。即使当前最好的 synthetic 结构也只能达到中等强度的 local / high-slot 对齐，因此需要显式 regularization 或 supervision。

1. **Attention-derived routing objective：** 让 attention score 高的 token pair 具有更相似的 router logits 或更高的 expert-overlap。这个方向最直接服务 reverse KV：如果 gate bucket 能预测 attention retrieval bucket，就可以用 routing bucket 做 KV reverse index。
2. **Logits-similarity regularization：** 在 synthetic 数据上，可以约束同 local / high slot 或高 attention mass token 的 router logits cosine similarity 更高，不相关 token 的 router logits 更低。
3. **Next-token-logits regularization：** 在真实数据上，可以约束输入同一 expert 的 token position，其最终 next-token logits 分布更相似。这对应 real-corpus feature 的 downstream behavior 定义。
4. **Load-balance loss：** 保证 expert usage 不 collapse，但它本身不是 specialization objective。它应作为稳定训练的辅助项，而不是核心目标。
5. **Common expert / top-k routing：** common expert 和 top-k 可以提升 NTP，但会让 hard specialization 指标变软。若目标是 reverse KV，top-1 routing 通常更干净；若目标是预测性能，top-k 和 common expert 可能更有价值。
6. **Ground-truth routing / supervised routing：** 在 synthetic 数据上可作为 upper bound 或 diagnostic，验证 feature-based routing 是否本身有收益；但它不是最终可部署方案。


### 评价指标

1. **NTP Acc：** NTP accuracy / loss 不显著变差，最好在困难样本或长程依赖样本上有收益；
2. **Feature selectivity：** expert assignment 与 synthetic ground truth 或真实语料 proxy feature 显著对齐；
3. **Deployability：** routing signal 能够在需要的位置提前产生，并能服务 KV cache reverse indexing 或其他下游系统目标。
