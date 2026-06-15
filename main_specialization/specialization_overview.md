# Specialization Overview

## 总体目标

我们的核心目标是实现 **MoE 模型中的 expert specialization**：不同 expert 应当对输入 token / context 中不同类型的 feature 形成稳定、可解释、可复用的分工。

如果 expert bucket 能够对应稳定 feature，那么它不仅可能提升预测与端侧部署，也可能服务 **KV cache reverse indexing**：decode 时当前 token 的 routing bucket 可以作为索引，优先检索历史上属于相同或相关 expert bucket 的 KV，从而减少无关 KV 的访问。

详细目标见 [00\_goal.md](./00_goal.md)。

## 1. 如何定义 Specialization

核心判断：feature 不应只被定义成 token id，也不应只停留在数据生成规则或 next-token distribution 的物理先验上；最新一轮需要进一步从模型参数空间中定义 feature。

Round3 的新假设是：模型参数矩阵的谱分解方向可能对应模型真正学到的 feature。这个定义更接近模型内部机制，但当前仍需要回答它如何与 Round1 / Round2 中的数据侧物理先验对应。

当前文件：

1. [Round3: Parameter-Space Feature Definition Draft](./01_round3_feature_definition_draft.md)

## 2. 为什么现在的 MoE 没做到 Specialization

核心判断：标准 MoE 有多个 expert，但 ordinary hidden-router 并不会自然形成我们希望的 feature-level specialization。它更容易学到 token id、target token、frequency、representation cluster 或其他局部 shortcut。

### 2.1 Baseline MoE 到底按什么分发

最新结论是：baseline MoE 没有自然按 distributional feature 分发。当 token id、input prefix、output token 与 distributional feature 解耦后，baseline gate 仍然更接近 surface-token / local shortcut，而不是硬的 distributional specialization。

当前文件：

1. [Round2 Baseline Routing](./02_round2_baseline_routing.md)

### 2.2 理想 Specialization 应当长什么样

核心判断：理想 specialization 不只是“同 token 同 expert”。在 synthetic 数据上，它应当对齐生成规则定义的 feature；在 reused-token 数据上，它应当对齐 future-distribution feature；在真实数据上，它应当让 next-token logits 分布相似的 token / context 更倾向于进入相同或相近 expert bucket。

当前文件：

1. [Round2 Ideal Specialization](./02_round2_ideal_specialization.md)

### 2.3 模型结构与训练范式如何影响 Specialization

核心判断：ground-truth feature routing 能提升学习效率，但普通训练目标和普通 load-balance 并不足以自动产生 feature specialization。真实数据上，load-balance 更主要改变 expert usage；残差路径通常帮助 gate 读出 feature；attention 当前只弱地捕捉 feature relation，还没有形成非常干净的 feature-internal attention structure。

当前文件：

1. [Round2 Training Effect](./02_round2_training_effect.md)

## 3. 什么结构能做到 Specialization

核心判断：在 controlled reused-token data 上，specialization 不是靠 ordinary hidden-router 自然出现的，而更依赖 router input 是否接近 attention retrieval feature，以及 expert input 是否保留完整预测信息。

### 3.1 当前结构结论

1. `k` 仍然是最强 router input；
2. `k/head` 对 different-input-same-output 更强，但 `full-k` 对 A/B 两类 feature 更平衡；
3. expert 输入仍应优先使用 full token vector / `attention_output + residual`；
4. head/head 结构能提高 routing-attention 对齐，但会损伤 NTP；
5. 后续 regularization 应从 local/high slot 对齐升级为 distributional feature 对齐。

当前文件：

1. [Round2 Architecture Conclusion](./03_round2_architecture_conclusion.md)

### 3.2 候选技术方案

后续可行结构可以按五个互相独立但需要组合设计的维度组织：

1. [Router Input](./03_candidate_router_input.md)
2. [Router Input Shape](./03_candidate_router_shape.md)
3. [Expert Input](./03_candidate_expert_input.md)
4. [Expert Input Shape](./03_candidate_expert_shape.md)
5. [Regularization](./03_candidate_regularization.md)

## 4. Scale、宽度与长尾学习

核心判断：频率不均匀会让高频 feature 主导训练梯度方向，导致低频 feature 的 loss 和 output margin 落后；加宽模型对低频 feature 的改善显著大于高频 feature。这个 tail 落后很大程度可由 loss reweight 或恢复 uniform 数据续训缓解，因此当前更支持“训练动力学主导 + 宽度缓解 long-tail 瓶颈”，而不是“纯不可逆表征污染”。

当前文件：

1. [Round4 Frequency-Width Conclusion](./04_round4_frequency_width_conclusion.md)
2. [Round4 Distribution Evidence](./04_round4_frequency_width_distribution.md)
3. [Round4 Gradient Dynamics](./04_round4_frequency_width_gradient_dynamics.md)
4. [Round4 Intervention Evidence](./04_round4_frequency_width_interventions.md)
5. [Round4 Linear Theory](./04_round4_frequency_width_linear_theory.md)

## 5. 评价指标

当前评价指标集中在三个层次：NTP 能力、feature selectivity、deployability。

详细定义见 [04\_metrics.md](./04_metrics.md)。
