# 第四轮：Baseline MoE 到底按什么分发，以及如何构造可用于 Reverse Indexing 的 Routing Objective

## 这一轮要回答的问题

前三轮实验已经说明：

1. Attention 已经能捕捉 hierarchy feature，尤其 higher-level slot 更接近模型真正使用的 retrieval bucket。
2. 直接按照 ground-truth higher-level slot 做 routing 能提升 next-token prediction，说明 feature-based expert bucket 是合理目标。
3. Baseline MoE 没有自然形成 ground-truth slot specialization。
4. Supervised gate 即使能达到很高 routing accuracy，也不能自动复现 ground-truth dispatch 的收益。
5. Naive inhibition loss 目前主要表现为让 routing 变 sharp 甚至 collapse，并不能证明它形成了有意义的 feature specialization。

因此，Round 4 暂时不继续围绕 naive inhibition 展开，而是聚焦两个问题：

1. **Baseline MoE 没有按 ground-truth slot 分发，那它到底在按什么分发？**
2. **能否用 attention-derived signal 训练一个 pre-attention routing，使 expert bucket 对齐 attention retrieval bucket？**

这两个问题分别对应 specialization 和 reverse indexing。前者解释现有 MoE 的自然 routing 机制；后者尝试构造一个真正能在 attention 前产生、并可用于 KV reverse index 的 routing signal。

## 问题 1：Baseline MoE 到底按什么分发

**目标：找出 baseline gate 的真实分发依据，而不是只证明它没有按 ground-truth slot 分。**

现有结论只说明 baseline MoE 与 local slot / higher-level slot 对齐不强，但这并不等于 routing 是随机的。它可能在按其他变量分发，例如 token id、slot 内位置、边界位置、频率、target token 或 attention cluster。

下一步应把 expert assignment 与一组 candidate features 做统一相关性分析。

### Candidate Features

需要测试的候选分发依据包括：

1. local slot；
2. higher-level slot；
3. token id；
4. slot 内位置；
5. local slot boundary / higher-level slot boundary；
6. next-token target；
7. feature frequency / Zipf rank；
8. attention top-mass cluster；
9. hidden representation / attention-output representation 的聚类结果。

其中前两项是我们希望模型按其分发的 ground-truth hierarchy feature；后面几项用于回答 baseline MoE 是否学到了其他更容易、更局部或更偏统计的 routing rule。

### Metrics

每个 candidate feature 都用同一组指标评估：

1. **feature-to-expert purity**：同一个 feature 是否主要被送到同一个 expert；
2. **expert-to-feature purity**：同一个 expert 是否主要接收少数 feature；
3. **same-feature same-expert rate**：同 feature 的 token 是否进入同 expert；
4. **MI / NMI**：feature id 与 expert id 的互信息，避免 collapse 导致的假高 purity；
5. **expert load entropy**：判断 routing 是否退化为单 expert 或少数 expert collapse；
6. **per-layer routing pattern**：判断不同层是否在按不同 feature 分发。

解释时必须同时看 purity、MI/NMI 和 load entropy。单独的 same-feature same-expert rate 不可靠，因为所有 token 进同一个 expert 时它也会接近 1。

### 预期输出

这个诊断实验最终应回答：

```text
baseline MoE 不是按 ground-truth slot 分；
它更接近按 A / B / C 分；
这种分发是否有稳定 layer pattern；
这种分发是否对 NTP 有用；
这种分发为什么不能直接作为 KV reverse index。
```

如果所有 candidate features 都无法解释 expert assignment，说明 baseline MoE 的 routing 更接近训练噪声或不稳定分工；如果某些 candidate feature 能解释 routing，则下一步需要判断它们是否可能服务于 reverse indexing。

## 问题 2：如何用 Attention-derived Signal 构造 Routing Objective

**目标：让 routing bucket 对齐模型实际使用的 attention retrieval bucket，并且这个 routing signal 必须能在 attention 前产生。**

Round 2 已经说明，attention 本身能捕捉有用 retrieval structure；只保留 same higher-level slot 的 KV 时，NTP 几乎不下降。Round 3 也说明，直接用 `v` 近似 gate input 的 early proxy 失败，原因是 `v` 与真正 attention 后 gate input 差距较大。

因此，新的方向不是事后用 gate 去解释 attention，而是在训练时显式让 pre-attention routing 学会预测 attention retrieval bucket。

### Attention Mass Coverage

不使用“top 75% token 数量”作为 label，因为 token 数量不能反映 attention 的真实贡献。应使用 attention mass coverage。

对每个 query token `i`，在可见历史 token 中按照 attention score 从高到低排序，取最小集合 `S_i`，使其累计 attention mass 达到阈值：

```text
sum_{j in S_i} attention(i, j) >= rho
rho = 0.75
```

`S_i` 就是 query token 的 attention-derived positive set。它表示模型当前真正依赖的 retrieval bucket，而不是人为指定的 ground-truth slot。

### Routing Objective

训练目标不是简单要求 `S_i` 中所有 token 进入同一个 expert。这样可能再次导致 expert collapse。更合适的目标是：用 `S_i` 中历史 token 的 routing 结果构造 teacher，再约束 query token 的 pre-attention routing 对齐这个 teacher，同时加入 anti-collapse regularization。

当前可以测试三种 teacher / loss 形式。三种方案都应支持 top-k expert routing，而不是只支持 top-1。

#### 方案 1：Routing Distribution KL

先把 router logits 归一化成 routing distribution：

```text
p_i = softmax(router_logits_i)
```

然后用 attention-derived positive set `S_i` 中历史 token 的 routing distribution 加权平均，作为 query token `i` 的 teacher：

```text
teacher_i = sum_{j in S_i} attention(i, j) * stop_grad(p_j)
loss_i = KL(teacher_i || p_i)
```

这个方案的优点是实现简单、梯度稳定，能让 query token 的 routing distribution 接近它高 attention token 的平均 routing distribution。缺点是它是 soft alignment：如果 teacher 本身不 sharp，它不一定强制 query token 与 high-attention token 进入同一个 expert。

#### 方案 2：Pairwise Same-expert Loss

直接最大化 query token 与 high-attention token 被分到同一 expert 的概率。对 top-k routing，可以把 routing distribution 看成 soft expert membership：

```text
same_prob(i, j) = dot(p_i, stop_grad(p_j))
loss_i = - sum_{j in S_i} attention(i, j) * log same_prob(i, j)
```

这个方案比 KL 更直接，因为它优化的就是 “query token 与 high-attention history token 进入同一 expert bucket” 的概率。缺点是它更容易出现 trivial solution：所有 token 都进入同一个 expert。因此必须配合 anti-collapse / load-balance 约束。

#### 方案 3：Top-k Logits Teacher

这个方案更接近实际 top-k dispatch。对 `S_i` 中每个历史 token `j`，先取它 raw router logits 的 top-k，只保留 top-k expert 的 logits，其余 expert 置为无效值或零权重：

```text
raw_logits_j = [0, 5, 3, 2], k = 2
topk_logits_j = [0, 5, 3, 0]
```

然后对 `S_i` 中所有历史 token 的 top-k logits 做 attention 加权平均：

```text
teacher_logits_i = sum_{j in S_i} attention(i, j) * stop_grad(topk_logits_j)
```

再对 `teacher_logits_i` 取 top-k，并 softmax 成 teacher distribution：

```text
teacher_i = softmax(topk(teacher_logits_i, k))
loss_i = KL(teacher_i || p_i)
```

这个方案是合理的，因为它保留了历史 token 的实际 top-k dispatch 偏好，而不是把所有 expert 的低置信度概率都混进 teacher。它比方案 1 更贴近真实 MoE routing，也比方案 2 更容易表达 top-k expert set。

需要注意两点：

1. raw logits 有尺度问题，因此不能直接用 MSE 对齐 raw logits；必须先经过 top-k 截断和 softmax，最终仍然在 probability distribution 上计算 loss。
2. 如果 top-k logits teacher 过于 sharp，也可能推动 collapse，因此仍然需要 anti-collapse / expert usage regularization。

### Anti-collapse Regularization

上述三种方案都存在 collapse 风险。最终训练 loss 至少应包含：

```text
loss = lm_loss
     + lambda_route * attention_derived_routing_loss
     + lambda_balance * anti_collapse_loss
```

其中 `anti_collapse_loss` 至少需要约束 expert usage 的有效熵，避免所有 token 被送到同一个 expert。解释实验结果时必须同时报告 same-feature same-expert rate、MI/NMI、expert load entropy 和 effective expert count。

这个 objective 的目标不是让 routing 复刻 ground-truth label，而是让 routing bucket 逼近 attention retrieval bucket。若 attention retrieval bucket 与 hierarchy feature 对齐，那么 routing 也会间接学到 hierarchy feature。

### Pre-attention Routing Input

为了让 routing 能用于同层 KV reverse indexing，routing input 不能依赖 attention output。候选输入包括：

1. layer input / hidden before attention；
2. `v`；
3. `O(v)`；
4. `residual + O(v)`。

其中 `v` proxy 已经作为失败 baseline：它不能可靠近似真实 gate input。因此后续重点应测试显式训练后的 pre-attention router 是否能比 naive `v` proxy 更好地预测 attention-derived retrieval bucket。

### Reverse Indexing 连接

如果 pre-attention routing 能预测 attention retrieval bucket，那么 decode 时可以：

```text
prefill: 为历史 token 记录 pre-attention routing bucket
decode: 用当前 token 的 pre-attention routing bucket 查询历史 KV
attention: 只访问同 bucket 或 top bucket 匹配的 KV
```

这才是可部署的同层 reverse indexing 路径。它不依赖 attention 后的真实 gate input，因此可以在 attention 计算前减少 KV 访问。

## 需要验证的内容

1. **Baseline routing diagnostic**
   分析 baseline expert assignment 与所有 candidate features 的关系，确定 baseline MoE 实际按什么分发。

2. **Attention-derived bucket quality**
   验证 attention mass coverage 得到的 `S_i` 是否稳定、是否与 higher-level slot 对齐、是否保留 NTP 所需信息。

3. **Pre-attention routing learning**
   用 attention-derived positive set 训练 pre-attention router，测试它能否预测 `S_i` 或其 bucket structure。

4. **Anti-collapse 检查**
   所有 routing objective 都必须报告 expert load entropy、MI/NMI 和 effective expert count，防止 naive inhibition 式 collapse。

5. **Reverse indexing eval**
   用 learned pre-attention routing 过滤 KV，测试 KV 保留比例、NTP loss / accuracy，以及相对 full attention 的下降。

## 已完成的新结构实验

### 实验 A：Attention-derived routing loss

**目的：验证能否用 attention top-mass positive set 训练 pre-router，使 routing bucket 对齐 attention retrieval bucket。**

这一组实验保留原始 dense attention，只额外训练 pre-router。测试了三种 attention-derived loss：Routing Distribution KL、Pairwise Same-expert Loss、Top-k Logits Teacher。

实验结果显示，这三种 loss 都能维持正常 NTP，但没有形成有意义的 feature specialization。主要问题是 expert collapse：

| loss | NTP acc | routed-attention acc | routing 现象 |
|---|---:|---:|---|
| KL | 93.96% | 92.04% | 有明显 collapse，后两层 effective experts 接近 1 |
| Pairwise | 93.96% | 93.46% | 后两层完全 collapse |
| Top-k logits | 93.98% | 93.98% | 三层完全 collapse |

因此，attention-derived loss 本身不足以学出可用于 reverse indexing 的 routing bucket。尤其是 top-k logits 虽然 routed acc 看起来不掉，但原因是所有 token 进同一个 expert，KV mask 实际退化为 dense attention。

### 实验 B：Attention-derived routing loss + soft entropy floor

**目的：验证轻量 anti-collapse 约束能否阻止 expert collapse，同时保留 attention-derived routing 的语义。**

这一组实验在三种 attention-derived loss 上增加 soft expert usage entropy floor：

```text
anti_collapse = relu(alpha * log(num_experts) - entropy(mean_router_probs))
alpha = 0.5
weight = 0.01
```

实验结果显示，soft entropy floor 只能轻微缓解 collapse，不能真正改变 hard top-1 routing：

| loss | NTP acc | routed-attention acc | effective experts | 结论 |
|---|---:|---:|---|---|
| KL + entropy floor | 93.96% | 91.27% | `[1.40, 1.06, 1.02]` | routed 性能更差 |
| Pairwise + entropy floor | 93.94% | 93.25% | `[1.29, 1.09, 1.09]` | 稍微缓解 collapse，但仍不干净 |
| Top-k logits + entropy floor | 93.96% | 93.91% | `[1.00, 1.14, 1.14]` | 仍基本 collapse |

关键解释是：entropy floor 约束的是 soft routing probability 的平均熵，而实际 dispatch 看 hard top-1 expert。模型可以让 soft probability 满足熵阈值，但 argmax 仍然几乎落到同一个 expert。因此这类软负载约束不能作为主线解决方案。

### 实验 C：Pre-router cluster attention + tied MoE routing

**目的：验证一个更结构化的 inverse KV 方案：pre-router 同时控制 attention 可见 KV bucket 和 MoE expert dispatch。**

结构如下：

1. 每层 attention 前，用 layer input 计算 pre-router logits；
2. 每个 token 被分到 top-k cluster，当前实验取 `topk=1`；
3. 第 `j` 个 token 只 attend 到历史中与自己 cluster 相同的 token；
4. MoE 使用同一份 pre-router logits dispatch，cluster 与 expert 一一对应；
5. position id 仍按原始序列位置计算，不重排 token。

这一结构不再把 routed attention 当成 eval-only mask，而是训练时就使用 routed attention。因此它直接测试 “routing 能否作为 KV reverse index”。

#### C.1 只有 next-token prediction loss

不加任何额外 routing loss 或 load-balance loss，只用 NTP 训练。

| 指标 | 结果 |
|---|---|
| NTP acc | 93.87% |
| NTP loss | 0.2108 |
| 历史 KV 保留比例 | 26.2% ~ 31.6% |
| 历史 KV 压缩比 | 3.17x ~ 3.82x |
| effective experts | 3.56 ~ 3.89 |
| max expert fraction | 30.4% ~ 47.0% |

结论：这个结构是目前最有价值的正结果。模型在训练时真的只能访问 routed KV，仍然能达到接近 baseline 的 NTP accuracy，并且实现了约 3x-4x 的历史 KV 压缩。它没有 collapse，说明 pre-router cluster 可以作为一种可训练的 KV reverse index。

但它学到的 bucket 并不是干净的 ground-truth hierarchy bucket：

| layer | local MI | local feature-to-expert purity | local same-feature same-expert | higher MI |
|---|---:|---:|---:|---:|
| 0 | 0.437 | 0.540 | 0.390 | 0.090 |
| 1 | 0.324 | 0.570 | 0.397 | 0.075 |
| 2 | 0.289 | 0.506 | 0.342 | 0.065 |

也就是说，routing 与 local slot 存在非随机对齐，但对齐不干净：每个 local slot 大约 50% ~ 57% token 落在主 expert；任意两个同 local-slot token 落到同 expert 的概率约 34% ~ 40%。higher-level slot 对齐更弱。

#### C.2 加标准 load-balance loss

在同一结构上增加 `moe_load_balance_loss_weight = 0.01`。

| 指标 | 结果 |
|---|---|
| NTP acc | 93.66% |
| NTP loss | 0.2174 |
| 历史 KV 保留比例 | 24.6% ~ 24.8% |
| 历史 KV 压缩比 | 4.03x ~ 4.07x |
| effective experts | 约 4.0 |
| max expert fraction | 约 25.5% ~ 25.8% |

结论：load balance 让 expert 使用几乎完全均匀，并带来更稳定的 4x KV 压缩，但 NTP 下降，feature selectivity 也下降。它优化了负载形状，而不是优化 routing 语义。

| layer | local MI | local feature-to-expert purity | higher MI |
|---|---:|---:|---:|
| 0 | 0.319 | 0.459 | 0.020 |
| 1 | 0.274 | 0.474 | 0.030 |
| 2 | 0.220 | 0.452 | 0.030 |

因此标准 load balance 不应作为主线目标。它能防止 collapse，但会削弱 feature specialization。

### 实验 D：新结构中的 attention 是否仍集中到 local / higher slot

**目的：检查 pre-router cluster attention 是否迫使 attention score 继续集中到 ground-truth hierarchy feature。**

这里必须区分两个口径：

1. `include_self_mass`：包含当前 token 自己，数值通常较高，因为当前 token 必然属于自己的 local / higher slot；
2. `history_mass`：只看历史 token，更能反映 attention 是否真的检索同 slot / same higher-level slot 的历史信息。

在 `ntp-only` 新结构下：

| layer | local history mass | local baseline | higher history mass | higher baseline |
|---|---:|---:|---:|---:|
| 0 | 12.2% | 9.3% | 29.7% | 29.1% |
| 1 | 12.4% | 9.3% | 30.9% | 29.1% |
| 2 | 9.1% | 9.3% | 24.7% | 29.1% |

结论：attention 对 local slot 有轻微 lift，但并不强；higher-level slot 基本接近随机可见比例，第三层甚至低于 baseline。这说明新结构能学到可用 routing bucket，但 attention 并没有自动、稳定地集中到 ground-truth local / higher slot。

在 `lb0.01` 新结构下：

| layer | local history mass | local baseline | higher history mass | higher baseline |
|---|---:|---:|---:|---:|
| 0 | 6.6% | 9.3% | 17.2% | 29.1% |
| 1 | 12.1% | 9.3% | 37.1% | 29.1% |
| 2 | 11.0% | 9.3% | 37.8% | 29.1% |

load-balance 版本在 layer 1/2 对 higher-level slot 的 history mass 更高，但 layer 0 更差，而且 NTP 与 feature selectivity 下降。因此它不能简单视为更优结构。

### 当前结论

Round 4 到目前为止最重要的结论是：

1. **attention-derived loss 失败**：只用 attention top-mass signal 训练 pre-router，会导致 collapse 或接近 collapse，不能形成可用 specialization。
2. **soft entropy floor 不够**：它约束 soft probability，不足以改变 hard top-1 dispatch。
3. **pre-router cluster attention 是当前最有价值的新结构**：训练时就使用 routed attention，仍能保持约 93.9% NTP acc，并实现约 3x-4x 历史 KV 压缩。
4. **当前 routing bucket 有用，但不是干净的 hierarchy bucket**：routing 与 local slot 有非随机对齐，但对 higher-level slot 对齐弱；attention history mass 也没有稳定集中在 ground-truth hierarchy slot。
5. **标准 load balance 不应作为主线**：它让负载均匀、压缩比稳定，但削弱 NTP 与 feature selectivity。

下一步研究重点应从 “如何防 collapse” 转为 “如何让 pre-router cluster 更语义化”。也就是说，要保留 pre-router-controlled attention 这个结构，同时研究更合适的 routing inductive bias 或 objective，让 bucket 不只是可用，而是更接近 hierarchy feature。

## 当前预期结论形式

这一轮最终需要给出三类结论：

1. **Baseline MoE 到底按什么分发：** 它是否按 token、位置、边界、频率、target、attention cluster 或 representation cluster 分发。
2. **Attention-derived routing 是否可学：** pre-attention router 是否能预测 attention retrieval bucket，且不发生 expert collapse。
3. **这对 inverse KV 是否有帮助：** 如果 pre-attention routing 能保留足够 NTP accuracy 并减少 KV 访问，它就是比当前 gate / `v` proxy 更合理的 reverse indexing signal。
