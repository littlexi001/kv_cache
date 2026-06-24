# 公共奇异方向的生命周期：理论故事、方法位置与 reweighting 验证

## 文档目的

本文整理当前关于大语言模型训练中 common direction / 大奇异值方向的工作理论。我们关心的不是“所有公共方向都不好”。语言确实需要共享结构、句法模板和可复用语义。真正的问题是：

> 高频 pattern 是否在训练早期过度占据表示、参数和梯度空间，使 residual / long-tail 方向得不到同等效率的优化。

当前理论分成三部分：

1. 高频 token、短语、功能性 pattern 首先产生 common direction。
2. nested 语言结构让这个 common direction 反复出现在大量 token / pattern 的表征中。
3. 大奇异值方向拥有更高局部优化收益，梯度更偏好它；residual 方向被冷落，低频 token 和长尾 pattern 学得慢。

![公共奇异方向的生命周期与干预位置](common_direction_lifecycle_and_interventions.svg)

## 0. 当前结论摘要

目前最强的 toy 实验证据支持以下版本：

```text
高频/shared pattern
→ early gradient mass 不均
→ first common direction 成核
→ nested / shared input 让 common component 广泛进入 hidden states
→ common direction 获得较大 scale / singular gain
→ 早期 update 变低秩，tail/residual loss decrease 很小
→ long-tail 收敛成为整体 bottleneck
```

reweighting 的作用不是简单“最后把谱压平”。它的主要作用是：

```text
从训练最早期削弱 frequency domination
→ early update 更高秩
→ residual / tail 方向更早获得有效 loss decrease
→ common 学得相对慢一点，long-tail 学得更快
→ 因为 bottleneck 原本是 long-tail，所以整体训练路径更健康
```

需要保留的边界：

- 这些结论目前来自 controlled toy experiments，不等价于真实 LLM 机制已被证明。
- nested 结构是否是实际 LLM 中 top singular directions 的主要强化来源仍需 checkpoint 级验证。
- “最终谱更平”不等价于“训练路径更 scale-balanced”。强行 clipping 后验谱没有改善 tail speed，说明路径本身很重要。

## 1. 理论建模故事

### 1.1 高频 pattern 首先产生 common direction

自然语言不是均匀分布。少数 token、短语、功能性结构和句法模板出现频率极高，例如：

- 高频 token：标点、助词、冠词、介词、连接词；
- 高频短语：`of the`、`in the`、`is a`、`there is`；
- 高频功能结构：列表、从句开头、问答模板、固定 syntactic slots；
- 高频 target：很多不同 context 都预测同一个 token；
- 高频 prefix / subpattern：很多长 pattern 共享同一个短结构。

这些 pattern 的共同点是：它们在 early training 中贡献了不成比例的梯度质量。

#### K 作为 target：many-to-one 成核

假设很多不同 context 都预测同一个高频 token `K`：

```text
context_1 -> K
context_2 -> K
context_3 -> K
...
```

训练早期模型还不会预测 `K`，所以这些样本都有较大 cross-entropy gradient。因为 target 相同，它们在 output side 的更新方向高度一致：都要求提高 `K` 的 logit。

在 tied embedding 中，这会同时带来两件事：

1. `K` 的 output embedding 被许多 context hidden states 拉向它们的条件均值；
2. 这些 context hidden states 也被拉向更能预测 `K` 的方向。

因此，最初的 common direction 不是 nested 结构凭空产生的，而是 high-frequency / shared-target 统计结构造成的 early gradient alignment。

#### K 作为 input：one-to-many 传播

高频 token 也经常作为 input 出现在许多 context 中：

```text
K, x_1 -> y_1
K, x_2 -> y_2
K, x_3 -> y_3
...
```

这时所有包含 `K` 的 hidden states 都共享一部分来自 `K` 的 component：

```text
hidden state = common component from K + residual component from current context
```

这不意味着这些 hidden states 完全一样。不同 `x_i` 和不同 target 仍然提供 residual information。但如果 common component 的 norm 很大，不同 hidden states 的角度会被 common component 主导，真正区分样本的信息被压到 residual subspace。

#### 实验证据

Stage 4 直接验证了这一点：

- `uniform_disjoint`：token、target、input prefix 和完整 pattern 都不共享，且频率均匀；
- `shared_target`：很多不同 context 预测同一个 target；
- `shared_target_reweight`：对 target frequency 做 reweighting。

关键结果：

| 指标 | uniform-disjoint | shared-target | shared-target reweight |
|---|---:|---:|---:|
| step 0 centered output-gradient top1 energy | 0.125 | 0.500 | 0.152 |
| repeated target gradient 与 context mean cosine | - | 0.99997 | - |
| final centered embedding top1 energy | 0.112 | 0.143 | 0.128 |

这说明：

```text
shared target statistics
→ step 0 gradient concentration
→ later parameter / embedding concentration
```

也说明 `reweighting` 能直接削弱这个成核过程。

### 1.2 nested 结构让 common direction 广泛进入表征

自然语言具有嵌套结构：

```text
A
A B
A B C
A B C D
```

更长的结构通常继承更短结构的表示，再加入额外约束。比如：

```text
in
in the
in the middle
in the middle of
in the middle of the sentence
```

如果短结构已经带有 strong common component，那么更长结构的 hidden state 往往也会继承这个 component：

```text
h_long = inherited common component + new residual component
```

这里要避免一个过强说法：我们目前不应说 nested residual feature 一定会被吸进同一个 top singular direction。Stage 2 没有强力支持这个版本。

更准确的说法是：

> nested 结构让 common component 反复出现在许多 token / pattern 的 hidden states 中；这会让 residual feature 的优化处在一个 common component 已经很大的背景下。

这和“residual feature 必然投影到 common direction”不是同一件事。

### 1.3 梯度偏好 common 大方向，residual 方向被冷落

一旦 common direction 获得较大 norm 或 singular gain，优化器面对的是不平衡的局部几何。

设一个 hidden state 近似为：

```text
h = c · u_common + r_residual
```

如果 `|c|` 很大，那么 hidden direction 主要由 `u_common` 决定。此时：

- 沿 common direction 小幅移动，可能产生较大 logit / attention score 改变；
- 沿 residual direction 同样幅度移动，产生的 logit 改变较小；
- tail feature 若要产生可见角度差异，需要更大的 residual norm；
- 但 low-frequency tail pattern 本来梯度就少，因此 residual 方向长得慢。

所以优化器不是“有意识地只喜欢 common direction”。更准确地说：

> 在相同 update budget 下，common high-gain direction 的即时 loss decrease 更大，residual direction 的有效学习效率更低。

这造成一个优化路径问题：

```text
common direction already large
→ update 更容易继续利用 common channel
→ residual / tail 方向 early loss decrease 小
→ tail 收敛慢
→ tail 成为整体训练 bottleneck
```

在后期，当 common pattern 的 loss 已经很低时，common gradient 会下降，tail 方向会相对获得更多学习机会。但这时早期路径已经塑造了表示空间和参数空间，tail 学习已经被延迟。

## 2. 基于建模故事的方法位置

不同方法对应不同因果环节。它们不应被混成一种“压谱方法”。

| 方法 | 对应环节 | 目标 | 主要风险 |
|---|---|---|---|
| loss reweighting / balanced sampling | 1.1 高频 pattern 成核 | 从源头削弱 common direction 的 early gradient domination | 降权过强会让必要 common pattern 学得太慢 |
| MoE / subspace routing | 1.2 nested 后的表征共享 | 把不同 feature 放入不同参数/表示空间，减少所有 feature 争同一 global common subspace | routing collapse 会在 expert 内复现同样问题 |
| 方向感知优化器 | 1.3 梯度偏好 common 大方向 | 限制 update 反复 align 到大奇异方向，给 residual directions 更高有效学习机会 | 硬删除 top direction 可能破坏有用共享结构 |
| 谱控制 / normalization | common scale / gain 过大 | 控制 semantic direction 与 confidence scale 的耦合 | 后验强行压谱不等于从头获得 scale-balanced path |

### 2.1 Reweighting loss：针对成核源头

reweighting 直接作用于第 1.1 步：

```text
高频 target / phrase 贡献过多 gradient
→ 通过 frequency-aware weight 降低其 early gradient mass
→ first common direction 变弱
```

这不是要删除 common structure，而是避免它在 tail feature 开始有效学习前垄断主要更新。

合理目标是 soft reweighting：

- common 仍然能学会；
- 但 common 不再过早吞掉 early update；
- tail 在更早阶段获得有效 loss decrease。

### 2.2 MoE：针对 nested 后的表征解耦

MoE 不一定要消灭 common direction。它更像是在改变“所有 feature 是否必须共享同一个 global 表征空间”。

如果每个 expert 或 routed subspace 可以形成自己的局部表示结构，那么：

```text
global common direction domination
→ local expert subspace specialization
→ common / tail 不必全部竞争同一个 top singular subspace
```

这对应第 1.2 步：nested 会让 common component 广泛进入 hidden states，而 MoE 尝试把不同语义区域分配到不同空间里学习，平衡不同参数/表征的强度。

验证 MoE 是否真的解决问题，不能只看 routing entropy 或 expert load balance，还要看：

- 每个 expert 内部 effective rank；
- common 与 tail feature 是否进入不同局部 subspace；
- tail feature 的 loss / margin / residual rank 是否改善；
- expert 内是否重新出现 common singular collapse。

### 2.3 优化器中对梯度做操作：针对 common 方向偏好

方向感知优化器针对第 1.3 步：

```text
当前 update = common component + residual component
```

它可以做：

- 限制 top-gradient / top-singular direction 的 update share；
- 给 common component 和 residual component 使用不同学习率；
- 对连续 align 到同一大奇异方向的梯度做 soft clipping；
- 提高 residual component 的有效 learning rate 或 preconditioning。

目标不是硬删除 common direction，而是防止优化器反复把 update budget 投给已经过强的 common channel。

### 2.4 为什么“强行 clip 谱”不是充分方法

Stage 3/5 中 `zipf_clip` 能降低部分谱集中，但没有改善 tail stable speed。它不应被解释成 reweighting 理论的反例。

更准确的解释是：

> 后验强行压平谱，不等价于从训练开始就让优化路径 scale-balanced。

如果 early training 已经让 tail 方向长期欠优化，那么 later clipping 不能自动补回早期没学到的 residual structure。

## 3. 如何验证建模是对的：reweighting 的预测与实验

如果上述理论建模是对的，那么 reweighting 应该按下面方式工作。

### 3.1 预测一：从一开始 common direction 就应弱化

理论预测：

```text
reweighting 降低高频 target / phrase 的 early gradient mass
→ step 0 或 early window 中 common direction 弱化
```

Stage 4 支持：

| 条件 | step 0 centered output-gradient top1 energy |
|---|---:|
| uniform-disjoint | 0.125 |
| shared-target | 0.500 |
| shared-target reweight | 0.152 |

解释：

- shared-target 会立刻制造强 common output-gradient mode；
- reweighting 把这个 early gradient mode 大幅削弱；
- 因此 reweighting 确实首先打断第 1.1 步。

### 3.2 预测二：尽管依旧有 nested，common component 的强度应减弱

理论预测：

```text
reweighting 不能删除 nested 结构
但因为最初 common direction 较弱
所以 nested 传播出去的 common component 也应较弱
```

Stage 3/5 支持一部分：

| 条件 | final centered embedding top1 energy | final tail residual rank |
|---|---:|---:|
| uniform | 0.217 | 3.199 |
| Zipf | 0.219 | 3.164 |
| Zipf + reweight | 0.215 | 3.292 |
| Zipf + clip | 0.218 | 3.172 |

解释：

- reweighting 后最终 common concentration 更低；
- tail residual rank 更高；
- 这说明 reweighting 没有改变 nested 数据结构本身，但改变了 nested 结构中 common component 的强度和 tail residual 的可用空间。

边界：

- 这不是证明 nested 在真实 LLM 中一定是主要传播机制；
- 但在当前 toy 中，reweighting 后 common direction 确实弱化，tail residual structure 更健康。

### 3.3 预测三：梯度不再那么偏好 common 方向，更多方向被利用

理论预测：

```text
Zipf baseline:
  early update 低秩
  common-update share 高
  tail next-step loss decrease 小

Zipf + reweight:
  early update 更高秩
  common-update share 降低
  residual update rank 提高
  tail next-step loss decrease 提高
```

Stage 5 在 nucleation window `0..50` 中支持：

| 指标，steps 0..50 | uniform | Zipf | Zipf + reweight | Zipf + clip |
|---|---:|---:|---:|---:|
| common-update share | 0.235 | 0.262 | 0.200 | 0.262 |
| embedding-update effective rank | 5.097 | 1.824 | 3.333 | 1.824 |
| residual-update effective rank | 4.752 | 1.982 | 3.020 | 1.982 |
| next-step tail loss decrease | 0.00516 | 0.00076 | 0.00391 | 0.00160 |

这说明：

- Zipf 让 early update 极低秩；
- reweighting 让 early update 更高秩；
- reweighting 提高 residual-update rank；
- reweighting 让 tail 在下一步获得更大 loss decrease。

因此，reweighting 的 work 原理与我们的建模一致：它确实从早期优化路径上减少 common domination，增加非 common / residual 方向的有效使用。

### 3.4 common 慢一点，long-tail 快一点，整体 bottleneck 改善

reweighting 不是让所有 pattern 都同时更快。它重新分配 early learning budget：

```text
common 少吃一点
tail 早吃一点
update 使用更多方向
```

在 Stage 3/5 中，stable full tail accuracy：

| 条件 | stable full tail accuracy step |
|---|---:|
| uniform | 520 |
| Zipf | 970 |
| Zipf + reweight | 770 |
| Zipf + clip | 970 |

解释：

- Zipf 下 tail 是整体瓶颈；
- reweighting 让 common 收敛相对慢一点，但 common 本来不是瓶颈；
- tail 更早获得有效学习，整体 bottleneck 缩短；
- 因而整体训练路径更接近 scale-balanced / multi-direction optimization。

### 3.5 当前证据边界

已支持：

1. 高频/shared target 可以在 step 0 产生强 common gradient direction。
2. uniform-disjoint 数据不会产生 shared-target 那种额外 common mode。
3. reweighting 明显削弱 early common gradient mode。
4. reweighting 在成核窗口提高 update effective rank 和 residual-update effective rank。
5. reweighting 提高 early next-step tail loss decrease。
6. reweighting 改善 tail stable accuracy step 和 final tail residual rank。
7. 后验 clipping 谱不能替代从头 scale-balanced 的优化路径。

仍未完全证明：

1. 真实 LLM 中最主要的 common directions 是否由同一高频/shared-target 机制产生。
2. nested 结构在真实 LLM 中是否是 common direction 广泛进入 hidden states 的主要传播机制。
3. MoE / optimizer intervention 是否能通过同一机制改善 long-tail，而不是通过容量或正则化副作用。
4. reweighting 是否在更大模型、更自然数据和非 tied embedding setting 中保持同样机制。

## 4. 当前最简理论表述

可以把当前故事写成：

```text
高频 token / phrase / target / prefix
→ early gradient mass 不均
→ common direction 成核
→ nested 结构让 common component 广泛进入 hidden states
→ common direction 获得高 norm / high singular gain
→ optimizer 的即时 loss decrease 偏向 common channel
→ residual / tail directions 早期欠优化
→ long-tail 收敛变慢，成为整体 bottleneck
```

reweighting 的位置是：

```text
在成核阶段降低高频 pattern 的 gradient domination
→ common direction 变弱
→ early update 更高秩
→ residual / tail 获得更大即时 loss decrease
→ tail bottleneck 缩短
```

因此，我们当前不应把目标写成“消灭所有 common direction”。更准确的目标是：

> 保留必要的 compositional reuse，同时防止少数 high-frequency common directions 在训练早期垄断 scale、gradient 和 representation geometry。
