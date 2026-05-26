# 第六轮：Reused-token Synthetic Data 下如何定义 Feature 与 Specialization

## 数学建模：语言作为概率状态机

Round6 的出发点是：如果要严肃定义 feature，必须先把语言本身形式化。一个自然的计算语言学视角是，把语言建模成一个概率状态机。

令 `s_t` 表示当前位置的语言状态。这个状态可以理解成完整 prefix、隐含语法状态、语义状态，或者模型 hidden state 试图近似的对象。语言生成过程可以写成：

```text
s_t -> P(s_{t+1} | s_t)
s_t -> P(x_{t+1} | s_t)
```

其中 `x_{t+1}` 是下一个可观测 token。语言模型真正学习的不是某个唯一正确的 next token，而是条件概率分布：

```text
P(x_{t+1} | x_{\le t})
```

因此，最细粒度的 feature 可以定义为未来分布等价类：

```text
s_i ~ s_j
iff
P(x_{t+1} | s_i) = P(x_{t+1} | s_j)
```

更强的定义可以比较完整未来 suffix distribution：

```text
s_i ~ s_j
iff
P(x_{t+1:t+k} | s_i) = P(x_{t+1:t+k} | s_j)
```

在这个定义下，两个状态是否具有相同 feature，不取决于它们表面 token 是否相同，而取决于它们诱导出的未来 token distribution 是否相同或足够相似。

## 粗粒度 Feature：未来分布的投影

完整 next-token distribution 是最细粒度的 functional feature。但 feature 也可以定义得更粗。例如两个状态满足：

```text
P(x_{t+1}=A | s_i) = P(x_{t+1}=A | s_j) = 0.6
```

即使它们在其他 token 上的概率不同，也可以认为它们共享一个 common feature：

```text
next-A probability = 0.6
```

更一般地，可以定义一组投影函数：

```text
g_k(s) = projection_k(P(x_{t+1} | s))
```

每个 `g_k` 都对应一种 feature：某个 token 的概率、某个 token group 的概率、某个语法类别的概率、某个语义 continuation 的概率，或者未来分布在某个子空间上的坐标。

因此，一个状态不是只属于一个 feature，而是由多个 feature 组合而成：

```text
state = composition of distributional features
```

这给 MoE specialization 一个更本源的定义：expert specialization 不应只是把 token state 分成互斥类别，而应当尽可能把状态中可复用的未来分布特征拆出来，并让 expert 或 expert bucket 对应这些稳定、可复用的 distributional feature。

## 这一轮要回答的问题

Round6 的核心问题是：当 synthetic 数据中允许 token 在不同 slot / 不同上下文中重复出现时，我们还能否清楚定义什么是 feature，以及什么是 MoE expert specialization。

前几轮实验默认 local slot / higher-level slot 可以作为 ground-truth feature。但从上面的数学建模看，这只是一个近似：slot label 只有在它对应稳定的 future distribution 或 distributional projection 时，才是真正有意义的 feature label。

这个定义在 clean synthetic 数据中比较自然，因为 token id、local slot、higher-level slot 之间的关系相对干净。一旦允许 token 复用，token identity 与 slot-level feature 不再一一对应，原来的定义就需要重新检查。

这一轮先不急于设计新 MoE 结构，而是先把数据定义、feature 定义和评价指标重新形式化。

## 动机

真实语言中，同一个 token 可以出现在不同上下文中并表达不同功能；不同 token 也可能在相似上下文中承担相似功能。因此，如果 synthetic 数据中每个 token 只属于唯一 slot，那么它会过于干净，容易让模型通过 token-id shortcut 完成任务。

Reused-token synthetic data 的目标是显式制造这种冲突：

```text
same token id does not necessarily mean same feature
different token id does not necessarily mean different feature
```

因此，这一轮数据应当用于区分：

1. gate 是否只是按 token id 分发；
2. gate 是否能利用上下文识别 local / higher-level feature；
3. gate 是否能接近真正决定 next-token distribution 的 functional feature。

## Round6 与 Round5 的关系

Round5 已经说明，在当前 synthetic 设定下，某些 router input 与 router shape 能带来更强 specialization，尤其是 `k/head`。但 Round5 的主要问题是：当前数据仍可能太容易，模型不一定需要真正按照 feature 分发也能完成 NTP。

Round6 要进一步提高诊断能力：通过 token reuse，让 token id 与 slot / functional feature 解耦，从而检验所谓 specialization 到底是在捕捉 token identity，还是在捕捉上下文相关的 feature。

## Round6 数据生成规则

Round6 采用更简洁的 slot-level grammar。每一层只关心自己这一层长度为 `slot_size` 的 slot：

```text
slot = (u_1, u_2, ..., u_{b-1}, v)
input = (u_1, u_2, ..., u_{b-1})
output = v
```

其中 `b = slot_size`。底层的 `u_i, v` 是 raw token id；更高层的 `u_i, v` 是下一层生成出的 unit id。也就是说，每一层都使用同一套规则，只是 symbol 的含义不同。

在每一层中，slot 被强制划分成三类互不重叠的结构集合：

```text
A. same-input-different-output slots
B. different-input-same-output slots
C. normal one-to-one slots
```

### A. Same Input, Different Output

这类 slot 共享相同 input prefix，但最后一个 output symbol 不同。例如 `slot_size=4` 时：

```text
ABCX
ABCY
ABCZ
```

它表示：

```text
input = ABC
output distribution over {X, Y, Z}
```

这里的“同输入不同输出”不是 deterministic classification 错误，而是语言模型的概率分布属性：同一个 input prefix 可以以不同频率接不同 next token。

对应超参：

| 参数 | 含义 |
|---|---|
| `same_input_diff_output_rate` | 每层有多少比例的 slot 属于这类结构 |
| `same_input_diff_output_size` | 每个 shared input prefix 对应多少个不同 output |
| `same_input_diff_output_distribution` | 同一个 group 内不同 output 的出现频率，`uniform` 或 `zipf` |
| `same_input_diff_output_zipf_alpha` | 当分布为 `zipf` 时的长尾强度 |

### B. Different Input, Same Output

这类 slot 具有不同 input prefix，但共享相同 output symbol。例如：

```text
ABCX
DEFX
GHIX
```

它表示多个不同 surface input 对应相同 output token / output feature，是 synthetic 中的“不同输入同输出”。

对应超参：

| 参数 | 含义 |
|---|---|
| `diff_input_same_output_rate` | 每层有多少比例的 slot 属于这类结构 |
| `diff_input_same_output_size` | 每个 shared output 对应多少个不同 input prefix |
| `diff_input_same_output_distribution` | 同一个 group 内不同 input 的出现频率，`uniform` 或 `zipf` |
| `diff_input_same_output_zipf_alpha` | 当分布为 `zipf` 时的长尾强度 |

### C. Normal One-to-One

剩余 slot 是普通 one-to-one slot：

```text
ABCX
DEFY
GHIZ
```

它们既不共享 input prefix，也不共享 output symbol，作为对照组存在。该比例由前两类结构的 rate 自动决定：

```text
normal_rate = 1 - same_input_diff_output_rate - diff_input_same_output_rate
```

第一版实现中，A/B/C 三类 slot 集合强制 disjoint，避免无意中生成 many-to-many 结构。后续如果需要，可以显式加入 many-to-many 作为更复杂的 ablation。

## Round6 数据超参

### 1. 数据规模与层次结构

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `max_seq_len` | LM 输入长度，dataset 实际生成 `max_seq_len + 1` 个 token | `128` |
| `num_samples` | Dataset 长度 | `100000` |
| `seed` | 随机种子 | `0` |
| `min_token_id` | 最小 raw token id，保留 `0` 做 pad | `1` |
| `content_token_count` | 底层 raw token 数量 | `512` |
| `num_hierarchy_layers` | 层次数量 | `2` |
| `slot_size` | 每层 slot 长度 | `4` |
| `num_units_per_layer` | 每层生成多少个 slot / unit | `512` |

### 2. Same-input-different-output 结构

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `same_input_diff_output_rate` | slot 中有多少比例属于同输入不同输出 | `0.3` |
| `same_input_diff_output_size` | 每个 input prefix 对应几个 output | `4` |
| `same_input_diff_output_distribution` | group 内 output 频率分布 | `zipf` |
| `same_input_diff_output_zipf_alpha` | zipf 长尾强度 | `1.0` |

### 3. Different-input-same-output 结构

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `diff_input_same_output_rate` | slot 中有多少比例属于不同输入同输出 | `0.3` |
| `diff_input_same_output_size` | 每个 output 对应几个 input prefix | `4` |
| `diff_input_same_output_distribution` | group 内 input 频率分布 | `zipf` |
| `diff_input_same_output_zipf_alpha` | zipf 长尾强度 | `1.0` |

### 4. 生成与采样

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `top_sampling_distribution` | 顶层 unit 的采样分布 | `zipf` |
| `top_sampling_zipf_alpha` | 顶层采样 zipf 强度 | `1.0` |
| `padding` | 是否 padding 到固定长度 | `false` |
| `return_metadata` | 是否返回每个 token 的层次 metadata | `false` |

## Round6 与已有数据的区别

这一轮需要区分三类数据设定。

### 1. Clean synthetic data

这是前几轮使用的理想化设定：

```text
token id -> local slot -> higher-level slot
```

在这个设定中，token id 与 slot feature 高度绑定。因此，同 token、同 local slot、同 higher-level slot 之间的界限比较清楚，适合做第一版 ground-truth feature 分析，但不适合测试 token-id shortcut。

### 2. Random reused-token data

当前 `HierarchicalPatternData` 的实现已经允许 token 在不同 base units 中随机复用。其 base local unit 由 token pool 随机采样得到，因此不同 local slot 之间可以共享 token。

这个设定比 clean synthetic 更接近真实语言，但它的复用是随机的，没有显式定义哪些 slot 应该相似、哪些 slot 应该不同。因此它可以用于观察 token reuse 是否破坏 token-id shortcut，但不一定足够支持清晰的 feature-family 分析。

### 3. Controlled reused-token data

Round6 更理想的目标设定应当是 controlled reuse：显式构造 slot family / template family。

例如：

```text
family 0:
  ABC
  ABD
  ABE

family 1:
  EFD
  GHD
  KLD
```

在这种设定下，数据生成规则可以明确规定：

1. exact same slot 表示最强 local feature 相同；
2. same slot family 表示 functional feature 相似；
3. different slot family 表示 next-token distribution 不同或相似度较低；
4. same token id 只表示 surface token 相同，不必然表示 feature 相同。

这类数据更适合回答：expert 应当按 token id、exact slot、slot family，还是 next-token distribution 分发。

## 初步 feature 定义

在 reused-token data 下，feature 不应再简单等同于 token id。建议同时保留三个层次的 feature 定义。

### 1. Surface token feature

同一个 token id 具有相同 surface feature。

这个定义最容易被模型利用，但它也是最容易产生 shortcut 的定义。若 gate 主要按 token id 分发，说明它未必学到了 context-dependent feature。

### 2. Contextual slot feature

一个 token position 所属的 local slot / higher-level slot 定义其上下文 feature。

这个定义适合 synthetic 数据，因为 slot label 由数据生成过程给出。但在 reused-token data 中，同一个 token 可以属于不同 slot，因此 contextual slot feature 与 surface token feature 会发生冲突。

### 3. Functional feature

两个 token position 如果对应相似的 next-token distribution，则它们具有相似 functional feature。

可以形式化为：

```text
feature(x_t) = equivalence class induced by P(x_{t+1} | x_{\le t})
```

这里的关键不是要求模型把某个输入坍缩到唯一正确的 next token，而是要求模型学到数据生成过程中的条件概率分布。对于同一个输入 pattern，后面可以跟多个不同输出；它们都可以是“对”的，只是频率不同。只有当模型输出的 next-token distribution 与数据中的真实条件频率分布一致时，cross entropy loss 才最低。

因此，在 synthetic 数据中，functional feature 应当由显式设计的条件分布定义，例如 `P(next unit | current unit, context)`；在真实数据中，只能通过 next-token logits similarity、representation similarity 或 attention retrieval pattern 等 proxy 近似。

## 初步 specialization 定义

在 reused-token data 下，specialization 不应写成“同 token 同 expert”或“同 slot 同 expert”这么简单，而应写成：

```text
expert assignment should be aligned with task-relevant feature similarity.
```

更具体地说：

1. exact same local slot 的 token / position 应当有最高 expert overlap；
2. same slot family 或 same higher-level functional group 的 token / position 应当有较高 expert overlap；
3. same token id but different conditional next-token distribution 的 token / position 不应被强制分到同一 expert；
4. different token id but similar conditional next-token distribution 的 token / position 可以被分到同一 expert；
5. unrelated slot / unrelated next-token distribution 的 token / position 应当进入不同或低重叠的 expert bucket。

因此，Round6 的核心指标不应只看 local slot purity，也要看 token id 与 contextual feature 冲突时，gate 到底选择跟随哪一个。

## 建议评价指标

Round6 至少需要下面几类指标。

### 1. Token identity alignment

衡量 gate 是否主要按 surface token id 分发：

1. same-token same-expert rate；
2. token-id feature-to-expert purity；
3. token-id NMI。

### 2. Contextual feature alignment

衡量 gate 是否按 local / higher-level slot 分发：

1. local-slot same-expert rate；
2. higher-slot same-expert rate；
3. local-slot / high-slot NMI；
4. fixed-token conditional slot purity：固定 token id 后，gate 是否还能区分不同 slot。

### 3. Functional feature alignment

衡量 gate 是否按 downstream behavior 分发：

1. same slot-family same-expert rate；
2. next-token logits similarity vs same-expert rate；
3. expert 内部 next-token logits similarity；
4. fixed-slot conditional token invariance：固定 functional slot 后，gate 是否能忽略 token id 差异。

### 4. Prediction utility

判断 specialization 是否真的有任务价值：

1. overall NTP accuracy / loss；
2. inside-local accuracy；
3. local-boundary-not-high accuracy；
4. reused-token ambiguous positions 的 accuracy；
5. slot-family boundary accuracy。

## 当前未定问题

后续需要继续讨论并确定：

1. Controlled reused-token data 中，slot family 应该如何构造；
2. token reuse 的比例应当多高；
3. 是否需要显式设计同义 token / 多义 token；
4. next-token distribution 是否应当由 exact slot 决定，还是由 slot family / higher-level context 决定；
5. 评价 specialization 时，主要 target 是 exact slot、slot family，还是 functional feature；
6. 当前 `HierarchicalPatternData` 是否应扩展一个新 mode，而不是直接修改原有数据生成逻辑。

## 暂定结论

Round6 的关键不是直接寻找更高 NTP 的模型结构，而是先构造一个能真正区分 token-id shortcut 与 context-dependent feature specialization 的 synthetic benchmark。

如果 gate 在这个 benchmark 上仍主要按 token id 分发，说明现有 MoE specialization 更接近 surface-token specialization；如果 gate 能在 token 复用冲突下按 slot family 或 functional feature 分发，才更接近我们想要的 feature-level specialization。

## 已完成实验：controlled reused-token broad sweep

### 实验设定

本轮实验使用 controlled reused-token 数据：

```text
slot_size = 4
num_hierarchy_layers = 2
content_token_count = 512
num_units_per_layer = 512
same-input-different-output rate = 0.1
same-input-different-output size = 4
same-input-different-output distribution = zipf
different-input-same-output rate = 0.1
different-input-same-output size = 4
different-input-same-output distribution = zipf
top sampling distribution = uniform
```

对应地，三类 slot 比例约为：

```text
A. same-input-different-output: 10%
B. different-input-same-output: 10%
C. normal one-to-one: 80%
```

实验结果见：

```text
fdong/experiments/round6_controlled_reuse_analysis.md
fdong/experiments/round6_controlled_reuse_analysis.json
```

### 1. NTP 能力

Dense 模型给出了清楚的数据难度门槛：

```text
dense-h32: 72.47%
dense-h64: 89.37%
dense-h128: 89.97%
```

这说明新数据对 `h32` 明显有难度，但 `h64` 已经基本接近 `h128`。因此后续 MoE 以 `2-layer h64` 作为主设定是合理的：它不是完全学不会，也不是过小导致所有结构都失败。

A/B 两类结构难度差异明显：

```text
dense-h64 A output acc: 85.14%
dense-h64 B output acc: 100.00%
```

其中 A 类 `same-input-different-output` 更难，因为同一个 input prefix 对应多个可能 output；B 类 `different-input-same-output` 更容易，因为不同 input prefix 共享相同 output。

### 2. Routing 与 feature 对齐

普通 hidden-router MoE 仍然主要表现为较弱的 feature specialization：

```text
moe-hidden-full-top1:
token same-expert: 66.30%
base unit same-expert: 31.32%
A group same-expert: 33.65%
B group same-expert: 37.68%
```

使用 `k` 作为 router input 后，specialization 明显增强：

```text
moe-rfull-k-eresid:
NTP acc: 89.56%
token same-expert: 93.25%
target same-expert: 61.66%
base unit same-expert: 56.27%
A group same-expert: 66.09%
B group same-expert: 66.32%
```

这说明 `k` router 不只是按 token id 分发，也更接近 output-side / next-token feature。

### 3. A/B 专项结果

A/B 两类都应被视为“同 feature”的结构约束，但它们对应的等价层次不同。

A 类 `same-input-different-output` 是同一个状态的分布 feature：

```text
ABC -> X / Y / Z
```

这里 `ABC` 对应同一个条件状态。不同样本中出现 `X/Y/Z`，不是因为存在多个唯一正确答案，而是因为同一状态诱导出一个 next-token distribution。因此，A group 内的样本应当有较高 expert overlap；如果 gate 把 `ABCX`、`ABCY`、`ABCZ` 完全割裂，说明它在按 observed output token 过度分发，而不是按 input-conditioned distributional feature 分发。

B 类 `different-input-same-output` 是不同状态共享 output-side projection：

```text
ABC -> X
DEF -> X
GHI -> X
```

这里不同 input prefix 共享 `P(next = X)` 高这一 projected feature；如果更强地设计成完整 next-token distribution 相同，它们就是完整 distributional equivalence class。因此，B group 内的样本也应当有较高 expert overlap；如果 gate 只按 input surface form 分发，就会错过这种 functional equivalence。

A 类 `same-input-different-output` 最强结果：

```text
moe-rfull-k-eresid:
A group same-expert: 66.09%
NTP acc: 89.56%
```

B 类 `different-input-same-output` 最强结果：

```text
moe-rhead-k-eresid:
B group same-expert: 75.57%
NTP acc: 89.45%

moe-k-head-ne4-topk1:
B group same-expert: 73.90%
NTP acc: 89.42%
```

因此，`full-k` 更平衡，A/B 两类 group 都较强；`k/head` 对 B 类 different-input-same-output 更强。

### 4. Attention 仍然捕捉 local/high 结构

新数据下，attention 仍然明显捕捉 local/high unit：

```text
dense-h64:
attention base mass: 62.30%
attention high mass: 83.48%

moe-rfull-k-eresid:
attention high mass: 83.20%

moe-k-head-ehidden:
attention high mass: 83.31%
```

一些 head/head 结构可以进一步提高 attention high mass 到 `89%~93%`，但通常会损害 NTP。

同时，attention 对 A/B group 的直接 mass 很低，大多只有 `5%~8%`。这说明 attention 主要捕捉 local/high structural unit，而不是直接把 A/B group 当作 retrieval cluster。

### 5. MoE routing 与 attention retrieval 的对应

MoE routing 与 attention retrieval 存在对应，但并不完美。较强结果包括：

```text
head/head-k-eattn:
attention-expert mass: 82.35%

head/head-k-ehidden:
attention-expert mass: 80.17%

k-head-eattn:
attention-expert mass: 79.63%
```

但这些结构通常会降低 NTP 到 `86%~88%`。因此当前仍然存在 tradeoff：越强地让 routing bucket 接近 attention bucket，specialization / retrieval alignment 更强，但预测性能更容易下降。

### 当前结论

Round6 支持 Round5 的主结论：`k`-based routing 仍然是当前最有效的 specialization 方向。

更细的结论是：

1. 为了 NTP，MoE 还没有明显超过 dense-h128；
2. 为了 specialization，`k` 明显优于 hidden / v / layer_input；
3. 为了 A/B feature，`full-k` 更平衡，`k/head` 对 B 类更强；
4. 为了 routing-attention 对齐，head/head 更强，但伤 NTP；
5. A 类 same-input-different-output 比 B 类 different-input-same-output 更难，是后续最值得重点分析的位置。

## 原始实验结果表

本节直接由 `fdong/experiments/round6_controlled_reuse_analysis.json` 汇总生成，保留各模型的 NTP、loss、routing purity / same-expert 指标，以及 attention mass 集中度。

### 1. NTP 与 A/B 专项预测结果

| Run | Type | Train loss | Eval loss | NTP acc | Output-token acc | A output acc | B output acc |
|---|---|---|---|---|---|---|---|
| `round6-controlled-dense-h32` | dense | 1.4260 | 1.4107 | 72.47% | 88.62% | 57.82% | 68.18% |
| `round6-controlled-dense-h64` | dense | 0.5060 | 0.4986 | 89.37% | 98.58% | 85.14% | 100.00% |
| `round6-controlled-dense-h128` | dense | 0.4080 | 0.4046 | 89.97% | 98.80% | 87.35% | 99.93% |
| `round6-controlled-moe-hidden-full-top1` | moe | 0.4810 | 0.4835 | 89.03% | 98.39% | 84.43% | 99.93% |
| `round6-controlled-moe-hidden-full-top2` | moe | 0.4420 | 0.4448 | 89.63% | 98.61% | 85.53% | 99.86% |
| `round6-controlled-moe-hidden-full-common` | moe | 0.4390 | 0.4433 | 89.76% | 98.66% | 86.11% | 99.93% |
| `round6-controlled-moe-hidden-full-lb001` | moe | 0.5340 | 0.5412 | 88.50% | 98.07% | 82.09% | 99.93% |
| `round6-controlled-moe-rfull-attn-eresid` | moe | 0.5040 | 0.5192 | 88.64% | 98.02% | 82.15% | 99.93% |
| `round6-controlled-moe-rfull-q-eresid` | moe | 0.4820 | 0.4740 | 89.29% | 98.50% | 84.82% | 100.00% |
| `round6-controlled-moe-rfull-k-eresid` | moe | 0.4530 | 0.4581 | 89.56% | 98.54% | 84.88% | 99.79% |
| `round6-controlled-moe-rfull-v-eresid` | moe | 0.4510 | 0.4532 | 89.48% | 98.58% | 85.27% | 100.00% |
| `round6-controlled-moe-rfull-layerin-eresid` | moe | 0.4540 | 0.4547 | 89.53% | 98.57% | 85.53% | 100.00% |
| `round6-controlled-moe-rfull-hidden-eresid` | moe | 0.4810 | 0.4850 | 89.01% | 98.39% | 84.10% | 99.86% |
| `round6-controlled-moe-rhead-attn-eresid` | moe | 0.5440 | 0.5341 | 88.32% | 97.99% | 81.70% | 100.00% |
| `round6-controlled-moe-rhead-q-eresid` | moe | 0.4660 | 0.4654 | 89.42% | 98.60% | 85.92% | 99.86% |
| `round6-controlled-moe-rhead-k-eresid` | moe | 0.4660 | 0.4619 | 89.45% | 98.53% | 85.07% | 99.93% |
| `round6-controlled-moe-rhead-v-eresid` | moe | 0.4670 | 0.4694 | 89.36% | 98.51% | 84.94% | 99.86% |
| `round6-controlled-moe-rhead-layerin-eresid` | moe | 0.4540 | 0.4605 | 89.44% | 98.46% | 84.04% | 99.93% |
| `round6-controlled-moe-rhead-hidden-eresid` | moe | 0.4690 | 0.4776 | 89.23% | 98.57% | 86.24% | 99.93% |
| `round6-controlled-moe-k-head-ne4-topk1` | moe | 0.4620 | 0.4622 | 89.42% | 98.55% | 84.88% | 99.86% |
| `round6-controlled-moe-k-head-ne4-topk2` | moe | 0.4510 | 0.4450 | 89.69% | 98.63% | 85.66% | 100.00% |
| `round6-controlled-moe-k-head-ne8-topk1` | moe | 0.4550 | 0.4556 | 89.44% | 98.57% | 85.40% | 100.00% |
| `round6-controlled-moe-k-head-ne8-topk2` | moe | 0.4270 | 0.4301 | 89.77% | 98.77% | 86.96% | 100.00% |
| `round6-controlled-moe-k-head-common` | moe | 0.4540 | 0.4454 | 89.72% | 98.65% | 85.85% | 100.00% |
| `round6-controlled-moe-k-head-lb001` | moe | 0.6130 | 0.6139 | 87.88% | 97.61% | 79.17% | 99.86% |
| `round6-controlled-moe-k-head-lb01` | moe | 0.7890 | 0.7020 | 85.91% | 96.36% | 76.38% | 97.52% |
| `round6-controlled-moe-k-head-eattn` | moe | 0.4800 | 0.4756 | 89.48% | 98.50% | 84.88% | 100.00% |
| `round6-controlled-moe-k-head-ehidden` | moe | 0.4530 | 0.4559 | 89.54% | 98.59% | 85.27% | 99.93% |
| `round6-controlled-moe-k-head-elayerin` | moe | 0.5380 | 0.5258 | 89.10% | 98.49% | 85.20% | 99.79% |
| `round6-controlled-moe-k-head-eq` | moe | 0.6100 | 0.6015 | 88.16% | 97.94% | 81.38% | 99.93% |
| `round6-controlled-moe-k-head-ek` | moe | 0.5880 | 0.5887 | 88.53% | 98.02% | 80.86% | 99.38% |
| `round6-controlled-moe-k-head-ev` | moe | 0.5830 | 0.5729 | 88.79% | 98.10% | 81.12% | 99.72% |
| `round6-controlled-moe-headhead-k-eresid` | moe | 0.6160 | 0.6132 | 88.71% | 98.17% | 81.83% | 99.79% |
| `round6-controlled-moe-headhead-k-eattn` | moe | 0.6300 | 0.6037 | 88.62% | 98.38% | 84.04% | 99.72% |
| `round6-controlled-moe-headhead-k-eq` | moe | 0.7450 | 0.7317 | 86.97% | 97.63% | 77.87% | 99.72% |
| `round6-controlled-moe-headhead-k-ek` | moe | 0.7520 | 0.7400 | 86.84% | 97.20% | 74.37% | 99.66% |
| `round6-controlled-moe-headhead-k-ev` | moe | 0.7150 | 0.7134 | 87.64% | 97.85% | 78.72% | 99.86% |
| `round6-controlled-moe-headhead-k-elayerin` | moe | 0.6430 | 0.6531 | 88.28% | 98.18% | 82.09% | 99.66% |
| `round6-controlled-moe-headhead-k-ehidden` | moe | 0.6060 | 0.6039 | 88.59% | 98.22% | 82.48% | 99.86% |

### 2. MoE routing purity / same-expert 原始结果

| Run | Token same-expert | Target same-expert | Base unit same-expert | High unit same-expert | A group same-expert | B group same-expert | Eff. experts | Attn-expert mass |
|---|---|---|---|---|---|---|---|---|
| `round6-controlled-dense-h32` | - | - | - | - | - | - | - | - |
| `round6-controlled-dense-h64` | - | - | - | - | - | - | - | - |
| `round6-controlled-dense-h128` | - | - | - | - | - | - | - | - |
| `round6-controlled-moe-hidden-full-top1` | 66.30% | 42.23% | 31.32% | 26.35% | 33.65% | 37.68% | 3.9955 | 49.39% |
| `round6-controlled-moe-hidden-full-top2` | 84.00% | 47.59% | 35.65% | 27.32% | 27.40% | 52.57% | 3.9894 | 49.74% |
| `round6-controlled-moe-hidden-full-common` | 70.00% | 40.88% | 32.35% | 26.98% | 31.47% | 36.42% | 3.9777 | 47.35% |
| `round6-controlled-moe-hidden-full-lb001` | 61.16% | 38.40% | 27.31% | 24.73% | 28.21% | 26.27% | 3.9999 | 49.74% |
| `round6-controlled-moe-rfull-attn-eresid` | 50.07% | 36.59% | 36.01% | 29.03% | 39.43% | 41.60% | 3.9446 | 58.34% |
| `round6-controlled-moe-rfull-q-eresid` | 85.86% | 46.38% | 37.84% | 28.79% | 44.43% | 57.49% | 3.9411 | 58.03% |
| `round6-controlled-moe-rfull-k-eresid` | 93.25% | 61.66% | 56.27% | 50.41% | 66.09% | 66.32% | 2.7642 | 68.75% |
| `round6-controlled-moe-rfull-v-eresid` | 84.73% | 43.83% | 32.48% | 28.17% | 40.55% | 37.66% | 3.8234 | 51.30% |
| `round6-controlled-moe-rfull-layerin-eresid` | 83.05% | 43.91% | 32.86% | 27.02% | 33.33% | 45.74% | 3.9589 | 48.29% |
| `round6-controlled-moe-rfull-hidden-eresid` | 64.37% | 42.37% | 31.73% | 26.46% | 33.98% | 38.22% | 3.9947 | 49.29% |
| `round6-controlled-moe-rhead-attn-eresid` | 53.07% | 39.06% | 37.36% | 30.14% | 40.94% | 52.62% | 3.9530 | 60.71% |
| `round6-controlled-moe-rhead-q-eresid` | 86.53% | 50.96% | 42.55% | 34.07% | 48.35% | 62.33% | 3.8801 | 61.27% |
| `round6-controlled-moe-rhead-k-eresid` | 89.32% | 58.22% | 49.24% | 41.85% | 46.43% | 75.57% | 3.9447 | 67.93% |
| `round6-controlled-moe-rhead-v-eresid` | 84.54% | 42.31% | 31.56% | 27.22% | 30.89% | 33.77% | 3.9753 | 49.98% |
| `round6-controlled-moe-rhead-layerin-eresid` | 83.93% | 45.21% | 35.08% | 27.83% | 36.55% | 47.34% | 3.9763 | 51.05% |
| `round6-controlled-moe-rhead-hidden-eresid` | 66.41% | 42.60% | 34.12% | 27.55% | 35.53% | 37.33% | 3.9895 | 51.81% |
| `round6-controlled-moe-k-head-ne4-topk1` | 89.89% | 58.08% | 49.45% | 42.22% | 46.99% | 73.90% | 3.9060 | 67.96% |
| `round6-controlled-moe-k-head-ne4-topk2` | 91.56% | 47.09% | 36.56% | 28.25% | 42.78% | 55.14% | 3.9922 | 54.06% |
| `round6-controlled-moe-k-head-ne8-topk1` | 85.11% | 44.84% | 33.85% | 25.03% | 29.70% | 59.10% | 7.1313 | 51.78% |
| `round6-controlled-moe-k-head-ne8-topk2` | 92.88% | 38.56% | 25.44% | 17.02% | 25.10% | 48.93% | 7.7846 | 41.99% |
| `round6-controlled-moe-k-head-common` | 91.24% | 52.53% | 45.72% | 36.53% | 44.94% | 66.00% | 3.8759 | 58.96% |
| `round6-controlled-moe-k-head-lb001` | 83.67% | 44.03% | 35.16% | 27.19% | 43.24% | 46.99% | 3.9990 | 56.46% |
| `round6-controlled-moe-k-head-lb01` | 83.78% | 43.74% | 34.34% | 27.22% | 34.95% | 44.26% | 3.9915 | 56.08% |
| `round6-controlled-moe-k-head-eattn` | 86.37% | 57.62% | 48.70% | 39.18% | 49.03% | 71.04% | 3.8368 | 79.63% |
| `round6-controlled-moe-k-head-ehidden` | 91.05% | 59.41% | 52.15% | 45.11% | 56.82% | 70.65% | 3.8121 | 68.82% |
| `round6-controlled-moe-k-head-elayerin` | 88.99% | 47.30% | 38.32% | 30.00% | 43.49% | 58.25% | 3.9608 | 57.15% |
| `round6-controlled-moe-k-head-eq` | 92.91% | 51.33% | 40.74% | 33.17% | 44.81% | 71.85% | 3.8306 | 52.60% |
| `round6-controlled-moe-k-head-ek` | 91.31% | 47.72% | 39.38% | 33.04% | 41.74% | 50.61% | 3.9273 | 53.17% |
| `round6-controlled-moe-k-head-ev` | 91.82% | 51.69% | 44.17% | 35.68% | 52.35% | 59.51% | 3.9433 | 68.23% |
| `round6-controlled-moe-headhead-k-eresid` | 92.65% | 56.24% | 49.50% | 39.39% | 51.45% | 71.95% | 3.7598 | 78.66% |
| `round6-controlled-moe-headhead-k-eattn` | 92.39% | 58.22% | 51.71% | 42.21% | 57.52% | 63.80% | 3.8074 | 82.35% |
| `round6-controlled-moe-headhead-k-eq` | 94.53% | 58.57% | 52.98% | 44.98% | 58.05% | 68.35% | 3.7593 | 72.78% |
| `round6-controlled-moe-headhead-k-ek` | 93.03% | 54.92% | 47.43% | 40.26% | 47.83% | 62.13% | 3.9429 | 72.11% |
| `round6-controlled-moe-headhead-k-ev` | 92.40% | 54.70% | 47.65% | 37.90% | 58.81% | 67.19% | 3.8388 | 76.20% |
| `round6-controlled-moe-headhead-k-elayerin` | 93.70% | 57.08% | 50.24% | 41.11% | 47.92% | 71.95% | 3.7245 | 77.13% |
| `round6-controlled-moe-headhead-k-ehidden` | 92.53% | 56.55% | 50.16% | 39.96% | 49.73% | 74.22% | 3.6994 | 80.17% |

### 3. Attention mass 集中度原始结果

| Run | Attn base mass | Attn high mass | Attn A group mass | Attn B group mass | Attn base history mass | Attn high history mass | Attn A history mass | Attn B history mass |
|---|---|---|---|---|---|---|---|---|
| `round6-controlled-dense-h32` | 46.61% | 70.38% | 4.85% | 4.58% | 27.53% | 59.45% | 2.89% | 3.26% |
| `round6-controlled-dense-h64` | 62.30% | 83.48% | 5.88% | 6.21% | 39.85% | 72.08% | 3.84% | 4.78% |
| `round6-controlled-dense-h128` | 51.63% | 75.63% | 4.69% | 5.75% | 35.90% | 66.74% | 3.09% | 4.54% |
| `round6-controlled-moe-hidden-full-top1` | 55.28% | 76.05% | 5.40% | 6.33% | 33.31% | 63.19% | 3.32% | 4.64% |
| `round6-controlled-moe-hidden-full-top2` | 53.79% | 76.51% | 5.15% | 5.94% | 33.49% | 64.97% | 3.20% | 4.38% |
| `round6-controlled-moe-hidden-full-common` | 53.91% | 78.94% | 4.76% | 5.66% | 35.33% | 69.51% | 3.07% | 4.15% |
| `round6-controlled-moe-hidden-full-lb001` | 58.31% | 77.68% | 5.71% | 6.17% | 34.43% | 63.73% | 3.40% | 4.29% |
| `round6-controlled-moe-rfull-attn-eresid` | 61.26% | 79.36% | 5.76% | 6.43% | 34.03% | 63.11% | 3.24% | 4.39% |
| `round6-controlled-moe-rfull-q-eresid` | 59.46% | 77.94% | 5.86% | 6.79% | 33.59% | 62.16% | 3.25% | 4.82% |
| `round6-controlled-moe-rfull-k-eresid` | 59.86% | 83.20% | 5.85% | 6.29% | 38.48% | 72.80% | 3.62% | 4.84% |
| `round6-controlled-moe-rfull-v-eresid` | 58.20% | 81.99% | 5.34% | 6.05% | 36.41% | 71.02% | 3.30% | 4.59% |
| `round6-controlled-moe-rfull-layerin-eresid` | 54.14% | 76.20% | 5.26% | 6.42% | 34.12% | 64.55% | 3.14% | 4.79% |
| `round6-controlled-moe-rfull-hidden-eresid` | 54.96% | 76.05% | 5.28% | 6.31% | 33.31% | 63.41% | 3.23% | 4.59% |
| `round6-controlled-moe-rhead-attn-eresid` | 62.27% | 78.88% | 6.29% | 6.46% | 34.31% | 61.72% | 3.84% | 4.53% |
| `round6-controlled-moe-rhead-q-eresid` | 59.35% | 82.22% | 5.48% | 6.23% | 36.78% | 70.74% | 3.30% | 4.58% |
| `round6-controlled-moe-rhead-k-eresid` | 61.59% | 82.24% | 5.72% | 6.61% | 36.42% | 68.66% | 3.19% | 4.79% |
| `round6-controlled-moe-rhead-v-eresid` | 55.46% | 76.42% | 5.66% | 6.36% | 33.48% | 63.45% | 3.42% | 4.78% |
| `round6-controlled-moe-rhead-layerin-eresid` | 54.37% | 76.00% | 5.17% | 6.25% | 33.08% | 63.60% | 3.24% | 4.59% |
| `round6-controlled-moe-rhead-hidden-eresid` | 55.46% | 75.86% | 5.36% | 6.19% | 32.75% | 62.25% | 3.18% | 4.41% |
| `round6-controlled-moe-k-head-ne4-topk1` | 61.33% | 82.31% | 5.76% | 6.55% | 36.72% | 69.24% | 3.37% | 4.76% |
| `round6-controlled-moe-k-head-ne4-topk2` | 56.23% | 80.37% | 5.35% | 6.18% | 35.59% | 69.82% | 3.27% | 4.45% |
| `round6-controlled-moe-k-head-ne8-topk1` | 53.89% | 76.05% | 5.15% | 6.05% | 33.88% | 64.37% | 3.19% | 4.59% |
| `round6-controlled-moe-k-head-ne8-topk2` | 52.62% | 77.67% | 5.03% | 5.85% | 34.93% | 68.36% | 3.29% | 4.60% |
| `round6-controlled-moe-k-head-common` | 56.38% | 80.24% | 5.26% | 5.88% | 36.94% | 70.38% | 3.49% | 4.38% |
| `round6-controlled-moe-k-head-lb001` | 60.39% | 78.48% | 6.08% | 6.38% | 35.34% | 63.60% | 3.57% | 4.37% |
| `round6-controlled-moe-k-head-lb01` | 62.28% | 83.47% | 5.81% | 6.41% | 38.53% | 71.32% | 3.49% | 4.67% |
| `round6-controlled-moe-k-head-eattn` | 76.86% | 89.24% | 7.32% | 7.19% | 37.35% | 66.85% | 3.79% | 4.68% |
| `round6-controlled-moe-k-head-ehidden` | 61.62% | 83.31% | 5.68% | 6.47% | 37.67% | 70.95% | 3.39% | 4.69% |
| `round6-controlled-moe-k-head-elayerin` | 54.26% | 75.55% | 5.28% | 6.33% | 32.36% | 62.48% | 3.12% | 4.39% |
| `round6-controlled-moe-k-head-eq` | 42.22% | 63.21% | 4.22% | 5.70% | 25.99% | 52.80% | 2.61% | 3.85% |
| `round6-controlled-moe-k-head-ek` | 50.46% | 73.75% | 4.91% | 5.94% | 31.51% | 63.32% | 2.91% | 4.22% |
| `round6-controlled-moe-k-head-ev` | 68.29% | 86.21% | 6.42% | 6.58% | 40.18% | 71.08% | 3.89% | 4.55% |
| `round6-controlled-moe-headhead-k-eresid` | 78.26% | 90.92% | 7.27% | 7.40% | 42.36% | 70.99% | 4.29% | 5.08% |
| `round6-controlled-moe-headhead-k-eattn` | 82.19% | 92.80% | 7.65% | 7.73% | 43.65% | 71.23% | 4.52% | 5.20% |
| `round6-controlled-moe-headhead-k-eq` | 63.98% | 80.92% | 6.03% | 6.76% | 34.25% | 63.49% | 3.27% | 4.51% |
| `round6-controlled-moe-headhead-k-ek` | 69.91% | 83.95% | 6.67% | 7.03% | 37.40% | 64.01% | 3.70% | 4.81% |
| `round6-controlled-moe-headhead-k-ev` | 77.35% | 89.59% | 7.57% | 7.22% | 42.10% | 69.35% | 4.48% | 4.91% |
| `round6-controlled-moe-headhead-k-elayerin` | 75.91% | 89.52% | 7.26% | 7.48% | 41.06% | 69.95% | 4.20% | 4.94% |
| `round6-controlled-moe-headhead-k-ehidden` | 79.24% | 91.18% | 7.36% | 7.48% | 41.92% | 70.38% | 4.20% | 5.10% |
