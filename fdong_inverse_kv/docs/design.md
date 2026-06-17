# Head-level KV/Expert Shared Bucket: Design

## 0. Objective and Current Conclusion

已有实验事实：在每层每个 head 中只保留 top-2% attention token，推理质量基本不下降，某些情况下还会优于 full attention。因此当前不是从零证明 sparse attention 是否可用，而是解释：

> 对某一层某个 head，top-2% key token 为什么被当前 query 选中？它们共享的 feature 在哪里？

这个问题直接决定 inverse KV 的设计。如果 top-2% token 共享的 feature 可以被 pre-attention 表征预测，那么这个 feature 就可以成为 KV bucket 和 expert bucket 的共同 index。

当前要区分三种可能：

1. **Hidden-native feature**：输入 residual hidden state 已经包含这种聚类，Q/K 只是在读出已有 feature；
2. **Q/K-extracted feature**：hidden 中相似性不明显，但该 head 的 `W_Q/W_K` 把当前 head 关心的 feature 抽取出来；
3. **Bilinear/SVD feature**：高 attention score 不是单一 hidden cosine 或 K-space cosine 能解释，而是来自完整 `W_Q/W_K` 奇异空间中的少数 query/key feature pairs。

本方法希望先解释这些 feature，再用这些 feature 指导 head-level KV bucket 与 expert bucket 的共同设计。

## 0.1 Architecture Objective

本方法希望从 Transformer block 的入口开始，在每个 attention head 内学习一个离散 bucket。这个 bucket 同时决定：

1. 当前 head 可以读取哪些历史 KV；
2. 当前 head 的表征进入哪个 expert。

核心结构是：

```text
one layer
-> split computation by attention head
-> each head predicts one bucket
-> same-bucket historical head states perform exact attention
-> the same bucket selects the expert for that head
-> merge heads and retain the ordinary residual stream
```

当前结论只是：该结构具有实验依据，值得在真实数据上训练。现有结果尚未证明共享 bucket 优于独立的 KV bucket 和 expert bucket。

## 1. Falsifiable Conjecture

当前更基础的 conjecture 是：

> top-2% attention token 不是任意稀疏集合；对每一层每个 head，它们共享该 head 定义的少数 feature directions。这些 directions 可以通过 hidden space、Q/K space，或完整 `W_Q/W_K` SVD 后按 head 切输出空间的方式被定位。

如果该 conjecture 成立，那么 inverse KV 的 bucket 不应按 token id 或位置定义，而应按 head-specific feature 定义。

以下结果会削弱该 conjecture：

1. top-2% pair 与匹配距离的 random pair 在 hidden/QK/SVD feature 指标上没有差异；
2. top-2% token 主要由 sink、position 或 norm 解释，而不是 feature direction 解释；
3. 不同 head 的 top-2% token 集合高度重叠，缺少 head-specific 分工；
4. top-2% 的有效性只来自后续层补救，而不是当前 head 的 feature retrieval。

在该问题之后，结构 conjecture 才是：

> 在真实语言训练中，使用 head-level、因果中心化的共享 bucket，同时约束 attention 可见性与 expert ownership，可以在显著减少 KV 访问的同时维持 NTP，并形成比 ordinary MoE 更稳定的 expert specialization。

更强、也更关键的版本是：

> 在相同 KV candidate ratio、active parameter count 和训练预算下，共享 KV/expert bucket 优于分别学习 KV bucket 和 expert bucket。

以下任一结果都会削弱该假设：

1. shared bucket 明显差于 separate buckets；
2. 模型只能通过 bucket collapse 或接近 dense attention 保持 NTP；
3. KV 访问减少，但 expert specialization 没有改善；
4. expert specialization 改善，但没有带来 retrieval 或系统收益；
5. bucket 计算开销抵消了减少 KV 访问的收益。

## 2. Physical Priors

### 2.0 Top-2% token should share a head-specific feature

Self-attention 在每个 head 上通过内积选择 token：

$$
\mathrm{score}(i,j,h)=q_{i,h}^{\top}k_{j,h}
$$

如果只保留每个 head 的 top-2% key token 仍能完成推理，那么这些 token 不是均匀可替换的历史上下文，而是当前 head 认为最相关的少数 feature carrier。

这个 feature 可以有三个来源：

1. `x_i` 和 `x_j` 在 residual hidden space 中已经相似；
2. `W_Q/W_K` 从 `x` 中抽取出当前 head 关心的 feature，使 `q_i` 与 `k_j` 相似；
3. 完整 `W_Q/W_K` 的 SVD 定义了全局 query/key singular feature；这些 feature 经由 $U_Q/U_K$ 写入不同 head，高 score 来自少数 query/key feature pair 的乘积。

因此实验必须先回答 feature 来源，再决定 router input 应该是 layer input、Q、K，还是一个学习到的 Q/K-like projection。

### 2.0.1 Mathematical model for full-WQ/WK feature source

对某一层，先对完整 Q/K projection 做 SVD：

$$
W_Q=U_Q\Sigma_QV_Q^{\top}
$$

$$
W_K=U_K\Sigma_KV_K^{\top}
$$

其中：

1. $V_Q,V_K$ 在输入 residual hidden space 中，表示 query/key 读取的全局输入 feature；
2. $\Sigma_Q,\Sigma_K$ 表示这些 feature 被放大的强度；
3. $U_Q,U_K$ 在 concat 后的 Q/K 输出空间中，可以按 head 坐标切分。

记第 $h$ 个 head 的输出坐标切片为 $U_{Q,h}$ 和 $U_{K,h}$，则：

$$
q_{h,i}=x_iV_Q\Sigma_QU_{Q,h}^{\top}
$$

$$
k_{h,j}=x_jV_K\Sigma_KU_{K,h}^{\top}
$$

第 $h$ 个 head 的 QK score 为：

$$
s_{h,i,j}
=x_iV_Q\Sigma_QU_{Q,h}^{\top}U_{K,h}\Sigma_KV_K^{\top}x_j^{\top}
$$

因此第 $r$ 个 query singular feature 和第 $s$ 个 key singular feature 的贡献为：

$$
c^{(h)}_{i,j,r,s}
=(x_iV_Q)_r
\sigma_{Q,r}
\left(U_{Q,h}^{\top}U_{K,h}\right)_{r,s}
\sigma_{K,s}
(x_jV_K)_s
$$

这里的可测变量是：

1. `cos(x_i, x_j)`：hidden-native feature 是否已经存在；
2. `q_i^T k_j` 或 `cos(q_i, k_j)`：Q/K 是否抽取出更强相关性；
3. $U_Q/U_K$ 的 head mass：某个全局 singular feature 被集中写入少数 head，还是分散写入多个 head；
4. $c^{(h)}_{i,j,r,s}$：某个 query/key singular feature pair 对 high score 的贡献；
5. top-contribution feature-pair overlap：top pair 是否共享少数稳定的 query/key feature pairs。

如果 top-2% pair 的 $c^{(h)}_{i,j,r,s}$ 集中在少数 feature pairs，而 matched negative pair 没有这种集中性，说明该 head 的 retrieval 可以被完整 $W_Q/W_K$ 奇异空间中的 feature pairs 解释。

对 $M_h=W_{Q,h}^{\top}W_{K,h}$ 做 SVD 只作为辅助 sanity check。它能说明组合后的双线性匹配是否集中，但不能回答原始 $W_Q/W_K$ 奇异 feature 如何分配到不同 head。

### 2.1 Strong QK relations have local closure

Qwen3-0.6B 的真实文本实验显示，在每个 head 的 strict-history top-2% QK 关系中：

1. 两跳闭包率约为 `27%-34%`；
2. 相对距离和 attention-target popularity 匹配基线，closure lift 约为 `1.70x-2.95x`；
3. 相似 query 的 retrieval-set overlap 高于位置匹配随机 query；
4. 关系越严格，closure lift 越高。

这支持把高分 QK neighborhood 作为 coarse retrieval bucket，但不支持把它当成严格等价类。因此 bucket 只负责候选生成，bucket 内仍需执行 exact qK attention。

### 2.2 The computation should be head-level

不同 layer/head 的 QK geometry、attention sparsity 和 long-range responsibility 差异明显。同一个 token 在不同 head 中可能表达不同 retrieval feature。

因此 bucket assignment 应为：

$b[\mathrm{layer},\mathrm{token},\mathrm{head}]$

而不是：

$b[\mathrm{layer},\mathrm{token}]$

### 2.3 Raw K contains a large common center

令某个 layer/head 的 key 为：

$$
k_i=c+r_i
$$

则：

$$
k_i^{\top}k_j=\lVert c\rVert^2+c^{\top}r_i+c^{\top}r_j+r_i^{\top}r_j
$$

大的 `||c||^2` 会让 raw K-K inner product 普遍很高，产生虚假相似性。

但对固定 query：

$$
q^{\top}k_i=q^{\top}c+q^{\top}r_i
$$

`q^T c` 对所有历史 token 相同，会在 token-wise softmax 中抵消。因此 router 应主要使用 residual K geometry，而不是 raw common center。

### 2.4 Similar representations produce similar routing only with margin

对线性 gate：

$$
z_i=Gx_i+b
$$

有：

$$
\lVert z_i-z_j\rVert \le \lVert G\rVert_{\mathrm{op}}\lVert x_i-x_j\rVert
$$

如果两个输入接近，router logits 也会接近。但 top-k expert assignment 保持一致还要求 routing margin 足够大。

因此需要同时测量：

1. router input distance；
2. router logits distance；
3. top-k expert overlap；
4. top-k / top-(k+1) routing margin。

### 2.5 Causal center is a coordinate statistic, not a learned communication path

历史均值只用于定义当前 token 的坐标原点。它不应成为当前 routing loss 向历史 token 传播梯度的路径。

因此 prefix center 必须 stop-gradient：

$$
\mathrm{center}=\mathrm{stop\_gradient}(\mathrm{prefix\_mean})
$$

这避免模型通过操纵历史均值取巧，也与 decode 时已经写入 cache 的历史状态保持一致。

## 3. Model Structure

设：

| Symbol | Meaning |
|---|---|
| `T` | sequence length |
| `d` | model hidden size |
| `H` | attention query-head count |
| `H_kv` | KV-head count |
| `d_h` | head dimension |
| `E` | bucket count and expert count per head |
| `X^l` | input residual stream of layer `l`, shape `[B,T,d]` |

第一版使用 query-head level bucket。GQA 的一个 KV head 会先按标准方式复制给其 query-head group；同组 query head 拥有独立 gate，因此复制后的 K/V 可以进入不同的逻辑 bucket。当前实现验证算法行为，尚未实现按 bucket 物理分区的 decode cache。

### 3.1 Standard QKV projection

每层先保留标准 pre-norm 和 QKV projection：

$$
X_{\mathrm{norm}}=\mathrm{RMSNorm}(X^l)
$$

$$
Q=X_{\mathrm{norm}}W_Q,\quad
K=X_{\mathrm{norm}}W_K,\quad
V=X_{\mathrm{norm}}W_V
$$

reshape：

```text
Q: [B,T,H,d_h]
K: [B,T,H_kv,d_h]
V: [B,T,H_kv,d_h]
```

Q/K/V 计算没有被 bucket 替代。Bucket 只改变 attention 可见集合和 expert dispatch。

### 3.2 Router input candidates

Router 必须在 attention 前得到。候选包括：

1. `layer_input_head = X_norm W_R`；
2. pre-RoPE `Q`；
3. pre-RoPE `K`；
4. pre-RoPE `V`。

第一版推荐：

$$
\mathrm{router\_input}=\mathrm{pre\text{-}RoPE}\ K
$$

原因：

1. K 是历史 token 作为可检索地址的表示；
2. K 在 attention 前可计算；
3. 既有 synthetic 实验中 K routing 的 specialization 最强；
4. pre-RoPE K 不混入当前位置旋转，更适合估计跨 token common center。

该选择仍是一个 operationalization，不代表 K 是唯一正确 router input。实验必须包含 layer input、Q 和 raw/centered K 对照。

### 3.3 Causal stop-gradient centering

对 layer `l`、head `h`、位置 `t` 的 router input `R^l_{t,h}`，定义 exclusive prefix mean：

$$
\mu^l_{t-1,h}=\frac{1}{t-1}\sum_{j<t}R^l_{j,h}
$$

中心化输入：

$$
R^l_{\mathrm{centered},t,h}
=R^l_{t,h}-\mathrm{stop\_gradient}(\mu^l_{t-1,h})
$$

随后归一化：

$$
R^l_{\mathrm{router},t,h}
=\mathrm{RMSNorm\_or\_L2Norm}(R^l_{\mathrm{centered},t,h})
$$

位置 `t=0` 时定义：

$$
\mu=0
$$

#### Gradient contract

允许的梯度：

```text
router loss
-> current R_t
-> current W_K / W_Q / W_R
```

禁止的梯度：

```text
router loss at t
-> prefix mean
-> R_0 ... R_{t-1}
```

该 stop-gradient 只作用于 center statistic。历史 K/V 参与普通 attention 时仍保留正常 NTP 梯度。

### 3.4 Head-level bucket assignment

每个 head 有独立 gate：

$$
z^l_{t,h}=G^l_hR^l_{\mathrm{router},t,h}+b^l_h
$$

其中：

$z\in\mathbb{R}^{B\times T\times H\times E}$

第一版使用 top-1：

$$
b^l_{t,h}=\arg\max_e z^l_{t,h,e}
$$

每个 head 的 bucket 数量等于 expert 数量：

$$
\mathrm{bucket}\ e \leftrightarrow \mathrm{expert}\ E^l_{h,e}
$$

历史 token 的 bucket id 在写入 KV cache 时确定，之后不因 prefix mean 更新而重新分桶。

### 3.5 Same-bucket causal attention

当前 token `t`、head `h` 的同 bucket 历史集合：

$$
C^l_{t,h}=\{j\le t\mid b^l_{j,h}=b^l_{t,h}\}
$$

第一版建议加入 local fallback：

$$
L_t=\{\max(0,t-w),\ldots,t\}
$$

$$
\mathcal{V}^l_{t,h}=C^l_{t,h}\cup L_t\cup \mathrm{sink\_tokens}
$$

attention 仍使用原始位置上的 post-RoPE Q/K：

$$
A^l_{t,h}
=\mathrm{softmax}\left(\frac{Q^l_{t,h}(K^l_{\mathcal{V}_t,h})^{\top}}{\sqrt{d_h}}\right)
$$

$$
O^l_{t,h}=A^l_{t,h}V^l_{\mathcal{V}_t,h}
$$

Bucket 不重排 token，也不改变 position id。

Local fallback 的作用是避免训练早期错误 bucket 完全切断局部语法梯度。它必须作为消融项，不能默认其收益来自 shared bucket。

### 3.6 Head-level expert computation

每个 head 拥有独立 expert 集合：

```text
Expert^l_{h,1}, ..., Expert^l_{h,E}
```

需要先为 residual stream 构造 head-level 表征。Qwen3-0.6B 的 `hidden_size/H=64`，而 attention `head_dim=128`。当前实现不增加额外 residual projection，而是把 normalized residual stream 均匀切成 `H` 个 64 维 slice：

$$
X^l_{\mathrm{head}}=\mathrm{reshape}(X_{\mathrm{norm}},[B,T,H,d/H])
$$

attention 先通过标准 `W_O` 合并回 residual stream；随后 post-attention RMSNorm 的第 `h` 个 slice 进入共享 bucket 对应的 head expert。每个 head expert 输出完整 `d` 维 residual update：

$$
U^l=\mathrm{RMSNorm}\left(X^l+W_O\,\mathrm{Concat}_h(O^l_h)\right)
$$

$$
F^l_{t,h}=\mathrm{Expert}^l_{h,b_{t,h}}(U^l_{t,h}),\quad F^l_{t,h}\in\mathbb{R}^d
$$

合并 heads：

$$
F^l_t=\frac{1}{\sqrt{H}}\sum_h F^l_{t,h}
$$

$$
X^{l+1}_t=X^l_t+W_O\,\mathrm{Concat}_h(O^l_{t,h})+F^l_t
$$

这里没有删除 Transformer 的外部 residual。结构变化只发生在：

1. attention visibility 变成 head-level bucket mask；
2. FFN 变成 head-level experts；
3. attention bucket 与 expert id 共享。

### 3.7 Expert parameterization

每个 head expert 的 MLP：

$$
d/H \rightarrow m \rightarrow d
$$

这与“每个 head 输出后拼接，再乘一个输出矩阵”等价：把公共输出矩阵按 head 输入列切分后，拼接后的线性映射可以写成各 head 映射结果之和。除以 `sqrt(H)` 用于控制独立 head update 相加后的初始化方差。

为了与 ordinary MoE 严格匹配参数量，设 ordinary expert 为 `d -> I -> d`。每层每个 bucket 的 ordinary expert 参数为 `3dI`；shared 结构的 `H` 个 head experts 总参数为 `dm(H+2)`。因此：

$$
m=\frac{3I}{H+2}
$$

对 Qwen3-0.6B：

$$
d=1024,\quad H=16,\quad I=3072,\quad m=512
$$

$$
\mathrm{head\ expert}:64\rightarrow512\rightarrow1024
$$

为了与 dense FFN 或 ordinary MoE 公平比较，必须分别控制：

1. active parameters per token；
2. total parameters；
3. FLOPs；
4. memory bytes accessed。

第一版不能同时放宽所有预算。建议先固定 active parameters，再单独报告 total parameter overhead。

## 4. Training Contract

### 4.1 Parallel causal centering

训练时对 router input 的 detached copy 做 exclusive cumulative mean：

```python
history = router_input.detach()
prefix_sum = history.cumsum(dim=1) - history
count = arange(T).clamp_min(1)
prefix_mean = prefix_sum / count
centered = router_input - prefix_mean
```

必须测试该并行实现与逐 token decode running mean 数值一致。

### 4.2 Bucket mask

训练 mask：

```text
causal_mask AND
(same_bucket OR local_window OR sink_token)
```

Bucket assignment 必须只依赖当前位置及其之前可得信息。不能使用完整序列均值或未来 token 的 bucket。

### 4.3 Gradient through discrete routing

Top-1 bucket 是离散选择。第一版需要明确使用一种训练方式：

1. straight-through estimator；
2. soft bucket mask during warmup, hard mask later；
3. Gumbel-softmax；
4. auxiliary dense attention teacher。

当前推荐的最小实现是：

```text
hard top-1 forward
+ straight-through softmax gradient for expert dispatch
+ local fallback for attention stability
```

但 attention mask 本身对 bucket assignment 不可微。因此仅靠 NTP 是否能训练 router，是本方法的首要风险。必须把它作为实验问题，而不是默认可训练。

### 4.4 Loss

最小实验首先使用：

$$
L=L_{\mathrm{NTP}}
$$

只有观察到 collapse 或 router 无梯度后，才逐项加入：

```text
L_balance
L_attention_mass_recall
L_bucket_stability
L_routing_margin
```

不能一开始同时加入多个 loss，否则无法判断结构本身是否有效。

## 5. Prefill Contract

对每层每头维护：

```text
running_sum[layer, head]
running_count[layer, head]
KV_bucket[layer, head, expert]
```

对 prefill token 按位置依次定义 bucket，但矩阵计算可用 causal prefix-sum 并行实现。每个 token 的：

```text
(K, V, position_id)
```

写入对应 head/bucket 的 cache block。

必须保存原始 position id，因为 bucket 内存储顺序不能代替序列位置。

## 6. Decode Contract

对一个新 token，每层执行：

1. 从 layer input 计算 Q/K/V；
2. 读取该层各 head 的 detached running mean；
3. 中心化 router input 并得到 bucket id；
4. 对每个 head，只读取同 bucket KV，加上 local/sink fallback；
5. 在候选上执行 exact qK attention；
6. 将 head attention state 送入同 id expert；
7. 合并 heads 并加入标准 residual；
8. 把当前 K/V 写入对应 bucket；
9. 用当前 router input 的 detached value 更新 running sum/count。

更新顺序必须保证当前 token 使用的是 exclusive prefix mean，而不是包含自身的 mean。

## 7. Implementation Stages

### Stage 1: Causal center module

输入：`[B,T,H,d_h]` router input。

输出：centered router input、prefix mean、running-state update。

通过条件：parallel prefill 与 token-by-token decode 的输出误差接近数值精度；prefix mean 无梯度。

失败原因：off-by-one、包含当前 token、head 维度混淆、detach 位置错误。

### Stage 2: Head-level router

输入：centered head representation。

输出：router logits、top-1 bucket、margin、load statistics。

通过条件：每个 layer/head 可独立路由；bucket 未 collapse；训练和 decode bucket 一致。

### Stage 3: Bucketed attention

输入：Q/K/V、bucket id、causal position、local fallback。

输出：head attention output、candidate count、attention mass recall。

通过条件：dense-bucket 特例与标准 attention 数值一致；mask 无未来泄露。

### Stage 4: Head-level experts

输入：head residual + head attention output、共享 bucket id。

输出：head expert output、active parameter statistics。

通过条件：bucket 与 expert 一一对应；合并后 shape 与原 residual stream 一致。

### Stage 5: Cache layout

输入：每层每头每 bucket 的增量 K/V。

输出：可增量写入和按 bucket gather 的 cache。

通过条件：实际读取 KV 数与 candidate accounting 一致，且不需要随均值变化重排历史 cache。

## 8. Initial Hyperparameters

| Parameter | Initial value | Meaning | Too small | Too large |
|---|---:|---|---|---|
| `num_buckets` | 4 | experts and KV partitions per head | compression/selectivity weak | buckets underpopulated, training unstable |
| `router_top_k` | 1 | active bucket/expert count | fixed at minimum | overlap weakens hard specialization |
| `local_window` | 32 | always-visible recent tokens | early training may fail | masks bucket contribution and reduces compression |
| `sink_tokens` | 4 | always-visible prefix tokens | may lose attention sinks | unnecessary fixed KV cost |
| `center_mode` | causal prefix mean | remove head common center | no centering leaves common bias | complex estimators add state and instability |
| `center_gradient` | stopped | prevent history-gradient coupling | gradients can manipulate center | not applicable |
| `router_input` | pre-RoPE K | pre-attention address representation | operational choice | must be ablated against Q/layer input |
| `expert_intermediate_size` | 3072 | ordinary-MoE reference width | insufficient baseline capacity | total parameters and optimizer memory increase |
| derived head expert width | 512 | `3I/(H+2)` gives equal parameters for `64 -> 512 -> 1024` head experts | shared experts under-capacity | shared model exceeds baseline budget |

## 9. Current Implementation Boundary

Implemented in `scripts/models/myqwen.py`:

1. full-sequence training and evaluation；
2. head-level top-1 routing；
3. router input ablation among `layer_input/q/k/v`；
4. detached exclusive causal mean centering；
5. same-bucket causal attention with local/sink fallback；
6. shared bucket id for `64 -> 512 -> 1024` head-level expert dispatch；
7. hard-forward/soft-gradient expert scale；
8. per-step routing and candidate-ratio metrics。

Not implemented yet:

1. bucketed KV-cache prefill/decode；
2. physical cache compaction and system latency measurement；
3. optimized grouped expert kernels；
4. auxiliary routing losses；
5. model-parallel sharding。

Therefore the current code can test whether NTP alone learns usable shared buckets, but cannot yet establish end-to-end KV memory or latency savings.

## 10. Required Controls

1. Full attention + dense FFN；
2. Full attention + ordinary MoE；
3. KV bucket only；
4. Head-level expert bucket only；
5. Separate KV bucket and expert bucket；
6. Shared KV/expert bucket；
7. Shared raw-K bucket；
8. Shared causal-centered-K bucket；
9. Token-level shared bucket；
10. Head-level shared bucket。

其中最关键的对照是：

```text
Separate buckets vs Shared bucket
```

它直接检验 Attention retrieval 与 expert computation 是否应共享 partition。

## 11. Required Measurements

### NTP and task quality

- training/validation loss；
- perplexity；
- downstream accuracy；
- boundary/long-range token accuracy。

### KV retrieval

- candidate ratio；
- attention mass recall；
- top-attention-token recall；
- sparse/full attention-output cosine；
- per-layer/head candidate distribution。

### Expert specialization

- effective experts；
- load entropy；
- routing margin；
- bucket ownership stability across checkpoints；
- same-expert representation and next-token-logit similarity。

### Shared-bucket consistency

- same-bucket attention mass；
- bucket/expert identity consistency；
- QK two-hop closure inside learned buckets；
- shared vs separate partition agreement。

### System

- KV bytes read per decode token；
- bucket lookup overhead；
- expert parameter bytes read；
- end-to-end decode latency；
- peak memory。

## 12. Failure Interpretation

| Observed failure | What it means |
|---|---|
| Router collapses under NTP-only | This training operationalization failed; it does not yet falsify shared specialization |
| Centering improves bucket balance but not retrieval | Common center affected routing load, not useful feature geometry |
| KV-only works, Shared does not improve it | Expert sharing adds no demonstrated value |
| Separate outperforms Shared | Attention and expert partitions should likely differ |
| Shared improves NTP but not KV ratio | Benefit may come from MoE architecture rather than reverse indexing |
| Shared reduces KV but loses NTP | Bucket recall or early-training routing is insufficient |
| Synthetic works, real data fails | The synthetic feature prior did not transfer |
| Offline metrics pass but latency does not improve | System index overhead dominates saved attention work |

## 13. Claim Boundary and Next Uncertainty

现有证据允许说：

1. 高分 QK relation 具有显著局部闭包；
2. head-level K geometry 可用于 coarse candidate generation；
3. pre-attention routing 控制 sparse attention 在 synthetic 上可训练；
4. common K center 应从 K-K routing geometry 中移除。

现有证据不允许说：

1. QK relation 是严格传递的 feature 等价类；
2. Attention 与 MoE 天然共享同一个最优 partition；
3. causal-centered K 一定是最佳 router input；
4. shared bucket 已在真实数据训练和真实系统上有效。

当前最小不确定性是：

> 在一个小型真实数据训练实验中，head-level causal-centered K router 能否仅靠 NTP 学出非坍缩 bucket，并在约 25% KV candidate ratio 下保持接近 full-attention 的 validation loss？

只有该问题通过，才值得进一步比较 Shared 与 Separate expert buckets。
