# Head-level KV/Expert Shared Bucket: Design

## 0. Objective and Current Conclusion

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

```text
b[layer, token, head]
```

而不是：

```text
b[layer, token]
```

### 2.3 Raw K contains a large common center

令某个 layer/head 的 key 为：

```text
k_i = c + r_i
```

则：

```text
k_i^T k_j = ||c||^2 + c^T r_i + c^T r_j + r_i^T r_j
```

大的 `||c||^2` 会让 raw K-K inner product 普遍很高，产生虚假相似性。

但对固定 query：

```text
q^T k_i = q^T c + q^T r_i
```

`q^T c` 对所有历史 token 相同，会在 token-wise softmax 中抵消。因此 router 应主要使用 residual K geometry，而不是 raw common center。

### 2.4 Similar representations produce similar routing only with margin

对线性 gate：

```text
z_i = G x_i + b
```

有：

```text
||z_i - z_j|| <= ||G||_op ||x_i - x_j||
```

如果两个输入接近，router logits 也会接近。但 top-k expert assignment 保持一致还要求 routing margin 足够大。

因此需要同时测量：

1. router input distance；
2. router logits distance；
3. top-k expert overlap；
4. top-k / top-(k+1) routing margin。

### 2.5 Causal center is a coordinate statistic, not a learned communication path

历史均值只用于定义当前 token 的坐标原点。它不应成为当前 routing loss 向历史 token 传播梯度的路径。

因此 prefix center 必须 stop-gradient：

```text
center = stop_gradient(prefix_mean)
```

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

```text
X_norm = RMSNorm(X^l)
Q = X_norm W_Q
K = X_norm W_K
V = X_norm W_V
```

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

```text
router_input = pre-RoPE K
```

原因：

1. K 是历史 token 作为可检索地址的表示；
2. K 在 attention 前可计算；
3. 既有 synthetic 实验中 K routing 的 specialization 最强；
4. pre-RoPE K 不混入当前位置旋转，更适合估计跨 token common center。

该选择仍是一个 operationalization，不代表 K 是唯一正确 router input。实验必须包含 layer input、Q 和 raw/centered K 对照。

### 3.3 Causal stop-gradient centering

对 layer `l`、head `h`、位置 `t` 的 router input `R^l_{t,h}`，定义 exclusive prefix mean：

```text
mu^l_{t-1,h} = 1/(t-1) sum_{j<t} R^l_{j,h}
```

中心化输入：

```text
R_centered^l_{t,h}
  = R^l_{t,h} - stop_gradient(mu^l_{t-1,h})
```

随后归一化：

```text
R_router^l_{t,h}
  = RMSNorm_or_L2Norm(R_centered^l_{t,h})
```

位置 `t=0` 时定义：

```text
mu = 0
```

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

```text
z^l_{t,h} = G^l_h R_router^l_{t,h} + b^l_h
```

其中：

```text
z: [B,T,H,E]
```

第一版使用 top-1：

```text
b^l_{t,h} = argmax_e z^l_{t,h,e}
```

每个 head 的 bucket 数量等于 expert 数量：

```text
bucket e <-> expert E^l_{h,e}
```

历史 token 的 bucket id 在写入 KV cache 时确定，之后不因 prefix mean 更新而重新分桶。

### 3.5 Same-bucket causal attention

当前 token `t`、head `h` 的同 bucket 历史集合：

```text
C^l_{t,h} = {j <= t | b^l_{j,h} = b^l_{t,h}}
```

第一版建议加入 local fallback：

```text
L_t = {max(0,t-w), ..., t}
V^l_{t,h} = C^l_{t,h} union L_t union sink_tokens
```

attention 仍使用原始位置上的 post-RoPE Q/K：

```text
A^l_{t,h}
  = softmax(Q^l_{t,h} K^l_{V_t,h}^T / sqrt(d_h))

O^l_{t,h}
  = A^l_{t,h} V^l_{V_t,h}
```

Bucket 不重排 token，也不改变 position id。

Local fallback 的作用是避免训练早期错误 bucket 完全切断局部语法梯度。它必须作为消融项，不能默认其收益来自 shared bucket。

### 3.6 Head-level expert computation

每个 head 拥有独立 expert 集合：

```text
Expert^l_{h,1}, ..., Expert^l_{h,E}
```

需要先为 residual stream 构造 head-level 表征。Qwen3-0.6B 的 `hidden_size/H=64`，而 attention `head_dim=128`。第一版不增加额外 residual projection，而是把 normalized residual stream 均匀切成 `H` 个 64 维 slice：

```text
X_head^l = reshape(X_norm, [B,T,H,d/H])
```

attention 先通过标准 `W_O` 合并回 residual stream；随后 post-attention RMSNorm 的第 `h` 个 slice 进入共享 bucket 对应的 head expert：

```text
U^l = RMSNorm(X^l + W_O Concat_h(O^l_h))
F^l_{t,h} = Expert^l_{h,b_t,h}(U^l_{t,h})
```

合并 heads：

```text
F^l_t = Concat_h(F^l_{t,h})
X^{l+1}_t = X^l_t + W_O Concat_h(O^l_{t,h}) + F^l_t
```

这里没有删除 Transformer 的外部 residual。结构变化只发生在：

1. attention visibility 变成 head-level bucket mask；
2. FFN 变成 head-level experts；
3. attention bucket 与 expert id 共享。

### 3.7 Expert parameterization

每个 head expert 的默认 MLP：

```text
d_h -> r d_h -> d_h
```

其中 `r` 是 expert expansion ratio。

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

```text
L = L_NTP
```

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
| `expert_intermediate_size` | 3072 | matches active FFN parameters to Qwen3 dense FFN; total model is about 1.389B with 4 experts | insufficient capacity | total parameters and optimizer memory increase |

## 9. Current Implementation Boundary

Implemented in `scripts/models/myqwen.py`:

1. full-sequence training and evaluation；
2. head-level top-1 routing；
3. router input ablation among `layer_input/q/k/v`；
4. detached exclusive causal mean centering；
5. same-bucket causal attention with local/sink fallback；
6. shared bucket id for head-level expert dispatch；
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
