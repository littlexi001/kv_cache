# Section 27: Token-wise KV Retrieval 的数学性质探索

Date: 2026-06-29

## 0. 背景判断

最近几个项目给出的信息可以合在一起看：

1. `qabs reuse` 说明：用 query 中幅值最大的少量通道做 partial-QK，再复用相邻 decode step 的候选集合，质量上是有效的；但继续手工试 `Dq/candidate_fraction/reuse` 的效率很低。
2. `counterfactual risk-gated KV cache` 说明：静态压缩规则不稳定，必须在局部窗口内估计风险；层级和 token 级别都需要 gate。
3. `influence-bounded synthetic KV` 说明：不一定要把问题限制成“找真实 token”，也可以把远程 KV 看成 attention function approximation。
4. `top2 head overlap` 说明：跨 head 重合 token 不是简单冗余，sink 和 broad recent token 的跨 head link 有因果必要性。
5. `tail10 top1 attention` 说明：不同层的职责明显不同。低层更偏局部形式，中高层更明显命中答案证据，最后层又偏向 tail / 输出格式。
6. `K-L2 energy analysis` 说明：KV 几何结构对某些 head 有效，但不是所有 head 都有效。

因此，下一步不应该继续只靠枚举 qabs 参数，而应该先建立一组可测的数学性质：

```text
给定 layer/head/token，KV retrieval 为什么可以被少量 query 通道、相邻 token reuse、K-L2 邻域或 synthetic prototype 近似？
这些近似在哪些层/head/token 上有理论上可解释的稳定性？
什么时候必须 fall back 到 full attention？
```

## 1. 基础形式化

对某层 `l`、head `h`、decode/query token `t`，真实 attention score 为：

```text
s_t(j) = q_t^T k_j / sqrt(d),      j < t
```

真实 top-k 检索集合为：

```text
T_t(k) = top-k_j s_t(j)
```

任意 cheap retrieval 方法先产生候选集合：

```text
C_t = R(q_t, K_{<t}, state_{t-1})
```

然后只在 `C_t` 内 rerank 或 attention。核心问题不是“C_t 大不大”，而是三件事：

```text
Recall:      T_t(k) 中的重要 token 是否被 C_t 覆盖？
Influence:   没覆盖的 token 对 attention output / logits / loss 的影响多大？
Stability:   这种覆盖关系是否能跨 token、跨层、跨文本片段泛化？
```

因此，应把 KV retrieval 研究从经验参数搜索转成：

```text
score ordering approximation
+ attention output perturbation
+ layer/head/token heterogeneity
+ online risk estimation
```

## 2. 性质一：query 通道贡献的集中性

`qabs` 的隐含假设是：

```text
q_t^T k_j = sum_i q_t[i] k_j[i]
```

这个和式里，少数维度贡献了足够多的 token 排序信息。当前 qabs 选择 `|q_t[i]|` 最大的维度，本质上是在假设：

```text
|q_t[i]| 大的通道更能决定 s_t(j) 的相对排序。
```

但这不是唯一合理的通道重要性。更好的数学量可能是：

```text
importance_i(t, l, h) = |q_t[i]| * std_j(k_j[i])
```

或者：

```text
importance_i = |q_t[i]| * MAD_j(k_j[i])
importance_i = Var_j(q_t[i] k_j[i])
importance_i = estimated mutual information with top-k membership
```

原因是：如果某个 query 通道幅值很大，但历史 key 在这个通道几乎是常数，它对 token 间排序帮助不大。真正决定 top-k 的是 `q_i k_j_i` 在不同 `j` 之间的可分性。

### 可测指标

对每个 `(layer, head, token)`，比较 full score 和 partial score：

```text
s_full(j) = q^T k_j
s_S(j)    = sum_{i in S} q[i] k_j[i]
```

记录：

```text
top-k recall:       |top_k(s_full) ∩ top_m(s_S)| / k
Kendall / Spearman: s_full 与 s_S 的排序相关
score error:        max_j |s_full(j) - s_S(j)|
margin coverage:    top-k 边界 margin 是否大于 partial error
energy coverage:    sum_{j in C} softmax(s_full)_j
```

关键不是只看平均 recall，而要看 tail：

```text
P(recall < 0.9)
P(energy < 0.95)
worst 5% tokens
```

因为 PPL 崩坏往往来自少数 high-risk token。

## 3. 性质二：top-k membership 的 margin 稳定性

如果 cheap score `s_hat` 对 full score 的误差是：

```text
e(j) = s_full(j) - s_hat(j)
```

对于真实第 `k` 名 token `a` 和候选外 token `b`，只要：

```text
s_full(a) - s_full(b) > |e(a)| + |e(b)|
```

那么 `a` 不会被 `b` 反超。也就是说，top-k 检索是否安全取决于：

```text
score margin / approximation error
```

而不是固定的 `Dq=8/16`。

这给 qabs 一个更 principled 的设计：

```text
动态选择通道数 Dq，直到 partial score 的误差上界或经验误差分位数小于 top-k 边界 margin。
```

工程上不一定能得到严格误差上界，但可以用 calibration 估计：

```text
err_q95(layer, head, Dq)
margin_q05(layer, head, token)
risk_score = err_q95 / max(margin, eps)
```

如果 `risk_score` 高，就增大候选集合或 fall back。

## 4. 性质三：相邻 decode token 的 retrieval 连续性

`reuse previous candidate/top2` 有效的数学原因应该来自 query 连续性。

相邻 token 的 score 变化为：

```text
s_t(j) - s_{t-1}(j) = (q_t - q_{t-1})^T k_j
```

若：

```text
||q_t - q_{t-1}|| * ||k_j|| <= small
```

并且 top-k margin 足够大，则 `T_t(k)` 和 `T_{t-1}(k)` 应该有高 overlap。反之，如果当前 token 触发语义跳转、格式切换或答案抽取，reuse 可能失效。

### 可测指标

对每层/head统计：

```text
query_delta_norm = ||q_t - q_{t-1}||
score_delta_max  = max_j |s_t(j) - s_{t-1}(j)|
topk_jaccard     = J(T_t, T_{t-1})
energy_reuse     = sum_{j in T_{t-1}} a_t(j)
```

然后拟合关系：

```text
topk_jaccard ≈ f(query_delta_norm, margin, layer, head, token_type)
```

这能回答一个关键问题：

```text
previous candidate reuse 到底在哪些层/head/token 类型上有数学依据？
```

如果某些层的 `query_delta_norm` 小、top-k overlap 高，就适合 aggressive reuse；如果某些层 query jump 很大，就应该降低 reuse 权重或直接 full/rerank。

## 5. 性质四：层的语义阶段不同，retrieval 目标也不同

从 `tail10 top1 attention` 看，层间功能并不一致：

```text
低层：局部形式、token 表面、tail context。
中层：逐渐出现远程答案证据。
中高层：语义检索最明显。
最后层：重新偏向 tail / 输出格式 / logits 形成。
```

因此，一个统一的 qabs 规则对所有层使用同样参数，本身就不合理。

更合理的层级分类是：

```text
local/syntax layers:
  主要保护 recent + sink，远程检索收益低。

semantic retrieval layers:
  需要高 recall 的远程候选，适合 qabs/K-L2/reuse/risk gate。

integration layers:
  远程证据仍有用，但 attention 更分散，适合较大候选或 output-risk gate。

logit/output layers:
  可能强依赖 tail，远程 token selection 不一定是主要瓶颈。
```

这也解释了 section26 的现象：固定压缩 `22,23` 或 `7,8` 有时失败，因为层的功能和当前文本/token 状态不匹配。

## 6. 性质五：跨 head overlap 不是冗余，而是 link 级结构

`top2 head overlap` 的核心教训是：

```text
token 被多个 head 选中，不等于这些 head-token link 是冗余的。
```

高 overlap token 主要有两类：

```text
sink token:    很多 head 同时依赖序列开头位置。
broad recent: 多个 head 同时依赖较宽的近邻上下文。
```

所以 KV retrieval 不应只做 token-level keep/drop，还要区分：

```text
token-level retrieval: 这个历史 token 是否进入候选？
head-link retrieval:   这个 token 对哪些 heads 必须可见？
```

对同一个 token，如果只保留少数 head link，可能破坏多 head 共同使用的计算结构。更安全的规则是：

```text
1. sink/recent 区域保留跨 head link；
2. 非保护区域才做 head cap 或 head-wise sparse retrieval；
3. cap 决策使用 normalized score gap，而不是固定 top3。
```

这和 `head_recent` 的正面结果一致：压缩需要落到 `(layer, head)` 粒度，而不是整层或全 head 粗暴处理。

## 7. 性质六：missed token 的风险应看 output influence，而不是只看 recall

即使漏掉一个 top-k token，也未必造成 loss 上升；反过来，漏掉少数高 influence token 可能导致严重错误。

attention output 为：

```text
y = sum_j a_j v_j
```

如果候选集合为 `C`，被丢弃集合为 `M`，一个粗略上界是：

```text
||y_full - y_C|| <= omitted_mass * value_range
```

其中：

```text
omitted_mass = sum_{j in M} a_j
```

更精确地，可以记录：

```text
delta_y = y_full - y_sparse
delta_logits
delta_loss
```

这正好连接到 section26 的 counterfactual risk：

```text
risk(token, method) = loss_sparse(token) - loss_full(token)
```

数学探索里应该建立从便宜特征到 risk 的映射：

```text
features:
  top-k recall
  attention energy coverage
  score margin
  qabs partial/full disagreement
  query_delta_norm
  layer/head id
  entropy / logit margin

target:
  delta_loss or rescue-needed label
```

这样就能把 token-wise retrieval 做成风险预测问题，而不是靠固定参数。

## 8. 性质七：decode 是逐 token 过程，风险有时间结构

KV retrieval 不是一次性处理完整序列，而是逐 token 发生：

```text
t = 1, 2, 3, ...
```

每一步的 retrieval 影响当前 token loss，也影响后续 token 的 hidden state、query 和 cache 写入。因此，token 级别有两种风险：

```text
instant risk:
  当前 token 的 loss gap。

propagation risk:
  当前 sparse/压缩输出改变后续 hidden/query，导致之后更多 token 风险升高。
```

当前 section26 的 block calibration 更关注 instant risk。后续可以补一个 propagation 诊断：

```text
在 block 内第 r 个 token forced sparse，
观察之后 r+1...r+n 的 delta_loss 是否扩大。
```

如果 propagation risk 很弱，token-level rescue 就足够；如果很强，则要在 block 级别更保守。

## 9. 性质八：query manifold 可能低维，适合 function approximation

section25 的 synthetic KV 方向可以和 qabs 统一起来。

如果某层/head 在一个局部 decode block 中的 queries：

```text
Q_block = [q_t, q_{t+1}, ..., q_{t+B}]
```

主要落在低维子空间，那么远程 KV 不需要对所有可能 query 都准确，只需要在这个 query manifold 上近似：

```text
Attn(Q_block, K_remote, V_remote)
```

这时 synthetic KV 的成功条件是：

```text
rank / intrinsic dimension of Q_block is low
and attention output function is smooth on this local manifold.
```

建议测：

```text
SVD(Q_block) explained variance
SVD(score_matrix = Q_block K_remote^T)
numerical rank of attention output Y_block
calibration-to-heldout output MSE gap
```

如果某些层/head 的 query manifold 很低维，它们更适合 synthetic prototype；如果 query 变化大、score matrix 高秩，则更适合真实 token retrieval 或 full fallback。

## 10. 方法设计：从 qabs 参数搜索升级为 Math-Guided Retrieval

建议下一版方法叫：

```text
Math-Guided Token-wise KV Retrieval
```

核心不是固定 `qabs16cand3reuse`，而是对每个 layer/head/token 动态选择 retrieval backend。

### 10.1 Backend 候选

```text
recent_sink:
  只保留 sink + recent，适合 local/syntax/output 层。

qabs_partial:
  用通道子集 partial-QK 产生候选，适合通道贡献集中、margin 稳定的 head。

qabs_reuse:
  当前 qabs candidate union previous candidate/topk，适合 query 连续性强的 head。

k_l2_expand:
  从历史高 attention seeds 出发扩展 K-L2 邻居，适合 K 几何和 attention 目标一致的 head。

synthetic_kv:
  用少量 prototype 近似远程 attention function，适合 query manifold 低维的 head。

full:
  高风险 token/layer/head fallback。
```

### 10.2 Online selector

每个 block 做短 calibration，估计：

```text
channel concentration
partial score recall
top-k margin
query continuity
energy coverage
delta_loss risk
```

然后选择最便宜且满足风险约束的 backend：

```text
choose cheapest backend b
subject to:
  mean_delta_loss(b) <= eps_mean
  p95_delta_loss(b) <= eps_tail
  min_energy_coverage(b) >= tau_energy
```

如果 block 内 token 出现 cheap risk signal：

```text
low logit margin
high attention entropy
high qabs/full disagreement estimate
large query_delta_norm
```

则 token-level rescue 到更保守 backend。

## 11. 最小实验路线

### Stage A: 数学 profiling，不改 forward

目标：先证明哪些数学性质存在。

对已有 Qwen3-0.6B，dump 若干文本和 needle prompt 的 full attention 信息，统计：

```text
1. layer/head 的 qabs top-k recall heatmap
2. |q| 通道 vs |q|*std(K) 通道的 recall 对比
3. top-k margin 分布
4. adjacent-token top-k Jaccard
5. query_delta_norm 与 reuse energy 的关系
6. omitted attention mass 与 delta_loss 的关系
7. Q_block / score_matrix / Y_block 的 numerical rank
```

成功标准：

```text
能把 heads 分成至少 3 类：
  local/recent heads
  qabs-stable retrieval heads
  unstable/full-risk heads
并且这种分类能解释 qabs reuse 的成功和失败。
```

### Stage B: 替换 qabs 通道选择规则

比较：

```text
top |q_i|
top |q_i| * std(K_i)
top contribution variance
random projection
learned per-layer/head channel mask
SVD/PCA subspace projection
```

指标：

```text
candidate recall
energy coverage
PPL
runtime
```

关键判断：

```text
如果 |q|*std(K) 明显优于 |q|，
说明 qabs 的正确数学对象不是 query 幅值，而是 score variance contribution。
```

### Stage C: layer-adaptive qabs/reuse policy

不要所有层同一配置。根据 Stage A 分类：

```text
low layers:
  recent_sink 或小候选

mid semantic layers:
  qabs_reuse + 较高 recall

high integration layers:
  qabs_reuse + risk gate

last layer:
  recent/tail-biased 或 full
```

比较：

```text
uniform qabs16cand3reuse
vs layer-adaptive qabs
vs risk-gated layer-adaptive qabs
```

### Stage D: token-wise risk rescue

对每个 token 记录 cheap features 和 counterfactual `delta_loss`，训练或手写一个 risk score：

```text
r_t = f(margin, entropy, query_delta, qabs_disagreement, layer-risk)
```

策略：

```text
if r_t > threshold:
  use safer retrieval or full attention
else:
  use cheap retrieval
```

成功标准：

```text
在相同平均候选预算下，risk rescue 的 PPL tail 明显好于固定 qabs。
```

### Stage E: synthetic KV 接入

只对 Stage A 中 query manifold 低维、calibration-to-heldout MSE 稳定的层/head 做 synthetic KV。

不要全层上来就替换。先验证：

```text
selected heads synthetic
vs selected heads qabs
vs selected heads top real token
```

这样 synthetic KV 不再是另一个孤立方向，而是作为数学 profiling 后的一种 backend。

## 12. 当前最值得验证的三个假设

### H1: qabs 成功来自 score variance contribution，而不是单纯 query abs

预测：

```text
top |q_i| * std(K_i)
```

会在相同维度数下比 `top |q_i|` 有更高 top-k recall 和 energy coverage。

如果成立，qabs 可以从启发式通道选择升级为 score decomposition 方法。

### H2: reuse 成功来自 query continuity + top-k margin

预测：

```text
topk_jaccard(T_t, T_{t-1})
```

可以被：

```text
query_delta_norm
score_margin
layer/head id
```

较好解释。

如果成立，reuse 不应该固定打开，而应该按 token/layer/head 动态打开。

### H3: 层级最优策略呈 U 形或分段结构

预测：

```text
低层和最后层更适合 recent/sink；
中高层更需要远程 semantic retrieval；
少数高风险层需要 risk gate 或 full fallback。
```

如果成立，统一 qabs 参数不是最优，layer-adaptive policy 应该在同等预算下更稳。

## 13. 和现有项目的关系

这个探索不是替代 section24/25/26，而是把它们合并成一条主线：

```text
section24 qabs reuse:
  提供 cheap retrieval backend。

section25 synthetic KV:
  提供 function approximation backend。

section26 counterfactual risk gate:
  提供 online backend selector 和 token rescue。

section21/19/15:
  提供 layer/head/token 的数学结构证据。
```

建议主线表述为：

```text
KV retrieval should be token-wise, layer-aware, and risk-gated.
The retrieval backend should be chosen from measurable geometric properties
of query-channel concentration, temporal continuity, attention influence,
and layer/head specialization.
```

中文表述：

```text
KV 检索不应该是固定规则，而应该是逐 token、分层、风险约束的动态决策。
选择哪种检索方法，应该由 query 通道贡献集中性、相邻 token 连续性、
attention output influence 和层/head 功能分化这些可测数学性质决定。
```

## 14. 下一步建议

最优先的新项目不是再写一个 sparse attention kernel，而是先建一个 profiling 项目：

```text
ymluo/projects/qwen3_tokenwise_kv_math_profile/
  src/
    dump_full_attention_stats.py
    analyze_channel_contribution.py
    analyze_temporal_reuse.py
    analyze_layer_head_taxonomy.py
    fit_risk_predictor.py
  scripts/
    run_math_profile.sh
```

第一批输出应该是几张 heatmap/table：

```text
1. layer x head: qabs recall / energy coverage
2. layer x head: reuse top-k Jaccard
3. layer x head: query_delta_norm
4. layer x head: margin-risk score
5. layer x head: recommended backend
```

只有这些图出来以后，后续参数搜索才有方向：

```text
不是问 qabs16 还是 qabs8，
而是问这个 layer/head/token 是否满足 qabs 的数学适用条件。
```

