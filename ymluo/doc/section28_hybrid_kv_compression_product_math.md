# Section 28: Hybrid KV Compression Product 的数学分析与产品方案

Date: 2026-06-29

## 0. 目标

最终成品应是一种混合 KV 压缩系统：

```text
对不同 layer、不同 head、不同 token/block，
根据可测数学性质选择不同 KV 压缩策略，
在质量风险受控的前提下降低 KV cache memory 和 attention compute。
```

它不应该是一个固定规则，例如：

```text
所有层都 qabs16cand3reuse
所有层都 landmark recent512 stride64
所有层都 synthetic KV
```

而应该是一个 selector：

```text
HybridKVSelector(layer, head, token/block features) -> backend
```

backend 可以是：

```text
full
recent/sink
landmark
qabs
qabs_reuse
headmix_qabs_reuse
head_recent
synthetic_kv
landmark_synthetic
```

现有代码里 `layerbudgetattn` 已经支持多种 layer budget：

```text
full
recent
landmark
qabs / qabs_reuse / qabs8cand3reuse
headmix_qabs_reuse
head_recent
synthetic
landmark_synthetic
```

因此产品路线不是从零开始，而是把这些 backend 用数学 profiling 和 risk gate 组织起来。

## 1. 统一优化目标

对模型层 `l`、head `h`、decode token `t`，full attention 输出为：

```text
y_full(l,h,t) = Attn(q_{l,h,t}, K_{l,h,<t}, V_{l,h,<t})
```

压缩 backend `b` 的输出为：

```text
y_b(l,h,t) = Attn_b(q_{l,h,t}, compressed(K,V), state_b)
```

最终系统要解的是一个受约束优化：

```text
min_policy   E[ Cost(policy; l,h,t) ]

subject to   E[ Δloss_t(policy) ] <= eps_mean
             Quantile_0.95(Δloss_t(policy)) <= eps_tail
             P(Δloss_t(policy) > eps_bad) <= delta
```

其中：

```text
Δloss_t(policy) = loss_compressed_t - loss_full_t
```

`Cost` 可以是 memory 或 compute 的加权和：

```text
Cost = α * KV_tokens_kept
     + β * QK_dot_products
     + γ * gather/index_overhead
     + η * calibration_overhead
```

这比只看 `candidate_fraction` 更准确，因为 qabs 的当前瓶颈不是 final selected token，而是 candidate generation、mask/indices compaction 和 small kernels。

## 2. 为什么必须混合，而不是单一压缩

已有实验给出的约束很明确：

1. 低层、最后层和中高语义层关注对象不同。
2. 某些 head 的 attention 目标与 K-L2 几何一致，某些不一致。
3. qabs 对某些 layer/head 有效，但不是所有层都适用。
4. cross-head overlap token 不是纯冗余，sink/recent 需要保护。
5. synthetic KV 可能适合 function approximation，但泛化风险高。
6. 静态 layer replacement 不稳定，需要 counterfactual risk gate。

因此，最合理的产品形态是：

```text
offline profile -> layer/head taxonomy -> initial policy map
online calibration -> block-level policy selection
token-level rescue -> fallback
```

## 3. 数学特征一：remote attention mass

首先要判断某个 layer/head 是否真的需要远程 KV。

定义 protected 区域：

```text
P_t = sink(S) ∪ recent(R)
```

远程区域：

```text
M_t = {j < t : j not in P_t}
```

remote attention mass：

```text
ρ_remote(l,h,t) = sum_{j in M_t} a_{l,h,t}(j)
```

如果：

```text
ρ_remote 很低
```

则该 head 基本可用 `recent/sink` 或 `head_recent`。如果：

```text
ρ_remote 高且稳定
```

则必须使用远程 retrieval backend，例如 qabs、landmark、synthetic 或 full。

建议统计：

```text
remote_mass_mean
remote_mass_p95
remote_mass_tail_ratio = P(ρ_remote > τ_remote)
```

分类规则初版：

```text
remote_mass_p95 < 0.05:
  local head -> recent/sink

remote_mass_mean > 0.15:
  retrieval head -> qabs/landmark/synthetic/full candidate
```

## 4. 数学特征二：qabs 通道可压缩性

qabs 的数学基础是 score decomposition：

```text
s(j) = q^T k_j = sum_i q_i k_{j,i}
```

原始 qabs 使用：

```text
S_D(q) = top-D dimensions by |q_i|
```

但更合理的通道贡献应考虑 key 在历史 token 间的方差：

```text
w_i(q,K) = |q_i| * std_j(k_{j,i})
```

因为只有能造成 token 间 score 差异的通道才有检索价值。

partial score：

```text
s_D(j) = sum_{i in S_D} q_i k_{j,i}
```

qabs 适用性看：

```text
Recall_D = | top_k(s) ∩ top_m(s_D) | / k
Energy_D = sum_{j in top_m(s_D)} softmax(s)_j
RankCorr_D = Spearman(s, s_D)
```

推荐的 head 级 qabs score：

```text
QABSScore(l,h) =
  0.4 * mean(Energy_D)
+ 0.3 * mean(Recall_D)
+ 0.2 * p05(Energy_D)
+ 0.1 * RankCorr_D
```

如果 `QABSScore` 高，则该 head 适合 `qabs` 或 `qabs_reuse`。

重要改进方向：

```text
qabs_abs:
  top |q_i|

qabs_var:
  top |q_i| * std(K_i)

qabs_cov:
  top estimated contribution variance Var_j(q_i k_{j,i})
```

如果 `qabs_var` 明显优于 `qabs_abs`，产品中的通道选择规则应替换成 variance-weighted qabs。

## 5. 数学特征三：top-k margin 与保序风险

qabs 是否会漏掉重要 token，关键取决于 score margin。

真实 score 和 cheap score 的误差：

```text
e(j) = s(j) - s_D(j)
```

若真实 top-k token `a` 和候选外 token `b` 满足：

```text
s(a) - s(b) > |e(a)| + |e(b)|
```

则 `a` 不会被 `b` 反超。

定义边界 margin：

```text
γ_k(t) = s_(k)(t) - s_(k+r)(t)
```

其中 `s_(k)` 是第 `k` 大 score，`r` 可以取 candidate over-sampling gap。

定义 qabs 保序风险：

```text
Risk_order = err_q95(D) / max(γ_k, eps)
```

如果：

```text
Risk_order << 1
```

qabs 安全；如果接近或超过 1，需要增大候选集、换 backend 或 full fallback。

## 6. 数学特征四：相邻 token reuse 稳定性

`qabs_reuse` 的数学依据是相邻 query 连续性：

```text
s_t(j) - s_{t-1}(j) = (q_t - q_{t-1})^T k_j
```

如果：

```text
||q_t - q_{t-1}|| * ||k_j|| 小
```

且 top-k margin 大，则当前 top-k 与上一 token top-k 高重合。

定义：

```text
ReuseJaccard(l,h,t) = |T_t ∩ T_{t-1}| / |T_t ∪ T_{t-1}|
ReuseEnergy(l,h,t)  = sum_{j in T_{t-1}} a_t(j)
QueryDelta(l,h,t)   = ||q_t - q_{t-1}|| / max(||q_{t-1}||, eps)
```

reuse 适用性：

```text
ReuseScore(l,h) =
  0.5 * mean(ReuseEnergy)
+ 0.3 * mean(ReuseJaccard)
+ 0.2 * (1 - p95(QueryDelta))
```

如果 `ReuseScore` 高，使用 `qabs_reuse`。如果 `qabs` 可用但 `ReuseScore` 低，使用 current-only `qabs` 或更大 candidate。

## 7. 数学特征五：K 几何可扩展性

K-L2 方法假设：

```text
attention target token 的 key 邻域在 K-space 中有可利用结构。
```

定义 K-neighbor candidate：

```text
C_t = seed_{t-1} ∪ NN_K(seed_{t-1})
```

K 几何适用性：

```text
KGeoEnergy(l,h,t) = sum_{j in C_t} a_t(j)
KGeoGap(l,h,t)    = OracleEnergy_same_budget - KGeoEnergy
```

如果：

```text
KGeoEnergy 高且 KGeoGap 低
```

该 head 适合 K-L2/landmark 类检索。否则说明它更 query-dependent，不适合仅靠 K 几何扩展。

产品初版可先把 K-L2 合并进 `landmark` 或后续 backend，不必第一阶段实现完整 K graph kernel。

## 8. 数学特征六：query manifold 低秩性与 synthetic KV

synthetic KV 的适用条件不是“这个层能不能压缩”，而是：

```text
局部 block 的 query manifold 是否低维；
远程 attention output 是否可被少量 prototype 拟合。
```

对 block queries：

```text
Q_B in R^{B x d}
S_B = Q_B K_remote^T
Y_B = softmax(S_B) V_remote
```

统计：

```text
rank95(Q_B)
rank95(S_B)
rank95(Y_B)
calib_mse
heldout_mse
mse_generalization_gap = heldout_mse / max(calib_mse, eps)
```

synthetic KV 适用性：

```text
SynthScore(l,h) =
  low rank95(Y_B)
  and low heldout_mse
  and low mse_generalization_gap
```

如果 `SynthScore` 高，该 head 可用 `synthetic_kv` 或 `landmark_synthetic`。

如果 calibration MSE 很低但 held-out MSE 高，说明过拟合，应禁止 synthetic backend。

## 9. 数学特征七：head-link overlap 风险

已有 top2 head overlap 实验证明：高 overlap token 不一定冗余。

定义某 token 被多少 head 选中：

```text
c_l,t,j = sum_h 1[j in T_{l,h,t}]
```

高 overlap token 的风险分类：

```text
if j in sink or broad recent:
  preserve cross-head links
else:
  allow head cap with score-gap protection
```

产品规则：

```text
sink/recent token:
  不做跨 head hard cap。

remote non-protected token:
  可以做 headmix 或 head-level qabs。

head cap:
  不能固定 top3，要用 normalized score gap。
```

归一化 score gap：

```text
gap_norm = (score_h - score_cutoff) / std_h(score_h over heads)
```

只删除：

```text
gap_norm < -α
```

的低价值 head-token link。

## 10. 层/头 taxonomy

最终产品应该给每个 `(layer, head)` 打标签。

建议第一版 taxonomy：

| 类型 | 数学条件 | 推荐 backend |
| --- | --- | --- |
| Local head | `remote_mass_p95` 低 | `recent/sink` |
| Streaming head | recent mass 高，query delta 小 | `head_recent` |
| QABS retrieval head | `QABSScore` 高 | `qabs` |
| QABS reuse head | `QABSScore` 高且 `ReuseScore` 高 | `qabs_reuse` |
| Geometry head | `KGeoEnergy` 高且 gap 小 | `landmark` / K-L2 |
| Synthetic head | output rank 低，heldout MSE 稳定 | `synthetic_kv` / `landmark_synthetic` |
| Fragile semantic head | remote mass 高但所有 cheap score 风险高 | `full` 或 conservative `landmark` |
| Output-format head | 最后层 tail mass 高 | `recent/sink` 或 `full` |

注意：分类粒度应是 `(layer, head)`，不是只按 layer。layer 级策略用于第一版工程简化，head 级策略才是最终产品目标。

## 11. Hybrid selector 设计

### 11.1 离线 selector

离线 profiling 生成：

```text
profile[layer][head] = {
  remote_mass_mean,
  remote_mass_p95,
  qabs_energy_mean,
  qabs_energy_p05,
  qabs_recall_mean,
  qabs_var_gain,
  reuse_jaccard,
  reuse_energy,
  query_delta_p95,
  margin_p05,
  kgeo_energy,
  synthetic_rank95,
  synthetic_heldout_mse,
  counterfactual_delta_loss_mean,
  counterfactual_delta_loss_p95,
}
```

然后生成初始 policy：

```text
policy[layer][head] = backend
```

或当前代码更容易支持的 layer map：

```json
{
  "default": {"type": "full"},
  "layers": {
    "0": {"type": "recent", "recent": 512},
    "8": {"type": "qabs8cand3reuse", "dims": 8, "candidate_fraction": 0.03},
    "16": {"type": "landmark_synthetic", "recent": 512, "stride": 64, "prototypes": 16}
  }
}
```

### 11.2 在线 selector

离线 policy 只能给默认策略。真正运行时还需要 block-level calibration：

```text
for each decode block:
  run full on calibration tokens
  run candidate backends on same tokens
  estimate Δloss distribution
  choose cheapest backend satisfying risk constraints
```

选择规则：

```text
backend* = argmin_b Cost(b)
subject to:
  mean(Δloss_b) <= eps_mean
  max(Δloss_b) <= eps_max
  positive_ratio(Δloss_b) <= eps_ratio
```

### 11.3 token-level rescue

对每个 token，用 cheap risk features：

```text
logit_margin
entropy
query_delta
qabs_margin_proxy
reuse_disagreement
backend_risk_prior(layer, head)
```

定义 rescue score：

```text
r_t = θ^T φ_t
```

如果：

```text
r_t > threshold
```

则升级 backend：

```text
recent -> qabs
qabs -> landmark/full
synthetic -> landmark/full
```

不要直接依赖 margin rescue；section26 已经显示 margin rescue 可能过救，也可能漏掉高风险 token。

## 12. 产品架构

建议产品分成四个模块。

### 12.1 Profiler

输入：

```text
model
calibration corpus
long-context / retrieval prompts
```

输出：

```text
profile.csv
profile_by_layer_head.csv
recommended_policy.json
```

复用现有脚本：

```text
analyze_q_sparsity_qabs.py
analyze_adjacent_stability.py
analyze_full_head_remote_mass.py
analyze_qabs_top2_overlap.py
```

需要补充：

```text
analyze_margin_order_risk.py
analyze_synthetic_rank.py
build_hybrid_policy.py
```

### 12.2 Policy Compiler

把 profile 编译成可执行 budget map：

```text
hybrid_policy.json -> layerbudgetattn map
```

第一版先 layer-level：

```text
layer -> backend
```

第二版 head-level：

```text
layer -> head -> backend
```

### 12.3 Runtime

基于现有：

```text
evaluate_qwen3_top2_head_limit3_ppl.py::layerbudgetattn
```

扩展：

```text
layer/head budget map
block calibration selector
token rescue state
```

### 12.4 Evaluator

必须同时评估：

```text
PPL
delta loss mean / p95 / max
retrieval accuracy
downstream accuracy
KV memory saved
attention work proxy
wall time
```

速度指标要拆开：

```text
QK work reduction
selected token count
candidate generation overhead
indices compaction overhead
final attention overhead
```

否则会重复 qabs 当前的问题：理论 token 少了，但 wall time 被索引开销吃掉。

## 13. 第一版产品策略

在现有代码能力下，第一版建议不要直接做完整 head-level runtime，而是做两步。

### V0: layer-level hybrid

使用 `layerbudgetattn` 的 layer map：

```text
低 remote mass 层:
  recent 或 head_recent

qabs stable 层:
  qabs8cand3reuse 或 qabs16cand3reuse

synthetic stable 层:
  landmark_synthetic

fragile 层:
  full
```

目标：

```text
证明 hybrid layer policy 比统一 qabs / 统一 landmark / 统一 synthetic 更稳。
```

### V1: headmix within selected layers

使用已有：

```text
headmix_qabs_reuse
head_recent
```

策略：

```text
full_heads = fragile / sink-heavy heads
qabs_heads = retrieval stable heads
recent_heads = local heads
```

目标：

```text
在相同 PPL 风险下，比 V0 有更高 KV saved。
```

### V2: real head-level policy

支持：

```text
layer_budget_map:
  layer:
    head:
      backend
```

这是最终产品形态。

## 14. 推荐的初始决策树

第一版可以用手写规则，不急着训练模型。

```text
if remote_mass_p95 < 0.05:
    backend = recent

else if counterfactual_delta_loss_p95 > eps_tail:
    backend = full

else if synthetic_heldout_mse < tau_mse
     and synthetic_gap < tau_gap
     and rank95(Y_block) <= r:
    backend = landmark_synthetic

else if qabs_energy_p05 > 0.95
     and qabs_recall_mean > 0.90
     and margin_p05 > tau_margin:
    if reuse_energy_mean > 0.90 and query_delta_p95 < tau_delta:
        backend = qabs_reuse
    else:
        backend = qabs

else if kgeo_energy_p05 > 0.90:
    backend = landmark

else:
    backend = full
```

然后在 block-level calibration 中校正：

```text
if selected backend violates risk:
    promote to safer backend
```

backend 安全序：

```text
recent < synthetic < qabs < landmark < head_recent/full_heads < full
```

这个序不是绝对的，最终要由 calibration 的 `Δloss` 决定。

## 15. 需要新增的关键实验

### Experiment 1: qabs_abs vs qabs_var

目的：验证通道选择数学。

比较：

```text
top |q_i|
top |q_i| * std(K_i)
top Var_j(q_i k_{j,i})
random D dims
```

指标：

```text
top-k recall
attention energy coverage
PPL
```

如果 `qabs_var` 赢，应把 qabs backend 升级为 variance-weighted qabs。

### Experiment 2: layer/head taxonomy heatmap

输出：

```text
remote mass heatmap
qabs energy heatmap
reuse energy heatmap
margin risk heatmap
recommended backend heatmap
```

目标：证明层/头存在明确分工，从而支撑 hybrid 产品。

### Experiment 3: hybrid map vs single backend

比较：

```text
baseline full
uniform qabs
uniform landmark
uniform synthetic
hybrid layer policy
hybrid headmix policy
hybrid + risk gate
```

目标：

```text
hybrid 在相同质量约束下有更高 compression；
或在相同 compression 下有更低 PPL/risk。
```

### Experiment 4: risk gate ablation

比较：

```text
offline policy only
offline policy + block calibration
offline policy + block calibration + token rescue
```

目标：证明 risk gate 不是装饰，而是解决静态策略不稳定的核心。

## 16. 论文/产品主张

可以形成的主张：

```text
LLM KV cache compression is not a single retrieval problem.
Different layers and heads expose different mathematical compressibility:
locality, query-channel concentration, temporal continuity, key-geometry structure,
and low-rank attention function approximation.

A hybrid selector can exploit these properties while using counterfactual risk
to prevent unsafe compression.
```

中文：

```text
KV cache 压缩不是一个统一的 token selection 问题。
不同层、不同 head 具有不同的数学可压缩性：
局部性、query 通道集中性、相邻 token 连续性、K 几何结构、
以及 attention function 的低秩可近似性。

最终系统应该根据这些性质选择混合 backend，并用 counterfactual risk gate
阻止不安全压缩。
```

## 17. 近期执行建议

接下来最值得做的是：

1. 新建 profiling 聚合脚本，汇总已有 `q_sparsity`、`adjacent_stability`、`remote_mass` 输出。
2. 补 `qabs_var` 通道选择实验。
3. 输出第一版 `recommended_hybrid_policy.json`。
4. 用 `run_influence_gated_hybrid_budget.py` 或 `layerbudgetattn` 跑 V0 layer-level hybrid。
5. 如果 V0 成立，再改 runtime 支持真正 head-level map。

第一版成品验收标准：

```text
hybrid policy 在至少 3 个数据集上：
  PPL ratio <= 1.01 或 delta_loss <= 0.01
  effective KV saved 明显高于只压 2 层的 PCIC ladder
  tail risk 小于 uniform qabs / uniform landmark
```

第二版成品验收标准：

```text
head-level hybrid:
  same PPL risk 下，比 layer-level hybrid 多节省 1.5x-2x KV
  block calibration 能稳定避免高风险层/head/token
```

