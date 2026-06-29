# Section 29: GPU-Friendly Hybrid KV Compression 方法设计

Date: 2026-06-29

## 0. 方法目标

目标不是再提出一个单一 sparse attention 规则，而是设计一套可产品化的混合 KV 压缩方法：

```text
对不同 layer / head / token block，
根据数学画像选择不同压缩 backend；
同时让 backend 的运行形态尽量 GPU-friendly。
```

核心约束：

1. 不能依赖每 token 大量动态 mask / non-contiguous gather。
2. 不能让每个 head 都产生不同长度候选，否则 kernel 很难高效。
3. 不能先 full-QK 再选 token，否则只是在做 oracle。
4. 不能只静态压缩，因为已有实验显示静态规则跨文本、跨 block 不稳定。
5. 必须保留创新性：不是简单复用 SparQ / Quest / DuoAttention / landmark / synthetic KV，而是把层头数学画像、page 级 GPU 模板和 counterfactual risk gate 合成一个新系统。

建议方法名：

```text
R2H-KV: Risk-Calibrated Routed Hybrid KV Compression
```

含义：

```text
Risk-Calibrated:
  用短 calibration 估计压缩风险。

Routed:
  每个 layer/head group 被路由到不同 backend。

Hybrid:
  同时使用 local、page retrieval、q-sketch、synthetic、full fallback。

KV:
  目标是 KV cache memory + attention compute 压缩。
```

## 1. 总体架构

系统分四层：

```text
Offline Profiler
  -> Policy Compiler
    -> GPU Runtime Backends
      -> Online Risk Gate
```

### 1.1 Offline Profiler

收集每个 `(layer, head)` 的数学特征：

```text
remote_mass
qabs / q-sketch energy coverage
adjacent reuse stability
score margin
query delta
K-geometry consistency
synthetic output reconstruction error
counterfactual delta_loss
```

输出：

```text
profile_by_layer_head.csv
```

### 1.2 Policy Compiler

把数学特征编译成 GPU-friendly policy：

```text
layer -> head_group -> backend template
```

注意是 `head_group`，不是完全自由的每 head 动态策略。head group 的好处是：

```text
同一组 head 使用相同 backend 和固定候选预算，
kernel shape 稳定，更适合 GPU。
```

### 1.3 GPU Runtime Backends

Runtime 只执行少量固定模板：

```text
FULL
LOCAL
PAGE_SELECT
QSKETCH_PAGE
SYNTH_PAGE
HYBRID_SAFE
```

每个模板固定：

```text
page size
selected page count
recent window
head group size
prototype count
```

避免 per-token 变长候选集合。

### 1.4 Online Risk Gate

每个 block 用少量 calibration token 比较 full 与 candidate backend：

```text
Δloss = loss_backend - loss_full
```

如果风险超过阈值，则升级 backend：

```text
LOCAL -> PAGE_SELECT -> QSKETCH_PAGE -> HYBRID_SAFE -> FULL
SYNTH_PAGE -> HYBRID_SAFE -> FULL
```

## 2. 创新点

### 2.1 从 token selection 改成 layer/head backend routing

已有很多方法问：

```text
给定 query，找哪些 token？
```

本方法先问：

```text
这个 layer/head 的数学可压缩性是什么？
它应该使用哪类 retrieval/compression backend？
```

这是从 token-level retrieval 扩展到：

```text
layer/head-level compression routing + token/block-level risk calibration
```

### 2.2 用 q-sketch 替代 per-token qabs top-dim

当前 qabs 的 GPU 问题是每个 token/head 都要：

```text
topk(|q|)
partial-QK
candidate topk
mask union
indices compaction
gather
```

这些步骤动态性强，小 kernel 多。

新方法改成 q-sketch：

```text
每个 layer/head group 离线选出固定 channel groups 或 projection groups。
运行时不再对每个 query 做 top-|q| dim selection。
```

score sketch：

```text
z_q[g] = sum_{i in G_g} q_i * scale_i
z_page[p,g] = summary over K_page[:, G_g]
score_hat(q, page) = z_q^T z_page
```

其中 channel groups `G_g` 由 profiler 根据：

```text
|q_i| * std(K_i)
Var_j(q_i k_{j,i})
top-k membership mutual information
```

离线选出。

创新点：

```text
不是 SparQ 式动态 top-|q| channel；
而是 layer/head-specific fixed q-sketch projection，
用固定形状矩阵乘法做 page routing。
```

这更 GPU 友好。

### 2.3 page 级路由，而不是 token 级动态候选

直接 token candidate 会产生 ragged indices。GPU 友好版本改成 page candidate：

```text
把历史 KV 分成固定 page，每页 P tokens，例如 P=64 或 128。
```

每个 page 存：

```text
K_page_raw
V_page_raw
K_page_summary
V_page_summary
qsketch_summary
optional synthetic prototypes
```

运行时先选固定数量 page：

```text
top B pages
```

然后对这些 page 内 token 做 attention，或者直接对 page prototype attention。

这样候选 shape 固定：

```text
selected_tokens = B * page_size + recent + sink
```

kernel 可以预编译和融合。

### 2.4 risk-calibrated backend promotion

静态 policy 只给初始 backend。每个 block 的真实风险由 counterfactual calibration 决定：

```text
如果当前 block 的语义状态变了，即使 layer/head profile 原本安全，也可以升级 backend。
```

这把 section26 的风险门控和 section28 的层头画像结合起来。

### 2.5 semantic-stage-aware budget prior

把层级语义理解作为 prior，但不硬编码：

```text
浅层倾向 LOCAL / PAGE_SELECT
中层倾向 QSKETCH_PAGE
中高语义层倾向 HYBRID_SAFE / FULL
最后层按 profile 分流，不能一刀切
```

最终由 profiler 和 calibration 修正。

## 3. KV 存储格式

使用 paged KV layout：

```text
KVCache[layer][head_group][page][token_in_page][dim]
```

建议 page size：

```text
P = 64 或 128
```

每页附加 metadata：

```text
PageMeta {
  k_mean
  k_max_norm
  qsketch_summary
  landmark_key
  synthetic_keys[m]
  synthetic_values[m]
  token_start
  token_end
}
```

为了 GPU 友好：

1. page 连续存储。
2. 每个 backend 固定选 `B` 个 page。
3. head group 内共享 page selection。
4. recent window 用连续内存，不走 scatter gather。
5. sink tokens 固定小块，直接常驻。

## 4. Backend 设计

### 4.1 FULL

完整 attention，不压缩。

用于：

```text
fragile semantic heads
high-risk block
calibration baseline
```

### 4.2 LOCAL

只看：

```text
sink + recent
```

适合：

```text
remote_mass 很低的浅层/local heads
最后层 output-format heads
```

GPU 形态：

```text
contiguous recent window + fixed sink
```

没有动态候选。

### 4.3 PAGE_SELECT

使用 page-level summaries 选远程 page：

```text
score_page = q^T k_page_summary
select top B pages
attend to raw tokens inside selected pages + sink + recent
```

适合：

```text
K-geometry稳定，但不适合 synthetic 的 heads。
```

GPU 形态：

```text
Q x page_summary matmul
top-B pages
fixed B * P raw token attention
```

### 4.4 QSKETCH_PAGE

使用 fixed q-sketch 进行 page routing：

```text
z_q = q W_sketch
z_page = page_sketch
score_page = z_q z_page^T
```

然后选固定 `B` 页。

适合：

```text
qabs 能量覆盖高、但 per-token qabs 太慢的 heads。
```

关键创新：

```text
把动态 top-|q| 通道选择改成离线编译的 fixed sketch groups。
```

GPU 形态：

```text
small dense matmul + top-B page selection + fixed page attention
```

### 4.5 SYNTH_PAGE

对远程 page 或 page group 存少量 synthetic prototypes：

```text
K_syn[p, m, d]
V_syn[p, m, d]
```

decode 时 attention 到：

```text
sink + recent + selected synthetic prototypes
```

适合：

```text
query manifold 低秩、attention output 可拟合的 heads。
```

GPU 形态：

```text
固定 m prototypes/page
固定 selected page count
无 ragged token gather
```

### 4.6 HYBRID_SAFE

保守混合 backend：

```text
sink + recent
+ selected raw pages
+ page synthetic prototypes
+ optional full for selected heads
```

用于：

```text
高 remote mass 但 full 过贵；
calibration 显示 LOCAL/QSKETCH/SYNTH 风险偏高但还不必须 FULL。
```

## 5. Layer/Head 策略

第一版可以按层段给 prior：

```text
Layer 0-5:
  LOCAL or PAGE_SELECT
  aggressive budget

Layer 6-13:
  QSKETCH_PAGE or PAGE_SELECT
  medium budget

Layer 14-22:
  HYBRID_SAFE or FULL for fragile heads
  conservative budget

Layer 23-27:
  mixed
  output-format heads -> LOCAL
  semantic integration heads -> HYBRID_SAFE/FULL
```

但最终不硬编码，而是根据 profile：

```text
if remote_mass_p95 low:
  LOCAL

elif qsketch_energy_p05 high and reuse stable:
  QSKETCH_PAGE with small B

elif kgeo_energy_p05 high:
  PAGE_SELECT

elif synth_heldout_mse low and rank low:
  SYNTH_PAGE

elif counterfactual_risk high:
  FULL

else:
  HYBRID_SAFE
```

## 6. GPU-Friendly Policy Compiler

Profiler 输出是连续指标，但 runtime 不能支持无限多配置。Policy compiler 要把它量化成少数 template。

例如：

```text
Template A: LOCAL_R512
  sink=16, recent=512

Template B: QSKETCH_B2_P64
  selected_pages=2, page_size=64, recent=512

Template C: QSKETCH_B4_P64
  selected_pages=4, page_size=64, recent=512

Template D: SYNTH_M8_B4
  prototypes_per_page=8, selected_pages=4

Template E: HYBRID_SAFE_B4_M8
  raw_pages=4, prototypes=8, recent=1024

Template F: FULL
```

每个 `(layer, head_group)` 只选择这些模板之一。

这保证：

```text
kernel 数量有限；
shape 有限；
CUDA graph / torch compile / fused kernel 更容易做。
```

## 7. Online Risk Gate

每个 block 开始用 `C` 个 calibration tokens：

```text
C = 8, 16, 32
```

对候选 templates 计算：

```text
mean_delta_loss
p95_delta_loss
max_delta_loss
positive_delta_ratio
```

选择：

```text
cheapest template satisfying risk constraints
```

为了 GPU 友好，候选 template 数不能太多：

```text
每个 head_group 最多 2-3 个候选：
  default
  safer
  full
```

不要在 runtime 枚举几十个组合。

## 8. Token-Level Rescue

token rescue 不应导致每个 token 都走完全不同 kernel。建议使用 block 内批量 rescue：

```text
先用 selected backend 跑一个 block；
标记 high-risk token；
对 high-risk token 统一用 safer backend 重算。
```

这比逐 token if-else 更 GPU 友好。

risk feature：

```text
logit_margin
entropy
query_delta_norm
qsketch_page_score_margin
selected_page_score_gap
calibration risk prior
```

rescue 规则：

```text
if risk_score > threshold:
  put token into rescue batch
```

rescue batch 统一执行：

```text
HYBRID_SAFE or FULL
```

## 9. 训练/学习组件

第一版不需要训练 selector，用规则即可。

第二版可以训练轻量模型：

```text
risk_predictor(features) -> P(Δloss > eps)
```

但注意 GPU runtime 不依赖复杂模型。risk predictor 可以只输出：

```text
promote / keep
```

或提前离线编译成阈值表。

## 10. 与已有方法的区别

### 10.1 不同于 SparQ / Quest

它们主要是：

```text
cheaply find important tokens for current query
```

本方法是：

```text
profile layer/head compressibility,
compile GPU templates,
then use risk-calibrated backend routing.
```

并且 q-sketch 使用固定 projection/page summaries，而不是 per-token 动态 top-|q| channel。

### 10.2 不同于 DuoAttention

DuoAttention 区分 retrieval heads 和 streaming heads。

本方法进一步区分：

```text
local
q-sketch retrieval
page geometry
synthetic approximation
fragile full
```

并用 counterfactual risk gate 在线校正。

### 10.3 不同于普通 landmark

landmark 是固定稀疏结构。

本方法中 page/landmark 只是 backend 之一，由 profile 和 risk gate 选择。

### 10.4 不同于纯 synthetic KV

synthetic KV 只用于 query manifold 低秩、heldout MSE 稳定的 heads。

不对所有层/head 强行使用 synthetic。

## 11. 实现路线

### V0: 文档和 profiling

新增：

```text
analyze_qsketch_channel_rules.py
build_hybrid_kv_profile.py
compile_hybrid_policy.py
```

先不改 CUDA kernel，只输出：

```text
profile_by_layer_head.csv
recommended_policy.json
```

### V1: layer-level template map

复用现有：

```text
layerbudgetattn
run_influence_gated_hybrid_budget.py
```

把 policy 编译成现有 layer map：

```json
{
  "default": {"type": "full"},
  "layers": {
    "0": {"type": "recent", "recent": 512},
    "8": {"type": "qabs8cand3reuse", "dims": 8, "candidate_fraction": 0.03},
    "16": {"type": "landmark_synthetic", "recent": 512, "stride": 64}
  }
}
```

验证 hybrid 是否优于 single-backend。

### V2: head-group runtime

扩展 map：

```json
{
  "layers": {
    "16": {
      "head_groups": [
        {"heads": [0,1,2,3], "template": "FULL"},
        {"heads": [4,5,6,7], "template": "QSKETCH_B4_P64"},
        {"heads": [8,9,10,11], "template": "LOCAL_R512"},
        {"heads": [12,13,14,15], "template": "HYBRID_SAFE_B4_M8"}
      ]
    }
  }
}
```

### V3: page-based fused kernels

实现核心 kernels：

```text
qsketch_page_score_kernel
topB_page_select_kernel
page_attention_kernel
synth_page_attention_kernel
hybrid_safe_attention_kernel
```

优先融合：

```text
page score -> topB -> page attention
```

避免 dense bool mask 和 padded index compaction。

### V4: risk-calibrated serving

加入：

```text
block calibration
template promotion
rescue batch
runtime accounting
```

## 12. 关键实验

### Experiment A: qabs_dynamic vs qsketch_fixed

比较：

```text
dynamic top-|q| qabs
fixed |q|*std(K) channel groups
fixed learned/information channel groups
random fixed groups
```

指标：

```text
attention energy coverage
top-k recall
runtime proxy
PPL
```

成功标准：

```text
fixed qsketch 接近 dynamic qabs 质量，
但 GPU runtime 更简单。
```

### Experiment B: page routing vs token routing

比较：

```text
token qabs candidate
page qsketch candidate
page landmark candidate
```

指标：

```text
energy coverage
selected token count
kernel launch count
wall time
```

### Experiment C: hybrid policy vs layer-depth prior

比较：

```text
manual shallow aggressive / deep conservative
profile-based hybrid
profile + risk gate
```

目标：

```text
证明 profile-based policy 能修正简单 depth prior 的错误。
```

### Experiment D: head-group granularity

比较：

```text
layer-level policy
head-level policy
head-group policy
```

预期：

```text
head-group policy 接近 head-level 质量，
但更 GPU 友好。
```

## 13. 预期论文主张

可以主张：

```text
KV cache compressibility is heterogeneous across layers and heads.
This heterogeneity is measurable through remote mass, query-channel sketchability,
temporal reuse stability, page-geometry consistency, and synthetic approximation error.

R2H-KV compiles these measurements into a small set of GPU-friendly execution templates
and uses counterfactual risk calibration to safely promote risky blocks.
```

中文：

```text
KV cache 的可压缩性在 layer/head 上高度异质。
这种异质性可以通过 remote mass、query-channel sketchability、
相邻 token 复用稳定性、page 几何一致性和 synthetic 近似误差来测量。

R2H-KV 把这些测量结果编译成少数 GPU 友好的执行模板，
再用 counterfactual risk calibration 对高风险 block 做安全升级。
```

## 14. 最近可做的最小闭环

最小闭环不要直接写 fused CUDA。先做：

```text
1. qsketch_fixed profiling:
   证明 fixed |q|*std(K) channel groups 接近 dynamic qabs。

2. layer/head taxonomy:
   输出 recommended backend heatmap。

3. V1 layer-level hybrid:
   用现有 layerbudgetattn 跑 profile-based layer map。

4. risk gate:
   在 V1 上加 block calibration，证明 tail risk 下降。
```

如果这四步成立，再进入 page-based kernel。

## 15. 方法一句话

```text
R2H-KV is a risk-calibrated, layer/head-routed hybrid KV compression system
that compiles measured attention geometry into a small set of GPU-friendly
page/sketch/synthetic/full attention templates.
```

