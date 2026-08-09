# MoR-KV 方法设计

## 1. 问题定义

对 layer `l`、query head `h` 和 decode query `q`，设历史 KV blocks 为 `B`。传统 head-aware 方法通常选择预算 `k_lh`，但所有 heads 仍共享一种 token/block importance 逻辑。

MoR-KV 将动作扩展为：

```text
a_lh(q) = (retrieval operator, head portfolio, budget, fallback level)
```

目标是在总物理 KV block 预算 `K` 下最小化：

```text
quality loss + lambda_neg * distractor exposure + lambda_cost * retrieval cost
```

这里的关键是 operator identity 也是决策变量，而不只是 budget。

## 2. Query score signature

对每个 query-head 的候选 block scores `s_lh,1 >= ... >= s_lh,m`，构造：

```text
z_lh(q) = [s_1, s_1 - s_4, std(s_1...s_m)]
z(q) = concat over (l,h) z_lh(q)
```

这些量在正式 gather 之前已经由低维 index scoring 得到，不需要 full attention。v1 用 train split 的每类 centroid 和标准化 Euclidean distance 路由：

```text
r(q) = argmin_c || normalize(z(q)) - centroid_c ||^2
```

论文版可将其替换成小型 cost-sensitive router，但 nearest-centroid 是重要的无隐藏容量基线。

## 3. Specialist head compilation

在 calibration train split 上，对每个 task/operator family 计算 head 的 evidence MRR：

```text
u_lh(c) = mean_{x in train_c} reciprocal_rank(gold block in head Top-m)
```

按 `u_lh(c)` 选择少量 specialist query heads。Qwen3-0.6B 使用 GQA，相邻两个 query heads 共享一个 KV head，因此默认只保留每个 `(layer, kv_head)` 中 calibration utility 更高的 query head：

```text
kv_head = query_head // group_size
```

这避免把 query-head specialization 错误地解释为可独立存储两份 K/V。

## 4. Operator library

### 4.1 Streaming operator

```text
S_stream = sink union recent_window
```

服务稳定 self/previous/recent/sink heads，不扫描远程 KV。

### 4.2 Lexical/structural operator

KV block 旁保存压缩 side metadata：token hash、delimiter bitmap、record boundary 和可选 n-gram sketch。query token/hash 对 block 做 lexical/structure score。

v1 使用 BM25 block scores 作为该算子的高质量研究原型；最终实现必须换成 GPU/CPU 可部署 side index并报告 metadata bytes/token。

### 4.3 Semantic QK operator

每个 KV head 使用 centered K-SVD basis，保存低秩 block/token profile。query 在相同 basis 中投影，独立产生每个 query-head 的 block ranking。

### 4.4 Dense/risk operator

当 router margin、retrieval score spread 或 predecode risk 超过阈值时扩大 budget，极端情况下回退 full attention。

## 5. Specialist-preserving aggregation

多数共识对一个 block 的得分近似随支持 head 数增加，因此会删除只被少数专业 head 找到的证据。MoR-KV v1 提供：

```text
weighted_rrf(b) = sum_h u_h / (60 + rank_h(b))
minority_max(b) = max_h u_h / rank_h(b)
```

`minority_max` 不要求其他 heads 同意。论文版将使用 group-saturating objective：

```text
F(S) = sum_g w_g * (1 - exp(-sum_{b in S} utility_g(b)))
```

`F` 是 monotone submodular function；cardinality budget 下 greedy selection 有 `1-1/e` 近似保证。指数饱和使已经获得很多 blocks 的通用组边际收益下降，为少数 specialist group 留出预算。

## 6. 两个基础理论界

### 6.1 Router regret

设 oracle operator action 的损失为 `L*`，router 错误率为 `epsilon`，任意错误动作相对 oracle 的最大额外损失为 `Delta`，则：

```text
E[L(router)] <= E[L*] + epsilon * Delta
```

因此应同时报告 router accuracy 和 wrong-router regret；只报告 oracle task route 不足以证明方法。

### 6.2 Omitted attention mass 与输出误差

对某 head，full attention 在被省略集合上的总质量为 `delta`，并假设所有 value norm 不超过 `M`。对保留集合重新归一化后的 sparse output `o_S`：

```text
||o_full - o_S|| <= 2 M delta
```

证明来自 `o_full=(1-delta)mu_S+delta mu_R`，因此差值为 `delta(mu_R-mu_S)`。这个界说明最终因果实验必须回到每 head omitted mass、attention output error 和 downstream loss，而不能只报告 block recall。

## 7. 训练、验证和冻结协议

```text
train: 编译 head utility 和 router centroids
dev:   选择 operator/head-count/depth/quota
test:  冻结一次评测，不再调参
```

v1 action space：

```text
head_count in {1,2,4,8,16,32}
depth in {1,2,4,8,16}
aggregation in {weighted_rrf, minority_max}
lexical quota in {0,1,K/4,K/2,3K/4,K-1,K}
```

## 8. 最终 runtime 形态

```text
query
  -> low-cost score signatures
  -> operator/head-group router
  -> parallel operator nomination
  -> GQA-aware physical block union
  -> specialist-preserving budget projection
  -> paged KV gather + sparse attention
  -> confidence/risk fallback
```

系统实现必须把动态动作编译到有限模板，避免破坏 CUDA Graph 和 PagedAttention；这部分应吸收 HARD-KV 的 static-dynamic mismatch 教训，而不是停留在 Python top-k。

## 9. Causal per-head distortion teacher

Natural zero-overlap holdout 证明 generic answer-NLL proxy 和 QK confidence 不能安全预测 operator regret。修订后的 dense teacher 对每个 `(layer, query_head, query, operator)` 直接计算：

```text
d_lh(a,q) = ||o_full_lh(q) - o_a_lh(q)|| / ||o_full_lh(q)||
```

teacher 使用 exact post-RoPE Q/K/V 和真实 value vectors，并同时输出 omitted attention mass。测试时 router 只使用 head identity、QK/lexical block-score top value、margin、spread、entropy 和 operator disagreement，不访问 full attention。

对 action `a` 预测 `d_hat_lh(a,q)` 后，使用 per-head one-sided conformal correction `c_lh,a(alpha)`：

```text
U_lh(a,q) = d_hat_lh(a,q) + c_lh,a(alpha)
a*_lh(q) = cheapest a with U_lh(a,q) <= epsilon
```

若没有 sparse action 满足约束则回退 full。该设计将“stable head prior”与“query activation”显式分离，并把 tail risk 直接写进选择规则。

## 10. End-to-end causal correctness reference

为了验证 teacher target 是否真正影响模型输出，在每个 transformer layer 内执行下列干预：

```text
ordinary attention forward
  -> exact full per-head output at scored query positions
  -> routed sparse per-head output at the same positions
  -> project (sparse - full) through the ordinary o_proj
  -> add delta to the layer output
  -> ordinary residual, MLP, later layers and LM head
```

只有被评分的 prompt/answer query positions 被替换；更早 token 状态保持不变。答案 continuation 使用与既有 NLL evaluator 相同的前导空格协议。实现只对被评分位置执行 LM head，避免生成 `[sequence_length, vocabulary]` 全量 logits。

这个路径是 correctness reference，不声称速度。它可以分别运行 fixed operator、exact risk oracle 和序列化的 deployable conformal bundle，并对同一 query 计算 paired delta NLL、bootstrap CI 与 p95 absolute delta。

## 11. GQA physical-union execution path

对每个 `(layer, physical_kv_head, query_position)`，先对共享该 KV head 的 query-head block decisions 求并集：

```text
I_lg(q) = union_{h shares g} selected_blocks_lh(q)
```

当前 reduced-compute reference 对每个物理 KV head：

1. `index_select` gather `I_lg(q)` 对应的 K/V tokens；
2. 让该 group 的 query heads 共享 gathered K/V；
3. 调用一次 PyTorch SDPA；
4. 拼接所有 physical groups 的输出。

这条路径确实不计算未选 KV，但每层需要 `num_kv_heads` 次 gather 与 SDPA launch，因此只是可验证的系统下界。最终 kernel 应融合 physical-group page gather 与 attention，并把 operator template 编译到有限 block layouts，以降低 Python 和 launch overhead。

### 11.1 Physical-union-aware structured routing

先独立最小化每个 query head 的 blocks、再对共享 KV head 求并集，不保证物理 KV 最优。对一个 GQA group `G(k)`，应直接求解：

```text
minimize    | union_h selected_blocks(a_h) |
subject to  upper_error_h(a_h, q) <= epsilon_h,  h in G(k)
```

候选算子较少时，可用 group 内动态规划精确编译 teacher；部署时再将其蒸馏为 structured router。Qwen3-8B 的 64-query exact audit 证明这个次序差异真实存在：epsilon=0.05 下，独立 head oracle 的物理节省为 `8.71%`，直接最小化 GQA union 为 `9.14%`；epsilon=0.10 时为 `21.50%` 对 `22.24%`。当前 gap 不大，不能单独支撑主张，但说明论文方法和 kernel 成本函数都应在 physical KV-head 层定义，而不是把 logical head blocks 当 proxy。

## 12. Cross-layer propagated risk budget

端到端实验表明，独立约束每个 head 的局部误差仍会累积成可测 logit drift。令第 `l` 层 attention intervention 经过 `o_proj` 后的残差扰动为 `Delta z_l`，后续 transformer suffix 到 logits 的局部放大系数为 `s_l`。一阶传播给出：

```text
||Delta logits|| <= sum_l s_l ||Delta z_l|| + higher-order remainder
```

结合 per-head conformal upper bound，可定义 conservative action risk：

```text
R_lh(a,q) = s_l * ||W_o,l,h|| * U_lh(a,q) * ||o_full_lh(q)||
```

最终 route 不再对所有 heads 使用同一个 epsilon，而解一个 cost-constrained projection：

```text
minimize    physical_GQA_union_blocks(actions)
subject to  sum_lh R_lh(a_lh,q) <= B(q)
            high-risk layers/heads may fall back to full
```

`s_l` 的第一版估计来自 query-disjoint layer-group causal interventions：在相同局部 threshold 下测量每个层组的 paired delta NLL / physical saving。论文版应进一步比较 Jacobian-vector product、finite-difference logit sensitivity 与这个低成本 empirical estimator，并单独控制 higher-order remainder。
