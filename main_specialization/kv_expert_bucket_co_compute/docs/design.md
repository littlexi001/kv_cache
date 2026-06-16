# KV Bucket and Expert Bucket Co-compute: Design

## Objective

在每个 attention head 内学习一个 attention 前可计算的 bucket，使同一个 bucket 同时：

1. 选择当前 query 需要读取的历史 KV；
2. 决定当前 token 使用哪个 expert。

目标不是让 Attention 和 MoE 在概念上“看起来一致”，而是检验共享 bucket 是否能同时改善 KV retrieval 和 expert specialization。

## Falsifiable Conjecture

> 对真实语言训练，head-level、去除 common center 的共享 bucket，在相同 KV candidate ratio 下比独立 K-index 具有更低 NTP loss，同时比普通 MoE 具有更稳定的 expert specialization。

如果共享 bucket 不优于两个独立 bucket，或者只能靠退化成 dense/collapse 保持质量，则“Attention 与 MoE 应共享 specialization”假设被削弱。

## Physical Priors

### Prior 1: Strong QK relations have local closure

Qwen3-0.6B 微测试显示，实际 post-RoPE QK top-k 图的两跳闭包高于距离和 popularity 匹配基线。严格 top-k 下闭包更强。

含义：某些 layer/head 的高分 QK neighborhood 可以近似为局部 retrieval bucket。

### Prior 2: The common K center is a nuisance for K-K routing

令：

```text
k_i = c_h + r_i
```

raw K-K inner product 包含大的 `||c_h||^2`，但固定 query 的 `q^T c_h` 在 attention softmax 中抵消。因此 raw K similarity 不能直接作为 bucket signal。

### Prior 3: Routing should be head-level

不同 layer/head 的 QK geometry、closure 和 long-range responsibility 明显不同。共享一个 token-level bucket 会混合多个 retrieval relation。

### Prior 4: Similar router inputs produce similar gate outputs

对线性 gate：

```text
z_i = G x_i + b
```

有：

```text
||z_i - z_j|| <= ||G||_op ||x_i - x_j||
```

若 top-k routing margin 大于 logit perturbation，则相近输入保持相同 top-k expert set。因此 QK-local geometry 有可能诱导 expert-local routing，但该结论依赖 gate margin，并非自动成立。

## Mathematical Model

### Centered head-level representations

对 layer `l`、head `h`：

```text
r_k,j = Normalize(k_j - mu_k,l,h)
r_q,i = Normalize(A_q,l,h q_i - mu_q,l,h)
```

`A_q` 是可选的轻量 alignment map，用来处理 Q/K 不在同一坐标系的问题。

### Bucket distributions

```text
p_k,j = softmax(G_k,h r_k,j / tau)
p_q,i = softmax(G_q,h r_q,i / tau)
```

KV candidate selection 使用 query/key bucket overlap：

```text
candidate(i,j) = TopMOverlap(p_q,i, p_k,j)
```

expert dispatch 使用当前 query-side bucket：

```text
expert(i,h) = top-k(p_q,i)
```

### Training objective

```text
L = L_NTP
  + lambda_retrieval L_attention_mass_recall
  + lambda_bucket L_qk_bucket_alignment
  + lambda_specialization L_expert_consistency
  + lambda_collapse L_anti_collapse
```

必须分别报告每个 loss 对 retrieval、routing 和 collapse 的影响，不能只报告总 NTP。

## Rival Hypotheses

1. Attention retrieval 与 expert computation 有关，但最优 partition 不同；应使用独立 bucket。
2. 共享 bucket 的收益完全来自 sparse attention，MoE specialization 没有额外贡献。
3. head-level 有效，但 center removal 无效或有害。
4. pretrained QK geometry 可用于 inference index，但无法在真实数据训练中稳定共同优化。

## Claim Boundary

现有实验已支持 QK retrieval 的局部闭包和 pre-router sparse attention 的可训练性；尚未证明共享 KV/expert bucket 优于独立索引，也未在真实语料上训练该结构。
