# Section 25: Influence-Bounded Synthetic KV

> 新增日期：2026-06-28  
> 对应项目：`ymluo/projects/influence_bounded_synthetic_kv`（计划）  
> 核心目标：不再保留真实远程 KV，而是为每层生成少量 synthetic K/V prototypes，使 compressed attention output 逼近 full attention output。

## 1. 研究动机

现有 sparse KV 方向大多仍在回答一个 token selection 问题：

```text
给定 query，哪些历史 token 的真实 KV 应该被保留或召回？
```

这会自然落到 SparQ、Quest、DuoAttention、top-k attention pruning、chunk routing 等路线附近。它们的共同点是：远程记忆仍然由真实 token KV 组成，只是访问方式、保留比例或召回策略不同。

本项目尝试换一个目标：

```text
不要求复原历史 KV，也不要求选中真实 token。
只要求在一小段 calibration query 上，compressed attention 的输出接近 full attention 输出。
```

也就是说，把 KV cache 压缩看成 attention function approximation，而不是 token selection。

## 2. 核心假设

对于某一层、某一 head，full attention 在远程历史 KV 上产生的输出为：

```text
Y_full(Q) = softmax(Q K_remote^T / sqrt(d)) V_remote
```

我们不保留 `K_remote, V_remote`，而是学习少量合成原型：

```text
K_syn in R^{m x d}
V_syn in R^{m x d_v}
m << remote_token_count
```

使得：

```text
Y_syn(Q_calib) = softmax(Q_calib K_syn^T / sqrt(d)) V_syn
```

尽量接近：

```text
Y_full(Q_calib)
```

其中 `Q_calib` 是 prefill 后的一小段 calibration query，可以来自：
- prompt 内尾部若干 token；
- prefill 后模拟 decode 的前若干步；
- 真实 decode 早期 query；
- 或上述 query 的混合采样。

## 3. 方法定义

### 3.1 压缩对象

每层、每个 attention head 单独生成 synthetic KV：

```text
layer l, head h:
  remote KV: tokens [0, ..., T_remote - 1]
  protected KV: recent tokens + sink tokens + optional special anchors
  synthetic KV: m prototypes
```

最终 decode 时 attention 看到的是：

```text
KV_decode = protected_real_KV union synthetic_KV
```

其中 remote real KV 被丢弃，不再参与后续 decode。

### 3.2 训练目标

基础目标：

```text
min_{K_syn, V_syn} ||Y_syn(Q_calib) - Y_full(Q_calib)||_2^2
```

更稳定的版本可以加上输出尺度和方向约束：

```text
L = mse(Y_syn, Y_full)
  + lambda_cos * (1 - cos(Y_syn, Y_full))
  + lambda_norm * | ||Y_syn|| - ||Y_full|| |
```

如果把 protected real KV 也放进 compressed attention，需要拟合的是 remote contribution residual：

```text
Y_full_all = Attn(Q, K_protected union K_remote, V_protected union V_remote)
Y_comp     = Attn(Q, K_protected union K_syn,    V_protected union V_syn)
min ||Y_comp - Y_full_all||^2
```

这个目标更接近真实 decode 行为。

### 3.3 Influence-Bounded 约束

直接优化 synthetic KV 可能产生异常大的 key norm 或 value norm，从而在非 calibration query 上产生不可控影响。因此需要 influence bound：

```text
||K_syn[i]|| <= k_bound
||V_syn[i]|| <= v_bound
max_q softmax(q K_syn^T)[i] <= mass_bound
```

工程上可先用简单约束：

```text
K_syn = normalize(K_syn) * clipped_norm
V_syn = clamp_or_norm_clip(V_syn)
attention_temperature >= tau_min
```

研究上更清楚的定义是：synthetic prototype 对 calibration query 的最大 attention mass 和输出范数不能超过 full remote attention 的统计上界。

## 4. 求解路线

### 4.1 固定 K，闭式求解 V

如果先固定 `K_syn`，则 attention weight：

```text
A_syn = softmax(Q_calib K_syn^T / sqrt(d))
```

此时问题变成线性 least squares：

```text
min_{V_syn} ||A_syn V_syn - Y_target||_2^2
```

闭式解：

```text
V_syn = (A_syn^T A_syn + lambda I)^{-1} A_syn^T Y_target
```

优点：
- 实现简单；
- 可快速验证 synthetic value 是否足够表达 full attention output；
- 适合先做 per-layer/per-head offline calibration。

风险：
- 如果 `K_syn` 初始化不好，`A_syn` 表达能力有限；
- 只优化 V，不能改变 query 到 prototype 的路由形状。

### 4.2 K/V 联合优化

直接用梯度下降优化：

```text
K_syn, V_syn = trainable parameters
loss = output_mse + influence_regularization
```

初始化候选：
- 对 remote K 做 k-means，cluster center 作为 `K_syn`；
- 对 full attention weight 加权的 remote V 做 prototype 初始化；
- 从 calibration query 的 top-attended token 中采样 key 作为初始 prototype；
- 使用随机正交 key，再闭式求一次 V。

推荐第一版流程：

```text
1. dump Q_calib, K_remote, V_remote, Y_full
2. initialize K_syn by k-means or top-attended token keys
3. solve V_syn by ridge regression
4. run 50-200 steps joint K/V optimization
5. norm clip K/V
6. evaluate output MSE and PPL
```

### 4.3 分层训练策略

三个粒度从易到难：

```text
head-local:
  每个 layer/head 单独拟合自己的 attention output

layer-local:
  每层所有 heads 合并，以 layer attention output 为目标

end-to-end:
  synthetic KV 替换后继续前向，直接最小化 logits 或 next-token loss
```

建议先做 `head-local`，因为它最容易定位失败原因。

## 5. 实验设计

### 5.1 最小可行实验

模型：

```text
Qwen3-0.6B 或当前已有 Qwen 小模型路径
```

数据：

```text
短文本 PPL eval
长 retrieval prompt
needle / synthetic retrieval
```

默认配置：

```text
protected_recent_tokens = 128
protected_sink_tokens = 16
calib_tokens = 32 / 64 / 128
synthetic_prototypes_per_head = 4 / 8 / 16 / 32
remote_start = 0
remote_end = seq_len - protected_recent_tokens
```

指标：
- calibration output MSE；
- held-out query output MSE；
- attention output cosine；
- final logits KL；
- PPL；
- long retrieval accuracy；
- decode latency and memory footprint。

### 5.2 关键 ablation

必须比较：

```text
real top-k KV
random real KV
k-means K + closed-form V
top-attended K + closed-form V
joint optimized K/V
joint optimized K/V with influence bound
```

必须扫描：

```text
m = 4, 8, 16, 32
calib query count = 16, 32, 64, 128
recent protection = 32, 128, 512
sink protection = 0, 16, 64
```

### 5.3 成功标准

第一阶段成功：

```text
在 held-out query 上，synthetic KV 的 attention output MSE 显著低于 real-token top-m KV。
```

第二阶段成功：

```text
在相同 KV budget 下，PPL 接近或优于 token selection baseline。
```

第三阶段成功：

```text
在 long retrieval 或真实长上下文任务上，少量 synthetic prototypes 能稳定替代大段 remote KV。
```

## 6. 与已有方法的边界

### 6.1 SparQ / Quest

SparQ / Quest 仍然主要是在真实历史 KV 中做高效召回或近似筛选。它们关心的是：

```text
如何便宜地找到重要 token？
```

本项目关心的是：

```text
是否可以不保留这些 token，只保留一组能模拟其 attention function 的合成原型？
```

### 6.2 DuoAttention

DuoAttention 区分 retrieval heads 和 streaming heads，核心仍然是 head-level 稀疏模式和真实 KV 的保留策略。

本项目不是给 head 分配不同保留规则，而是在每个需要远程信息的层/head 内，用 synthetic KV 替换远程真实 KV。

### 6.3 Pyramid / summary KV

summary KV 通常希望压缩或汇总一段 token 的语义状态。本项目的 synthetic KV 不要求可解释为文本 summary，也不要求对应某段真实 token。它只需要对 calibration query 产生正确 attention output。

## 7. 主要风险

### 7.1 Calibration overfit

Synthetic KV 可能只拟合 calibration query，对后续 decode query 泛化差。

缓解：
- 使用 prefill 尾部 query + 模拟 decode query；
- held-out query 评估；
- 限制 K/V norm 和 attention mass；
- 增加 query dropout 或噪声扰动。

### 7.2 Softmax 非线性导致闭式解不足

固定 K 后闭式求 V 只能解决 value mixing，不能解决 attention routing。

缓解：
- 先用闭式解做下界；
- 再做 K/V joint optimization；
- 比较不同 K 初始化。

### 7.3 合成原型不可控

优化可能制造高 norm key，导致某些未来 query 被 synthetic prototype 吸走。

缓解：
- key/value norm clipping；
- attention mass penalty；
- prototype dropout；
- held-out query worst-case mass 检查。

### 7.4 系统实现复杂

每层每 head 都要 dump calibration、求解、替换 cache，并保持 decode path 正确。

缓解：
- 第一版离线实现，不追求速度；
- 只验证单 batch、单 prompt、单层或少数层；
- 确认质量收益后再考虑 fused kernel 或在线压缩。

## 8. 第一版实现计划

建议新建：

```text
ymluo/projects/influence_bounded_synthetic_kv/
  README.md
  src/
    dump_calibration.py
    fit_synthetic_kv.py
    evaluate_synthetic_kv.py
  scripts/
    run_dump_calib.sh
    run_fit_closed_form.sh
    run_eval_ppl.sh
```

最小实现顺序：

1. 在 dense baseline 前向中 dump 每层每 head 的 `Q_calib, K_remote, V_remote, Y_full`。
2. 只对单层单 head 做 `k-means K + ridge V`。
3. 用 held-out query 评估 output MSE。
4. 扩展到所有 head，替换 remote KV，跑短 PPL。
5. 加 joint optimization 和 influence bound。
6. 与 top-m real KV、recent-only、qabs sparse decode 做同预算比较。

## 9. 当前判断

这个方向的创新点比较清楚：它把长上下文 KV 压缩从 token selection 转成 attention function approximation。

如果结果成立，它和 SparQ/Quest/DuoAttention 的边界会很明确：

```text
它们优化“保留或召回哪些真实 token”；
本项目优化“少量合成 KV 能否替代远程 token 对 attention output 的函数贡献”。
```

最大风险是实现和泛化。建议先不追求速度，也不直接做完整 serving path，而是用 output reconstruction + PPL 两级实验快速判断方向是否值得继续。
