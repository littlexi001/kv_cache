# 长度退化的数学解释与无硬阈值候选方法

## 1. 当前可证伪结论

截至 2026-08-03 的固定目标实验支持以下判断：

1. `64K` 不是模型或算法中的自然常数。
2. 低比特 QK 排序和 sampled threshold 在 32K--96K 的额外 NLL 很小，不是首要矛盾。
3. 固定 top-1,280 随历史增长承载的完整 attention mass 持续下降。
4. attention mass 的遗漏与单层 attention 输出误差近乎同步增长。
5. rank-16 INT4 Value-tail 能恢复困难医学文本，但固定 `alpha=0.5` 会在部分宗教文本上过补或欠补，因此不能作为最终统一规则。

困难主题的阶段结果如下。质量均相对同长度 Full KV；`local error` 是精确 QK top-1,280 的逐 head attention 输出相对 L2 误差。

| 主题 | 长度 | Exact top-1,280 | QKSieve sampled | `alpha=0.5` Value-tail | selected mass | local error |
|---|---:|---:|---:|---:|---:|---:|
| 医学 | 32K | 90.33% | 92.15% | 100.43% | 92.23% | 13.20% |
| 医学 | 64K | 88.88% | 91.69% | 100.30% | 90.89% | 14.59% |
| 医学 | 96K | 86.42% | 89.40% | 99.76% | 88.57% | 17.25% |
| 体育 | 32K | 99.67% | 99.72% | 99.93% | 94.90% | 8.70% |
| 体育 | 64K | 99.55% | 99.65% | 99.97% | 93.83% | 10.20% |
| 体育 | 96K | 99.64% | 99.68% | 99.76% | 91.38% | 13.59% |
| 宗教 | 32K | 106.86% | 107.01% | 103.06% | 91.47% | 14.35% |
| 宗教 | 64K | 107.80% | 106.46% | 101.50% | 88.85% | 17.31% |
| 宗教 | 96K | 102.63% | 100.42% | 96.95% | 86.57% | 19.27% |

9 个困难主题/长度案例中：

```text
corr(1 - selected_mass, local_attention_relative_error) = 0.9899
```

该相关性是机制证据，不等价于端到端 NLL 的单调相关。后续层、残差连接和目标 token 类型会放大或抵消局部误差。

## 2. 精确误差恒等式

对一个 query head，将选中集合记为 `S`，遗漏集合记为 `T`。完整 softmax 在 `S` 上的概率质量为 `p`，两部分的条件 Value 均值分别为 `mu_S` 与 `mu_T`。

```text
o_full   = p * mu_S + (1 - p) * mu_T
o_sparse = mu_S

o_sparse - o_full = (1 - p) * (mu_S - mu_T)
```

因此，长度本身不会直接制造误差。误差由两个连续量共同决定：

```text
tail_mass = 1 - p
value_contrast = ||mu_S - mu_T||
```

经输出投影 `W_o` 后有：

```text
||W_o(o_sparse - o_full)||
<= (1 - p) * ||W_o||_2 * ||mu_S - mu_T||
```

新诊断代码直接计算等式两边。数值闭合误差应接近浮点误差；若不闭合，说明实现或统计口径有问题。

## 3. 为什么固定 k 在任意长度上不可能普遍安全

考虑 attention score 中不含显著语义 needle 的 diffuse bulk。若 bulk score `z_i` 具有有限矩母函数，例如近似次高斯，则大数定律给出：

```text
Z_bulk = sum_i exp(z_i) = Theta(N)
```

而固定 `k` 个极值的 score 只按 `O(sqrt(log N))` 增长，因此其指数和至多按下式增长：

```text
Z_topk <= k * exp(O(sqrt(log N)))
```

于是：

```text
p_topk = Z_topk / Z_all
       <= k * exp(O(sqrt(log N))) / Theta(N)
       -> 0,  N -> infinity
```

这不是说真实 attention 一定服从独立高斯，而是说明：只要上下文中存在不会随长度消失的 diffuse bulk，固定 token 数就没有任意长度质量保证。真实语义 needle 可以让 top-k 质量暂时很高，但不能消除 bulk 的总贡献。

固定目标实验中 selected mass 随长度持续下降，正是该机制的经验对应。

## 4. 为什么当前首先不应增加 Key bit

令精确 score 为 `z_i`，低比特 proxy 为：

```text
z_tilde_i = z_i + e_i
```

若精确 top-k 边界为：

```text
m_k = z_(k) - z_(k+1)
```

则一个简单充分条件是：

```text
2 * max_i |e_i| < m_k
```

满足时 top-k 集合不变。长度增长会增加潜在 crossing 数并压小边界间隔，因此 Key 排序风险确实存在；但当前固定目标替换实验中，proxy full-top-k 与 exact top-k 的质量差远小于 exact top-k 与 Full 的差，sampled threshold 的额外差也很小。因此现阶段继续堆 Key bit 不能修复主要误差。

## 5. 候选方法：Control-Variate Tail QKSieve

工作名为 `CV-Tail QKSieve`。目标是去掉固定 `64K` gate 和固定 `alpha`，仍不回退 Full attention。

### 5.1 两部分计算

1. 稀疏异常项：QKSieve 低比特索引定位 top-k，选中 token 使用原始 K/V 与 exact QK。
2. 稠密 bulk：扫描低比特 Key/Value sketch，估计遗漏 partition 与遗漏 Value numerator。

记：

```text
Z_T = sum_{i in T} exp(z_i)
N_T = sum_{i in T} exp(z_i) * v_i
```

低比特 proxy 给出便宜但有偏的 `Z_tilde_T` 与 `N_tilde_T`。对每个 query head 均匀抽取 `m` 个 tail token，计算其 exact QK，并作 control-variate 修正：

```text
Z_hat_T = Z_tilde_T
        + |T|/m * sum_{i in U} [exp(z_i) - exp(z_tilde_i)]

N_hat_T = N_tilde_T
        + |T|/m * sum_{i in U}
          [exp(z_i) - exp(z_tilde_i)] * v_hat_i
```

其中 `v_hat_i` 是 rank-r、低比特 Value sketch。条件于当前 cache 和 query，只要 `U` 是 tail 的均匀样本：

```text
E[Z_hat_T] = Z_T
E[N_hat_T] = sum_{i in T} exp(z_i) * v_hat_i
```

第一式消除固定 `alpha`；第二式把剩余误差严格隔离为 Value sketch 残差，而不是 score 与 Value 的混合偏差。

最终输出：

```text
o_hat = (N_S_exact + N_hat_T) / (Z_S_exact + Z_hat_T)
```

### 5.2 输出误差分解

设 `Delta_Z = Z_hat_T - Z_T`，`Delta_N = N_hat_T - N_T`，则有精确关系：

```text
o_hat - o_full
= [Delta_N - o_full * Delta_Z]
  / [Z_S + Z_T + Delta_Z]
```

因此：

```text
||o_hat - o_full||
<= [||Delta_N|| + ||o_full|| * |Delta_Z|]
   / [Z_S + Z_hat_T]
```

这给出了三个可独立验证和优化的量：

1. partition 误差 `|Delta_Z|`；
2. Value numerator 误差 `||Delta_N||`；
3. 最终归一化分母。

### 5.3 连续、无长度阈值的 rank 选择

对每个 KV head，在 prefill 时估计 rank-r Value 重构残差 `epsilon_v(r)`。运行时用 tail mass 与采样方差形成风险上界：

```text
R(r) = estimated_tail_mass * epsilon_v(r)
     + partition_confidence_radius
```

选择满足 `R(r) <= tau` 的最小 rank。决策依赖当前 query、head 和 Value 数值，而不依赖 `N < 64K` 或任务标签。

`tau` 是输出误差容忍度，不是按 benchmark 调出的长度补丁。正式方法若使用该规则，必须在 held-out topic、未见长度和第二模型上冻结同一个 `tau`。

## 6. 必须主动覆盖的失败情况

1. proxy 与 exact score 非仿射、重尾或发生少量巨大 crossing；control variate 需要报告方差与置信半径。
2. 采样恰好漏掉罕见高分 tail token；要比较 systematic、随机相位和分层采样。
3. Value 残差与 score 正相关；仅按无权 PCA 能量选 rank 可能失效。
4. selected mass 很高但 `mu_S-mu_T` 极大；不能只用 mass gate。
5. selected mass 很低但 tail Value 接近 selected Value；不能仅因长度扩大预算。
6. GQA 四个 query head 的风险不同；共享 candidate 时必须按归一化输出风险合并，而不是直接取原始 score 最大值。
7. 短文本上校准和 tail kernel 的固定开销可能超过 Full attention；质量规则与成本规则必须分开报告。

## 7. 当前实验门槛

在实现新 CUDA kernel 前，依次通过以下小实验：

1. 32K 与 128K real-QKV trace：比较固定 `alpha=0.5/1`、sample ratio、control variate、oracle mass 与 exact-score Value sketch。
2. 判断主要剩余误差来自 partition 还是 Value rank；若 control variate 无法稳定降低 p90/maximum，停止该方向。
3. 固定 sample count 与误差阈值，在六主题、未见 seed、32K--128K 上做短 PPL。
4. 仅候选通过上述门槛后实现 fused sample-QK + proxy scan + tail accumulation kernel。
5. 独立计时 attention 子系统和稳态 decode；索引构建单独报告并计算多轮 Agent 场景 break-even。

