# QKSieve 当前方法、实现与内核优化

## 1. 当前冻结判断

截至 2026-07-30，建议冻结的研究方向是：

1. 保留 **QK-balanced 投影基底**，不退回纯 Key-PCA。
2. 冻结部署位宽分配为更简单的 **Key-MSE**；Query-weighted qMSE 保留为
   更一般的理论目标和对照消融，不再作为最快部署路径的默认配置。
3. 保留每层、每个 KV head 独立的混合位宽索引。
4. 保留长度相关的精确 attention 预算形式。32K--128K 已验证配置仍是：

   ```text
   B(N) = min(1280, max(256, ceil(0.06 * N)))
   ```

   严格 256K 四窗口 Exact-QK oracle 已证明 `B=1,280` 本身足够：

   ```text
   exact FP16 QK top-1280:
     0.488% active, 100.24% PPL retention, KL 0.00516
   ```

   但是当前低比特 QK-balanced proxy 在相同预算下只有 80.77% 保持。因此
   不能把 `1,280` 写成“已否决的 attention 预算”，应写成“当前 proxy 尚未
   达到的 oracle 目标”。现有 proxy 的两个可执行工作点为：

   ```text
   speed-oriented:  B = 26,214  (9.98% active)
   high-fidelity:   B = 62,900 (23.96% active)
   ```

   前者为 96.26% PPL 保持、5.35x steady；后者为 100.25% PPL 保持、
   2.99x steady。24% 是当前 proxy 通过扩大候选池补偿排序误差的工作点，
   不是理论必要 active ratio。该路径不依赖 router 或任务标签。512K 尚未
   冻结质量预算，因为当前模型的原生位置上限只有 256K。

5. 保留完整历史 FP16 K/V，并只对候选 token 做精确 QK、softmax 和 AV。
6. 速度实现使用模型级冻结模板、WMMA Query 投影、GQA-4 共享扫描和
   sampled-quantile 单遍候选写出。
7. 不加入 router、任务规则、Full fallback、exact-QK rerank 或 cold-token
   硬跳过。

最新 256K 四窗口 oracle 诊断如下；每点包含 256 个与 Full 严格配对的
target token。Exact oracle 用完整 FP16 QK 排序，proxy 用冻结
QK-balanced + Key-MSE 240-bit 索引排序；二者最终都在原始 FP16 K/V 上做
exact sparse attention。

| Selector | Exact/head | Active | Oracle attention mass | PPL 保持（95% CI） | Top-1 | KL |
|---|---:|---:|---:|---:|---:|---:|
| Exact FP16 QK oracle | 1,280 | 0.488% | 90.002% | **100.240% [99.48, 100.88]** | 90.625% | **0.00516** |
| Exact FP16 QK oracle | 2,560 | 0.977% | 93.057% | **100.148% [99.54, 100.68]** | 92.969% | **0.00297** |
| QK-balanced low-bit proxy | 1,280 | 0.488% | 未测 | 80.771% [77.84, 83.92] | 67.188% | 0.24246 |
| QK-balanced low-bit proxy | 2,560 | 0.977% | 未测 | 86.132% [84.68, 87.77] | 71.094% | 0.15092 |

Exact oracle 的运行时间包含全历史 FP16 QK 与诊断统计，不能作为可部署速度；
它只用于确认预算是否足够。结果说明 1,280 已经是可行的 oracle 目标，当前
主要缺口是低比特 selector 的排序精度。

同一冻结 Key-MSE 模板、完整 proxy top-k、四窗口协议的严格长度对照为：

| Selector / budget | 128K PPL 保持 | 256K PPL 保持 |
|---|---:|---:|
| Exact-QK top-1,280 | 100.805% | 100.240% |
| Exact-QK top-2,560 | 100.345% | 100.148% |
| 低比特 proxy top-1,280 | **98.993%** | **80.771%** |
| 低比特 proxy top-2,560 | **100.005%** | **86.132%** |

因此 128K 的 99%+ 质量记忆是正确的，而且现在已用同路径排除了方法版本
差异。断崖发生在 128K 到 256K 的 proxy 排序稳定性，不发生在 Exact-QK
预算充分性。

同一四窗口、top-1,280 的单因素归因进一步得到：

| Selector | PPL 保持（95% CI） | Top-1 | KL | Steady decode | Online decode |
|---|---:|---:|---:|---:|---:|
| Exact FP16 QK | **100.240% [99.48, 100.88]** | 90.625% | 0.00516 | 诊断路径 | 诊断路径 |
| 冻结坐标 + 冻结 allocation | **80.771% [77.84, 83.92]** | 67.188% | 0.24246 | **7.785x** | **6.668x** |
| 冻结坐标 + local allocation | **89.911% [84.13, 94.28]** | 77.344% | 0.13332 | **7.766x** | **5.259x** |
| Local 坐标 + fixed allocation | **100.443% [99.80, 101.02]** | 91.016% | 0.00620 | **7.671x** | **4.955x** |
| Local 坐标 + local allocation | **100.430% [99.78, 100.99]** | 91.016% | 0.00622 | **7.733x** | **4.055x** |
| 冻结全 INT8 | **100.267% [99.48, 100.91]** | 91.016% | **0.00508** | 1.882x | 1.846x |

冻结坐标只刷新 bit allocation 仍只有 89.91%，而 request-local 坐标配固定
`(4,1,1,1,1,1,0,0)` 已恢复到 100.44%。真正的问题主要是冻结 QK-balanced
坐标无法在 256K 请求上继续对齐 joint score energy；allocation 失配是次因。
当前质量/速度首选因此改为 **request-local QK-balanced + fixed 240-bit +
完整 proxy top-k + top-1,280 exact attention**。自动 allocation 在本组没有
质量收益，并把 64-token Online 从 4.96x 降到 4.06x。表中速度是共享
prefill 后的整模型 decode，不包含共同 prefill。

当前可部署 proxy 的 256K 四窗口预算前沿如下：

| Active | PPL 保持 | 95% window-bootstrap CI | KL | Steady | Online |
|---:|---:|---:|---:|---:|---:|
| 7.98% | 94.44% | [93.38, 95.68] | 0.04365 | 5.51x | 4.93x |
| 9.98% | 96.26% | [94.53, 97.73] | 0.03266 | 5.35x | 5.10x |
| 11.93% | 96.95% | [94.79, 98.43] | 0.02696 | 4.45x | 4.28x |
| 15.95% | 97.42% | [95.43, 99.07] | 0.02250 | 3.64x | 3.38x |
| 20.06% | 99.07% | [97.62, 100.55] | 0.01349 | 2.96x | 2.89x |
| **23.96%** | **100.25%** | **[99.40, 101.06]** | **0.00733** | **2.99x** | **2.91x** |

24% 点只能解释为与 Full 统计一致，不能解释为普遍优于 Full。完整 K/V 仍驻留
GPU；active 比例表示进入精确 attention 的 token 比例，不是 KV 显存压缩率。

这里必须区分三个容易混淆的设计轴：

| 设计轴 | 选项 | 当前判断 |
|---|---|---|
| 投影坐标 | Key-PCA / QK-balanced | 保留 QK-balanced |
| 位宽目标 | Key-MSE / Query-weighted qMSE | 部署冻结 Key-MSE；qMSE 作为理论/消融参考 |
| 候选选择 | 完整 proxy top-k / sampled quantile | 前者是质量参考，后者是速度实现 |

“Key-MSE 分配足够好”不等于“只需要 Key-PCA”。前者只决定 240 bit 如何分给
八个 band；后者决定在哪个坐标系中压缩和近似 QK score。

## 2. 为什么不采用 pure Key-PCA

### 2.1 严格控制实验

Qwen3-4B、32K history、六个主题、每个主题两个独立窗口，共 12 个严格配对
窗口。两个方法共享：

- 240 bit/token/KV-head 索引预算；
- Key-MSE 位宽分配；
- 每个 Query head 1,280 个精确 attention token；
- 完整 proxy top-k；
- 原始 FP16 K/V 和 exact sparse attention；
- 无 rerank、无 fallback。

唯一变化是投影基底。

| 指标 | Key-PCA + Key-MSE | QK-balanced + Key-MSE |
|---|---:|---:|
| Full PPL | 16.4020 | 16.4020 |
| Sparse PPL | 16.4387 | **16.2843** |
| PPL 质量保持率 | 99.7766% | **100.7229%** |
| Top-1 与 Full 一致率 | 93.6198% | **95.5729%** |
| KL(Full \|\| Sparse) | 0.02106 | **0.00844** |
| Index / Full FP16 K+V | 5.7921% | 5.8571% |
| Steady decode | **57.698 ms/token** | 58.167 ms/token |

配对统计量定义为：

```text
delta_nll(Key-PCA + Key-MSE)
  - delta_nll(QK-balanced + Key-MSE)
```

结果为：

```text
mean       = +0.009440
95% CI     = [+0.000781, +0.018906]
P(Key-PCA better) = 1.48%
```

正值表示 QK-balanced 更好，且置信区间不跨零。QK-balanced 的 steady 时间只
慢约 0.81%，但 KL 降低约 60%，Top-1 一致率提高 1.95 个百分点。当前证据
不支持为了代码简单而退回纯 Key-PCA。

### 2.2 可以简化的是 allocation

已有四窗口 32K 消融中，同一 QK-balanced 基底下：

| 位宽分配 | PPL 质量保持率 | Top-1 一致率 | KL |
|---|---:|---:|---:|
| Query-weighted qMSE | 100.198% | 94.531% | 0.00952 |
| Key-MSE | **100.296%** | **94.727%** | **0.00945** |

LongBench m20 严格配对因果消融进一步得到：

| 方法 | Macro score | 相对 Full |
|---|---:|---:|
| Full KV | 0.459694 | 100% |
| QK-balanced + qMSE | 0.458648 | 99.772% |
| Key-PCA + Key-MSE | 0.458986 | 99.846% |
| **QK-balanced + Key-MSE** | **0.459338** | **99.923%** |

QK-balanced + Key-MSE 相对 QK-balanced + qMSE 的 macro 差值为
`+0.000690`，任务分层 paired bootstrap 95% CI 为
`[-0.000717, +0.002160]`，`P(Key-MSE better)=82.51%`。两种 allocation
在 m20 上统计等价，不能宣称 Key-MSE 的质量显著更高，但可以说明简化没有
可检测的质量代价。

同一 WMMA + GQA-4 + sampled-quantile 优化路径、同一个 32K/256-token
窗口的部署对比为：

| 部署 allocation | PPL 保持率 | Steady decode | 相对 Full |
|---|---:|---:|---:|
| qMSE | 100.731% | 46.696 ms/token | 1.889x |
| **Key-MSE** | **100.751%** | **45.577 ms/token** | **1.936x** |

Key-MSE 在该受控窗口中快 2.46%，质量持平。最终部署配置因此冻结为：

```text
QK-balanced coordinates + Key-MSE allocation
```

它仍利用 Query 分布找到适合近似 QK score 的坐标，但分配 bit 时只统计投影
Key 的重建误差。这里简化的是位宽目标，不是 QK-balanced 坐标本身。

### 2.3 为什么 QK-balanced 中的 Key-MSE 仍然包含 Query 信息

QK-balanced 构造满足：变换后的 Query 与 Key 构造矩共享联合谱
`Sigma = diag(sigma_j)`。因此在理想化的比例量化噪声模型中：

```text
E[e_j^2] = c_j(bit_j) * sigma_j

Key-MSE   = sum_j c_j(bit_j) * sigma_j
qMSE      = sum_j c_j(bit_j) * sigma_j^2
```

所以这里的 Key-MSE 不是原坐标中的普通 Key reconstruction，也不是纯
Key-PCA；QK-balanced 基底已经把 Query 二阶矩写进了坐标方向和尺度。
qMSE 会再乘一次 Query 权重，近似把联合谱平方，因此更倾向把 bit 集中到头部。

两个模型级模板的实际结构与该预测一致：

| 统计 | qMSE | Key-MSE |
|---|---:|---:|
| 平均 active band | 3.649 | 5.948 |
| 前四 band 的物理 bit 比例 | 99.812% | 73.957% |
| 后四 band 的物理 bit 比例 | 0.188% | 26.043% |
| 最常见布局 | `4-4-4`，32.3% | `4-1-1-1-1-1`，97.6% |

这解释了 Key-MSE 为何可能对 Query drift 和长尾证据更稳健，但不能证明它在
所有任务上必然更好。特别是 Key-MSE 的 288 个 layer/head 中有 281 个都得到
`4-1-1-1-1-1-0-0`，说明逐 head allocator 可能进一步简化成固定布局。
固定布局的六主题、12 窗口因果实验已经注册；在结果完成前不把该简化写成最终
结论。

## 3. 问题定义

对某层 Query head `h` 和其对应的 KV head `g(h)`，Full attention 为：

```text
s[h,i] = q[h]^T k[g(h),i] / sqrt(d)
p[h]   = softmax(s[h])
o[h]   = sum_i p[h,i] v[g(h),i]
```

历史长度为 `N` 时，每个 decode step、每层都要读取并计算全部 `N` 个 K/V。
QKSieve 的目标不是删除 KV，而是找到 Query 相关候选集合 `S[h]`，随后只在
`S[h]` 上读取原始 K/V 并执行精确 attention：

```text
S[h] = retrieve(q[h], low_bit_key_index)
o_tilde[h] = ExactAttention(q[h], K[S[h]], V[S[h]])
```

所有历史 K/V 仍可被检索，同一 token 在不同层、不同 head 和不同 decode step
可以被选择或不选择。

## 4. QK-balanced 坐标

### 4.1 二阶矩

对每层、每个 KV head 收集 post-RoPE Key 和映射到该 KV head 的 Query，构造：

```text
C_k = E[k k^T]
C_q = E[q q^T]
```

Query 样本少于 Key 样本，因此使用固定收缩：

```text
Cq_tilde = (1 - lambda) C_q
           + lambda * trace(C_q) / d * I
lambda = 0.75
```

对两个矩阵的特征值使用一致的正数 floor，避免逆平方根不稳定。

### 4.2 双正交分解

计算：

```text
Cq_tilde^(1/2) C_k^(1/2) = U Sigma V^T

A = Cq_tilde^(-1/2) U Sigma^(1/2)
D = C_k^(-1/2) V Sigma^(1/2)

q' = A^T q
k' = D^T k
```

在使用相同正则化矩阵时：

```text
A D^T = I
q'^T k' = q^T k
A^T Cq_tilde A = D^T C_k D = Sigma
```

因此完整 128-D 变换本身不近似 QK dot product。近似只来自后续低比特 Key
索引。`Sigma` 的顺序反映 Query 和 Key 的联合 score energy，不是单纯的 Key
重建能量。

Key-PCA 只在 Query 二阶矩近似各向同性时成为 QK-balanced 的特例。真实 Query
各向异性明显时，Key-PCA 可能把 bit 花在 Key 方差大但当前 Query 不敏感的
方向上。

## 5. 混合位宽索引

### 5.1 Band 和物理格式

head dimension 为 128，按连续 16 维分成八个 band。每个 band 可选：

```text
bits_per_value in {0, 1, 2, 4, 8}
```

非零 band 还保存一个 FP16 scale，即每个 16-D band 额外 16 bit。物理预算：

```text
R(b) = 16 * sum_g (b_g + I[b_g > 0]) <= 240 bit
```

所以每个 token、每个 KV head 的索引上限是 30 byte。

完整 FP16 K+V 为：

```text
2 tensors * 128 dimensions * 2 byte = 512 byte/token/KV-head
```

索引上限占：

```text
30 / 512 = 5.859%
```

最终 Key-MSE 模型级冻结模板的实测平均为 5.854%。注意该比例是**额外索引**；
原始 K/V 仍保留在 GPU，所以当前方法不会把总 KV 显存降低到 5.854%，而是降低每步
attention 实际读取和计算的 K/V 数量。

### 5.2 量化规则

- 2/4/8 bit：per-token、per-band 对称 max-absolute scale；
- 1 bit：sign 乘该 band 的 mean absolute magnitude；
- 0 bit：不存 code 和 scale；
- Query 在运行时按 band INT8 量化；
- proxy score 只用于排序，最终 attention 使用原始 FP16 K/V。

### 5.3 位宽分配

qMSE 分配对每个 band 和每个候选 bit 计算：

```text
D_g(b) = E_q,k [(q'_g^T (k'_g - Quant_b(k'_g)))^2]
       = trace(C'_q,g C_e,g(b))
```

然后枚举八个 band 的可行配置：

```text
b* = argmin_b sum_g D_g(b_g)
subject to R(b) <= 240 bit
```

最终部署版本使用 Key-MSE：

```text
D_key,g(b) = trace(C_e,g(b))
```

Query-weighted qMSE 仍是分析 Query drift 和 logit distortion 的一般形式。
由于只有八个 band、五个 bit level 和 15 个归一化预算单位，完整枚举或动态
规划都很小；最终每层、每个 KV head 的分配被写入模型级冻结模板。

## 6. 两条执行路径

### 6.1 质量参考路径

用于当前完整 Llama LongBench 主结果：

1. 当前请求 prefill 完成后，从 prompt Key 和最后八个 prompt Query 构造
   request-local QK-balanced 坐标。
2. 构造当前请求的混合位宽索引。
3. 每个 decode step 物化所有 token 的 proxy score。
4. 对 proxy score 做完整 top-k，直接得到 `B(N)` 个候选。
5. 不做 exact-QK rerank。
6. 在原始 FP16 K/V 候选上做精确 QK、softmax 和 AV。

注册方法：

```text
qksieve_fullprompt_auto_plain_fulltopk
```

这一条路径提供干净、确定的质量参考，但完整 score materialization 和通用
top-k 不是最快系统实现。

### 6.2 优化部署路径

用于最新 32K/64K/120K/128K/256K 系统速度实验：

1. 用彼此独立的 32K `sports`、`medicine`、`mixed_a` 校准文本构造模型级
   QK-balanced 模板。
2. 每层、每个 KV head 的 `(A,D)` 与 mixed-bit pattern 离线冻结。
3. 运行时加载约 19 MB 模板，不在每个请求上做协方差、SVD 或 allocation。
4. prefill 时分块投影并编码历史 Key；decode 时只增量追加一个 Key 索引项。
5. WMMA kernel 同时完成 GQA-4 Query 投影和 band-wise INT8 量化。
6. 令目标比例 `r(N)=B(N)/N`。先根据期望尾部样本数 `c` 计算
   `ceil(c/r(N))`，再向上对齐到 256：

   ```text
   m(N) = min(N, 8192,
              max(256, 256 * ceil(c / (256 * r(N)))))
   ```

   4K--128K 和严格 256K 质量前沿使用 `c=64`。512K、24% 的系统压力点
   使用 `c=128`，目的是把置信候选容量限制在当前 16-way ragged kernel
   可以执行的范围内；它不是 512K 质量超参数。
7. GQA-4 kernel 一次读取 Key 索引，同时为四个 Query head 扫描。
8. score 超过阈值的 token 直接写入候选数组，不物化 `N` 长度 score tensor，
   不调用通用 `torch.topk`。
9. 在候选对应的原始 FP16 K/V 上做精确 sparse attention。

sampled threshold 的候选数均值通常在目标附近，例如 120K 平均约
1,282 token/head；但“均值接近”不等于每个 head 都稳定。若阈值是 `m` 个
均匀样本中的第 `j` 大值，连续分数的随机秩模型给出：

```text
tail_mass_hat ~ Beta(j, m + 1 - j)
relative_std(candidate_count) ~= 1 / sqrt(m * r) = 1 / sqrt(c)
```

在稀有尾部极限下，归一化候选比例还近似满足
`tail_mass_hat / r -> Gamma(c,1) / c`。对 `H` 个 layer-head 使用 Chernoff
界和 union bound，可得到：

```text
Pr[max_h(candidate_h / B) >= 1 + eps]
  <= H * exp(-c * (eps - log(1 + eps)))
```

取 Qwen3-4B 的 `H=36*8=288`、失败概率 5%，`c=16/64/256` 对应的最大候选
上界因子约为 2.43/1.61/1.28。该式是随机秩参考模型，不是确定性分层采样的
形式证书，但它解释了为什么系统延迟取决于 headwise 尾部，而不只取决于均值。

旧 `c=16` 的相对标准差约为 25%。严格 256K、4% 目标的实际范围曾达到
2,586--26,408 token/head，虽然全 head 均值约为 10,095。因此 16 个尾部样本
只解决“几乎完全采不到目标尾部”的问题，没有解决分位数和 kernel straggler。
提高到 `c=64` 后，6% 单窗口候选范围从 4,293--35,044 收窄到
9,343--23,507，sampled 路径的 PPL/KL 追平完整 proxy top-k，稳态速度还从
5.48x 提高到 6.30x。原因是 ragged attention 的最大容量和 split 尾部缩小，
并不是“采样越多天然越快”。

四个独立 256K 窗口已经完成复核。它证明 `c=64` 足以稳定 sampled selector；
但进一步的 Exact-QK oracle 对照表明，“当前 proxy 需要约 24%”并不等于
“attention 本身需要约 24%”。这里必须把三个误差源分开：

1. `c` 控制 sampled threshold 相对完整 proxy top-k 的额外误差和候选方差；
2. 低比特 proxy 控制候选排序是否接近 Exact-QK oracle，是当前 256K
   质量下降的主因；
3. `B(N)` 控制 oracle 最终能看到多少证据；四窗口 oracle 已证明
   `B=1,280` 在本协议下足够。

当前论文不能把 reference profile 的 99.881% LongBench 与旧 deployment
profile 的 4.57x whole-model speed 写成“同一条路径同时得到”。不过严格
256K 四窗口实验已经在同一条
`QK-balanced + Key-MSE + WMMA + sampled-quantile + exact sparse attention`
路径上同时测得质量和速度：24% active 为 100.25% PPL 保持和 2.99x steady。
该结论仍是跨窗口 causal-PPL 机制证据；deployment LongBench/RULER 任务质量
完成前，不能升级成下游任务主结论。

## 7. CUDA 内核优化

### 7.1 已进入当前速度版本

| 优化 | 做法 | 局部收益 | 状态 |
|---|---|---:|---|
| WMMA Query projection | 16x128 tile，FP16 WMMA、FP32 accumulation，并融合 INT8 quantization | 36 层 2.298 -> 0.715 ms，3.21x | 保留 |
| GQA-4 shared scan | 一个 KV-head 索引读取同时服务四个 Query head | 避免四次重复读取 | 保留 |
| Tail-resolution quantile | 先算 `ceil(c/r(N))`，再向上对齐到 256，并限制在 `[256,8192]`；扫描时直接写候选 | 删除完整 score tensor 和通用 top-k；256K 单窗口支持 `c` 从 16 提高到至少 64 | 机制保留，`c=64` 暂定 |
| Dynamic sample shared memory | 按 `next_power_of_two(m(N))` 分配，而不是按最大样本数固定分配 | 放宽到 8,192 时不惩罚短序列 occupancy | 保留 |
| Capacity-aware ragged split | 按候选容量自动选 4/8/16 splits，使每块动态 shared memory 约不超过 44 KB | 修复 512K、4% 候选的 CUDA launch failure | 保留 |
| Warp quantile | warp 内排序并汇总阈值 | scan 约 1.044x | 保留 |
| Pattern specialization | 为八种常见 bit pattern 编译专用 kernel | scan 约 1.019x | 保留 |
| Incremental append | 仅编码新生成 token 的 Key | 避免重建历史索引 | 保留 |
| GPU-resident exact KV | 候选后直接 gather 原始 K/V | 避免 CPU/PCIe 取回 | 保留 |

WMMA 的 Query code 一致率为 99.9974%，最大 code 差为 1，最大 scale 绝对差
为 `3.05e-5`。在 32K 和 120K PPL 上没有观察到同方向的系统性质量变化。

### 7.2 120K 阶段分解

Qwen3-4B、36 层、RTX 3090：

| 阶段 | Scalar Query | WMMA Query |
|---|---:|---:|
| Key index append | 2.428 ms | 2.410 ms |
| Query prepare | 2.459 ms | **1.819 ms** |
| Packed-index retrieval | 3.149 ms | 3.317 ms |
| Exact sparse attention | 3.986 ms | 3.940 ms |
| 四阶段合计 | 12.022 ms | **11.486 ms** |

完整 WMMA decode 为 61.130 ms/token，因此剩余约 49--50 ms 来自 QKV/O
projection、MLP、LayerNorm、HF cache/control flow、跨层调度和 kernel launch。
即使把 retrieval 的 3.317 ms 完全删除，整模型也只能再快约 5%。

### 7.3 已验证但否决

| 尝试 | 结果 | 否决原因 |
|---|---:|---|
| Retrieval 与 exact attention 强融合 | 0.67x--0.72x | occupancy 和局部候选缓冲恶化 |
| Band-major layout | 0.86x--0.90x | 地址计算超过访存收益 |
| Metadata shared-memory broadcast | 约 1.00x | metadata 不是瓶颈 |
| Sparse workspace 预分配 | 约 1.00x | allocator 不是瓶颈 |
| 更紧 candidate capacity / split8 | E2E 基本持平 | workspace 降低但无稳定速度 |
| 固定 quantile sample 上限 2,048 | 256K、1,280-token 时只能覆盖约 10 个目标尾部样本 | 长序列阈值分辨率不足；已由 8,192 上限和解析缩放替代 |
| 跨 decode step 候选复用 | refresh=2 只保留约 96% QKSieve mass | 质量风险大 |
| Cold-token 硬跳过 | 256K 相对 QKSieve 质量仅 91.60%--92.52% | E2E 仅预计再快 2%--4% |

## 8. 当前质量结果

### 8.1 完整 LongBench 质量参考

Llama-3.1-8B-Instruct，16 个英文任务，3,750 个严格 Full-QKSieve 配对：

| 方法 | Macro score | 相对 Full |
|---|---:|---:|
| Full Attention | 0.459398 | 100% |
| QKSieve reference | 0.458852 | **99.881%** |

任务分层 paired bootstrap：

```text
macro difference 95% CI = [-0.002647, +0.001591]
retention 95% CI        = [99.424%, 100.347%]
```

这是 request-local QK-balanced + qMSE + full proxy top-k 路径，不是最新
sampled-quantile 速度路径。

### 8.2 优化部署路径 PPL

| 长度/数据 | Full PPL | QKSieve PPL | 质量说明 |
|---|---:|---:|---|
| 32K，六窗口 | 19.6910 | 19.6337 | 100.29% PPL 保持 |
| 64K，sports 两窗口 | 6.0606 | 6.0089 | 100.86% |
| 120K，三次交叉顺序 | 3.7273 | 3.6271 | 单窗口诊断 |
| 旧 256K 边界压力，mixed_b 16 token | 5.7518 | 5.1052 | history 已占满原生上限，不能作正式质量 |

PPL 小幅优于 Full 不能解释成稳定质量提升。严格原生 256K 补测使用
`history=262080, eval=64`，使总长度恰好等于 262,144，并扩展到四个独立
窗口；完成前不能用旧 16-token 边界压力结果作正式质量结论。

首个严格原生窗口还原了旧的 2,048-sample 硬上限：

| 路径 | Full PPL | Sparse PPL | PPL 保持 | Top-1 | Steady decode |
|---|---:|---:|---:|---:|---:|
| Key-MSE、1,280 exact/head、sample cap 2,048 | 2.1605 | 5.6713 | **38.09%** | 70.31% | **9.06x** |

该结果说明“1,280 token + 旧 sample cap”在 256K 上虽然很快，但质量不可用，
不能作为正面结果。解析式要求的 quantile 样本数约为
`ceil(16 / (1280/262080)) = 3276`；原实现被 2,048 截断。解除样本上限并在
四个独立窗口上使用完整 proxy top-k 后，1,280-token 的 PPL 保持率为
80.77%；Exact FP16 QK oracle 则为 100.24%。因此 quantile 分辨率不是唯一
根因，低比特 proxy 相对 oracle 的错排才是当前主要瓶颈；最终
`1,280/262,080=0.488%` 的 exact-attention 预算本身并未失败。

同一个严格原生窗口、同一份 shared prefill 的部署前沿如下。`Steady` 包含
低比特扫描和 exact sparse attention；`Online` 还把一次索引准备固定成本均摊到
64 个评估 token，但两者都不包含共同的 full prefill。

| Exact/head | 实际 active | PPL 保持 | Top-1 | KL(Full \|\| Sparse) | Steady | Online |
|---:|---:|---:|---:|---:|---:|---:|
| 2,554 | 0.975% | 62.99% | 81.25% | 0.45153 | 8.88x | 8.25x |
| 3,802 | 1.451% | 81.30% | 90.63% | 0.18137 | 8.89x | 8.26x |
| 5,051 | 1.927% | 84.61% | 93.75% | 0.12558 | 8.87x | 8.23x |
| 7,588 | 2.895% | 92.26% | 95.31% | 0.06616 | 8.62x | 8.02x |
| 10,095 | 3.852% | **95.12%** | 93.75% | 0.04691 | **7.57x** | **7.10x** |
| 12,531 | 4.781% | 95.06% | 95.31% | 0.04661 | 6.64x | 6.28x |
| 15,441 | 5.892% | 93.75% | **98.44%** | **0.03141** | 5.84x | 5.56x |

完整 proxy top-k 在 3.91%/4.88%/6.00% 下的 PPL 保持分别为
95.50%/95.92%/95.70%，KL 则从 0.03899 降到 0.02821。也就是说，增加预算
确实持续改善分布逼近，但单窗口目标 PPL 会被少数位置主导而非单调变化。本窗口
64 个目标中，单个 `Part` token 贡献约 1.9--2.2 NLL 增量；去掉它仅用于诊断时，
4%--6% 配置为 98.3%--99.1% 保持。正式论文不能删除该 token，而应补独立窗口、
配对置信区间，并同时报告 PPL、KL 和 Top-1。

提高 quantile 采样精度后的同窗口结果如下。完整 proxy top-k 是质量参考；
sampled 路径不物化全长 score，也不执行通用 top-k：

| 目标预算 | Selector | 样本数/head | PPL 保持 | Top-1 | KL | Steady |
|---:|---|---:|---:|---:|---:|---:|
| 4% | 完整 proxy top-k | 全量 | 95.501% | 96.875% | 0.03899 | 6.56x |
| 4% | sampled，`c≈16` | 410 | 95.390% | 93.750% | 0.04644 | 7.66x |
| 4% | sampled | 1,024 | 95.397% | 95.312% | 0.04047 | **7.91x** |
| 4% | sampled | 4,096 | **95.532%** | **96.875%** | 0.04031 | 7.54x |
| 6% | 完整 proxy top-k | 全量 | 95.698% | 95.312% | 0.02821 | 5.48x |
| 6% | sampled，`c≈16` | 267 | 93.830% | 95.312% | 0.03032 | 5.87x |
| 6% | sampled，`c≈61` | 1,024 | 95.704% | 95.312% | 0.02840 | **6.30x** |
| 6% | sampled | 4,096 | **95.996%** | **96.875%** | **0.02766** | 6.12x |

更高采样数没有让稳态路径变慢，反而在 6% 档把 5.48x 提高到 6.30x。原因是
候选数方差下降：6% 的跨-head 范围从 4,293--35,044 收窄到
9,343--23,507，减少了 ragged sparse attention 的极端容量、split 和尾部等待。
这不是“多做采样天然更快”，而是当前 kernel 对候选长尾敏感。`c=64` 的统一
解析规则先计算 `ceil(64/r)`，再向上对齐到 kernel 支持的 256 倍数，因此在
4%/5%/6% 下分别请求 1,792/1,536/1,280 个样本，保证实际期望尾部样本不少于
64。该规则已进入多窗口验证；完成前上表仍是单窗口机制证据，不是最终 256K
质量均值。

512K 超过 Qwen3-4B 的原生 `max_position_embeddings=262144`，且无 RoPE
scaling。`history=524256, eval=32` 的 Full prefill 在 7 张 RTX 3090 上成功，
但 Full PPL 已达到 5510.60，说明模型本身已严重外推失真。因此下表只表示
“相对同一外推 Full 的分布逼近”，不能当作原生 512K 任务质量：

| 实际 active | PPL 保持 | Top-1 | KL | Steady | Online | Head 候选范围 |
|---:|---:|---:|---:|---:|---:|---:|
| 2.951% | 105.34% | 93.75% | 0.0978 | 10.70x | 9.00x | 5,117--33,365 |
| 3.947% | 101.63% | 96.88% | 0.0794 | **10.72x** | 8.13x | 6,999--48,828 |
| 4.880% | 99.97% | 93.75% | 0.0754 | 8.98x | 7.82x | 8,805--49,759 |
| 5.872% | 98.57% | 96.88% | **0.0592** | 7.72x | 6.85x | 11,359--66,293 |

完整 proxy top-k 在 3%/4%/5%/6% 的 KL 为
0.0903/0.0804/0.0646/0.0581。sampled 路径跟随同样趋势，但候选范围仍很宽，
进一步证明下一步应提高 quantile 有效尾部样本，而不是只看平均 active ratio。

## 9. 当前速度结果

### 9.1 整模型 steady decode

不同长度点来自不同文本或运行协议，只能作为 scaling diagnostic，不能拟合成
精确 crossover 曲线。

| 长度 | Full | 本地 FIER | QKSieve | QKSieve / Full | QKSieve / FIER |
|---|---:|---:|---:|---:|---:|
| 32K | 88.282 ms | **44.778 ms** | 46.539 ms | 1.90x | 0.96x |
| 64K | 159.178 ms | 51.949 ms | **47.330 ms** | 3.36x | 1.10x |
| 120K | 279.379 ms | 80.647 ms | **61.130 ms** | **4.57x** | **1.32x** |

120K、64-token 在线总时间：

```text
Full              17.601 s
local FIER          5.534 s
QKSieve             4.553 s

QKSieve vs Full     3.87x
QKSieve vs FIER     1.22x
```

本地 FIER 是按论文公式实现和审计的本地版本，不是官方作者 CUDA 实现，不能
跨硬件与论文中的速度数字直接相除。

256K 的 7-GPU、64-token 共享 prefill 质量 runner 中，当前可用的探索点是：

```text
Full steady decode                    587.744 ms/token
QKSieve, 10,240 configured/head        77.696 ms/token
steady decode speedup                   7.565x
fixed-cost-amortized decode speedup      7.097x
active exact KV                         3.852%
PPL retention                          95.124%
```

共同 prefill 为 950.64 秒，因此只生成 64 token 时，包含 prefill 的请求级加速
仅为 1.034x。按实测固定成本和每 token 斜率外推，生成 1,024/4,096 token 时
请求级加速约为 1.51x/2.65x。该外推需要真实自回归长生成复核。

这不是与 32K--120K 单卡 WMMA sweep 完全相同的协议，应单独标注；而且
256K 目前只有一个 64-token 窗口，不能当作最终均值。

512K、4% 探索点的对应整模型数字为：

```text
Full steady decode                   1,155.533 ms/token
QKSieve steady decode                  107.749 ms/token
steady decode speedup                   10.724x
fixed-cost-amortized decode speedup       8.134x
common full prefill                  4,297.18 s
```

只生成 32 token 时，包含 prefill 的请求级加速仅为 1.008x；按实测斜率外推，
生成 1,024/4,096/8,192 token 时约为 1.24x/1.91x/2.66x。由于 512K 非原生
上下文，这里只用于系统 break-even 诊断。

### 9.2 Attention 子系统

以下为包含 Query 准备、低比特扫描、候选写出和 exact sparse attention 的
物理 CUDA 路径；不包含模型 MLP 等非 attention 工作和首次索引构建。

| 长度 | Full SDPA | QKSieve | 子系统加速 |
|---|---:|---:|---:|
| 128K | 2.439 ms/layer | 0.360 ms/layer | 6.78x |
| 256K，1,280 exact/head | 175.008 ms/36 layers | 12.650 ms/36 layers | 13.83x |
| 512K，1,280 exact/head | 380.282 ms/36 layers | 17.513 ms/36 layers | 21.71x |

后两行是固定 1,280 候选的物理算子伸缩。Exact-QK oracle 已证明 256K 下
该预算可保持 Full 质量，但当前低比特 proxy 尚不能准确找到这些 token。
因此 13.83x 是“若 selector 达到 oracle 精度”的物理计算目标，不是当前
端到端可部署质量合格点；512K 还不是原生质量结论。

128K 对本地 FIER：

```text
FIER     1.179 ms/layer
QKSieve  0.360 ms/layer
QKSieve / FIER = 3.28x
```

## 10. 当前方法的真实资源含义

以 120K 为例：

```text
exact attention tokens/head = about 1280
active exact-token ratio     = 1280 / 120000 = 1.067%
index ratio                  = about 5.854% of Full FP16 K+V
raw exact K/V retained       = 100%
```

因此应分别报告：

1. `active KV ratio`：本步真正进入 exact attention 的 token 比例；
2. `index ratio`：低比特检索索引相对完整 K+V 的额外字节；
3. `resident exact KV ratio`：当前 GPU-resident 版本为 100%；
4. `attention bytes read`：每步实际读取的索引和候选 K/V。

只写“KV ratio 约 1%”会误导读者以为显存减少到 1%。当前论文主张应是
training-free sparse retrieval attention 和 decode acceleration，不是 KV cache
容量压缩。

## 11. 仍需闭合的论文证据

优先级从高到低：

1. 已完成严格正交归因：只刷新 allocation 为 89.91%，只刷新坐标并固定 bit
   为 100.44%，证明 256K 主因是冻结 QK 坐标而非总 rate。
2. 把 request-local QK-balanced 坐标的固定准备成本改成 prefill 增量统计、
   流式更新或异步构建，目标是在保留 7.67x steady 的同时把 64-token Online
   从 4.96x 推近冻结路径的 6.67x。
3. 按 layer/head 补 frozen-vs-local principal angle、joint-spectrum drift、
   proxy-to-oracle recall、score RMSE 和 crossing margin，建立无需训练的
   coordinate-refresh certificate。
4. 用与 20-Newsgroups 不同的独立长文本和至少一个新模型复核
   Exact-QK top-1,280 的预算充分性，以及修复后 proxy 的高保真点。
5. 在冻结后的同一 optimized request-local QK-balanced + fixed-bit +
   sampled-quantile 路径上完成 LongBench m20 验证，再扩大到 3,750 样本。
6. 用同一路径跑完整 RULER 4K--128K，重点补 64K/128K。
7. 在 H100 或 A100 上用同一 backend、同一模型、同一候选预算比较 Full、
   FIER 和 QKSieve；同时报告预展开纯 SDPA 与真实 HF GQA+SDPA。
8. 做严格同文本的 8K/16K/24K/32K/48K/64K/96K/128K whole-model sweep，
   报告固定成本、每 token 斜率和 break-even。
9. 报告 prefill、index build、steady decode、固定生成长度总延迟和自然 EOS
   请求延迟，避免只报告稳态。
9. 增加 Qwen3 和 Mistral 的完整 LongBench，验证跨模型泛化；512K 正式质量
   只在原生支持 512K 的模型上测试。

## 12. 复现入口

### 12.1 主要实现

```text
src/run_sample_calibrated_longbench_20260717.py
src/run_direct_countcap_denseprompt_ppl_20260725.py
src/qksieve_query_cuda_20260728.py
src/mixedblock_spectral_cuda_20260729.py
src/build_global_qksieve_template_20260729.py
src/rewrite_qksieve_template_allocation_20260729.py
```

### 12.2 速度与内核验证

```text
src/benchmark_qksieve_global_pattern_specialization_20260729.py
src/benchmark_qksieve_fused_select_attention_20260729.py
src/benchmark_qksieve_metadata_broadcast_20260729.py
src/benchmark_qksieve_warpselect_20260729.py
src/benchmark_qksieve_bandmajor_20260729.py
src/benchmark_preallocated_sparse_attention_20260729.py
scripts/run_qksieve_frozen_template_frontier_20260729.sh
```

### 12.3 Key-only 决策实验

```text
scripts/launch_qksieve_keyonly_decision_ppl_2gpu_20260730.sh
scripts/launch_qksieve_global_keymse_template_3gpu_20260730.sh
scripts/launch_qksieve_qmse_keymse_deploy_compare_3gpu_20260730.sh
src/summarize_qksieve_keyonly_decision_ppl_20260730.py

results/20260730_qksieve_keyonly_decision_ppl_32k_2gpu/summary.json
results/20260730_qksieve_keyonly_decision_longbench_m20_5gpu/
results/20260730_qksieve_global_keymse_template_3gpu/
results/20260730_qksieve_qmse_keymse_deploy_compare_3gpu/summary.json
```

### 12.4 长上下文与 cold-token 否决实验

```text
src/benchmark_qksieve_per_head_cold_skip_20260730.py
src/run_qksieve_coldskip_longcontext_quality_20260730.py
src/summarize_qksieve_longcontext_frontier_20260730.py
src/summarize_qksieve_longcontext_multiwindow_20260730.py

results/20260730_qksieve_per_head_coldskip_cuda_256k_512k_gpu6.json
results/20260730_qksieve_coldskip_quality_256k_sharedprefill/
results/20260730_qksieve_keymse_256k_budget_frontier_7gpu_v6/
results/20260730_qksieve_keymse_256k_highbudget_frontier_7gpu/
results/20260730_qksieve_256k_exact_oracle_vs_proxy_k1280_k2560_4window_7gpu/
results/20260730_qksieve_128k_exact_oracle_vs_proxy_k1280_k2560_4window_7gpu/
results/20260730_qksieve_256k_selector_cause_split_4window_7gpu/
scripts/launch_qksieve_keymse_256k_highbudget_frontier_7gpu_20260730.sh
scripts/launch_qksieve_keymse_256k_tailvariance_7gpu_20260730.sh
scripts/launch_qksieve_keymse_256k_multiwindow_7gpu_20260730.sh
scripts/launch_qksieve_256k_exact_oracle_vs_proxy_7gpu_20260730.sh
scripts/launch_qksieve_256k_selector_cause_split_7gpu_20260730.sh
scripts/launch_qksieve_keymse_512k_budget_frontier_7gpu_20260730.sh
```

## 13. 给复现者的最短描述

QKSieve 对每层、每个 KV head 建立一个保持完整 dot product 的 QK-balanced
双正交坐标，把 128 维分成八个 16-D band，并用固定
`(4,1,1,1,1,1,0,0)` 240 bit/token/head 量化 Key，作为只服务检索的低比特索引。
每步用当前 Query 扫描索引，按长度相关预算 `B(N)` 选择 token，再在原始
GPU-resident FP16 K/V 上执行精确 sparse attention。4K--128K 的已验证预算
上限为 1,280；严格 256K Exact-QK oracle 也证明 top-1,280 足够。冻结低比特
模板只有 80.77%，但 request-local 坐标加固定 240-bit 在相同预算恢复到
100.44%，其整模型 Steady/Online decode 为 7.67x/4.96x（共享 prefill 后）。
质量参考实现使用 request-local moments 和完整 proxy top-k；下一版速度实现
应在 prefill 增量构建同一 request-local 坐标，再结合 WMMA Query 投影、GQA-4 共享扫描和
sampled-quantile 候选写出。完整 K/V 不删除，方法不依赖训练、router、任务
规则、rerank 或 Full fallback。
