# QKSieve 256K/512K 速度与质量实验

## 1. 测试目的

本实验回答两个问题：

1. 当前 QKSieve 部署路径在 256K 原生上下文边界上需要多少 active KV，质量和
   decode 速度分别是多少。
2. 在 512K 超长历史下，CUDA 路径是否仍能运行和扩展，以及相对同一个 Full
   forward 的近似误差和速度如何。

测试方法为 QK-balanced 坐标、Key-MSE 混合位宽分配、240 bit/token/KV-head
低比特 Key 索引、sampled-quantile 候选选择和原始 FP16 K/V 上的精确 sparse
attention。没有 router、任务规则、exact-QK rerank 或 Full fallback。完整
FP16 K/V 均驻留 GPU；active ratio 是本步进入精确 attention 的 token 比例，
不是显存 KV 保留比例。

## 2. 协议

| 项目 | 256K | 512K |
|---|---:|---:|
| 模型 | Qwen3-4B-Instruct | Qwen3-4B-Instruct |
| GPU | 7 x RTX 3090 | 7 x RTX 3090 |
| history tokens | 262,080 | 524,256 |
| target tokens | 64 | 32 |
| 总长度 | 262,144 | 524,288 |
| 模型原生上限 | 262,144 | 262,144 |
| 上下文状态 | 原生边界 | 超上限外推 |
| Full PPL | 2.160468 | 5510.596547 |
| 共同 Full prefill | 950.81 s | 4297.18 s |

质量指标定义：

```text
PPL retention = Full PPL / Sparse PPL
Top-1         = Sparse 与 Full 下一 token argmax 的一致率
KL            = KL(Full distribution || Sparse distribution)
```

`PPL retention` 越接近 100%、`Top-1` 越高、`KL` 越低越好。PPL retention
偶尔超过 100% 只表示目标 token 在 sparse 分布下的 NLL 更低，不能解释成方法
普遍优于 Full。

速度指标定义：

```text
Steady = 仅稳态 decode，每 token 包含低比特扫描和精确 sparse attention
Online = 把一次索引/selector 固定开销均摊到本窗口的 target token
Request = prefill + 固定开销 + 全部生成 token
```

## 3. 256K 原生边界

### 3.1 被否决的是低比特 selector，不是 1,280 的 oracle 预算

最初的单窗口实验中，`1,280 / 262,080 = 0.488%`。完整低比特 proxy top-k
的 PPL retention 只有 47.44%，旧 sampled 路径为 38.09%。当时据此判断
“最终精确 attention 预算本身不足”，但后续四窗口 Exact-QK oracle 对照证明
这个因果判断是错误的。

四个独立窗口、256 个严格配对 target token 的结果如下。Oracle 使用原始 FP16
Q/K 计算全部历史 score，再严格选择每个 Query head 的 top-k；proxy 使用冻结
QK-balanced 坐标、Key-MSE 240-bit 索引和完整 proxy top-k。两者最终都在原始
FP16 K/V 上执行 exact sparse attention。

| Selector | Exact/head | Active | Full attention mass | PPL retention（95% CI） | Top-1 | KL |
|---|---:|---:|---:|---:|---:|---:|
| Exact FP16 QK oracle | 1,280 | 0.488% | 90.002% | **100.240% [99.48, 100.88]** | 90.625% | **0.00516** |
| Exact FP16 QK oracle | 2,560 | 0.977% | 93.057% | **100.148% [99.54, 100.68]** | 92.969% | **0.00297** |
| QK-balanced low-bit proxy | 1,280 | 0.488% | 未测 | **80.771% [77.84, 83.92]** | 67.188% | 0.24246 |
| QK-balanced low-bit proxy | 2,560 | 0.977% | 未测 | **86.132% [84.68, 87.77]** | 71.094% | 0.15092 |

因此 1,280 个 exact attention token 在这些 256K 窗口上已经足够。当前质量
断崖来自低比特 proxy 没有找对 oracle 的 1,280 个 token。把 proxy 候选翻倍
到 2,560 虽有改善，但仍远未闭合 oracle gap。24% active 能恢复质量，是因为
大候选池覆盖了更多排序错误，而不是 Full attention 天然需要 24% token。

为了排除“128K 与 256K 使用了不同方法版本”的混淆，又在同一冻结模板、
同一 Key-MSE 240-bit 索引、同一完整 proxy top-k、同一四组文本和随机种子上
补做了 128K 对照：

| Selector / budget | 128K PPL retention（95% CI） | 256K PPL retention（95% CI） |
|---|---:|---:|
| Exact FP16 QK，top-1,280 | **100.805% [99.40, 102.23]** | **100.240% [99.48, 100.88]** |
| Exact FP16 QK，top-2,560 | **100.345% [99.14, 101.57]** | **100.148% [99.54, 100.68]** |
| 冻结低比特 proxy，top-1,280 | **98.993% [96.53, 102.99]** | **80.771% [77.84, 83.92]** |
| 冻结低比特 proxy，top-2,560 | **100.005% [97.66, 103.02]** | **86.132% [84.68, 87.77]** |

这是严格的长度交叉证据：Exact-QK oracle 在两个长度都基本无损，低比特
proxy 却从 128K 的约 99%/100% 降到 256K 的约 81%/86%。因此不能用
“预算比例从 1% 减半到 0.5%”解释断崖，因为 oracle 在 0.488% 仍足够。
更符合数据的机制是：固定误差下，历史 token 增多会使 top-k 边界附近的
score 更密，极值量化误差和 crossing 数增加；冻结模板的分布漂移还会进一步
放大每个 token 的 proxy error。

进一步的受控归因只改变一个因素，仍使用相同四窗口、256 个配对 target token、
top-1,280 和原始 FP16 K/V consumer：

| 256K selector | 索引策略 | PPL retention（95% CI） | Top-1 | KL | Steady decode | Online decode |
|---|---|---:|---:|---:|---:|---:|
| Exact FP16 QK oracle | 无代理索引 | **100.240% [99.48, 100.88]** | 90.625% | 0.00516 | 诊断路径 | 诊断路径 |
| 冻结坐标 + 冻结 allocation | 冻结 QK-balanced + Key-MSE 240-bit | **80.771% [77.84, 83.92]** | 67.188% | 0.24246 | **7.785x** | **6.668x** |
| 冻结坐标 + local allocation | 冻结 QK-balanced + request-local Key-MSE 240-bit | **89.911% [84.13, 94.28]** | 77.344% | 0.13332 | **7.766x** | **5.259x** |
| Local 坐标 + fixed allocation | Request-local QK-balanced + `(4,1,1,1,1,1,0,0)` 240-bit | **100.443% [99.80, 101.02]** | 91.016% | 0.00620 | **7.671x** | **4.955x** |
| Local 坐标 + local allocation | Request-local QK-balanced + Key-MSE 240-bit | **100.430% [99.78, 100.99]** | 91.016% | 0.00622 | **7.733x** | **4.055x** |
| 冻结高位宽 proxy | 同一冻结坐标，全部 band 使用 INT8 | **100.267% [99.48, 100.91]** | 91.016% | **0.00508** | 1.882x | 1.846x |

归因已经闭合，而且比此前更精确：只刷新 allocation 从 80.77% 恢复到
89.91%，但只刷新 QK-balanced 坐标并使用固定 allocation 就恢复到 100.44%。
因此 256K 断崖的主因是**冻结 QK 坐标在当前请求上的外推**，allocation
失配是次因。冻结全 INT8 说明提高精度可以掩盖过期坐标，但速度代价很大。
最终 attention budget 和 240-bit 物理 rate 都不是根因。

速度列是共享 prefill 后的整模型 decode：Steady 排除首次模板准备，
Online 将固定 selector 开销均摊到本实验约 63 个 timed step，但二者都不包含
约 1,058 秒的共同 256K prefill。当前首选是 **request-local 坐标 + fixed
240-bit allocation**：质量与 local auto allocation 相同，Online 从 4.06x
提高到 4.96x。下一步应把坐标统计和变换增量化或异步化，而不是退回
全 INT8 或强制每个请求动态 allocation。

### 3.2 当前单窗口前沿

下表优先列出提高 quantile 分辨率后的路径。`Full proxy top-k` 是需要全量
proxy score 和通用 top-k 的质量参考，不是最快部署路径。

| 目标预算 | Selector | 实际 active | PPL | PPL retention | Top-1 | KL | Steady | Online |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4% | Full proxy top-k | 3.907% | 2.2622 | 95.501% | 96.875% | 0.03899 | 6.56x | 5.76x |
| 4% | sampled, 1,024 samples | 3.886% | 2.2647 | 95.397% | 95.312% | 0.04047 | **7.91x** | **7.40x** |
| 4% | sampled, 4,096 samples | 3.900% | 2.2615 | **95.532%** | **96.875%** | 0.04031 | 7.54x | 7.08x |
| 6% | Full proxy top-k | 6.001% | 2.2576 | 95.698% | 95.312% | 0.02821 | 5.48x | 5.22x |
| 6% | sampled, 1,024 samples | 6.017% | 2.2574 | 95.704% | 95.312% | 0.02840 | **6.30x** | **5.97x** |
| 6% | sampled, 4,096 samples | 6.000% | 2.2506 | **95.996%** | **96.875%** | **0.02766** | 6.12x | 5.81x |

当前最合理的两个工作点：

| 取向 | active KV | 单窗口质量 | 稳态 decode |
|---|---:|---|---:|
| 速度优先 | 约 4% | 约 95.4%-95.5% PPL retention | 7.5x-7.9x |
| 质量优先 | 约 6% | 约 95.7%-96.0% PPL retention | 6.1x-6.3x |

该窗口中一个 `Part` 目标 token 贡献约 1.9-2.2 的 excess NLL，所以 PPL
retention 对单点非常敏感。不能删除该 token；正式结论要依赖多窗口聚合、KL
和 Top-1，而不是挑选后的 PPL。

### 3.3 quantile 方差

若目标 active rate 为 `r`，sampled threshold 使用 `m` 个规则分层样本，目标
尾部中的期望样本数记为 `c = m*r`。候选比例的有限样本相对标准差近似为：

```text
relative std(candidate count) ~= 1 / sqrt(c)
```

旧 `c=16` 的相对标准差约 25%。256K、6% 预算把采样从 267 提高到 1,024
后，跨-head 候选范围从 4,293-35,044 收窄到 9,343-23,507，质量追平 Full
proxy top-k，稳态速度还从 5.48x 提高到 6.30x。速度提升来自极端候选容量和
ragged attention 尾部减少，不表示采样越多永远越快。

当前正在复核统一规则：

```text
m_raw(N) = ceil(64 / r(N))
m(N)     = min(8192, max(256, 256 * ceil(m_raw(N) / 256)))
```

在 4%/5%/6% 下分别为 1,792/1,536/1,280 个样本，从而保证期望目标尾部
样本不少于 64。四个独立洗牌窗口结果出来前，`c=64` 和最终 256K active
budget 都只算临时选择。

### 3.4 四窗口预算闭合

单窗口容易被少数高 excess-NLL token 主导，因此最终使用四个独立洗牌窗口，
每个窗口 64 个 target token，共 256 个严格配对 token。PPL 保持率先在每个
窗口内计算 NLL 差，再跨窗口聚合；95% 区间按窗口 bootstrap，只有四个 cluster，
因此区间仍较粗，不能替代后续跨数据集验证。

| 实际 active | PPL retention（95% CI） | Top-1 | KL | Steady | Online |
|---:|---:|---:|---:|---:|---:|
| 7.977% | 94.440% [93.38, 95.68] | 78.516% | 0.04365 | 5.511x | 4.925x |
| 9.977% | 96.260% [94.53, 97.73] | 82.031% | 0.03266 | 5.345x | 5.097x |
| 11.927% | 96.949% [94.79, 98.43] | 83.984% | 0.02696 | 4.453x | 4.283x |
| 15.951% | 97.418% [95.43, 99.07] | 83.594% | 0.02250 | 3.642x | 3.377x |
| 20.055% | 99.072% [97.62, 100.55] | 84.766% | 0.01349 | 2.959x | 2.887x |
| **23.958%** | **100.250% [99.40, 101.06]** | **92.188%** | **0.00733** | **2.989x** | **2.911x** |

`23.958%` 是当前 256K 高保真工作点：它与 Full 的 PPL 在窗口 bootstrap
不确定性内一致，不能解释为普遍优于 Full。`20%` 与 `24%` 的速度差很小，
24% 略快来自 split 调度边界与有限计时噪声，不应宣称更大预算本身会加速。

四窗口平均绝对时间：

```text
Full steady               588.942 ms/token
QKSieve-24 steady          197.019 ms/token
QKSieve-24 online          202.347 ms/token
QKSieve-24 fixed overhead    0.336 s
shared full prefill       1058.826 s
```

QKSieve-24 保留完整 FP16 K/V 在 GPU；23.958% 是每个 query head 实际进入
精确 QK-softmax-AV 的 token 比例，不是物理 KV 保留率。低比特 Key 索引是
额外约 5.85% full-KV 等价空间。

### 3.5 单窗口速度优先点的请求级速度

以 4%、1,024-sample 点为例：

```text
Full steady              587.610 ms/token
QKSieve steady            74.286 ms/token
QKSieve fixed overhead      0.326 s
shared full prefill       950.367 s
```

| 生成 token 数 | 包含 prefill 的请求级加速 |
|---:|---:|
| 64 | 1.034x |
| 1,024 | 1.512x |
| 4,096 | 2.675x |
| 8,192 | 3.697x |

约生成 2,166 token 后请求级加速达到 2x，约 5,214 token 后达到 3x。这里是
根据实测固定成本和每 token 斜率计算的外推，仍需真实长自回归生成复核。

## 4. 512K 外推压力测试

Qwen3-4B 的原生上下文上限是 262,144。512K Full PPL 已达到 5510.60，
说明 Full 模型本身发生严重位置外推失真。因此下表只能回答：

- sparse 分布是否接近同一个已经失真的 Full 分布；
- CUDA 路径在 512K 是否可运行、速度如何扩展。

它不能回答原生 512K 任务质量，也不能作为论文的 512K accuracy claim。

| 实际 active | PPL retention | Top-1 | KL | Steady | Online | 候选范围/head |
|---:|---:|---:|---:|---:|---:|---:|
| 2.951% | 105.34% | 93.75% | 0.0978 | 10.70x | 9.00x | 5,117-33,365 |
| 3.947% | 101.63% | 96.88% | 0.0794 | **10.72x** | **8.13x** | 6,999-48,828 |
| 4.880% | 99.97% | 93.75% | 0.0754 | 8.98x | 7.82x | 8,805-49,759 |
| 5.872% | 98.57% | 96.88% | **0.0592** | 7.72x | 6.85x | 11,359-66,293 |

4% 点的绝对时间：

```text
Full steady             1155.533 ms/token
QKSieve steady           107.749 ms/token
QKSieve fixed overhead     1.064 s
shared full prefill      4296.483 s
```

| 生成 token 数 | 包含 prefill 的请求级加速 |
|---:|---:|
| 32 | 1.008x |
| 1,024 | 1.243x |
| 4,096 | 1.905x |
| 8,192 | 2.657x |
| 16,384 | 3.831x |

约生成 4,573 token 后请求级加速达到 2x，约 10,328 token 后达到 3x。

### 4.1 内核修复

512K、4% 的单个 head 候选容量可超过 52K。旧 ragged attention 固定使用
4-way split，动态 shared memory 超过 RTX 3090 的 launch 上限并触发
`CUDA invalid argument`。当前实现按候选容量自动选择 4/8/16-way split，
将每个 block 的动态 shared memory 控制在约 44 KB；52K、80K、160K 容量
smoke test 均通过。该修复只解决执行正确性，不改变质量算法。

### 4.2 真实 HF GQA 基线与理想 SDPA 基线

独立 attention 基准使用 24% active、`c=128`、完整历史扫描，不使用
cold-token 跳过。QKSieve 时间包含 Query 投影/量化、低比特索引扫描、
sampled-quantile 选取和原始 FP16 K/V 上的精确 sparse attention；不包含
索引构建和非 attention 模型计算。

| 长度 | 预展开纯 SDPA | HF `repeat_kv` + SDPA | QKSieve-24 | 相对预展开 | 相对 HF |
|---:|---:|---:|---:|---:|---:|
| 256K | 175.104 ms | 567.610 ms | 199.992 ms | 0.876x | **2.838x** |
| 512K | 348.520 ms | 1135.721 ms | 373.800 ms | 0.932x | **3.038x** |

预展开纯 SDPA 是理想化算子下界：GQA 的 K/V 已在计时前复制成 32 个 query
heads。真实 Hugging Face decode 每步还执行 `repeat_kv`，所以 QKSieve-24
相对理想纯 SDPA 不占优，但相对真实 HF attention 路径仍有约 2.8x-3.0x。
这与 256K 七卡整模型 runner 的 2.989x steady decode 相互闭合。

512K、24% 时若仍用 `c=64`，置信候选容量会超过当前 16-way ragged kernel
的安全上限。将期望尾部样本提高到 `c=128` 后，规则样本数为 768，候选容量
下降且阈值方差更小。该设置只支持 512K 系统伸缩结论；Qwen3-4B 的原生上限
仍是 256K。

## 5. 当前结论

1. Exact FP16 QK oracle 证明 256K 下 1,280 token/head（0.488%）已经可以
   保持 100.24% PPL；被否决的是冻结低比特 selector，而不是这个预算本身。
2. 冻结低比特 proxy 在 top-1,280 下只有 80.77%；只刷新 allocation 为
   89.91%，只刷新坐标并固定 bit 为 100.44%。因此根因主要是冻结坐标外推。
3. Request-local 坐标 + fixed 240-bit 的整模型 Steady/Online decode 为
   7.67x/4.96x
   （共享 prefill 后）；约 24% active 的 `100.25% / 2.99x` 只是旧冻结模板
   通过扩大候选池补偿排序误差的工作点，不再是首选修复。
4. `c=64` 的 sampled-quantile 规则显著降低 256K 候选 straggler；512K、
   24% 时使用 `c=128` 以满足 16-way kernel 容量约束。
5. QKSieve-24 相对预展开纯 SDPA 为 0.88x-0.93x，相对真实 HF GQA attention
   为 2.84x-3.04x。论文必须同时报告两种基线，不能只选有利口径。
6. 512K Full PPL 为 5510.6，说明模型本身已位置外推失真。512K 只能报告
   kernel/系统伸缩和相对失真 Full 的诊断，不能作为原生任务质量结论。
7. 256K、24% 点包含共同 prefill 后，生成 64/1,024/4,096/8,192/16,384
   token 的请求级加速约为 1.02x/1.32x/1.86x/2.20x/2.50x；约 5,436
   个生成 token 后达到 2x。

## 6. 结果与复现入口

```text
results/20260730_qksieve_keymse_256k_tailvariance_7gpu_v1/
results/20260730_qksieve_keymse_512k_extrap_budget_frontier_7gpu_v4_splitfix/
results/20260730_qksieve_keymse_256k_multiwindow_c64_7gpu/
results/20260730_qksieve_keymse_256k_highbudget_corrected_c64_7gpu/
results/20260730_qksieve_keymse_256k_qualityclosure_c64_7gpu/
results/20260730_qksieve_qkbalanced_keymse_256k_k1280_fulltopk_4window_7gpu/
results/20260730_qksieve_256k_exact_oracle_vs_proxy_k1280_k2560_4window_7gpu/
results/20260730_qksieve_128k_exact_oracle_vs_proxy_k1280_k2560_4window_7gpu/
results/20260730_qksieve_256k_selector_cause_split_4window_7gpu/
results/20260730_qksieve_keymse_attention_256k_512k_b24_c128_dualbaseline_gpu0.json

src/run_qksieve_coldskip_longcontext_quality_20260730.py
src/summarize_qksieve_longcontext_frontier_20260730.py
src/summarize_qksieve_longcontext_multiwindow_20260730.py
src/benchmark_qksieve_per_head_cold_skip_20260730.py
scripts/launch_qksieve_keymse_256k_tailvariance_7gpu_20260730.sh
scripts/launch_qksieve_keymse_256k_multiwindow_7gpu_20260730.sh
scripts/launch_qksieve_256k_exact_oracle_vs_proxy_7gpu_20260730.sh
scripts/launch_qksieve_256k_selector_cause_split_7gpu_20260730.sh
scripts/launch_qksieve_keymse_256k_highbudget_supplement_7gpu_20260730.sh
scripts/launch_qksieve_keymse_256k_512k_attention_gpu0_20260730.sh
scripts/launch_qksieve_keymse_512k_budget_frontier_7gpu_20260730.sh
```
