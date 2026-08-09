# QKSieve 公平对比、128K 稳健性与直接测速报告

日期：2026-08-01

## 1. 当前结论

当前应冻结的主方法是 **QKSieve auto-240**，而不是激进的 `[4,1]` 版本。

- 主配置：request-local QK-balanced 坐标、每层每 KV head 自动 240-bit Key-MSE/QK-MSE 位宽分配、plain symmetric scale、原始 FP16 K/V 上的 exact sparse attention。
- 质量参考路径：完整 proxy score + deterministic top-k。
- 速度部署路径：sampled-quantile + fused packed scan/compaction，不做 exact proxy top-k，也不做 exact QK rerank。
- 预算：`B(N)=min(N, 1280, max(256, ceil(0.06N)))`。
- 不使用 router、任务规则、sink/recent 保留、exact rerank 或 Full fallback。
- `[4,1]` 只作为低索引开销消融，不作为通用默认值。

当前证据支持以下表述：

1. 在完整 3,750-pair LongBench 上，reference profile 保持 Full 的 **99.881%**，95% paired-bootstrap CI 为 **[99.424%, 100.347%]**。
2. 在原生 Llama-3.1-8B 128K 六主题诊断中，auto-240 保持 **99.883%**，topic-bootstrap CI 为 **[97.15%, 102.83%]**。
3. RTX 3090 上，deployment profile 的完整单层 BF16 attention call 在 8/16/32/64/128K 分别是 **0.60/1.06/2.90/4.58/6.80x**。
4. 整模型 32-token decode 直接测速在 8/16/32/64/128K 分别只有 **0.20/0.29/0.45/0.77/1.38x**；包含 prefill 的请求分别是 **0.48/0.68/0.88/0.98/1.01x**。
5. 因而 QKSieve 当前只在长历史、足够长 decode 时有系统收益。短 LongBench 请求更慢是事实，不能用 attention 子系统速度替代请求速度。
6. 同一 160-pair LongBench 筛查集上，FIER RTN-1 g32 保持 **100.010%**，QKSieve 为 **99.789%**；0.100 个绝对 macro 百分点的差距不足以支持显著优越性结论。
7. BinaryPC 的发布 selector 在同一 128K 单层 BF16 协议下达到 **11.30x**，明显快于 QKSieve 的 **6.80x**。QKSieve 当前不能把 selector kernel 的绝对速度作为相对 BinaryPC 的优势。
8. Targeted RULER 的 91 个严格 pair 上保持 **99.928%**，95% CI 为 **[99.559%, 100.294%]**；但每任务、每长度仅 2/2/2/1 条，只能作为机制筛查。

## 2. 问题定义

对 decoder layer，Query head `h` 映射到 KV head `g(h)`。Full attention 为：

```text
s[h,i] = q[h]^T k[g(h),i] / sqrt(d)
p[h]   = softmax(s[h])
o[h]   = sum_i p[h,i] v[g(h),i]
```

目标是在每个 decode step 为每个 Query head 找到集合 `S_hat[h]`，只在原始 FP16 K/V 的这些位置计算精确 QK、softmax 和 AV。所有历史 K/V 仍可寻址，因此它是 query-aware sparse retrieval，不是永久 eviction。

## 3. 方法

### 3.1 Request-local QK-balanced 坐标

每层、每 KV head：

1. 按 stride 32 采样 post-RoPE prompt Keys。
2. 采样 prompt 末尾 8 个 Query 位置，并合并映射到该 KV head 的 GQA Query heads。
3. 构造未中心化二阶矩 `C_k` 与 `C_q`。
4. 对 Query 矩阵使用 `lambda=0.75` 的 isotropic shrinkage。
5. 对 `C_q^(1/2) C_k^(1/2)` 做 SVD，并构造双正交变换 `A,D`。

变换后 `q'=A^Tq`、`k'=D^Tk`，并保持：

```text
q'^T k' = q^T k
```

因此全维变换本身不近似 QK；误差只来自后续分 band 量化与候选截断。

### 3.2 8 个 16-D band 与自动位宽

128 维坐标按联合 QK 能量排序后分成 8 个连续 16-D band。每个 band 的位宽来自 `{0,1,2,4,8}`。

一个 active band 还为 16 个系数共享一个 FP16 scale，等价于每系数 1 metadata bit。物理 rate 为：

```text
R(b) = 16 * sum_g (b_g + I[b_g>0]) <= 240 bit/token/KV-head
```

位宽分配通过枚举小规模可行集合精确求解：

- reference：最小化 Query-weighted logit MSE。
- deployment：最小化 Key reconstruction MSE，减少运行成本。

量化规则：

- 8/4/2 bit：max-absolute symmetric quantization。
- 1 bit：sign × mean-absolute magnitude。
- 0 bit：该 band 不存 code 和 scale。

240 bit 即 30 byte/token/KV-head。FP16 K+V 为 `2*128*2=512 byte`，所以索引相对完整 K+V 的额外逻辑存储为 **5.859%**。

### 3.3 Reference selector

reference profile 用于质量审计：

1. 当前 Query 变换并量化。
2. 扫描全部 packed index，形成所有 token 的 proxy score。
3. 对 proxy score 做 deterministic exact top-`B(N)`。
4. 从 GPU-resident 原始 FP16 K/V gather 被选 token。
5. 在选中 token 上重新计算精确 QK、softmax、AV。

该路径不使用 exact QK rerank。所谓 exact top-k 指 proxy score 的精确排序，不是先扫描 Full FP16 QK。

### 3.4 Deployment selector

deployment profile 用于速度审计：

1. 使用模型级冻结的 QK-balanced/Key-MSE template。
2. 融合 Query projection、量化与 GQA-4 数据复用。
3. 从 packed index 规则分层采样 `m(N)` 个位置：

```text
r(N) = B(N)/N
m(N) = min(N, 8192, max(256, ceil(16/r(N))))
```

4. 从样本估计目标分位阈值。
5. 单次扫描 packed index，把超过阈值的 token 直接 compact 到 bounded buffer。
6. 在原始 K/V 上执行 exact ragged sparse attention。

该路径不物化长度为 `N` 的 score tensor，也不调用通用 top-k workspace。实际候选数可围绕 `B(N)` 波动。

## 4. 128K 通用配置选择

Llama-3.1-8B、128K、6 个主题/seed、96 个 target token 的原生诊断：

| Profile | 索引/完整 K+V | PPL 质量保持 | Exact top-k recall | Attention mass recall |
|---|---:|---:|---:|---:|
| Exact FP16 QK | 100% K scan | 99.877% | 100.000% | 100.000% |
| `[4,1]` | 2.734% | 100.519% | 65.264% | 83.723% |
| `4-2-2-1` | 5.078% | 100.068% | 71.683% | 85.191% |
| `4-4-2-1` | 5.859% | 100.083% | 75.827% | 85.770% |
| auto-240 | 5.859% | 99.883% | 73.813% | 85.545% |

关键判断：

- `[4,1]` 在这 96 个 token 上没有出现 PPL cliff，但 exact-set recall 明显更低，跨主题最差 case 也更差，因此证据不足以把它冻结为通用主方法。
- `4-2-2-1`、`4-4-2-1` 与 auto-240 都闭合了 128K 质量。
- auto-240 能随 layer/head 的数值分布改变位宽，方法上更通用，故作为主配置。

合成机制实验也显示长度增加时更高位宽更重要。128K 下 `[4,1]` 到 `4-4-2-1` 的 exact-set recall 从 69.29% 升到 77.24%，但 mass recall 只从 89.03% 升到 89.75%。这解释了为什么低 bit 会降低边界 token 排序，但 PPL 不一定同步崩溃：被错排的许多 token 对 softmax 总质量贡献很小。

## 5. 与公开方法的公平小规模质量对比

### 5.1 160-pair LongBench selector screening

协议：Llama-3.1-8B、16 个英文任务、每任务 10 条、160 个严格 Full/sparse 配对；统一 prompt、scoring harness、active-token schedule 和原始 K/V exact consumer。

Full macro 为 **0.453019**。

| 方法 | Index bit/token/KV-head | 实际 active | 相对 Full 质量 |
|---|---:|---:|---:|
| Full | 0 | 100.000% | 100.000% |
| QKSieve reference | 240 | 7.440% | 99.789% |
| FIER RTN-1 g32 | 256 | 7.440% | 100.010% |
| Quest-P16，整页加载 | 256 | 7.562% | 92.217% |
| RaBitQ RTN-1 公式参考 | 224 | 7.440% | 100.012% |
| SparQ-R32 selector-only | 0 | 7.440% | 99.623% |
| SparQ-R32 完整公式 | 0 | 7.440% | 99.950% |

解释边界：

- FIER 使用审计过的 RTN-1 g32 reference selector，并与 QKSieve 严格共享样本、prompt、active-token 预算和 exact-KV consumer。
- Quest、RaBitQ、SparQ 是本仓库中经过审计的公式/reference quality control。
- 它们不是各论文官方优化 kernel，因此这张表不能用于系统速度排名。
- FIER 的 100.010%、RaBitQ 的 100.012% 与 QKSieve 的 99.789% 都在小样本波动范围内，不能宣称谁显著更好。

### 5.2 FIER matched reference

同一批 160 个严格配对样本上：

| 方法 | Macro score | 相对 Full | Index bit | Active |
|---|---:|---:|---:|---:|
| Full | 0.453019 | 100.000% | 0 | 100.000% |
| QKSieve | 0.452064 | 99.789% | 240 | 7.440% |
| FIER RTN-1 g32 | 0.453066 | 100.010% | 256 | 7.440% |

该质量实验不包含可用于系统结论的时间。FIER 与 QKSieve 的优化 CUDA 路径在独立直接测速中比较。

### 5.3 BinaryPC released projection

80 个严格配对样本、每任务 5 条：

| 方法 | Index bit | Active | 相对 Full 质量 |
|---|---:|---:|---:|
| QKSieve | 240 | 7.413% | 101.358% |
| BinaryPC released | 64 | 7.413% | 100.942% |

样本太少，不支持显著优越性结论。它只说明两者在该筛查集上都没有质量失败。

公开代码版本：

- BinaryPC official commit：`544b6d20977da16fb02493708096b702a6928bfa`
- RaBitQCache official commit：`9eddbbdb5979fd518ef651041a1f9ca5546d46b1`

## 6. 真实独立 CUDA 测速

硬件：单张 RTX 3090。模型结构：32 Query heads、8 KV heads、head dim 128、GQA-4、BF16。所有完整路径均由 CUDA event 直接测量；不通过独立 stage 求和构造。

### 6.1 完整单层 attention call

| 长度 | 目标 token/head | Full ms | QKSieve ms | QKSieve | BinaryPC official |
|---:|---:|---:|---:|---:|---:|
| 8K | 492 | 0.172 | 0.288 | 0.595x | 1.145x |
| 16K | 984 | 0.324 | 0.307 | 1.056x | 1.632x |
| 32K | 1,280 | 0.631 | 0.218 | 2.898x | 2.990x |
| 64K | 1,280 | 1.237 | 0.270 | 4.583x | 5.734x |
| 128K | 1,280 | 2.439 | 0.359 | 6.798x | 11.295x |

BinaryPC 使用发布的 BF16 selector、GQA-shared max、64-bit hash、10% reconstruction-error rescue，以及相同 exact sparse consumer。QKSieve 使用 fused sampled-quantile packed scan/compaction。

### 6.2 独立操作测速

这些操作各自单独调用，只用于定位瓶颈，**不能相加预测完整路径**。

| 独立 CUDA 操作 | 8K ms | 128K ms |
|---|---:|---:|
| QKSieve Query projection/quantization | 0.0450 | 0.0440 |
| QKSieve sampled scan/compaction | 0.0305 | 0.1357 |
| QKSieve exact ragged attention | 0.1851 | 0.1657 |
| QKSieve 完整 attention call | 0.2881 | 0.3588 |
| QKSieve 单 token index append | 0.1004 | 0.0972 |
| QKSieve 历史 index build | 0.6748 | 3.5912 |
| BinaryPC Query probe | 0.0202 | 0.0210 |
| BinaryPC fused hash scan | 0.0175 | 0.0398 |
| BinaryPC error rescue | 0.0194 | 0.0198 |
| BinaryPC top-k | 0.0368 | 0.0606 |
| BinaryPC exact ragged attention | 0.0988 | 0.0987 |
| BinaryPC 完整 attention call | 0.1525 | 0.2156 |

QKSieve 的三个明确速度差距：

1. Query 准备约为 BinaryPC probe 的两倍。
2. 240-bit scan 随长度增长快于 64-bit hash scan。
3. 每 Query head 独立阈值候选使 ragged consumer 比 GQA-shared 固定集合更慢。

### 6.3 整模型直接测速与短上下文边界

| 历史长度 | Steady forward | 32-token decode | Prefill + 32-token request |
|---:|---:|---:|---:|
| 8K | 0.843x | 0.196x | 0.484x |
| 16K | 1.217x | 0.292x | 0.680x |
| 32K | 1.552x | 0.452x | 0.880x |
| 64K | 2.656x | 0.767x | 0.980x |
| 128K | 4.937x | 1.384x | 1.011x |

结论：

- attention crossover 约在 16K。
- 整模型 steady crossover 也约在 16K。
- 包含初始化的 32-token decode 到 128K 才加速。
- 包含 prefill 的短请求到 128K 仍仅约 1.01x。
- LongBench 通常 prompt 更短、答案也短，因此当前部署路径在其请求级测速中变慢是预期现象，必须如实报告。

## 7. 256K 外推缺陷

128K 原生范围内 auto-240 已闭合，但不能据此宣称冻结 template 可无限外推。

在 256K 四窗口诊断中：

- Exact FP16 QK top-1,280：100.24% retention。
- 冻结 128K 坐标 + 冻结 240-bit proxy top-1,280：80.77%。
- 冻结坐标 + local bits：89.91%。
- Local 坐标 + fixed 240-bit allocation：100.44%。
- Local 坐标 + local bits：100.43%。

因此该 stress test 的主因是 **QK-balanced coordinate extrapolation**，不是 exact attention token 预算不足，也不是单纯总 bit 数不足。解决方向应是无监督、请求局部的坐标更新或稳定边界精化，而不是学习任务 router。

## 8. 当前推荐的下一步

优先研究“嵌套低 rate 全扫 + 仅边界高 rate 精化”：

1. 用低位宽前 band 扫描全部历史，得到粗分数和 top-k 阈值附近候选。
2. 只对可能跨越第 `B(N)` 名边界的 token 读取额外 band。
3. 使用数值误差界决定是否停止，而不是学习 router。
4. 候选确定后仍在原始 K/V 上做 exact sparse attention。

目标是同时保留 auto-240 的质量和接近 BinaryPC 64-bit scan 的速度。评价必须报告：

- 额外 band 实际读取比例。
- exact top-k recall 与 attention-mass recall。
- PPL/LongBench 小样本质量。
- 独立 low-rate scan、boundary refine、candidate compact、exact consumer 时间。
- 完整单层 attention 直接时间，禁止用 stage 求和。

## 9. 复现入口

主要代码：

- `src/run_head_top2_targeted_ppl_20260714.py`
- `src/run_sample_calibrated_longbench_20260717.py`
- `src/benchmark_qksieve_deployment_direct_stages_20260801.py`
- `src/benchmark_qksieve_direct_stage_matrix_20260801.py`
- `src/benchmark_binarypc_official_direct_stages_20260801.py`
- `src/summarize_qksieve_bit_profile_robustness_20260801.py`
- `src/summarize_qksieve_direct_decode_length_20260801.py`

主要结果：

- `results/20260801_overnight/native128k_bit_profiles.json`
- `results/20260801_overnight/public_selector_m10_summary.json`
- `results/20260801_overnight/fier_matched_m10_summary.json`
- `results/20260801_overnight/binarypc_matched_m5_summary.json`
- `results/20260801_overnight/qksieve_deployment_bf16_direct_summary.json`
- `results/20260801_overnight/binarypc_official_direct_summary.json`
- `results/20260801_overnight/direct_length_summary.json`
- `results/20260801_overnight/synthetic_length_bit_profiles.json`

远端完整 LongBench：

- `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260728_qksieve_fulltopk_longbench_6gpu`

远端 targeted RULER：

- `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260801_qksieve_targeted_ruler_6gpu`

## 10. 论文表述边界

可以写：

- reference QKSieve 在完整 LongBench 上保持 99.881%。
- auto-240 在原生 128K 六主题诊断上保持 99.883%。
- deployment attention 在 128K 直接测得 6.80x。
- 整模型 32-token decode 在 128K 直接测得 1.38x。
- BinaryPC official selector 在同协议下更快，达到 11.30x。

暂时不能写：

- 99.881% LongBench 与 6.80x 是同一 deployment path 的联合结果。
- QKSieve 在所有短请求上加速。
- QKSieve 显著优于 RaBitQ 或 BinaryPC。
- 冻结 128K template 可直接泛化到 256K/512K。
- 单独 stage 时间相加等于完整路径时间。
