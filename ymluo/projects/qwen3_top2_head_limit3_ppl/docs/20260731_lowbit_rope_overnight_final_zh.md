# 低比特 QKSieve 与 RoPE 解耦检索：最终实验报告

## 1. 这轮工作回答了什么

这轮工作围绕三个问题展开：

1. 更低 bit 的浮点量化是否优于整数；
2. 在不训练 router 的前提下，能否进一步改善质量和速度；
3. 能否利用 pre-RoPE 语义相关性，使稀疏 attention 在受控长链检索上超过 Full Attention。

当前结论是：

- 同码率 minifloat 没有转化为真实 PPL 优势，不继续开发；
- 整数 `[4,1]` 是比 `[4,2]` 更轻的实用点，索引仅占完整 FP16 K/V 的 2.734%；
- 低比特 post-RoPE 过取、早期层 pre-RoPE 语义重排、最终精确 post-RoPE attention 的组合，在普通 PPL 上接近 Full；
- 该组合在 128K、8 个非重复主题上达到 101.44% Full 和 3.98x 稳态加速；
- 在冻结的受控两跳检索分布上，32K 和 64K 可以显著超过 Full，64K 的 24 个独立种子同时改善 Gold NLL 和正确率；
- 该增益不能直接外推为 LongBench 或任意自然文本均达到 110% Full；
- 简单的 pre/post 等权混合在 128K 失败，已经否定；
- 128K 的 2% 预算归因和相位缓存测速见第 7、8 节。

## 2. 低比特浮点与整数

QKSieve 将每个 128 维 Key 划分为 8 个 16 维 band。每个激活 band
保存量化 payload 和一个 FP16 scale：

```text
bits_per_token_head
  = sum_b(16 * bit_b + 16 * I[bit_b > 0])

index_ratio
  = bits_per_token_head / (2 * 128 * 16)
```

分母是同一 KV head 的完整 FP16 K 和 V。

Qwen3-4B、32K、8 个主题、256 个严格配对预测 token：

| 格式 | PPL | 相对 Full | Top-1 一致率 | 索引/完整 K+V | 稳态加速 |
|---|---:|---:|---:|---:|---:|
| Full FP16 | 5.4560 | 100.000% | 100.00% | 0% | 1.000x |
| 整数 `[4,2]` | 5.4564 | 99.992% | 97.66% | 3.125% | 1.515x |
| minifloat `[4,2]` | 5.4674 | 99.791% | 97.66% | 3.125% | 仅质量模拟 |
| 整数 `[4,1]` | 5.4576 | 99.970% | 97.27% | 2.734% | 1.509x |
| 整数 `[2,1,1]` | 5.6305 | 96.900% | 91.41% | 2.734% | 1.477x |
| 整数 `[2,2]` | 5.6573 | 96.441% | 92.19% | 2.344% | 1.479x |
| 整数 `[4]` | 5.4806 | 99.551% | 97.27% | 1.953% | 1.522x |
| minifloat `[4]` | 5.4988 | 99.222% | 97.27% | 1.953% | 仅质量模拟 |
| 整数 `[2]` | 5.8874 | 92.672% | 90.63% | 1.172% | 1.520x |

核心发现：

- minifloat128 / int128 的质量比为 99.799%；
- minifloat80 / int80 的质量比为 99.670%；
- `[2]` 已跨过明显质量断崖；
- `[4,1]` 与 `[2,1,1]` 物理码率相同，但质量相差 3.07 个百分点，
  证明高 QK 能量 band 的位宽位置比总 bit 数本身更重要；
- bit 数继续下降几乎不再提高速度，瓶颈已经转移到 top-k、最终稀疏 attention 和模型非 attention 部分；
- bit 应优先给第一个高 QK 能量 band，不能平均摊给更多低能量坐标。

因此不开发 minifloat 专用 kernel，也不把 48-bit 整数作为主配置。

## 3. 当前 RoPE 解耦方法

### 3.1 动机

标准 attention 分数为：

```text
s_post(i) = <RoPE(q, t), RoPE(k_i, i)> / sqrt(d)
```

最新 RoPE 实验表明，长度只变化 64 token 时：

- Query cosine 仍可保持在约 0.87--0.94；
- 部分关键 head 的证据 QK 却下降 4--9；
- 证据 attention 可从 11.44% 降到 0.48%；
- 早期残差差异会在后层 attention 和 MLP 中继续放大。

因此，post-RoPE 分数适合最终位置感知 attention，但不一定是最稳定的
远程语义证据定位分数。

### 3.2 可部署流程

当前实用分支为：

```text
QK-balanced [4,1] low-bit post-RoPE scan
-> overfetch 2x remote candidates
-> exact pre-RoPE rerank on layers 0--8 only
-> keep 16 sink + 128 recent tokens
-> reduce to final budget
-> exact post-RoPE K/V attention
```

关键点：

1. 全历史只保存一份 `[4,1]` 低比特 QKSieve 索引；
2. 不构建第二份完整 pre-RoPE 索引；
3. pre-RoPE 只计算在约 2 倍候选池上；
4. 最终 softmax 和 Value 聚合仍使用原始 post-RoPE K/V；
5. 没有训练 router，也没有按任务标签选择动作。

现有预算：

```text
B_old(N) = min(N, max(256, min(ceil(0.06 * N), 1280)))
```

物理索引：

```text
16 * 4 + 16 scale + 16 * 1 + 16 scale = 112 bit
112 / 4096 = 2.734375% of full FP16 K/V
```

注意：2.734% 是辅助检索索引比例。当前速度实现仍让精确 K/V 常驻 GPU，
但每步最终 attention 只消费被选中的 token。

## 4. 为什么只改早期层

冻结种子上的分层干预结果：

| 长度 | L0--8 | L9--17 | L18--26 | L27--35 |
|---:|---:|---:|---:|---:|
| 32K 相对 Full | 155.31% | 166.11% | 142.70% | 136.64% |
| 64K 相对 Full | 147.29% | 87.54% | 103.87% | 120.93% |

中层 L9--17 在 64K 出现明显负迁移。所有层都改写时，不同层的增益和损失
会互相抵消。因此当前只保留跨 32K/64K 均为正的 L0--8。

这仍是需要扩大验证的经验性层范围，不能声称对所有模型架构都固定成立。

## 5. 普通文本 PPL

Qwen3-4B、8 个主题、每个主题 64 个 token：

| 长度 | 方法 | 相对 Full PPL 质量 | ms/token | 稳态加速 |
|---:|---|---:|---:|---:|
| 32K | `[4,1]` 基础 QKSieve | 100.563% | 61.111 | 1.467x |
| 32K | L0--8 pre-RoPE 重排 | 99.916% | 63.611 | 1.410x |
| 64K | `[4,1]` 基础 QKSieve | 100.157% | 61.244 | 2.657x |
| 64K | L0--8 pre-RoPE 重排 | 99.678% | 63.164 | 2.574x |

64K L0--8 的主题 bootstrap 95% 区间为
98.926%--100.411%。因此普通文本结论应写为“接近 Full”，不能写成显著提高。

这些数字来自相位表缓存优化之前。缓存后的复测见第 8 节。

## 6. 受控两跳检索能否超过 Full

### 6.1 32K

8 个冻结种子：

| 方法 | 相对 Full Gold-NLL 质量 | 95% CI | 改善种子 |
|---|---:|---:|---:|
| L0--8 pre-RoPE 重排 | 155.31% | 117.16%--212.23% | 7/8 |

区间下界高于 110%，但正确率均为 25%。因此这里证明的是 Gold token
概率显著改善，不是准确率提高。

### 6.2 64K

24 个独立种子：

| 指标 | Full | L0--8 pre-RoPE 重排 |
|---|---:|---:|
| Gold 几何 PPL | 5.9011 | 3.9515 |
| 相对 Full Gold-NLL 质量 | 100% | 149.34% |
| 95% CI | - | 117.92%--189.00% |
| 正确率 | 50.0% | 62.5% |
| 改善种子 | - | 23/24 |

这是当前最强的“超过 Full”机制结果：区间下界高于 110%，且正确率也提高。

合理解释是：

1. pre-RoPE 排名补回被相对位置相位削弱的语义证据；
2. 稀疏集合删除大量 softmax distractor；
3. 最终仍使用 post-RoPE 精确 attention，保留位置结构。

### 6.3 不能外推的部分

该 benchmark 是冻结模板的合成两跳检索，不是 LongBench、RULER 或自然文本
语言建模。论文中可以把它写为因果机制验证，不能写成通用任务质量为
149.34% Full。

## 7. 128K 归因

### 7.1 固定 1280 token

8 个种子：

| 方法 | Token 比例 | 相对 Full | 改善种子 |
|---|---:|---:|---:|
| `[4,1]` 基础 QKSieve | 0.977% | 81.99% | 2/8 |
| L0--8 纯 pre 重排 | 0.977% | 103.73% | 2/8 |
| L0--8 pre/post 等权质量混合 | 0.977% | 80.96% | 2/8 |

pre/post 等权混合被否定。它在 32K/64K 有正点估计，但没有解决 128K
尾部风险。

### 7.2 保持 2% 的 2560-token 归因

同一组 8 个合成种子的严格配对结果：

| 方法 | Token 比例 | 相对 Full | 改善种子 | 正确率 |
|---|---:|---:|---:|---:|
| Exact post-RoPE oracle，1280 | 0.977% | 96.88% | 4/8 | 12.5% |
| Exact post-RoPE oracle，2560 | 1.953% | 92.98% | 4/8 | 12.5% |
| `[4,1]` post-RoPE proxy，2560 | 1.953% | 96.50% | 3/8 | 12.5% |
| pre/post 等权质量混合，2560 | 1.953% | 91.64% | 3/8 | 12.5% |
| L0--8 纯 pre-RoPE 重排，2560 | 1.953% | 124.67% | 4/8 | 25.0% |

最后一行的均值较高，但 95% 区间为 64.38%--289.10%，没有统计稳定性。
因此，增加到 2% 能把正确率从 12.5% 提高到 25.0%，却不能证明 128K
合成检索已被稳健解决。更大的 post-RoPE 集合也不必然更好，因为它会重新
引入 softmax distractor。

自然文本的质量优先预算候选仍可写为：

```text
B_new(N) = min(2560, max(B_old(N), ceil(0.02 * N)))
```

但不能把它冻结为所有任务的通用规则；128K RoPE 尾部风险仍是未解决限制。

已有 128K 自然文本四窗口结果显示，标准 QKSieve 的 2560-token 版本为
100.01% Full，online decode 为 3.83x；1280-token 版本为 98.99% Full、
3.39x。该结果支持 2% 预算，但它不是本节的合成 RoPE 归因。

### 7.3 当前组合方法的 128K 普通文本

本轮另外直接测试了当前 `[4,1] + 2x overfetch + L0--8 pre-RoPE rerank`
的 2560-token 配置，而不是继续借用旧 KeyMSE 路径的数字。

设置：

- Qwen3-4B-Instruct；
- 每个请求使用两张 RTX 3090，让完整精确 K/V 常驻 GPU；
- 8 个非重复主题，每个主题 64 个严格配对 token，共 512 token；
- 单类 baseball 训练集不足 128K，因此用 baseball+hockey 的
  `sports_both` 替代循环文本；循环版 sports 不计入汇总。

| 指标 | Full | 当前方法 |
|---|---:|---:|
| 几何 PPL | 11.8553 | 11.6875 |
| 相对 Full 质量 | 100% | **101.436%** |
| 主题 bootstrap 95% CI | - | **100.305%--102.924%** |
| Token bootstrap 95% CI | - | 99.644%--103.368% |
| Top-1 一致率 | 100% | 96.09% |
| Full-to-sparse KL | 0 | 0.01343 |
| 稳态 ms/token | 304.034 | **76.460** |
| 稳态加速 | 1.000x | **3.976x** |
| Attention token/head | 131072 | 2560，1.953% |
| 辅助索引/完整 FP16 K+V | 0% | 2.734% |

分主题质量：

| 主题 | 相对 Full |
|---|---:|
| computer | 99.95% |
| medicine | 101.22% |
| mixed_a | 100.18% |
| mixed_b | 100.74% |
| politics | 99.76% |
| religion | 103.10% |
| space | 100.74% |
| sports_both | 105.96% |

首次编译 CUDA extension 的两个并发 worker 各产生约 116 秒的一次性假开销，
不能计入部署固定成本。其余 6 个预编译 worker 的固定成本均值为 3.300 秒，
8 个主题固定成本的稳健中位数为 3.356 秒。按中位数计算，128K 相对 Full
约生成 15 个 token 即可跨过 break-even；64-token 在线速度约为 2.36x。

该结果证明当前实际组合在 128K 普通文本上没有质量断崖，并且平均 PPL
略优于 Full。但只有 8 个主题，不能把 101.44% 外推为 LongBench、
开放生成或所有模型均显著优于 Full。

## 8. 相位缓存与速度

### 8.1 已验证的 CUDA 数值等价性

Qwen3-8B、64K、实际 YaRN 配置：

| 指标 | 结果 |
|---|---:|
| 最大绝对误差 | 9.54e-6 |
| 平均绝对误差 | 1.11e-6 |
| Top-k overlap | 100% |
| 融合 pre-RoPE candidate kernel | 0.0704 ms |
| 物化逆旋转参考 | 1.1338 ms |
| kernel 加速 | 16.10x |

### 8.2 相位表缓存

旧实现每个激活层都会对同一张 configured YaRN cosine/sine 表做切片、
幅值归一化和 contiguous copy。64K 的旧阶段剖析中，9 个早期层累计
pre-RoPE candidate score 为约 7.15 ms/token，而真正的融合候选 kernel
远小于该值。

新实现按：

```text
(model_config identity, device, head_dim, rounded capacity)
```

缓存单位幅值、split-half 的相位对表，只在首次使用时归一化。

缓存前后的严格复测：

| 长度 | 相对 Full | 缓存前 ms/token | 缓存后 ms/token | 缓存后加速 |
|---:|---:|---:|---:|---:|
| 32K | 99.919% | 63.611 | 62.922 | 1.426x |
| 64K | 99.687% | 63.164 | 61.834 | 2.627x |

逐阶段剖析中，9 个早期层累计的 pre-RoPE candidate score：

| 长度 | 缓存前 | 缓存后 | 降低 |
|---:|---:|---:|---:|
| 32K | 4.198 ms/token | 2.355 ms/token | 43.9% |
| 64K | 7.151 ms/token | 3.807 ms/token | 46.8% |

子阶段约减半，但整模型稳态只降低 1.1%--2.1%。这说明缓存实现正确且有用，
但剩余瓶颈已经转移到 Key append、全历史 top-k、最终 sparse attention
以及非 attention 模型底座。

当前测得的固定成本分别为 3.197 秒和 3.639 秒。按

```text
G_break = T_fixed / (T_full - T_sparse)
```

计算，32K 和 64K 相对 Full 的 break-even 约为 119 和 36 个生成 token。
固定成本跨运行有波动，因此这些值用于系统规划，不作为精确硬件常数。

### 8.3 sampled-quantile 替代 top-k

又测试了用 256 点 sampled-quantile compaction 替代早期层
`torch.topk`。结论是否定的：

- 2x 过取在 32K/64K/128K 分别出现 0.54%、4.20%、1.90% 的
  underfill row，最坏目标召回只有 87.9%、40.8%、54.2%；
- 保守 3x 过取在 32K 的选择算子为 1.59x，但 64K 为 1.00x，
  128K 反而只有 0.81x；
- 4x 过取虽基本消除漏召回，64K 仍无速度收益，128K 仅为 0.80x。

根因是较大的 ragged candidate workspace 初始化和写出抵消了省下的
通用 top-k。该分支不接入主方法。

### 8.4 128K 阶段瓶颈

2560-token 方法在 128K 的 16-token 阶段剖析中，增强路径为
84.03 ms/token；剖析事件本身会同步 CUDA，因此正文速度仍采用上一节
512-token 汇总的 76.46 ms/token。

增强路径的 36 层累计阶段：

| 阶段 | ms/token |
|---|---:|
| Key index append | 5.689 |
| Query prepare | 1.472 |
| 低比特 proxy scan | 2.718 |
| proxy top-k | 5.885 |
| 9 个早期层 pre-RoPE candidate score | 7.013 |
| candidate 内 top-k | 0.371 |
| 最终 sparse attention | 5.023 |

已验证的单层 128K、4832 候选到 2560 token 的融合 pre-RoPE 流水线为
0.362 ms；相对物化逆旋转参考的核心候选打分为 13.46x，Top-k overlap
为 100%。因此接下来最大的系统收益不在继续降低索引 bit，而在：

1. 把 Key index append 融入 K projection epilogue；
2. 用专用 selection backend 替代通用 top-k，但必须避免 sampled
   compaction 的大 workspace；
3. 用 CUDA Graph 或 vLLM/FlashInfer backend 降低未归因的模型底座和
   launch/control-flow 时间。

### 8.5 LongBench 小规模严格配对结果

随后用 Qwen3-4B 在 8 个 LongBench 任务上做了方向性 probe：每任务 6 条，
共 48 个样本、144 条方法结果；prompt 上限为 32K，实际 prefix 为
1.7K--32K、均值 12.25K，生成上限为 128 token。基础方法和 pre-RoPE
方法严格共享 112-bit `[4,1]` 索引与相同 attention token 预算。

| 方法 | Macro | 相对 Full | 相对基础方法 |
|---|---:|---:|---:|
| Full KV | 0.44871 | 100.000% | 99.676% |
| 基础 `[4,1]` post-RoPE QKSieve | 0.45017 | 100.325% | 100.000% |
| 加 L0--8 pre-RoPE 重排 | 0.44486 | 99.143% | **98.821%** |

pre-RoPE / 基础方法的 task-bootstrap 95% 区间为
96.345%--100.178%，逐样本 win/tie/loss 为 4/36/8。它仅在 GovReport
上有正点估计，在 NarrativeQA 和 QMSum 分别下降到基础方法的 91.42% 和
94.71%。按长度分桶也没有观察到增益，超过 24K 的 3 条样本为 91.61%，
但该桶样本太少，不能做强统计结论。

该 probe 支持将 pre-RoPE 从默认主方法中移除，仅保留为受控 RoPE
失真机制实验。由于这是 8 任务、每任务 6 条的小样本结果，不能替代完整
LongBench。当前 Hugging Face 质量 harness 在平均 12K prompt 下的稀疏
路径仍慢于 Full SDPA；pre-RoPE 相对基础稀疏路径的 decode 时间仅增加
约 0.09%，因此这里主要用于质量决策，不作为最终系统速度数字。

## 9. 当前应冻结什么

论文主线建议分成两层：

1. 通用主方法：QK-balanced `[4,1]` 低比特索引、精确 post-RoPE sparse
   attention；
2. 机制扩展：2x overfetch + 早期层 pre-RoPE semantic rerank。

在 LongBench、RULER 和多模型独立验证补齐前，不应把第二层默认开启为
所有任务的通用配置。

明确否定：

- minifloat 替代整数；
- 48-bit `[2]` 主配置；
- 完整 pre-RoPE 索引替代 QKSieve；
- pre/post 等权质量混合；
- 仅凭 8 个合成种子宣称通用 110% Full。

## 10. 复现入口

主要实现：

```text
src/run_head_top2_targeted_ppl_20260714.py
src/run_direct_countcap_denseprompt_ppl_20260725.py
src/run_qksieve_coldskip_longcontext_quality_20260730.py
src/qabs_cuda_kernels.py
```

汇总和验证：

```text
src/summarize_qksieve_lowbit_ppl_20260731.py
src/summarize_qksieve_rope_retrieval_20260731.py
src/validate_prerope_candidate_cuda_20260731.py
```

启动脚本：

```text
scripts/launch_qksieve_qwen8_rope_retrieval_8gpu_20260731.sh
scripts/launch_qksieve_qwen8_rope_retrieval_2gpu_pairs_20260731.sh
scripts/launch_qksieve_dualmass_ppl_8gpu_20260731.sh
scripts/launch_qksieve_prerope_128k_ppl_2gpu_pairs_20260731.sh
scripts/run_qksieve_prerope_stage_profile_20260731.sh
```

核心远端结果：

```text
results/20260731_qksieve_qwen8_i112_l00to08_64k_expanded/
results/20260731_qksieve_qwen8_i112_l00to08_128k_2gpu_pairs/
results/20260731_qksieve_qwen8_i112_dualmass_l00to08_128k_2gpu_pairs/
results/20260731_qksieve_qwen8_128k_budget2pct_attribution_2gpu_pairs/
results/20260731_qksieve_64k_i112_rope_l00to08_ppl_8gpu/
results/20260731_prerope_candidate_cuda_yarn_qwen8_64k_v2/
results/20260731_qksieve_i112_l00to08_phasecache_ppl_8gpu/
results/20260731_qksieve_i112_l00to08_k2560_128k_ppl_2gpu_pairs/
results/20260731_qksieve_i112_l00to08_k2560_stage_profile_128k/
results/20260731_sampled_overfetch_selection/
```

本地最终小型归档：

```text
artifacts/20260731_lowbit_rope_overnight_final/
```
