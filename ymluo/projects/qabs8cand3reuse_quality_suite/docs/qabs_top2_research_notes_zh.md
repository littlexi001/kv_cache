# QABS Top-2% Token Discovery 研究笔记

日期：2026-06-29  
本机：Windows + ROCm PyTorch，AMD Radeon RX 7900 XTX  
模型：`Qwen/Qwen3-0.6B`  
评测设置：`prefill_tokens=512`，`eval_tokens=32`，6 个主题文本 + 3 个 needle-in-a-haystack PPL case。  

## 1. 当前核心判断

你的 oracle 现象非常关键：如果在大模型前向时，每层每个 head 只保留真实 attention score 最高的 2% token，PPL 甚至可能比 dense baseline 更好。这个现象把问题重新定义为：

> 不再是“稀疏 attention 会不会损失质量”，而是“如何用足够快的方法找到真实 top-2% token”。

这条主线是有论文潜力的。当前 `qabs8cand3reuse` 的主题 PPL 已经接近 baseline，但速度慢主要来自当前 eager PyTorch/ROCm Windows 实现，不应把这个本机速度当作方法上限。

## 2. 已跑实验摘要

### 2.1 baseline / QABS / SPARQ 对比

来源：`outputs_fast/quality_suite_combined.csv`

| 任务 | 方法 | PPL ratio vs baseline | decode 时间比 | 平均 tok/s |
|---|---:|---:|---:|---:|
| 主题文本 | baseline | 1.000 | 1.00x | 76.4 |
| 主题文本 | `qabs8cand3reuse` | 1.056 | 5.99x | 12.8 |
| 主题文本 | `sparqfast8cand3` | 2.022 | 3.01x | 25.4 |
| Needle PPL | baseline | 1.000 | 1.00x | 78.6 |
| Needle PPL | `qabs8cand3reuse` | 1.287 | 5.80x | 13.6 |
| Needle PPL | `sparqfast8cand3` | 4.547 | 3.40x | 23.3 |

结论：
- `qabs8cand3reuse` 在质量上明显优于当前 SPARQ attention baseline。
- 当前实现速度不行，但这不是理论复杂度问题，主要是没有 fused kernel，并且 Windows ROCm/eager PyTorch 对这种小粒度 decode workload 很吃亏。

### 2.2 Top-2 rerank / shared / 固定刷新实验

来源：`outputs_refresh_sweep/quality_suite_combined.csv`

| 任务 | 方法 | PPL ratio vs baseline | decode 时间比 | 平均 tok/s |
|---|---:|---:|---:|---:|
| 主题文本 | `qabs8cand3reuse` | 1.056 | 6.18x | 12.2 |
| 主题文本 | `qabs8cand3top2globalattn` | 1.072 | 3.31x | 22.6 |
| 主题文本 | `qabs8cand3top2attn` | 1.081 | 1.63x | 46.0 |
| 主题文本 | `qabs8cand3sharedtop2attn` | 1.180 | 1.66x | 45.1 |
| 主题文本 | `qabs8cand3sharedr2attn` | 2.741 | 2.37x | 31.6 |
| 主题文本 | `qabs8cand3sharedr4attn` | 7.704 | 2.84x | 26.9 |
| 主题文本 | `qabs8cand3sharedr8attn` | 16.604 | 2.72x | 27.6 |
| Needle PPL | `qabs8cand3reuse` | 1.287 | 6.71x | 11.7 |
| Needle PPL | `qabs8cand3top2attn` | 1.448 | 1.73x | 44.9 |
| Needle PPL | `qabs8cand3top2globalattn` | 1.451 | 3.60x | 21.7 |
| Needle PPL | `qabs8cand3sharedtop2attn` | 1.545 | 1.86x | 41.8 |
| Needle PPL | `qabs8cand3sharedr2attn` | 1.785 | 2.65x | 29.5 |
| Needle PPL | `qabs8cand3sharedr4attn` | 1.803 | 2.81x | 27.8 |
| Needle PPL | `qabs8cand3sharedr8attn` | 2.026 | 2.63x | 29.6 |

结论：
- `qabs8cand3top2attn` 是当前最值得推进的路线：先用 QABS 找 3% candidate，再在 candidate 内用 exact QK rerank 到 top-2%。
- 直接复用 shared mask 或低频刷新会明显伤 PPL，尤其主题文本上 `sharedr4/sharedr8` 退化很严重。
- `globaltop2` 主题 PPL 略好于 per-head top2，但速度更差；后续可以做“全局预算分配”但需要更好的实现。

### 2.3 自适应刷新实验

来源：`outputs_adaptive_sweep/quality_suite_combined.csv`

| 任务 | 方法 | PPL ratio vs baseline | decode 时间比 | 平均 tok/s |
|---|---:|---:|---:|---:|
| 主题文本 | `qabs8cand3reuse` | 1.056 | 5.78x | 13.2 |
| 主题文本 | `qdriftqabs8cand3` | 1.085 | 4.79x | 15.8 |
| 主题文本 | `qdriftshareqabs8cand3` | 1.223 | 5.29x | 14.4 |
| 主题文本 | `lagrefresh2qabs8cand3` | 1.826 | 4.67x | 16.3 |
| 主题文本 | `lagrefresh4qabs8cand3` | 3.948 | 4.64x | 16.4 |
| 主题文本 | `pconf50qabs8cand3` | 51.893 | 5.47x | 14.0 |
| Needle PPL | `qabs8cand3reuse` | 1.287 | 5.75x | 13.7 |
| Needle PPL | `qdriftqabs8cand3` | 1.370 | 4.64x | 17.0 |
| Needle PPL | `qdriftshareqabs8cand3` | 1.555 | 5.25x | 15.0 |
| Needle PPL | `lagrefresh2qabs8cand3` | 1.548 | 5.45x | 14.7 |
| Needle PPL | `lagrefresh4qabs8cand3` | 1.642 | 4.34x | 18.2 |
| Needle PPL | `pconf50qabs8cand3` | 2.335 | 5.96x | 13.3 |

结论：
- 简单固定刷新不够。刷新间隔越大，candidate recall 掉得越快，PPL 恶化明显。
- `qdriftqabs8cand3` 能省一部分时间，但质量仍弱于 `qabs8cand3reuse`；它可以作为“刷新门控”的起点，而不是最终方法。
- 当前 `pconf50` 不可靠，说明仅用 previous final mask 的注意力分布置信度做刷新触发，会把一些需要召回的新 token 漏掉。

## 3. 最有论文价值的方法方向

### 3.1 主线：QABS-Recall + Exact Top-2 Rerank

建议把方法重新命名为一个清晰 pipeline：

1. **QABS-Recall**：每层每 head 选择 query 绝对值最大的 `d_s=8` 个维度，用低维 partial dot-product 快速召回 top `c=3%` KV token。
2. **Candidate Exact Rerank**：只在 3% candidate 内计算完整 QK score。
3. **Top-2% Sparse Attention**：每层每 head 保留 rerank 后的 top 2% token 做最终 attention。

这个方向最贴近 oracle 目标，也最容易写出理论复杂度：

Dense attention decode 复杂度近似：

```text
O(H * T * d)
```

QABS 两阶段复杂度近似：

```text
O(H * T * d_s) + O(H * cT * d) + O(H * rT * d_v)
```

其中 `d_s=8`，`c=3%`，`r=2%`。理论上，当上下文长度 T 变大时，这条路线有明显加速空间。

### 3.2 不建议把“简单刷新率”作为主创新

当前实验显示，`sharedr2/r4/r8`、`lagrefresh2/4` 的质量退化比较明显。简单说“每 N 步刷新一次”不够稳，论文风险较高。

更好的写法是：

> Refresh is not a fixed interval problem; it is a candidate recall reliability problem.

也就是说，刷新策略应该服务于“保持 oracle top-2% recall”，而不是为了省计算盲目复用 mask。

### 3.3 可发论文的创新版本：Recall-Reliable Adaptive QABS

建议设计一个新方法：**Recall-Reliable Adaptive QABS**。

核心思想：

1. 每步仍然做便宜的 query-side 信号，例如 top query dims、query drift、candidate threshold margin。
2. 不直接决定“刷新/不刷新”，而是决定 candidate budget：
   - 低风险 head：`cand=2%`
   - 中风险 head：`cand=3%`
   - 高风险 head：`cand=5%` 或强制 exact rerank
3. 始终在 candidate 内做 exact top-2 rerank，保证最终稀疏 attention 更接近 oracle。

这比固定刷新更稳，也更符合你最终目标：快速找到真实 top-2% token。

### 3.4 Candidate Recall Oracle 分析必须补上

现在只有 PPL，不够支撑论文。下一步应该直接测：

| 指标 | 含义 |
|---|---|
| `oracle_top2_recall@cand3` | QABS 3% candidate 覆盖真实 top-2% token 的比例 |
| `oracle_mass_recall@cand3` | candidate 覆盖真实 attention mass 的比例 |
| `missed_top2_score_gap` | 漏掉的 oracle top2 token 与保留 token 的 score gap |
| `head_sensitivity` | 哪些 layer/head 对漏召回最敏感 |
| `refresh_hit_rate` | 自适应刷新触发是否真的覆盖高风险 head |

如果实验能证明 `cand3` 对 oracle top2 的 recall 很高，而 PPL 接近 baseline，那么论文论证会很完整。

## 4. 建议的新实验矩阵

### 4.1 质量实验

优先跑以下模式：

```text
baseline
sparqfast8cand3
qabs8cand2top2attn
qabs8cand3top2attn
qabs8cand4top2attn
qabs8cand5top2attn
qabs8cand3top2globalattn
qdriftqabs8cand3
qabs8cand3reuse
```

重点看：
- `cand` 从 2% 到 5% 时，PPL 是否快速接近 baseline。
- `cand3top2attn` 是否能在不同主题和 needle 上稳定优于 SPARQ。
- `globaltop2` 是否只是在小模型上偶然有效，还是普遍有效。

### 4.2 长上下文实验

当前 `512+32` 只能验证功能，论文需要长上下文：

```text
prefill: 1k / 2k / 4k / 8k / 16k
eval: 128 或 256
```

长上下文才是 KV cache 稀疏化的主战场。短上下文下 overhead 会掩盖理论优势。

### 4.3 Needle 实验

当前 needle 只是 PPL，不足够。需要补生成式准确率：

```text
needle position: 0%, 25%, 50%, 75%, 100%
context length: 1k / 2k / 4k / 8k
metric: exact match / contains needle / answer token logprob
```

如果生成式 needle 太慢，先做 answer span 的 teacher-forced logprob，也比整体 PPL 更有针对性。

## 5. Kernel/性能路线

本机 7900 XTX 的当前速度不能代表方法潜力。现在主要问题：

- eager PyTorch 每 token 每层调很多小算子；
- `torch.topk`、dense bool mask、`nonzero`、`gather` 都有高调度开销；
- Windows ROCm 对小粒度 decode kernel 的优化不如成熟 CUDA 路线；
- 当前 sparse path 还没有真正 compact KV index + fused attention。

真正应该实现的 CUDA/HIP kernel 路线：

1. fused query-dim topk；
2. fused partial QABS score；
3. approximate top candidate select，避免 dense mask；
4. candidate 内 exact QK rerank；
5. compact index sparse attention；
6. 输出 logits 前不再 materialize 大 bool mask。

论文里可以先报告：
- 理论 FLOPs / memory traffic；
- PyTorch prototype 的 PPL 和 recall；
- CUDA kernel 后的真实 speedup。

但正式投稿前必须有 NVIDIA CUDA kernel 或 Triton kernel 的速度结果，否则 reviewers 会质疑“理论快但实现慢”。

## 6. 推荐论文叙事

可用标题方向：

```text
Recovering Oracle Sparse Attention: Fast Top-2% Token Discovery for KV Cache Decoding
```

或者：

```text
QABS: Query-Amplitude Guided Token Discovery for Sparse KV Cache Decoding
```

核心贡献可以写成：

1. 发现并系统验证 oracle top-2% attention 在 decode 阶段可以保持甚至改善 PPL。
2. 将 KV cache 稀疏化问题转化为“fast high-recall top token discovery”问题。
3. 提出 QABS 两阶段方法，用 query amplitude 低维召回 candidate，再 exact rerank 到 top-2%。
4. 提出 recall-aware adaptive budget/refresh，使不同 layer/head 动态分配 candidate 预算。
5. 在主题文本、needle retrieval、长上下文上对比 dense baseline、SPARQ 和多种 ablation。

## 7. 下一步落地建议

最短路径：

1. 先实现 `oracle_top2_recall@candidate` 统计，把“QABS 是否能找到真实 top2%”量化出来。
2. 跑 `cand=2/3/4/5` sweep，确认质量-召回曲线。
3. 把 `qabs8cand3top2attn` 作为当前主线，而不是只盯 `qabs8cand3reuse`。
4. 设计 adaptive candidate budget，而不是简单固定刷新。
5. 等质量故事稳定后，再做 fused CUDA/Triton kernel。

当前结论：`qabs8cand3reuse` 证明了 QABS 方向质量可行；但论文主方法更建议推进为 **QABS candidate recall + exact top-2 rerank + recall-aware adaptive budget**。
