# Direct CountCap：三模型独立验证与长上下文边界

更新时间：2026-07-26

## 1. 冻结设置

所有模型使用同一无训练配置：first-2K sampled uncentered PCA48、grouped log-scale INT4 Key、INT8 Query、256 点 sampled-quantile、候选内原始 FP16 Q/K/V direct sparse attention；不使用 exact-QK 重排、任务 router、recent 特判、前层 Full 或 Full fallback。

$$
B(N)=\min\left(N,1280,\max\left(256,\left\lceil0.06N\right\rceil\right)\right).
$$

LongBench 使用 16 个英文任务、每任务 100 条；完整 prompt（含模板、上下文和问题）不超过 7500 tokens。每个样本均为 Full/CountCap 严格配对。

## 2. LongBench 16 任务

| 模型 | 配对样本 | Full Macro | CountCap Macro | 保持率 | Macro 差值 95% CI | 目标 token/head | 目标比例 | online/token speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 1600 | 0.44558 | 0.44609 | 100.11% | [-0.00228, +0.00321] | 382.5 | 7.40% | 0.896x |
| Qwen3-4B-Instruct | 1600 | 0.40988 | 0.41051 | 100.15% | [-0.00288, +0.00439] | 383.8 | 7.35% | 0.777x |
| Qwen2.5-7B-Instruct | 1600 | 0.44007 | 0.43385 | 98.59% | [-0.01023, -0.00264] | 383.8 | 7.35% | 0.881x |

短 LongBench 的 `online/token speed` 包含检索和生成，但索引固定开销在短 prompt/短输出上不能摊销；质量表和长上下文速度表必须分开解释。

### 分任务

| 任务 | Llama Full/CountCap | Qwen3 Full/CountCap | Qwen2.5 Full/CountCap |
|---|---:|---:|---:|
| 2wikimqa | 0.4581/0.4614 | 0.3962/0.4040 | 0.4628/0.4677 |
| gov_report | 0.3320/0.3387 | 0.3020/0.3064 | 0.3451/0.3372 |
| hotpotqa | 0.4904/0.4904 | 0.4358/0.4472 | 0.5076/0.5034 |
| lcc | 0.6305/0.6264 | 0.6472/0.6424 | 0.6381/0.6354 |
| multi_news | 0.2681/0.2675 | 0.2405/0.2407 | 0.2565/0.2601 |
| multifieldqa_en | 0.5730/0.5737 | 0.4952/0.4881 | 0.5237/0.5077 |
| musique | 0.2567/0.2644 | 0.1321/0.1365 | 0.1890/0.1835 |
| narrativeqa | 0.2277/0.2284 | 0.1781/0.1833 | 0.2184/0.2105 |
| passage_count | 0.0740/0.0758 | 0.0067/0.0000 | 0.0500/0.0500 |
| passage_retrieval_en | 0.6200/0.6200 | 0.6000/0.6000 | 0.6000/0.5800 |
| qasper | 0.4533/0.4265 | 0.4210/0.4126 | 0.4001/0.3926 |
| qmsum | 0.2277/0.2323 | 0.2129/0.2082 | 0.2318/0.2253 |
| repobench-p | 0.5045/0.5147 | 0.5057/0.5169 | 0.5861/0.5675 |
| samsum | 0.4457/0.4493 | 0.4443/0.4414 | 0.4616/0.4613 |
| trec | 0.6700/0.6700 | 0.6800/0.6800 | 0.6900/0.6900 |
| triviaqa | 0.8977/0.8977 | 0.8605/0.8605 | 0.8801/0.8693 |

## 3. 64K/128K 连续文本 PPL 与速度

| 模型 | 长度 | cases | PPL 保持率 | 实际 token/head | per-head 范围 | Decode speed | $\Delta T_{fixed}$ | break-even token | 256-token protocol speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 64000 | 4 | 97.55% | 1560.8 | 86-3851 | 2.643x | -0.028s | 0 | 1.316x |
| llama31_8b | 128000 | 4 | 93.26% | 1500.4 | 13-6411 | 4.590x | -0.014s | 0 | 1.240x |
| qwen3_4b | 64000 | 4 | 91.37% | 1467.2 | 116-3851 | 2.772x | +0.591s | 6 | 1.395x |
| qwen3_4b | 128000 | 4 | 86.50% | 1440.0 | 15-6410 | 4.118x | +0.374s | 2 | 1.261x |

Decode speed 包含 PCA/INT4 scan、sampled threshold、候选 gather 和精确稀疏 attention；物理 FP16 K/V 仍完整常驻 GPU。最差连续文本 PPL 保持率为 86.50%（qwen3_4b，128K）。因此不能把 LongBench 质量外推成通用 PPL 无损，也不能把 attention 消费比例写成物理 KV 存储比例。

## 4. Qwen2.5 的 centered QK 谱外推

| 模型 | cases | Key 有效秩 | centered QK 有效秩 | 最优 rank-48 fidelity | Full Key-PCA48 fidelity | First-2K PCA48 fidelity | First-2K score cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 40 | 21.60 | 6.29 | 98.80% | 91.67% | 60.37% | 0.7864 |

该诊断只验证谱结构能否跨到第三个模型，不把 query-aware QK-SVD 上界当成可部署方法，也不替代 LongBench/PPL 结果。

## 5. 与公开强基线的边界

### Self-Indexing 11-task 子集的协议级对照

| 来源/模型 | Full | 稀疏方法 | 稀疏分数 | 保持率 | attention 预算 | 物理 KV |
|---|---:|---|---:|---:|---:|---:|
| 本次 runner/Llama-3.1-8B-Instruct | 53.25 | CountCap | 53.20 | 99.91% | 375.2 token/head | 100% FP16 |
| 本次 runner/Qwen3-4B-Instruct | 50.51 | CountCap | 50.60 | 100.18% | 376.9 token/head | 100% FP16 |
| 本次 runner/Qwen2.5-7B-Instruct | 53.32 | CountCap | 52.51 | 98.48% | 376.9 token/head | 100% FP16 |
| Self-Indexing 论文/Llama-3.1-8B | 58.70 | 16-bit | 58.40 | 99.49% | 160 token | 压缩 K/V |
| Self-Indexing 论文/Llama-3.1-8B | 58.70 | 2-bit | 58.20 | 99.15% | 160 token | 2-bit K/V |
| Self-Indexing 论文/Qwen2.5-14B | 56.90 | 16-bit | 55.90 | 98.24% | 160 token | 压缩 K/V |
| Self-Indexing 论文/Qwen2.5-14B | 56.90 | 2-bit | 55.70 | 97.89% | 160 token | 2-bit K/V |

任务集合相同，但模型尺寸、prompt、stop policy、实现和硬件不同；因此只比较各自相对 Full 的保持率与资源目标，不比较绝对分数排名。

- AdaKV 的 Llama-3.1-8B Table 5 是 question-aware、16 任务、固定 B=128--2048；本实验是长度预算且完整 prompt cap 不同。
- RaBitQCache 在 Llama-3.1-8B LongBench 13 任务公开报告 Full 50.58、RaBitQ 50.63、平均预算 17.33%，且前两层 Full；CountCap 应只比较同 13 任务相对保持率与预算量级。
- AAAI 2026 Self-Indexing KVCache 在压缩 Key 上直接检索，LongBench 使用 11-task、160-token 预算，其中 64 个固定 sink；这是 CountCap 必须正面对比的最近邻。
- NeurIPS 2025 SALS 使用 RoPE-free latent Q/K 选择后仅重构候选；ICML 2025 RocketKV 结合永久驱逐与低维 top-k；ICLR 2026 ProxyAttn 使用代表 head 和动态预算。它们分别覆盖低秩、两阶段和跨 head 路线。
- Loki 已使用离线 PCA 低维 top-k 和候选内完整维度 attention；LRQK 已使用联合 Q/K 低秩检索。STAR-KV 与 Thin Keys 还直接压缩隐藏维度。因此 PCA/SVD 低秩本身不是新颖点。

公开论文数字来自不同硬件、框架、prompt 和 stop policy，不作为同环境排名。当前最可信的结论来自本报告内部的严格 Full 配对。

## 6. 结论

三模型结果用于检验同一冻结配置能否跨架构迁移；64K/128K PPL 则用于暴露 LongBench 之外的失败模式。投稿时应同时保留正结果与负结果：CountCap 是低预算、无 Full fallback 的 question-aware sparse attention 系统，但不是任意连续文本上的无条件等价替换。

理论推导见 `docs/20260726_countcap_spectral_stability_mathematical_appendix_zh.md`；复现协议见 `docs/20260726_direct_countcap_frozen_method_reproduction_zh.md`。
