# CountCap 与最近工作的边界及公平比较方案

更新时间：2026-07-26

## 1. 结论先行

当前最需要对比的不是只有 AdaKV、SnapKV，而是下面四组工作：

1. **低秩 QK 检索/重构**：Loki、SALS、LRQK、RocketKV、ShadowKV、SVDq。
2. **压缩 Key 自索引**：Self-Indexing KVCache、RaBitQCache、PQCache、Double Sparsity。
3. **动态稀疏预算与跨 head 代理**：Quest、ProxyAttn、Twilight、Double-P、RaBitQCache。
4. **隐藏维物理压缩**：Thin Keys、STAR-KV；它们与 token sparsity 的目标不同，
   但直接限制了“低秩本身”的新颖性主张。

其中最接近当前 CountCap 核心流程的是：

- Loki：PCA 低维 QK 排名，选择 token 后用完整维度计算 attention；
- SALS：RoPE-free latent Q/K 检索，只重构被选中的 KV；
- LRQK：联合低秩 Q/K 代理分数、top-k、recent、精确 KV；
- Self-Indexing KVCache：同一低比特压缩 Key 同时承担存储与 top-k 检索；
- RaBitQCache：低比特 Key 索引、INT4 query、近似 QK、候选内 full-precision KV、top-p；
- RocketKV：永久 token 驱逐后，再做利用 head/sequence 降维的动态 top-k；
- ProxyAttn：用少量代表 attention heads 代理其余 heads，并分配动态预算；
- SVDq：按 Key 奇异谱构造 latent channels，再进行混合精度量化，并与稀疏方法结合。

因此，论文不能再声称：

- 首次发现 Key 低秩；
- 首次使用 PCA/SVD 近似 QK 做 sparse attention；
- 首次使用量化 Key 索引检索 full-precision KV；
- 首次让压缩 Key 同时承担检索索引；
- 首次根据 attention mass 分析稀疏输出误差。

当前仍可建立的差异是：

> 在线、逐序列、逐 KV-head 的首个 2048-token chunk sampled uncentered PCA48，
> 加上 grouped log-scale INT4 Key/INT8 query 索引；该基底随后固定，索引直接输出
> 每个 query head 的候选，不做全维精确重排；使用 $256/6\%/1280$ 长度封顶的
> 目标预算；所有层都执行稀疏 attention，不依赖前两层 Full；并提供从奇异谱、
> prefix-basis 失配、代理误差、遗漏 attention mass、attention 输出到 logits/PPL
> 的逐项误差账本。

这里的 `1280` 是 sampled-quantile 的目标数量，不是当前 kernel 的绝对 hard cap。
为了吸收 256 点分位数估计误差，执行路径预留

$$
C(N)=\min\left(
N,\,
\max\left\{
2fN,\,
(f+0.04)N
\right\}
\right)
$$

的候选容量，其中 $f=B(N)/N$。实际参与 attention 的数量由超过采样阈值的
token 数决定，通常接近目标，但最坏可达到该容量。正式结果必须同时报告目标
预算、实际消费均值、p95 和最大值，不能把目标比例直接写成严格执行比例。
LongBench m4 审计中，Llama/Qwen 的实际消费为 7.263%/7.262%，目标为
7.116%/7.063%；overflow head 仅 0.0218%/0.0316%。

固定 256 点分位数采样在极低比例下是当前明确限制。样本中期望位于目标尾部的
点数为 $mf$：32K/4%、64K/2%、128K/1% 分别只有
10.24/5.12/2.56。理想 i.i.d. 近似下，相对候选数标准差为

$$
\sqrt{\frac{1-f}{(m+2)f}},
$$

对应 30.5%/43.6%/61.9%。64K 实测单 head 范围已达约 116--3851；
128K 已完成窗口达到约 15--6410。这个问题与 PCA/SVD 或 INT4 精度不同，
可能造成少数 head 预算饥饿，必须在长 PPL 中单独报告。

该组合是否足以形成 ICLR 级创新，最终取决于三件事：

1. 最终 Direct LongBench 的跨模型质量是否稳定；
2. 128K 真实 decode 速度是否在同等质量和预算下优于强近邻；
3. 理论与实验证据是否能解释为什么有偏的 data-adaptive 谱索引在更低预算下仍然可靠。

还必须明确方法目标：冻结 CountCap 仍让完整 FP16 K/V 常驻 GPU，PCA48-INT4
只是附加检索索引。因此它当前证明的是 **decode-time KV read/attention compute
reduction**，不是物理 KV 存储压缩。与 Self-Indexing、SALS、STAR-KV 比较时，
表格必须把 `attention consumption ratio`、`index overhead` 和
`physical KV storage ratio` 分成三列，不能把前者写成 KV cache ratio。

---

## 2. 方法级对照

| 方法 | 代理表示 | 选择策略 | 候选后计算 | 是否训练/校准 | 关键系统特点 |
|---|---|---|---|---|---|
| SnapKV | prompt 尾部精确 attention | fixed top-k eviction | 保留后的 KV | 无训练 | 直接删除未保留 KV |
| AdaKV | SnapKV 类分数 | head-wise 自适应分配 | 保留后的 KV | 无训练 | 跨 head 分预算 |
| Quest | page min/max Key | fixed top-k page | page 内精确 attention | 无训练 | page 粒度 |
| Double Sparsity | 静态重要通道 | fixed top-k token | 精确稀疏 attention | 需要离线校准 | token + channel sparsity |
| Loki | PCA 低维 Key | fixed top-k token | 完整维度 attention | PCA 来自校准数据 | 保留全部 PCA channels |
| SALS | RoPE-free latent Q/K | latent-space top-k | 仅重构候选 KV | 每序列低秩变换 | 压缩物理 KV、避免全量重构 |
| LRQK | 联合低秩 Q/K | fixed top-k + recent | 精确 KV attention | 每序列分解 | GPU/CPU hit-miss |
| RocketKV | head/sequence 双重降维代理 | 永久驱逐 + 动态 top-k | 稀疏 attention | 无训练 | 两阶段物理压缩 |
| ShadowKV | pre-RoPE 低秩 Key + landmarks | chunk/token selection | 恢复 Key、拉取 Value | 每序列分解 | CPU Value offload |
| SVDq | centered SVD latent channels | 与 ShadowKV sparsity 组合 | 重建近似 Key | 每序列 SVD | 谱感知混合位宽 |
| PQCache | product quantization | MIPS top-k | full KV attention | PQ codebook | CPU/GPU overlap |
| Self-Indexing KVCache | 1-bit sign VQ + 低比特 K/V | LUT-GEMV top-k + 64 sink | fused dequant sparse attention | 无任务训练 | 同一压缩格式兼作索引 |
| RaBitQCache | 随机旋转 1-bit Key + INT4 Q | adaptive top-p + local | full-precision KV | 无任务训练 | 无偏估计、异步索引、lazy update |
| ProxyAttn | 代表 heads 的精确/压缩分数 | multi-head 动态预算 | 稀疏 attention | 无训练 | 跨 head 共享路由信号 |
| Thin Keys | 权重 SVD 后的低维 Key | dense attention | 低维 QK + full Value | 需要 QK 微调 | 物理缩小 Key cache |
| STAR-KV | head/block 自适应低秩 K/V | dense/low-rank attention | 低秩 K/V | 可微阈值训练 | 自适应 rank + 混合精度 |
| CountCap | 首 2K sampled uncentered PCA48 + INT4 K + INT8 Q | $256/6\%/1280$ per-head target | 原始 FP16 Q/K/V | 无训练、逐序列首段建基底 | direct no-rerank、fused scan、预分配 KV cache |

### 与 Loki 的关键区别

相同点：

- 都利用 Key 的低秩性；
- 都在 PCA/SVD 子空间计算近似 QK；
- 都按 query 选择 token；
- 都在选中 token 上使用完整维度 attention。

区别：

- Loki 的 PCA 变换来自离线 calibration，并保存全部 PCA-transformed Key；
  CountCap 根据当前请求首个 2048-token prefill chunk 建立
  per-layer/per-KV-head 基底，随后固定。
- Loki 的主要压缩来自低维计算；CountCap 进一步把 PCA48 Key 索引压成 grouped INT4，query 压成 INT8。
- CountCap 的候选直接进入 attention，无额外 exact-QK rerank，并使用长度封顶的目标预算与预分配 KV cache。

这意味着 CountCap 不能把“PCA 低维选 top-k”作为主要新意，必须证明 **online spectral indexing + low-bit direct retrieval + bounded cap + system realization** 的整体优势。

Loki 还报告了一个与本方法直接相关的现象：在 Llama-3 和 Mistral 上，离线校准的
post-RoPE PCA 明显弱于 pre-RoPE PCA。其解释是 post-RoPE 子空间会绑定校准数据的
位置分布。CountCap 使用当前请求自身的 post-RoPE Key 建立子空间，因此一个更有
价值的假设是：

> prefix-conditioned online PCA 不是普通离线 PCA 的实现细节，而是在不修改
> RoPE 数学的前提下，用当前请求首段降低跨请求、跨域子空间失配。

使用与测试样本隔离的四个 LongBench calibration prompt，已经在 Llama 与
Qwen 上完成 16 任务、每模型 320 个严格 online/fixed 配对。两者使用相同的
Direct、INT4/INT8、sampled-quantile 和预算后端，只改变 basis 来源：

| 模型 | Online macro | Fixed macro | Fixed-Online 95% CI | Fixed/Online | Prediction agreement | 固定基索引构建加速 | 固定基整样本加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.45824 | 0.45498 | [-0.00861, +0.00245] | 99.29% | 60.00% | 15.38x | 1.144x |
| Qwen3-4B | 0.41801 | 0.41957 | [-0.00346, +0.00717] | 100.37% | 41.25% | 15.39x | 1.166x |

两个模型的配对 bootstrap 区间均跨过 0，结果没有支持“请求内 online basis 在
Macro 上稳定优于 fixed basis”，也没有证明二者严格等价。固定基显著减少了索引
构建时间，在两个模型上的整体质量接近 online；但逐预测一致率很低，
而且 calibration prompt 仍来自 LongBench。因而：

1. `sequence-conditioned basis` 不能作为已经被实验证明的核心优势；
2. fixed basis 可以作为短上下文系统优化候选；
3. 在独立语料、未见模型和跨域设置上验证之前，不能把 fixed basis 写成无需
   校准的通用替代；
4. 冻结主方法仍使用请求内首 2K basis，本消融不修改主方法。

### 与 SVDq 的关键区别

SVDq 将 centered Key 投影到完整 SVD latent space，对不同 latent channel 使用混合位宽，解码时重建近似 Key；它主要解决 Key cache 本身的压缩精度。

CountCap 使用 uncentered sampled PCA48 作为检索索引，但最终 attention 读取原始 FP16 Key/Value。PCA/INT4 误差只影响候选选择，不继续进入候选内 softmax 和 Value 聚合。因此 CountCap 的理论目标是 omitted-mass stability，而不是完整 Key 重构误差。

### 与 LRQK 的关键区别

LRQK 已经明确把完整 $QK^\mathsf T$ 的低秩性作为核心观察，并在 prefill 中联合
分解 Query 和 Key；所以“QK 奇异值快速衰减”也不能单独作为 CountCap 的创新点。

CountCap 当前实现更轻：只用当前请求首个 2048-token chunk 的 sampled Key
二阶矩建立 48 维正交基，不联合优化 Q/K，不在 decode 中迭代更新，也不需要
recent buffer 才维持质量。代价是这个 Key-only prefix basis 不保证等于完整 QK
的最优 rank-48 子空间。

QK 谱实验回答的不是“QK 是否低秩”这一已知问题，而是：

1. 首段 Key-PCA48 对完整 QK 的 fidelity 是否接近 full-Key PCA48；
2. full-Key PCA48 与 QK 最优 rank-48 的 normalized regret 有多大；
3. 这种接近是否能在 Llama/Qwen、体育/医学和不同层上保持。

32K 双模型结果已经给出否定但有价值的答案。softmax 有效的中心化 QK 最优
rank-48 保能率为 99.09%/98.52%，证明低秩结构很强；full-history Key-PCA48
为 91.10%/91.45%，与最优 QK-SVD 相差约 7--8 个百分点；真实 first-2K
basis 进一步降到 62.39%/59.07%。因此不能把 CountCap 定位为“几乎无损复现
联合 Q/K 分解”的方法，也不能把 sequence-conditioned basis 的高谱 fidelity
作为主贡献。

更准确的定位是：CountCap 接受一个明显有偏、但非常便宜的 Key-only prefix
代理；它不追求 QK 矩阵重构，而是依靠低比特全历史扫描保留高 attention-mass
候选，再用原始 FP16 Q/K/V 做直接稀疏 attention。production 4% 候选的集合
recall 只有 47.34%，但 mass-weighted recall 为 92.38%，最终 NLL 只变化
0.0068--0.0110 nat/token。与 LRQK 的关键区别应落在 **优化目标从矩阵重构转为
attention-mass 保留，以及更轻的 direct 系统路径**，而不是声称低秩近似更精确。

### 与 RaBitQCache 的关键区别

相同点：

- 都保留低比特 Key 索引；
- 都对 query 做低比特处理；
- 都扫描近似 QK；
- 都拉取候选 full-precision KV 做最终 attention；
- 都有索引构建、融合 kernel 和系统流水。

区别：

- RaBitQCache 使用随机正交旋转和 1-bit Key，代理分数是有理论界的无偏估计；CountCap 使用数据自适应的 dominant spectral subspace，代理有偏，并依赖自然 Q/K 的谱集中。
- RaBitQCache 使用 top-p；CountCap 当前使用无学习的长度封顶目标预算和 sampled-quantile 阈值。
- RaBitQCache 的正式比较中前两层使用 Full attention；CountCap 最终方法所有层稀疏，无 Full 回退。
- RaBitQCache 在 LongBench 的平均实际预算为 17.33%；CountCap 在本次
  LongBench 混合长度 m4 审计的平均实际消费为 7.263%/7.262%，32K/128K
  的解析目标分别为 4%/1%。不同 prompt 长度分布下必须同时报告目标和实际
  消费，不能直接把这些百分比跨协议比较。

这是投稿中最重要的强对手。

更严格地说，“无偏”与“低误差”不是同一件事。RaBitQCache 的无偏性与
$O(1/\sqrt D)$ 高概率误差界依赖其随机旋转、中心化和高维球面假设；论文自身也
指出实际 Q/K 可能呈聚簇分布。CountCap 对固定 Q/K 是确定性的有偏投影，误差为

$$
-\frac{q^\mathsf T(I-P)k}{\sqrt d},
$$

但当自然 query 与 Key dominant subspace 对齐时，该 bias 可以很小。公平比较
不能只写“无偏/有偏”，而应在相同预算下报告 proxy score MSE、top-k recall、
attention-mass recall、实际消费和端到端速度。

### 与 Self-Indexing KVCache 的关键区别

这是当前最直接的压缩 Key 检索对手之一。AAAI 2026 的 Self-Indexing KVCache
先对 Key 做通道均值中心化，再按 4 维 group 建立 1-bit sign VQ codebook；
decode 时用 LUT-GEMV 直接在压缩 Key 上估计相似度并取 top-k。相同的压缩格式
还用于 K/V 低比特存储，候选通过融合 dequantization 与 sparse attention
消费。它在 LongBench 使用固定 160-token 预算，其中 64 个是始终保留的
full-precision sink，只有 96 个动态选择。

其 Llama-3.1-8B 公开 11-task LongBench 结果为：

| 方法 | Full | 稀疏 | 相对 Full | 预算 |
|---|---:|---:|---:|---:|
| Self-Indexing 16-bit retrieval | 58.7 | 58.4 | 99.49% | 160 token |
| Self-Indexing 2-bit K/V + 1-bit index | 58.7 | 58.2 | 99.15% | 160 token |

其 32K RULER 报告 Full 90.8、16-bit 89.4、2-bit 89.2，动态消费比例为 7.5%；
16K、batch 10 的模块表中 sparse attention 为 0.116 ms，Full
FlashAttention2 为 0.776 ms，约 6.69x；整模型 decode throughput 最高约 2x。
这些数字来自论文环境，不能直接与 CountCap 的单请求 RTX 3090/Hugging Face
测速相除。

两者共同点是“压缩 Key 代理直接选择候选，不做 full-QK 重排”。主要区别是：

1. Self-Indexing 的压缩表示同时承担物理 K/V 存储；CountCap 的 INT4 PCA
   只是索引，完整 FP16 K/V 仍在 GPU。
2. Self-Indexing 使用全历史一遍构建的 sign codebook、64 sink 和固定 top-k；
   CountCap 使用首 2K 谱基、无 sink/recent 特判和长度封顶 sampled threshold。
3. Self-Indexing 对 Value 也量化；CountCap 候选内始终读取原始 FP16 Value。
4. CountCap 的理论优势不能写成“首次自索引”，而应落在有偏谱代理的
   attention-mass 误差账本、无重排全层稀疏和长上下文数值/系统实现。

正式严格 LongBench 完成后，汇总器会额外计算同一个 11-task 子集。公平表只比较
各自相对 Full 的保持率，并同时列出动态 token、固定 sink、物理存储比例和硬件。

### 与 SALS、RocketKV 和 ProxyAttn 的边界

- **SALS（NeurIPS 2025）**：在 RoPE-free latent 空间检索并只重构候选，
  公开报告 4K attention operator 5.7x、端到端吞吐 4K 1.4x/32K 4.5x。
  它解决 latent KV 的物理压缩与 RoPE 重构；CountCap 不重构候选 Key，但保留
  完整 FP16 K/V。
- **RocketKV（ICML 2025）**：先永久驱逐 prompt token，再用 head 与 sequence
  两个维度的低维代理做动态 top-k，公开报告最高 400x compression、A100
  端到端 3.7x。CountCap 不做永久驱逐，因而多步 query 可重新选取任何历史
  token，但没有同等级物理显存收益。
- **ProxyAttn（ICLR 2026）**：通过代表 heads 估计其他 heads，并进行 multi-head
  动态预算。它的主要节省来自跨 head 共享；CountCap 仍为每个 query head
  独立扫描同一 KV-head 索引，跨 head 复用不是当前贡献。

这三篇工作意味着“低秩、两阶段、动态预算、跨 head 代理”都不能单独作为新意。
CountCap 必须用相同预算和相同硬件的实测证明其 direct low-bit spectral scan
在质量/速度 Pareto 上有优势。

---

## 3. 已有同环境基线

模型：Llama-3.1-8B-Instruct；LongBench 16 个英文任务；每任务前 100 条，共 1600 条；最大原文上下文 7500 tokens。

| 方法 | Macro | Full-relative | KV 预算 | 平均运行时间 |
|---|---:|---:|---:|---:|
| Full Attention | 36.58 | 100.00% | 100% | 4.780 s |
| SnapKV | 36.44 | 99.62% | 1024 token/head | 4.523 s |
| AdaKV | 36.36 | 99.39% | 约 1024 token/head | 9.294 s |
| H2O | 33.48 | 91.53% | 1024 token/head | 7.591 s |

这组结果已经完成严格样本配对，可以与当前正在运行的 Llama final-direct m100 结果做同样本质量比较。绝对时间跨 runner 不完全可比，因此主表应分开报告质量、attention kernel、整模型 decode 和完整 protocol。

---

## 4. RaBitQCache 已发表结果

RaBitQCache 已被 ICML 2026 接收。其论文在 Llama-3.1-8B-Instruct、LongBench 13 任务上报告：

| 方法 | LongBench | 平均预算/比例 |
|---|---:|---:|
| Full | 50.58 | 100% |
| Oracle top-1024 | 50.31 | 11.4% |
| RaBitQCache, top-p=0.95 | 50.63 | 17.33% |
| Quest-1024 | 46.52 | 11.38% |
| Double Sparsity-1024 | 50.28 | 11.42% |
| SparQ | 50.15 | 25% |

论文还报告在 30K context 下最高 3.88x decode speedup，但使用 Hopper GPU、vLLM、FlashInfer 和自定义 kernel，不能与当前 RTX 3090/Hugging Face 数字直接做绝对比较。

协议差异：

- RaBitQCache 报告 13 个 LongBench 任务；
- 它在前两层使用 Full attention；
- CountCap 当前正式运行 16 任务，完成后可以额外取同样 13 任务的子集；
- 两篇工作的实际 prompt、stop token 和评分包仍需逐项核对，因此论文数字只能列为 published reference，不能伪装成同框架复现。

---

## 5. 当前正式公平实验

运行目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  results/20260726_final_direct_multimodel_m100_prompt7500/
```

设置：

- GPU 0--1：Llama-3.1-8B-Instruct；
- GPU 2--3：Qwen3-4B-Instruct；
- 16 个 LongBench 英文任务；
- 每任务 100 条，每模型 1600 个严格 Full/CountCap 配对；
- 每模型 3200 行，共 6400 行；
- 完整 prompt（包含模板、上下文和问题）不超过 7500 tokens；
- Full 与 final Direct CountCap；
- final Direct 使用 PCA48、INT4 K、INT8 Q、无 exact rerank、无 Full 回退；
- frozen target budget：

$$
B(N)=\min\left(N,1280,\max\left(256,\left\lceil0.06N\right\rceil\right)\right).
$$

完成后必须同时报告：

1. 16-task Macro；
2. 与 RaBitQCache 相同的 13-task Macro；
3. 与 Self-Indexing KVCache 相同的 11-task Macro；
4. 每任务 Full/CountCap；
5. 平均实际 attention token 数与比例；
6. query、decode、online 和 total 时间；
7. attention 消费比例、索引开销与物理 KV 存储比例分栏；
8. Llama 与 Qwen 的质量保持率；
9. 短文本下没有速度收益这一负面结果。

该实验已通过两模型 smoke test：两模型均走 final Direct 路径，约 7.5K prompt
使用 452--454 个 attention tokens/head，没有 Full 回退。它当前排在
64K/128K 冻结方法配对测速之后，使用 GPU 0--3。

---

### 5.1 已完成的 body-cap preliminary 结果

旧目录 `20260726_final_direct_multimodel_m100_ctx7500` 只把 LongBench 原始
context body 截到 7500；加上 chat template 和 question 后，完整 prompt
最高达到 Llama 9493、Qwen 10415。因此它只能作为 preliminary：

| 模型 | Full Macro | CountCap Macro | 保持率 | Macro 差值 95% CI | online speed |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.44770 | 0.44943 | 100.39% | [-0.00169, +0.00520] | 0.902x |
| Qwen3-4B | 0.41388 | 0.41314 | 99.82% | [-0.00423, +0.00271] | 0.777x |

该结果支持短 LongBench 上的质量，但同时确认短文本没有速度优势。正式主表必须
使用上面的 full-prompt-cap 版本，不得把两个协议的绝对分数混在一起。

---

## 6. 已完成的数学诊断

32K、四主题、每模型 3072 个 token-level Full/CountCap 配对已经完成，记录：

- full-to-sparse KL；
- Jensen--Shannon divergence；
- top-1 agreement；
- Full top-1 margin；
- shift-invariant logit perturbation range；
- 满足严格 margin 充分条件的比例；
- margin flip rate；
- target NLL delta。

该实验补齐：

$$
\text{attention output perturbation}
\Longrightarrow
\text{final logits and NLL stability}.
$$

| 模型 | top-1 agreement | margin certificate | KL | 平均 NLL 差 |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 94.69% | 40.27% | 0.01554 | +0.01098 |
| Qwen3-4B | 91.73% | 28.00% | 0.03201 | +0.00682 |

KL/NLL 的确定性范围界通过率均为 100%。它不是 router，也不改变冻结方法。

---

## 7. 投稿时的正确定位

更稳妥的故事不是“又一个 PCA top-k 方法”，而是：

> 已有低秩方法证明 Key 可压缩，但没有回答一个有偏、低比特、在线构建的谱索引在不做精确重排时，为什么仍能可靠地直接驱动 token-level sparse attention。CountCap 把该问题拆成谱截断、采样子空间、量化、候选遗漏 mass、attention 输出和最终 logits 六个可测误差阶段，并通过长度封顶目标、融合扫描和预分配 KV cache 把长上下文检索成本转化为整模型 decode 加速。

要让这个定位成立，最终论文至少需要：

- 对 Loki、LRQK、RaBitQCache、Quest、SnapKV/AdaKV 的明确比较；
- 对 Self-Indexing、SALS、RocketKV、ProxyAttn 的最近邻边界；
- 跨 Llama/Qwen 的 final-direct LongBench；
- 64K/128K RULER；
- 32K/64K/128K 的统一真实 decode speed；
- 全部层稀疏与前两层 Full 的 ablation；
- PCA48 FP、PCA48 INT4、随机投影/随机旋转、Loki-style calibration PCA 的 matched-budget ablation。

## 主要资料

- Loki: <https://arxiv.org/abs/2406.02542>
- LRQK: <https://arxiv.org/abs/2510.23649>
- ShadowKV: <https://arxiv.org/abs/2410.21465>
- SVDq: <https://arxiv.org/abs/2502.15304>
- PQCache: <https://arxiv.org/abs/2407.12820>
- RaBitQCache: <https://arxiv.org/abs/2606.31519>
- Quest: <https://arxiv.org/abs/2406.10774>
- Double Sparsity: <https://arxiv.org/abs/2408.07092>
- Twilight: <https://arxiv.org/abs/2502.02770>
- Double-P: <https://arxiv.org/abs/2602.05191>
- AdaKV: <https://arxiv.org/abs/2407.11550>
- SALS: <https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html>
- RocketKV: <https://proceedings.mlr.press/v267/behnam25a.html>
- Self-Indexing KVCache: <https://ojs.aaai.org/index.php/AAAI/article/view/39988>
- ProxyAttn: <https://openreview.net/forum?id=m3HXHQYmZu>
- Thin Keys: <https://arxiv.org/abs/2603.04427>
- STAR-KV: <https://arxiv.org/abs/2606.08382>
