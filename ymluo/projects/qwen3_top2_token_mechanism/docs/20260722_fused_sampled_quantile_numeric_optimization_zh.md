# 融合采样分位数 KV 检索：方法与实验结果

更新时间：2026-07-22

## 1. 问题

已有实验确认：长上下文 decode 时，每个 query head 最终只对原始 KV 中精确 top-2% token 做 attention，PPL 可以接近甚至优于 Full KV。真正的效率问题不是最终 2% attention，而是如何快速找到它们。

上一版 QK-Metric48 的流程为：

```text
扫描 128K 个 PCA48 + INT4 近似分数
-> 对全部近似分数做全局 top-6%
-> 读取候选的原始 K，精确重排到 top-2%
-> 用原始 V 做稀疏 attention
```

其中第一次全局 `top-k` 需要物化每个 head 的 128K 个 FP32 近似分数，成为长上下文下的主要检索开销。

## 2. 机制启发

参考文档 `why_long_context_hurts_needle_retrieval_softmax_rope_20260720.pdf` 的结论是：

1. 上下文越长，softmax 分母中的无关 token 越多，相关 token 的概率质量会被稀释；
2. RoPE 会进一步改变远距离 QK logit；
3. 因此 top-2% 稀疏 attention 不只是压缩，还会去掉大量 softmax 噪声。

这意味着主方法必须保留“原始 K 精确重排到 2%”这一质量锚点，但用于产生宽候选池的近似全局 `top-k` 可以被更便宜的统计选择替代。

## 3. 最终方法

### 3.1 QK-Metric48 低秩空间

令 query/key 二阶矩为 `Cq` 和 `Ck`，直接最小化低秩近似后的 QK score 均方误差：

```text
minimize_A  E[(q^T k - q^T A k)^2],  rank(A) <= r
```

对白化后的矩阵做截断 SVD：

```text
M = Cq^(1/2) Ck^(1/2)
M ~= U_r Sigma_r V_r^T

L = Cq^(-1/2) U_r Sigma_r^(1/2)
R = Ck^(-1/2) V_r Sigma_r^(1/2)

q^T k ~= (q^T L)(k^T R)^T
```

当前高效点使用 `r=48`。K 投影使用 logscale16 INT4 存储，query 投影在线量化为 INT8。前 4 个 query 用闭式统计估计 `Cq`，之后一次性重建 QK-Metric 索引；没有 router、任务标签、答案或训练模块。

### 3.2 融合采样分位阈值

对每个 query head：

1. 在 128K 近似分数域中等距抽取 256 个位置；
2. 在 shared memory 中求样本 top-6% 分位阈值；
3. 扫描 INT4 索引时直接比较阈值，并把超过阈值的 token 压紧到候选数组；
4. 候选数组设置 12% 置信容量，避免常规 head 溢出；
5. 对实际候选读取原始 FP16 K，计算精确 QK；
6. 精确选择最终 top-2%，再对原始 V 做稀疏 attention。

```text
256-point sampled quantile
          |
QK-Metric48 INT4 scan + threshold + compact
          |  mean ~= 6.1%, transient
exact FP16 QK rerank
          |  exactly 2.0%
sparse softmax + original V
```

候选生成不再产生 `[head, 128K]` FP32 分数张量，也不再执行第一次全局 `top-k`。最终精确 top-2% 使用 `sorted=False`；attention 对 token 排列不敏感，因此选中集合和数学结果不变。

### 3.3 CountCap：按证据数封顶，而不是固定百分比

参考文档指出长上下文会不断增加 softmax 分母中的无关极值，但真正有用的证据数量未必随长度线性增长。128K 支持集扫描进一步验证：最终比例从 2% 降到 1% 后，四主题 PPL 几乎不变；Medicine 在 0.5% 时也仍优于 Full。于是最终支持集改为长度感知的固定数量上限：

```text
final_count(N) = min(round(0.02 * N), 1280)
final_fraction(N) = final_count(N) / N
candidate_fraction(N) = clamp(4 * final_fraction(N), 0.03, 0.06)
```

对应 32K/64K/128K/256K 的最终比例为 2%/2%/1%/0.5%，候选比例为 6%/6%/4%/3%。128K、180K、192K 和 256K 均已做严格 Full/CountCap 配对；32K/64K 保持原 2% 预算，因此不会改变此前长度实验。

## 4. 三种比例不能混用

| 比例 | 当前值 | 含义 |
|---|---:|---|
| 辅助检索索引 / Full FP16 K+V | 约 5.596% | PCA48 INT4 K 索引、scale、投影状态 |
| 瞬时候选 token | 约 6.1% | 每步送入原始 K 精确重排的平均比例 |
| 最终 attention links | 约 2.0% | 真正参与 softmax 和 V 聚合的 token 比例 |

12% 是临时候选数组的容量上限，不是平均计算比例，也不是常驻 KV 比例。

当前严格速度 harness 为了隔离检索与 attention 的计算收益，仍把完整原始 K/V backing store 保留在 GPU，供 6% exact K 重排和 2% V 聚合随机读取。因此 `5.596%` 是辅助索引大小，`2%` 是 attention 计算比例，不能写成总 GPU KV 占用只有 5.596% 或 2%。若要同时实现物理显存压缩，还需要把原始 K/V 量化或分层放置，并单独统计候选搬运开销。

## 5. 速度结果

### 5.1 候选生成

32 query heads、8 KV heads、128K、RTX 3090：

| 候选生成方式 | 每层每 token | 相对速度 |
|---|---:|---:|
| INT4 扫描 + 全局 top-6% | 0.4008 ms | 1.00x |
| 采样阈值 + 融合扫描压紧 | 0.2457 ms | **1.63x** |

### 5.2 Attention 流水线与 Full 基线口径修正

这里使用真正的 Qwen3-4B 128K layer-16 trace。旧微基准在计时前已经把 GQA 的 8 个 KV heads 展开成 32 个 query heads，只测预展开后的 SDPA 数学核；它漏掉了 HuggingFace Full decode 每层、每 token 执行的 `repeat_kv`、K/V contiguous 物化和输出 transpose。修正后分别报告两个口径：

| 口径 | Full | CountCap 4% -> 1% | 加速 |
|---|---:|---:|---:|
| K/V 已预展开的纯 SDPA kernel | 3.336 ms | 0.808 ms | **4.129x** |
| 真实 HF attention 接口 | 10.926 ms | 0.808 ms | **13.525x** |

第二次独立重复的 HF 接口结果为 `11.091 / 0.793 = 13.994x`，因此保守报告 **13.5x**。真实 HF 路径的额外成本来自：

```text
8 KV heads
-> repeat/reshape 成 32 heads
-> contiguous 物化完整 128K K 和 V
-> SDPA
-> output transpose + contiguous
```

CountCap 直接在未展开的 8-head K/V 上做检索和稀疏聚合，因此不仅减少 SDPA links，也消除了 Full 路径的大规模 GQA 展开。旧报告中的 `3.069x` 只是历史预展开 kernel 口径，不再作为当前 attention 子系统主结果。

冻结方法的 CUDA profiler 显示：INT4 阈值扫描约 0.218 ms、候选精确 QK 约 0.126 ms、最终 top-k 约 0.061 ms、V attention 约 0.057 ms。下一阶段的主要优化对象应是原始 K 的随机读取、精排和算子间 host gap，而不是再缩短 V attention。

另外实现了与标量 INT4 点积逐项等价的 DP4A 扫描。128K 下候选数、阈值、候选集合和候选分数完全一致，候选生成由 0.2561 ms 降至 0.1366 ms，CUDA 自耗时由约 0.704 ms 降至 0.583 ms。但完整 attention 流水线仅从 3.124x 变为 3.129x，8K 整模型在线时间还从 41.284 s 轻微变为 41.679 s。它暴露了当前 Python/CUDA 算子间隙，但没有形成可兑现的 wall-clock 收益，因此不替换冻结主路径。

### 5.3 整模型在线速度

协议：Qwen3-4B-Instruct、2x RTX 3090、128K history、512 query warm-up、512 PPL token，共 1023 次同步逐 token model forward。在线时间不含 prefill，但包含 QK-Metric 校准、一次索引重建、36 层检索、attention 和模型其余计算。

| 顺序 | Full PPL | Ours PPL | Full online | Ours online | 加速 |
|---|---:|---:|---:|---:|---:|
| Full -> Ours | 9.8551 | **9.7527** | 366.578 s | 107.228 s | **3.419x** |
| Ours -> Full | 9.8551 | **9.7541** | 366.777 s | 107.084 s | **3.425x** |

两种运行顺序结果基本重合。平均在线加速约 `3.422x`；Ours PPL 比 Full 低约 `1.0%`，不是用质量换速度。

### 5.4 长度拐点

| 长度 | Full PPL | Ours PPL | Full online | Ours online | 加速 |
|---:|---:|---:|---:|---:|---:|
| 32K | 17.1928 | **16.7450** | 104.844 s | 96.062 s | **1.091x** |
| 64K | 8.4242 | **8.3450** | 191.722 s | 98.835 s | **1.940x** |
| 128K | 9.8551 | **9.7534** | 366.678 s | 107.156 s | **3.422x** |

32K 已出现正加速，但收益较小；64K 接近 2x，128K 超过 3.4x。Ours 每步时间只从 32K 的约 93.9 ms 增长到 128K 的约 104.8 ms，而 Full 从 102.5 ms 增长到 358.3 ms。

### 5.5 128K 跨主题严格配对

| 主题 | Full PPL | Ours PPL | PPL变化 | Full online | Ours online | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Medicine | 9.8551 | **9.7527** | **-1.04%** | 366.578 s | 107.228 s | **3.419x** |
| Politics | 10.2315 | **10.0737** | **-1.54%** | 366.791 s | 107.254 s | **3.420x** |
| Computer | 6.6885 | **6.6189** | **-1.04%** | 367.383 s | 107.451 s | **3.419x** |
| Space | **24.9416** | 25.0043 | +0.25% | 367.012 s | 107.647 s | **3.409x** |

四主题几何 PPL 质量保持率为 `100.852%`，汇总在线加速为 `3.417x`，平均候选比例为 `6.109%`。Space 有轻微回退，但远小于 5% 质量容忍线；结果说明主结论不是 Medicine 特例。

### 5.6 CountCap：128K、四主题、每主题 512 token

先固定 6% 候选集，只改变最终支持比例。Medicine 的 128-token 前沿如下：

| 最终 links | PPL | online |
|---:|---:|---:|
| 0.5% | 10.2561 | 28.598 s |
| 1.0% | 10.2257 | 28.559 s |
| 1.5% | **10.1800** | 28.767 s |
| 2.0% | 10.1803 | 28.913 s |

1.5% 到 2% 已完全饱和；1% 与 2% 的四主题几何质量保持率分别为 `101.201%` 和 `101.135%`。因此 128K 使用 1% 最终 links，并把候选比例同步降到 4%。严格 512-token 结果为：

| 主题 | Full PPL | CountCap PPL | PPL变化 | Full online | CountCap online | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Medicine | 9.8551 | **9.7353** | **-1.22%** | 366.578 s | 104.528 s | **3.507x** |
| Politics | 10.2315 | **10.0241** | **-2.03%** | 366.791 s | 104.539 s | **3.509x** |
| Computer | 6.6885 | **6.6037** | **-1.27%** | 367.383 s | 104.447 s | **3.517x** |
| Space | **24.9416** | 25.2124 | +1.09% | 367.012 s | 104.646 s | **3.507x** |

汇总结果：

- 四主题几何 PPL：Full `11.3884`，CountCap `11.2901`；质量保持率 `100.870%`；
- 整模型在线加速 `3.510x`；最终 attention links `1.000%`；平均候选比例 `4.176%`；
- Medicine 重复运行得到 PPL `9.7150`、online `104.480 s`，说明结果稳定；
- Space 把候选比例从 4% 恢复到 6% 仅把 PPL 从 `25.2124` 改善到 `25.1576`，在线时间反而增至 `106.174 s`，因此不值得回退。

用保守的 `13.525x` HF attention 接口加速和 `3.510x` 整体加速做 Amdahl 反推：Full decode 中约 `77.2%` 属于 HF attention 路径，约 `22.8%` 是其余模型计算。对应每 token 约为：

```text
Full:      276.7 ms attention + 81.6 ms other = 358.3 ms
CountCap:   20.5 ms attention + 81.6 ms other = 102.1 ms
```

因此整模型 `3.510x` 与 attention 接口约 `13.5x` 是一致的。此前用 32K sparse 的 90-94 ms 总时间充当 128K 公共非-attention底座是不成立的，不能据此推出 20x 以上。

### 5.7 180K-256K 长度外推验证

所有点均使用不重复的真实主题文本。Medicine、Space 和 Computer 的原始主题流不足 256K，因此 256K 只在具有 288K 可用 token 的 Politics 上测试；没有通过循环文本补齐。180K/192K 使用两卡，256K 因 Full KV 两卡 OOM 改用四卡，但每个点内部的 Full 与 CountCap 硬件完全一致。

| 长度/主题 | Full PPL | CountCap PPL | 质量保持率 | 候选比例 | 最终 links | 在线加速 |
|---|---:|---:|---:|---:|---:|---:|
| 180K Medicine | 16.3794 | 16.4515 | 99.562% | 3.121% | 0.711% | **3.510x** |
| 192K Religion | 18.8425 | **18.6395** | 101.089% | 2.974% | 0.667% | **3.742x** |
| 192K Space | 12.8989 | **12.5844** | 102.499% | 3.083% | 0.667% | **3.748x** |
| 256K Politics | 13.9245 | 14.0182 | 99.331% | 3.052% | 0.500% | **3.607x** |

四个异构长度点的几何 PPL 质量保持率为 `100.612%`，配对在线汇总加速为 `3.650x`。prefill 汇总加速仅为 `0.9999x`，说明当前收益来自在线 decode attention；不能声称 prefill 已被加速。

256K 固定最终 0.5% 后的候选比例消融为：

| 候选目标 | 实际候选 | 质量保持率 | 在线加速 |
|---:|---:|---:|---:|
| 3% | 3.052% | 99.331% | **3.607x** |
| 4% | 4.205% | 99.086% | 3.578x |
| 6% | 6.138% | **99.699%** | 3.524x |

质量不随候选比例单调变化，因为更接近 Full attention 不保证更低 PPL；但三点都在 1% 质量损失内。默认保留 3% 作为速度点，6% 作为更保守的质量配置。该消融也说明主要发现是 1280-token 最终支持上限，而不是某个唯一最优候选比例。

## 6. 本轮被否定的方向

| 方向 | 结果与结论 |
|---|---|
| block/microblock 层次筛选 | 真实高分 token 空间分布过于集中，局部 quota 丢失全局极值；不进入主方法 |
| 谱级联与 INT4 bit-plane | 多一次筛选和 launch 的成本超过少读 bit 的收益；实测更慢 |
| 被丢弃 V 的矩补偿 | attention 输出 L2 可改善，但 PPL变差；重新引入了 PDF 所揭示的 softmax 尾部噪声 |
| q1024 + 8% 容量 | 阈值更稳定，但采样成本抵消更小容量，速度与 q256 接近、PPL略差 |
| q256 + 10% 容量 | 约 0.3% 速度收益，但 PPL 约差 0.2%；不值得替换 12% 安全点 |
| 第二个采样阈值代替最终 exact top-k | 8K PPL 恶化到 187.66；最终 2% 边界必须精确 |
| 近似 top-2%/4% 直接作为支持集 | 8K PPL 分别约 619/642；离线 mass 召回高，但在线闭环分布漂移，必须保留精排 |
| 复用 INT4 分数二次筛到 3%/4% | Qwen 128K 的 4% attention-mass 召回仍为 99.953%，但 top-k 版和二次采样版流水线仅 2.59x/2.75x，均慢于 3.11x 主路径 |
| 48维 INT4 + 尾部80维 INT2 级联 | 6% -> 4% 后 mass 召回为 99.9941%，几乎等于直接 6% 的 99.9943%；但新增索引和二次筛选，尚无正向实测速度，不进入主方法 |
| 修正后的 DP4A | 候选和分数逐项等价，扫描阶段 1.875x；但完整流水线和整模型 wall-clock 无稳定收益，只保留为 kernel 消融 |
| 扫描时直接融合 exact QK | 扫描与精排需要相反的并行粒度，候选归约也发生偏差；1.276 ms，比分离式 0.263 ms 慢约 4.8x，已删除在线入口 |

## 7. 当前结论

当前冻结方案是：

```text
QK-Metric48
+ logscale16 INT4 K index
+ 256-point sampled quantile
+ CountCap length-aware support budget
+ fused threshold/compaction (128K candidate ~= 4%)
+ original-K exact rerank (128K final ~= 1%)
+ unsorted sparse V attention
```

它是一条纯数值、无训练、无任务标签的方法。在 128K 严格复现实验中已经同时达到：

- 辅助检索索引约为 Full FP16 K+V 的 5.596%；
- 1.0% 最终 attention links，4.176% 候选精排；
- 4.129x 预展开纯 SDPA kernel 加速，13.525x 真实 HF attention 接口加速；
- 128K 四主题汇总 3.510x 整模型在线加速；
- 四主题几何 PPL 质量保持率 100.870%。
- 180K-256K 四个长度/主题点汇总 3.650x，质量保持率 100.612%。

这轮形成了两个相互独立的数值结论：第一，把“物化全分数 + 全局 top-k”改写为“统计阈值 + 扫描时压紧”；第二，最终支持集应随上下文长度按证据数量封顶，而不是永远保留固定百分比。真实 PPL 同时证明最终 original-K exact rerank 仍是在线稳定性的必要组成。

当前结果已经解决“完整原始 K/V 在 GPU 时，如何显著减少在线 attention 计算”的问题；尚未完成“只保留 5%-10% 总 GPU KV 时仍保持同样速度”的物理缓存系统。两项结果必须分开陈述。

## 8. 复现入口

- CUDA kernel：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/qabs_cuda_kernels.py`
- attention 主路径：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_head_top2_targeted_ppl_20260714.py`
- 融合流水线测速：`ymluo/projects/qwen3_top2_token_mechanism/src/benchmark_fused_qkmetric_pipeline.py`
- CUDA profiler：`ymluo/projects/qwen3_top2_token_mechanism/src/profile_fused_sampleq_pipeline.py`
- DP4A 逐项一致性验证：`ymluo/projects/qwen3_top2_token_mechanism/src/validate_dp4a_sampleq.py`
- 多精度级联分析：`ymluo/projects/qwen3_top2_token_mechanism/src/analyze_progressive_precision_rerank.py`
- CountCap 策略：`ymluo/projects/qwen3_top2_token_mechanism/src/count_capped_support_policy.py`
- CountCap 严格实验：`ymluo/projects/qwen3_top2_token_mechanism/scripts/run_countcap_ppl_20260722.sh`
- CountCap Full 配对：`ymluo/projects/qwen3_top2_token_mechanism/scripts/run_matched_countcap_ppl_20260722.sh`
- CountCap 自动汇总：`ymluo/projects/qwen3_top2_token_mechanism/src/summarize_support_scaling.py`
- 严格 Full/Ours 配对：`ymluo/projects/qwen3_top2_token_mechanism/scripts/run_matched_full_sparse_128k_20260722.sh`
- 自动汇总：`ymluo/projects/qwen3_top2_token_mechanism/src/summarize_fused_sampleq_matched.py`
