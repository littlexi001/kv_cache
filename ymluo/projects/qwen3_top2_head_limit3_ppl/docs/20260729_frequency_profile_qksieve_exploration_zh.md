# Frequency-Profile QKSieve 探索结果

## 1. 结论

本轮没有运行 LongBench。实验只使用已有 Q/K/V 轨迹、六主题短 PPL
窗口和独立 CUDA 微基准，目标是判断“历史召回频率”能否同时改善
QKSieve 的质量、索引大小和执行速度。

当前最好的平衡版本暂称 **Frequency-Profile QKSieve**：

1. 用 prompt 尾部 8 个 query 统计每个 KV head 的历史 top-2% 召回频率；
2. 以 32 token 为一块，对块内频率求和；
3. 频率最高的 15% 块保留原 QKSieve 的 per-head qMSE 高精度 profile；
4. 其余 85% 块统一使用 `[8, 4, 0, ..., 0]` profile；
5. 用 `uint16` 热块前缀和恢复每块地址，不保存逐块双 `int64` 基址；
6. 用 profile-specialized DP4A kernel 扫描索引；
7. 用无偏 sampled-quantile 产生变长候选，再从原始 FP16 K/V 做精确
   sparse attention；没有 router、训练、rerank、recent 或 Full fallback。

主要结果：

| 指标 | 结果 |
|---|---:|
| 六主题 sampled-threshold PPL 质量保持率 | **99.640%** |
| 三模型 held-out attention-mass 保持率 | **99.534%** |
| 辅助检索索引 / Full FP16 K+V | **5.519%** |
| 32K attention 子系统加速，中位数 | **4.346x** |
| 64K attention 子系统加速，中位数 | **6.911x** |
| 128K attention 子系统加速，中位数 | **8.659x** |

该版本已达到本轮“平均质量 99.5%+、比旧 QKSieve 更快、索引更小”的
探索目标，但还没有接入真实模型 decode kernel，因此不能把上述数字称为
整模型端到端加速。

## 2. 召回频率现象

Qwen3-4B、32K、体育和医学轨迹中：

| token 范围 | 覆盖全部历史召回事件 |
|---:|---:|
| 最热 0.1% | 4.56% |
| 最热 0.5% | 19.39% |
| 最热 1% | 33.49% |
| 最热 2% | 52.79% |
| 最热 4% | 73.43% |

全部 token 中只有约 15.00% 曾被 prompt-tail query 选中，85.00% 从未被
选中。这说明频率高度稀疏，但不能直接把低频 token 删除：每两步只全面
刷新一次、其余步骤复用热门 token 时，attention-mass 保持率只有约
96.22%，达不到 99.5%。

因此频率不用于跳过 token，而用于决定近似索引的数值精度。

## 3. 方法

令第 `h` 个 KV head、第 `i` 个 token 在校准 query 集合中的召回次数为
`f[h,i]`。对长度为 32 的块求和：

```text
F[h,b] = sum(f[h,i]), i in block b
```

每个 head 独立选取 `F` 最高的 15% 块作为 hot block。Hot block 使用
QKSieve 原有的 qMSE profile；cold block 固定使用两个有效 band：

```text
cold profile = [8, 4, 0, 0, 0, 0, 0, 0]
```

选择 `[8,4]` 而非码率更低的 `[4,4,1]` 是计算成本感知的决定：
`[8,4]` 只解码两个 band，后者要解码三个 band。更少的 bit 不必然更快。

逐块目录使用热块前缀和：

```text
hot_before[b] = number of hot blocks before block b
```

块基址由 `b`、`hot_before[b]`、冷热 stride 直接计算。目录从原型的
`profile + code_base + scale_base = 17 B/block` 降到约
`2 B/block`，即约 `0.063 B/token/head`。

CUDA 中对常见 `[8,4]`、`[8,1,1,1]`、`[4,4,4]` 和 `[4,4,1]`
profile 使用固定展开的 DP4A 路径，避免每个 token 循环 8 个 band 并
动态判断 bit 数。

## 4. 质量

### 4.1 六主题 32K PPL

协议：Qwen3-4B-Instruct、每主题一个独立 32K 窗口、64 个 held-out
token。以下结果使用与 CUDA 相同的 sampled-quantile 候选协议。

| 主题 | Full PPL | Ours PPL | 质量保持率 |
|---|---:|---:|---:|
| sports | 10.2995 | 10.4205 | 98.838% |
| medicine | 14.9265 | 15.1189 | 98.728% |
| space | 24.0565 | 23.9307 | 100.525% |
| computer | 36.0186 | 36.0049 | 100.038% |
| politics | 15.4847 | 15.1975 | 101.890% |
| religion | 9.0622 | 9.2588 | 97.876% |
| **几何汇总** | **16.2909** | **16.3497** | **99.640%** |

平均 KL 为 `0.008815`，top-1 agreement 为 `95.83%`。平均每 head
实际使用 `1263.1 / 32000 = 3.947%` token。单主题最差值没有达到
99.5%，当前结论仅是跨主题宏平均达到 99.5%。

作为对照，精确全局 top-k 版本质量保持率为 `99.706%`。从
`99.706%` 到 `99.640%` 是 sampled threshold 的实际代价。

### 4.2 跨模型轨迹

Qwen3-4B、Llama-3.1-8B 和 Qwen2.5-7B 的体育/医学 held-out
Q/K 轨迹共 1680 个 layer-step 条件上，cold `[8,4]`、hot 15% 的
attention-mass 保持率为 `99.534%`。这些是轨迹筛选指标，不代替完整
下游 benchmark。

## 5. CUDA 速度

协议：RTX 3090、32 query heads、8 KV heads、head dimension 128。
完整 attention 子系统包含近似索引扫描、sampled threshold、候选压缩、
原始 K 的精确 QK、稀疏 softmax 和 V 聚合；不含索引构建和模型的 MLP。
下表是 5 个随机种子的中位数。

| 历史长度 | Full SDPA | Ours | 加速 | 实际候选比例 |
|---:|---:|---:|---:|---:|
| 32K | 0.6291 ms | 0.1449 ms | **4.346x** | 3.858% |
| 64K | 1.2402 ms | 0.1792 ms | **6.911x** | 2.003% |
| 128K | 2.4362 ms | 0.2814 ms | **8.659x** | 0.994% |

旧的未专门化 QKSieve 分别为 `3.708x / 5.769x / 8.257x`。新版本
相对旧版本的完整子系统提速约为 `17.2% / 19.8% / 4.9%`。

用既有 128K 延迟分解中 Full decode 的 attention 占比 77.2% 做
Amdahl 估计，`8.659x` attention 加速对应约 `3.15x` 整模型加速。
这是外推，不是本轮实测；正式结果必须在接入模型后重新测量。

## 6. 存储口径

`5.519%` 指辅助检索索引相对于 Full FP16 K+V 的大小，不表示总 KV
只剩 5.519%。当前 CUDA 原型仍让原始精确 K/V 100% 常驻 GPU，用于
最终 sparse attention：

```text
GPU resident storage = 100% exact K/V + 5.519% retrieval index
```

本方法减少的是每步实际访问和计算的 KV token，以及相对旧 QKSieve 的
辅助索引。若论文要声称降低总 GPU KV memory，下一阶段必须把未激活的
精确 K/V 放到 CPU、分层显存或压缩后备存储，并单独测传输开销。

## 7. 被否定的方向

| 方向 | 结果 |
|---|---|
| 直接复用高频 token，隔步跳过全扫描 | mass 约 96.22%，质量不足 |
| 每个 32-token 块强制局部 quota | mass 约 80.89%，重要 token 在块间不均匀 |
| 给低码率分数加频率先验 | 最好约 99.11%，频率会污染当前 query 排序 |
| 热/冷双索引、双 top-k 再合并 | CUDA 慢约 17%--20% |
| 仅按频率重排物理块 | 目录更小，但扫描没有提速 |
| 全部 token 固定 `[4,4,1]` | Qwen2.5 跨模型质量明显不足 |
| qMSE profile codebook 只按校准误差选 | held-out mass 不足 99.5% |

## 8. 可复现文件

主要实现：

```text
src/analyze_qksieve_frequency_hotset_20260729.py
src/analyze_qksieve_frequency_tiered_index_20260729.py
src/mixedblock_spectral_cuda_20260729.py
src/variablebit_spectral_cuda_20260727.py
src/benchmark_qksieve_mixedblock_cuda_20260729.py
src/run_head_top2_targeted_ppl_20260714.py
src/run_direct_countcap_denseprompt_ppl_20260725.py
```

关键结果：

```text
results/20260729_qksieve_frequency_hotset_32k/summary.json
results/20260729_qksieve_computeaware_profiles_multimodel_32k/summary.json
results/20260729_qksieve_computeaware_sampled_six_topic_ppl_32k/case_summary.json
results/20260729_qksieve_computeaware_fixed84_hot15_cuda_multiseed/
```

复现实验脚本：

```text
scripts/launch_qksieve_computeaware_sampled_six_topic_ppl_gpu5_20260729.sh
```

## 9. 下一步

下一步不应先跑 LongBench，而应：

1. 把 mixed-block prefix-directory kernel 接入真实逐层 decode；
2. 在 32K、64K、128K 各做一次 256--512 token 严格 Full/Ours 配对；
3. 同时报 attention、整模型 steady decode、索引构建和 break-even；
4. 若 128K 实测接近 `3.15x` 且六主题质量仍超过 99.5%，再冻结方法；
5. 然后才进入 LongBench/RULER 和多模型论文表格。

