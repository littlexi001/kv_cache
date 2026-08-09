# Prefill 增量建索引与 sampled-quantile 融合实验

## 结论

本轮同时验证了两个方向：

1. 在 prefill 期间增量或异步构建 Key-PCA INT4 索引。
2. 融合 sampled-quantile 全表扫描与后续候选消费。

最终结论如下：

- **同步增量建索引有效地消除了首个稀疏 decode step 的建索引突刺。**
- 在 warm 运行状态下，16K、生成 64 token 时，完整总时间约为 Full 的 **1.05x**；8K 仍只有约 **0.85x**。
- **同 GPU 异步 stream 没有进一步加速。** Prefill 已接近占满 RTX 3090，索引 stream 与模型 stream 争抢计算资源。
- sampled-quantile 扫描中取消无用 proxy-score 写回，只带来 **0%–1.2%** 的扫描微内核收益。
- 扫描时直接计算候选 exact-QK 的融合内核没有稳定收益，并改变了边界附近候选，不能进入主方法。
- DP4A 扫描保持候选集合完全一致，微流水线快约 **2%–4%**，但整模型 decode 仅在噪声范围内变化。

因此，当前可保留的工程改动是 **prefill 同步增量建索引**；async、exact-scan 和 DP4A 暂不作为主方法卖点。

## 实现

### Prefill 增量索引

Prefill 每处理完一个 2048-token chunk，就执行一次索引回调：

1. 第一个 chunk 的 K 用于估计每层、每个 KV head 的 48 维 PCA 基。
2. 只投影尚未索引的新增 K。
3. 立即打包为 chunk-major PCA48 INT4 和 log-scale16。
4. 后续 chunk 继续追加，不重新处理旧 token。
5. Dense question suffix 完成后，再补入 suffix 的 K。

该路径通过以下方法名启用：

```text
countcap_fullprompt_keypca_direct_qkvfused_prefillindex
```

新增统计字段：

```text
index_build_seconds
```

`prefill_seconds` 和 `total_seconds` 都包含实际索引成本；因此不能通过移动计时边界虚构端到端加速。

### 异步索引

异步版本使用专用 CUDA stream：

```text
countcap_fullprompt_keypca_direct_qkvfused_asyncprefillindex
```

第 j 个模型 prefill chunk 与第 j-1 个 chunk 的 PCA/INT4 建索引尝试并行。CUDA event 保证索引读取 K 前，模型已经生成对应 cache。

### Quantile 扫描融合

原 sampled-quantile 路径本来已经在一次全表扫描内完成：

1. PCA48 INT4 近似分数。
2. sampled quantile 阈值判断。
3. 候选索引压紧。

本轮进一步测试了：

- 不再写回后续未使用的 proxy score。
- 在阈值扫描内直接消费候选并计算 exact-QK。
- 使用 DP4A 加速 INT4 点积，再交给现有 QK+V fused kernel。

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct |
| GPU | RTX 3090，仅使用 GPU 0–3 |
| 样本 | LongBench GovReport，固定 sample offset 115 |
| Prompt | 8,192 / 16,000 token |
| 生成长度 | 64 token |
| Prefill chunk | 2,048 token |
| 重复 | 3 次 |
| 公平性 | 轮换 Full、标准 fused、实验方法的执行顺序 |

## Prefill 索引结果

三次轮换实验的中位数如下。速度均相对同一次 repeat 的 Full 计算。

| 长度 | 方法 | Online | Online speed | Total | Total speed |
|---:|---|---:|---:|---:|---:|
| 8K | Full | 2.571 s | 1.000x | 5.269 s | 1.000x |
| 8K | 同步 prefill-index | 2.969 s | 0.866x | 6.673 s | 0.853x |
| 16K | Full | 3.944 s | 1.000x | 10.277 s | 1.000x |
| 16K | 同步 prefill-index | 2.985 s | 1.323x | 10.316 s | 1.026x |

16K warm allocator 状态下：

| 方法 | Total | 相对 Full |
|---|---:|---:|
| Full | 约 10.27 s | 1.000x |
| 同步 prefill-index | 约 9.81 s | 约 1.05x |

冷启动时，索引构建约为 1.1 秒；warm 状态约为 0.51 秒。因此必须同时报告冷、暖状态，不能只选择后执行的方法。

本实验中，prefill-index、旧 sparse 和对应 DP4A 版本的 prediction 完全一致。这里只验证了一个固定样本，不能替代完整 LongBench 质量表。

## Async 结果

控制稀疏索引是冷启动还是 warm 状态后：

| 长度 | 状态 | 同步 Total | Async Total | 结论 |
|---:|---|---:|---:|---|
| 8K | warm | 6.133 s | 6.134 s | 相同 |
| 8K | cold | 约 7.06 s | 7.20 s | Async 略慢 |
| 16K | warm | 9.818 s | 约 9.83 s | 相同 |
| 16K | cold | 约 10.65 s | 10.88 s | Async 略慢 |

说明当前 prefill 和索引构建无法在同一张 3090 上有效重叠。两个 stream 只是共享并争抢 SM、内存带宽和 cuBLAS 资源。

## Quantile 微基准

### 取消 proxy-score 写回

候选 count、boundary、overflow 和候选集合完全一致。

| 长度 | 写 proxy | 不写 proxy | 加速 |
|---:|---:|---:|---:|
| 8K | 约 0.088 ms | 约 0.087 ms | 1.01x |
| 16K | 约 0.100 ms | 约 0.099 ms | 1.01x |

收益太小，说明扫描耗时主要来自 INT4 解码和点积，而不是 proxy-score 写回。

### Scan + exact-QK

该内核把阈值判断和 exact-QK 放在一次 scan 中，再单独执行 V 聚合。

结果：

- 8K 相对同候选 DP4A 流水线更慢。
- 16K 没有稳定收益，多次测量在略快和明显更慢之间波动。
- 由于浮点累加顺序不同，边界附近候选集合不完全一致。
- 8K 随机数值测试中，最终输出最大差异达到约 0.0199。

因此该路径被否决。

### DP4A

DP4A 与原标量扫描的候选 count 和候选集合完全一致。

| 长度 | 原微流水线 | DP4A 微流水线 | 微加速 |
|---:|---:|---:|---:|
| 8K | 0.279 ms | 0.268 ms | 1.04x |
| 16K | 0.446 ms | 0.438 ms | 1.02x |

但整模型三次配对结果为：

- 8K decode 中位约快 0.4%。
- 16K decode 中位约慢 1%。
- 所有 prediction 一致。

微收益被其他算子和运行波动淹没，暂不替换默认路径。

## 代码与结果

主要实现：

```text
src/run_controlled_public_kv_benchmark_v1.py
src/run_head_top2_targeted_ppl_20260714.py
src/run_sample_calibrated_longbench_20260717.py
src/qabs_cuda_kernels.py
```

验证脚本：

```text
src/validate_sampled_quantile_no_proxy_20260724.py
scripts/launch_countcap_prefillindex_8k16k_4gpu_20260724.sh
scripts/launch_countcap_asyncprefillindex_8k16k_2gpu_20260724.sh
scripts/launch_countcap_dp4a_prefillindex_8k16k_2gpu_20260724.sh
```

结果目录：

```text
results/20260724_countcap_prefillindex_g64_4gpu
results/20260724_countcap_asyncprefillindex_g64_2gpu
results/20260724_countcap_dp4a_prefillindex_g64_2gpu
```

## 下一步

继续优化 quantile scan 的价值不高。更值得做的是：

1. 把 PCA 投影和 INT4 打包融合进 prefill 的 K projection 或其 epilogue，避免额外读取完整 K。
2. 使用离线校准或跨请求固定 PCA 基，使所有 chunk 从第一个 token 起即可直接打包。
3. 在 32K/64K/128K 上测试 prefill-index，因为索引固定成本在更长上下文和更长生成下更容易摊薄。
4. 在独立 LongBench 子集上验证“首个 chunk 学 PCA 基”是否保持完整 prompt PCA 的质量。
