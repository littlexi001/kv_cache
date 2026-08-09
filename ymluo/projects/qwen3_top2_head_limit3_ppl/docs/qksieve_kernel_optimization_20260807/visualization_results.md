# 最终长度扫描结果

## 稳态与固定开销

RTX 3090、Qwen3-4B-Instruct-2507，均为 3 次独立进程中位数：

| 上下文 | Full ms/token | QKSieve ms/token | 稳态加速 | Warm fixed | Cold fixed | Warm 回本 | Cold 回本 | PPL 质量保持 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 44.99 | 56.25 | 0.81x | 0.323 s | 0.602 s | 不存在 | 不存在 | 100.001% |
| 16K | 52.19 | 55.29 | 0.95x | 0.288 s | 0.586 s | 不存在 | 不存在 | 99.392% |
| 32K | 87.76 | 56.38 | 1.56x | 0.316 s | 0.675 s | 10.1 token | 21.5 token | 100.087% |
| 64K | 169.23 | 72.13 | 2.35x | 0.443 s | 0.937 s | 4.7 token | 10.4 token | 99.398% |
| 128K | 301.79 | 90.52 | 3.32x | 0.585 s | 1.415 s | 2.8 token | 6.7 token | 99.998% |

64K 的三次稳态结果为 2.23x、2.89x、2.35x，说明并行实验有明显资源干扰；表中使用中位数，但论文最终数字必须独占 GPU 复测。32K 与 128K 相对稳定。

## 输出长度对总 decode 的影响

| 上下文 | Warm G=8 | Warm G=16 | Warm G=32 | Warm G=64 | Cold G=8 | Cold G=16 | Cold G=32 | Cold G=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.43x | 0.56x | 0.66x | 0.73x | 0.33x | 0.46x | 0.59x | 0.68x |
| 16K | 0.57x | 0.71x | 0.81x | 0.88x | 0.41x | 0.57x | 0.71x | 0.82x |
| 32K | 0.92x | 1.15x | 1.32x | 1.43x | 0.62x | 0.89x | 1.13x | 1.31x |
| 64K | 1.31x | 1.68x | 1.96x | 2.14x | 0.86x | 1.24x | 1.61x | 1.91x |
| 128K | 1.84x | 2.37x | 2.77x | 3.02x | 1.13x | 1.68x | 2.23x | 2.67x |

这张表直接回答短输出问题：32K 冷请求约 22 个输出 token 后才开始获益；128K 即使冷请求也约 7 个 token 回本。8K/16K 不存在摊销回本点。

## 已验证的无损优化

### Fused W_o-metric Value append

- 27,648 个 packed 值逐位一致。
- Value append 阶段：17.27 降至 4.72 ms/token，阶段加速 3.66x。
- 真实模型稀疏 decode：32K、64K、128K 分别约再快 1.21x、1.21x、1.33x。

### 跨层 batched qMSE allocation

- qMSE allocation：102.5 ms 降至 21.1 ms，阶段加速 4.86x。
- 36/36 层 allocation 和 active packed-Key hash 一致。
- 32K 固定开销：0.413 秒降至 0.346 秒，约快 1.19x；steady decode 基本不变。

### 直接读取预分配 KV cache

旧的精确候选 attention 接口要求 K/V 完全连续。真实推理中的 K/V 是从容量大于当前历史长度的预分配 cache 切出的视图，因此旧接口会在每个 token、每一层先复制完整历史 K/V。新 kernel 显式读取 batch/head stride，并复用输出 workspace；候选集合和数值公式不变。

只测候选消费阶段、物理 cache 容量固定为 262,144 token 时：

| 逻辑历史 | 旧分配并复制 K/V | stride-aware 标量 | 阶段加速 | 最大绝对误差 |
|---:|---:|---:|---:|---:|
| 32K | 0.433 ms | 0.100 ms | 4.32x | 0 |
| 64K | 0.742 ms | 0.093 ms | 8.00x | 0 |
| 128K | 1.396 ms | 0.093 ms | 15.05x | 0 |

真实 Qwen3-4B 模型 A/B 的关键结果：

| 上下文 | 旧路径稳态 | stride-aware 标量稳态 | 端到端 decode 改善 | Full 对比 | PPL |
|---:|---:|---:|---:|---:|---:|
| 32K | 61.825 ms/token | 60.487 ms/token | 1.022x | 1.415x -> 1.448x | 进程间候选存在微小非确定性，不能单独归因 |
| 128K | 88.298 ms/token | 80.178 ms/token | 1.101x | 3.391x -> 3.740x | 两路径均为 99.6315% Full |

128K 是两次独立进程中位数，并使用双 RTX 3090 容纳 dense prefill。它证明删除整段 K/V 复制能在不改变 PPL 的情况下兑现约 10.1% 的整模型稳态收益。32K 收益只有约 2.2%，说明短上下文更受固定 runtime 和模型底座限制。

warp-tiled 版本在连续张量微基准中让候选 attention kernel 再快约 1.19--1.27 倍，但在真实模型 A/B 中没有稳定超过 stride-aware 标量版本。主方法因此冻结为标量版本，tiled 版本只作为系统消融。

## 图

- `figures/decode_latency_and_speedup.{pdf,png}`：逐 token 延迟与稳态加速。
- `figures/warm_speed_surface.{pdf,png}`：warm 生命周期二维加速。
- `figures/cold_speed_surface.{pdf,png}`：cold 生命周期二维加速。
- `figures/break_even_tokens.{pdf,png}`：回本生成长度。

图和表必须与 LongBench 的实际输出长度一起解读。旧的对齐 LongBench 运行中，2WikiMQA、HotpotQA、Musique、NarrativeQA、PassageCount 和 PassageRetrieval-en 的平均输出仅约 3--7 token；GovReport 和 MultiNews 的约 440--488 token 把全局平均值显著抬高。因此不能用平均输出长度代替分任务请求时延。
