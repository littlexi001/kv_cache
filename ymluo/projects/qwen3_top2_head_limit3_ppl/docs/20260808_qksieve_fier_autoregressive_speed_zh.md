# QKSieve 与 FIER 的真实生成速度：GQA 基线更正版

## 重要更正

最初实验使用 Transformers 4.53.1 的标准 `sdpa_attention_forward` 作为 Full。该实现先用 `repeat_kv` 将 Qwen3 的 8 个 KV heads 展开为 32 个 query heads，再调用 SDPA。展开结果随后被 `.contiguous()`，因此会产生真实的 K/V 复制和额外显存流量。

这不是公平的 GQA Full 基线。旧 Full 的 159--208 ms/token 以及由它得到的约 3x 整模型加速全部作废。旧数据仅保留用于审计，不能进入论文主表或摘要。

更正后的 Full 在单 token decode 中直接调用 PyTorch SDPA，并设置 `enable_gqa=True`，始终以 8 个 KV heads 保存和读取历史 K/V。64K prefill 仍使用标准实现，因为 PyTorch 原生 GQA 在多 token prefill 上会退化到高显存后端；本文实验只统计已有 KV 前缀上的 decode。

## 实现核验

Qwen3-4B-Instruct-2507 配置为 32 个 query heads、8 个 KV heads、head dimension 128、36 层。

64K 单层微基准：

| Full attention 实现 | 延迟 | 峰值临时显存 |
|---|---:|---:|
| HF `repeat_kv` 后 SDPA | 3.881 ms | 1280 MB |
| 原生 GQA SDPA | **0.350 ms** | **256 MB** |

两条路径输出的最大绝对误差为 `1.5e-5`。4K 整模型校验中，二者生成的 32 个 token 完全一致。64K 正式运行中，每次 256-step decode 都记录到 `9216 = 36 x 256` 次原生 GQA attention 调用，四张 GPU 的生成哈希也完全一致。

## 测试口径

- 模型：Qwen3-4B-Instruct-2507，FP16，RTX 3090。
- 固定 65,536-token 前缀，连续 greedy 生成 256 token。
- 每一步将当前 `argmax` 输出反馈为下一步输入，不使用 teacher forcing。
- 四轮 GPU 轮换；QKSieve、FIER 和正确 Full 均覆盖物理 GPU0--3。
- QKSieve 完全关闭 ValueSketch 补偿，保留 1,280 token/head。
- FIER 使用 RTN-1、group size 32、ragged attention `split=8`。
- 稳态统计排除前 16 步；含索引结果从索引构建开始计时，不包含 prefill。

## 更正后的稳态结果

| 方法 | token/head | 跨卡中位延迟 | 同 GPU 对正确 Full 加速中位数 |
|---|---:|---:|---:|
| **Full native GQA** | 65,536 | 54.443 ms/token | 1.000x |
| QKSieve no-Value | 1,280 | **51.357 ms/token** | **1.036x** |
| FIER RTN-1 g32 | 1,280 | 55.530 ms/token | 0.982x |
| FIER RTN-1 g32 | 512 | 55.450 ms/token | 0.984x |

QKSieve 在四张卡中的三张上有 3.5%--8.4% 的稳态提升，但在 GPU1 上为 0.830x；四卡同 GPU 配对中位数为 1.036x，几何平均为 0.991x。因此目前只能说 QKSieve 在典型卡上略快，不能声称获得稳定、显著的整模型加速。FIER 两个预算都略慢于正确 Full。

## 含索引成本

下表为四卡平衡轮换后的中位 ms/generated-token，越低越好。

| 生成长度 | Full native GQA | QKSieve top1280 | FIER top1280 | FIER top512 |
|---:|---:|---:|---:|---:|
| 16 | **54.356** | 130.834 | 95.138 | 94.977 |
| 32 | **54.328** | 91.142 | 75.542 | 75.660 |
| 64 | **54.174** | 71.331 | 65.682 | 65.640 |
| 128 | **54.217** | 61.364 | 60.708 | 60.666 |
| 256 | **54.553** | 56.431 | 58.365 | 58.167 |

在已测试的 256-token 生成范围内，三种稀疏路径都没有摊平索引成本。按跨卡中位数线性估计，QKSieve 约需要 420 个生成 token 才可能追回固定成本；GPU1 因稳态本身更慢，不存在回本点。缓存索引被多轮对话或 Agent 直接复用时，可以消除固定成本，但稳态收益仍只有几个百分点。

## 质量结果是否受影响

不受影响。此前 teacher-forced PPL、Top-1 一致率、LongBench 和 RULER 的质量结果比较的是相同 Full logits 与稀疏 logits，GQA 展开与原生 GQA 的数值误差很小，4K/64K greedy 哈希也保持一致。

需要作废的是所有以展开 Full 延迟为分母的速度结论，包括：

1. 约 3x 的 64K 整模型稳态生成加速。
2. 16--256 token 时已经达到 1.2x--3.0x 的含索引加速。
3. QKSieve 在约 160--170 token 后超过 FIER且获得显著 Full 加速的表述。

QKSieve 与 FIER 的直接比较仍显示：QKSieve 的典型稳态延迟比 FIER top1280 低约 7%，但这个差值相对正确 Full 已经很小。

## 当前判断

当前方法的质量和索引压缩结论仍然成立，但 64K 整模型速度尚不足以支撑论文中的强加速主张。下一步不应继续调候选数，而应针对 selector、top-k、索引构建和 kernel launch 做融合；同时在 128K/256K 上用原生 GQA Full 重新寻找长度交叉点。

## 结果文件

- 正确 Full：`results/20260808_full_native_gqa_decode_64k/`
- QKSieve/FIER：`results/20260808_qksieve_fier_autoregressive_64k/`
- 原始 HF 展开 Full 仅作为错误基线审计保留在后一目录中。
